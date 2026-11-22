"""批次与拆检记录的内存存储（含拟合缓存与单飞合并）。

并发与一致性
============
* 拆检记录保存在单进程内存中，所有读写经同一把 ``asyncio.Lock`` 串行化。
  两个浏览器窗口同时登记/更正/撤回（含同一台机器）时，两笔修改都不丢失，
  接口返回的机器明细、实检数与估数始终来自持锁期间的同一份快照。
* 最小二乘是纯 CPU 计算，被卸载到线程池执行且 **不持锁**：拟合进行期间
  登记、更正、撤回、换批均可照常生效。
* 拟合输入（归一化混淆矩阵 ``P`` 与未检初判分布 ``q``）相同即同一最小
  二乘问题。同批最近一次成功拟合按输入键缓存，快照一致时直接复用
  ``pi`` 与 ``sse``；并发请求遇到相同输入时共享同一个在途任务
  （single-flight），求解只执行一次。
* 每次修改递增快照版本号 ``version``，每次换批递增世代号
  ``generation`` 并清空缓存与在途登记表：旧世代的迟到拟合结果不会
  写回新批次，也不会覆盖窗口中已显示的较新记录（前端按版本号丢弃
  迟到响应）。
* 实检数、未检折算、机器清单始终按各自请求时的快照计算，复用的只有
  与快照解耦的拟合输出。

注意：该实现与 uvicorn 单 worker 绑定（docker-compose 已固定
``--workers 1``）；多进程部署需替换为数据库等共享存储。
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from .estimation import (
    CATEGORIES,
    CAT_INDEX,
    FIT_ACTUALS,
    FIT_CACHE,
    FIT_FAILED,
    FIT_GATED,
    FIT_INFLIGHT,
    FIT_SOLVED,
    FitError,
    FitOutput,
    SnapshotStats,
    assemble_result,
    build_snapshot_stats,
    fit_input_key,
    fit_proportions,
)


class StoreError(Exception):
    """业务校验错误，映射为 HTTP 400。"""


def run_fit(P, q):
    """最小二乘求解的模块级入口（同步，线程池内执行）。

    单独成函数便于测试替换为计数/阻塞/抛错替身；生产实现即
    :func:`app.estimation.fit_proportions`。
    """
    return fit_proportions(P, q)


class BatchStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._machines: list[dict] = []
        self._id_index: dict[str, int] = {}
        # 快照版本：任何一次登记/更正/撤回后递增；换批归零。
        self._version = 0
        # 批次世代：换批递增，用于隔离旧批次的迟到拟合结果。
        self._generation = 0
        # 同批最近一次成功拟合：fit_input_key -> FitOutput。
        self._fit_cache: dict[tuple, FitOutput] = {}
        # 在途拟合：fit_input_key -> asyncio.Future（单飞合并）。
        self._inflight: dict[tuple, asyncio.Future] = {}
        # CPU 求解线程池：拟合不占事件循环、不持锁。
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="lstsq-fit"
        )

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    # ------------------------------------------------------------------
    # 持锁的基础操作
    # ------------------------------------------------------------------
    def reset_locked(self, machines: list[dict]) -> None:
        """载入新批次（调用方须持锁）：清空记录、缓存与在途登记。"""
        self._machines = machines
        self._id_index = {m["id"]: i for i, m in enumerate(machines)}
        self._version = 0
        self._generation += 1
        self._fit_cache.clear()
        # 旧在途任务不取消：其结果仅会尝试写回已被换批清空的旧字典，
        # 世代校验保证它不会影响新批次；这里直接丢弃登记引用。
        self._inflight.clear()

    def is_empty_locked(self) -> bool:
        return not self._machines

    def apply_inspection_locked(
        self, machine_id: str, category: str | None
    ) -> None:
        """登记/更正（三类之一）或撤回（None）一台机器（持锁调用）。"""
        if machine_id not in self._id_index:
            raise StoreError(f"机器编号 {machine_id} 不存在于当前批次")
        if category is not None and category not in CAT_INDEX:
            raise StoreError(
                f"非法实检类别 {category!r}，允许：{', '.join(CATEGORIES)}"
            )
        machine = self._machines[self._id_index[machine_id]]
        machine["inspected"] = category
        self._version += 1

    def _capture_locked(self) -> tuple[SnapshotStats, list[dict], int]:
        """持锁抓取当前快照统计、机器清单副本与世代号。"""
        stats = build_snapshot_stats(self._machines, version=self._version)
        machines = [dict(m) for m in self._machines]
        return stats, machines, self._generation

    # ------------------------------------------------------------------
    # 对外的快照构建（异步）：快速统计持锁，重活在线程池且不持锁
    # ------------------------------------------------------------------
    async def build_snapshot(self) -> dict:
        """构建当前批次快照；拟合按需求解、缓存或单飞复用。

        循环不变量：返回的拟合输出必须与 *当前* 快照的拟合键一致。
        若等待在途拟合期间快照被登记/更正/撤回推进，则放弃迟到结果、
        按新键重新处理，保证迟到拟合不会覆盖较新记录。
        """
        while True:
            async with self._lock:
                stats, machines, generation = self._capture_locked()

                # 全检：实数，无需拟合。
                if stats.n_uninspected == 0:
                    result = assemble_result(stats, fit_source=FIT_ACTUALS)
                    return self._payload(machines, result)

                # 门控撤下：缺样本 / 不满秩，纯统计即可，不进入拟合。
                if stats.gate_reason is not None:
                    result = assemble_result(stats, fit_source=FIT_GATED)
                    return self._payload(machines, result)

                key = fit_input_key(stats.P, stats.q)

                # 复用同批最近一次成功拟合：输入与当前快照完全相同。
                cached = self._fit_cache.get(key)
                if cached is not None:
                    result = assemble_result(
                        stats, fit=cached, fit_source=FIT_CACHE
                    )
                    return self._payload(machines, result)

                # 单飞：相同拟合输入的并发请求只求解一次。
                future = self._inflight.get(key)
                if future is None:
                    future = asyncio.ensure_future(
                        self._run_fit(generation, key, stats.P, stats.q)
                    )
                    self._inflight[key] = future
                    source = FIT_SOLVED
                else:
                    source = FIT_INFLIGHT

            # 锁外等待（可能由其他请求发起），期间登记/更正/撤回不受阻塞。
            try:
                fit = await future
            except FitError as exc:
                # 失败不写缓存、在途登记已摘除，同一输入下次请求可重试。
                async with self._lock:
                    stats_now, machines_now, generation_now = (
                        self._capture_locked()
                    )
                    key_now = (
                        fit_input_key(stats_now.P, stats_now.q)
                        if stats_now.P is not None and stats_now.q is not None
                        else None
                    )
                    # 快照已推进或已换批：迟到的失败提示同样不能覆盖
                    # 较新记录，回到循环按新快照重新处理。
                    if key_now != key or generation_now != generation:
                        continue
                result = assemble_result(
                    stats_now,
                    fit_source=FIT_FAILED,
                    fit_error=str(exc),
                )
                return self._payload(machines_now, result)

            # 拟合完成后必须回到锁内核对快照是否仍是同一拟合输入。
            async with self._lock:
                stats_now, machines_now, generation_now = self._capture_locked()
                if stats_now.P is None or stats_now.q is None:
                    continue  # 已全检或进入门控，循环走对应分支
                if fit_input_key(stats_now.P, stats_now.q) != key:
                    continue  # 快照推进：迟到结果不得覆盖，按新键重来
                if generation_now != generation:
                    continue  # 已换批：旧世代结果一律丢弃
                result = assemble_result(
                    stats_now, fit=fit, fit_source=source
                )
                return self._payload(machines_now, result)

    async def _run_fit(
        self,
        generation: int,
        key: tuple,
        P,
        q,
    ) -> FitOutput:
        """线程池执行最小二乘；完成后做世代/在途校验并登记成功缓存。

        失败时通过 finally 摘除在途登记且不写缓存，保证同一输入的后续
        请求可以重新求解（可重试）。
        """
        loop = asyncio.get_running_loop()
        try:
            fit = await loop.run_in_executor(self._executor, run_fit, P, q)
        except BaseException:
            async with self._lock:
                # 仅当登记的还是本任务时摘除，避免误删后来的重试任务。
                if self._inflight.get(key) is asyncio.current_task():
                    self._inflight.pop(key, None)
            raise
        async with self._lock:
            if self._inflight.get(key) is asyncio.current_task():
                self._inflight.pop(key, None)
            # 换批后旧世代结果不得写回；同世代才缓存成功结果。
            if generation == self._generation:
                self._fit_cache[key] = fit
        return fit

    @staticmethod
    def _payload(machines: list[dict], result) -> dict:
        return {
            "machines": machines,
            "estimate": result.to_payload(),
        }

    async def update_inspection(
        self, machine_id: str, category: str | None
    ) -> dict:
        """登记/更正/撤回的对外入口：先落盘修改，再构建最新快照。"""
        async with self._lock:
            self.apply_inspection_locked(machine_id, category)
        return await self.build_snapshot()


store = BatchStore()
