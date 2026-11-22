"""拟合缓存 / 单飞 / 快照版本 / 失败重试 / 换批隔离的测试。"""
from __future__ import annotations

import asyncio
import threading

import pytest
from httpx import ASGITransport, AsyncClient

from app import storage as storage_mod
from app.estimation import (
    FIT_CACHE,
    FIT_FAILED,
    FIT_GATED,
    FIT_INFLIGHT,
    FIT_SOLVED,
    FitError,
    fit_proportions,
)
from app.main import app
from app.storage import store

CATS = ("power", "drive", "control")


@pytest.fixture(autouse=True)
async def _reset_store():
    async with store.lock:
        store.reset_locked([])
    yield
    async with store.lock:
        store.reset_locked([])


def _csv(n_per_initial: int = 6) -> bytes:
    rows = ["机器编号,初判"]
    mid = 1
    for cat in CATS:
        for _ in range(n_per_initial):
            rows.append(f"M{mid},{cat}")
            mid += 1
    return "\n".join(rows).encode("utf-8")


async def _upload(client: AsyncClient) -> dict:
    r = await client.post(
        "/api/batch",
        files={"file": ("b.csv", _csv(), "text/csv")},
    )
    assert r.status_code == 200
    return r.json()


async def _make_full_rank(client: AsyncClient) -> None:
    """登记 18 台，构造行线性无关（行列式非零）的混淆计数矩阵，保证 P 满秩。

    机器 M1-M6 初判 power、M7-M12 初判 drive、M13-M18 初判 control。
    计数矩阵（行=实检，列=初判）为 [[2,1,0],[0,2,1],[1,0,3]]，
    行列式 13 ≠ 0，因此归一化后的 P 秩为 3。
    """
    plan = {
        "M1": "power", "M2": "power", "M7": "power",
        "M8": "drive", "M9": "drive", "M13": "drive",
        "M3": "control", "M14": "control", "M15": "control",
        "M16": "control", "M4": "power", "M10": "drive",
    }
    for mid, cat in plan.items():
        r = await client.post(
            f"/api/machines/{mid}/inspect", json={"category": cat}
        )
        assert r.status_code == 200


async def _wait_version(target: int) -> None:
    """轮询存储内的快照版本（不经过会阻塞在拟合上的 HTTP 接口）。"""
    for _ in range(250):
        async with store.lock:
            current = store._version
        if current >= target:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"等待版本 {target} 超时，当前 {current}")


class _PatchFit:
    """临时替换 storage.run_fit 的上下文管理器（线程池内为同步调用）。"""

    def __init__(self, fake):
        self._fake = fake
        self._real = None

    def __enter__(self):
        self._real = storage_mod.run_fit
        storage_mod.run_fit = self._fake
        return self

    def __exit__(self, *exc):
        storage_mod.run_fit = self._real


# ---------------------------------------------------------------------
# 1. 缓存复用：同输入重复快照只求解一次，π/SSE 完全一致
# ---------------------------------------------------------------------
async def test_identical_snapshot_reuses_fit() -> None:
    transport = ASGITransport(app=app)
    calls = {"n": 0}
    real_fit = fit_proportions

    def counting_fit(P, q):
        calls["n"] += 1
        return real_fit(P, q)

    with _PatchFit(counting_fit):
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            await _upload(c)
            await _make_full_rank(c)
            # 登记过程中已求解并缓存；清掉后从干净状态验证缓存链路。
            async with store.lock:
                store._fit_cache.clear()
            calls["n"] = 0

            r1 = await c.get("/api/batch")
            e1 = r1.json()["estimate"]
            assert e1["fit_source"] == FIT_SOLVED
            assert calls["n"] == 1

            # 快照未变（轮询）：必须命中缓存，不再求解。
            r2 = await c.get("/api/batch")
            e2 = r2.json()["estimate"]
            assert e2["fit_source"] == FIT_CACHE
            assert calls["n"] == 1
            assert e2["pi"] == e1["pi"]
            assert e2["sse"] == e1["sse"]
            assert e2["confusion_matrix"] == e1["confusion_matrix"]
            assert e2["q"] == e1["q"]

            # 实检数与未检折算随当前快照返回，守恒关系成立。
            for k in CATS:
                assert e2["estimated_total"][k] == pytest.approx(
                    e2["inspected_counts"][k]
                    + e2["estimated_uninspected"][k]
                )


# ---------------------------------------------------------------------
# 2. 单飞：同批并发请求相同拟合输入，求解只发生一次
# ---------------------------------------------------------------------
async def test_concurrent_same_input_solves_once() -> None:
    transport = ASGITransport(app=app)
    calls = {"n": 0}
    entered = threading.Event()
    release = threading.Event()
    real_fit = fit_proportions

    def slow_fit(P, q):  # 在线程池工作线程内执行，使用线程同步原语
        calls["n"] += 1
        entered.set()
        assert release.wait(timeout=10)
        return real_fit(P, q)

    async with AsyncClient(transport=transport, base_url="http://t") as c:
        # 先在无补丁情况下完成满秩登记，避免登记请求被慢替身拖死。
        await _upload(c)
        await _make_full_rank(c)

        with _PatchFit(slow_fit):
            # 清掉登记过程中产生的缓存，保证从同一未求解键开始。
            async with store.lock:
                store._fit_cache.clear()

            # 先发起三个并发请求（不能用 gather 后再放行，那会形成等待环）。
            tasks = [
                asyncio.create_task(c.get("/api/batch")) for _ in range(3)
            ]
            # 等首个任务真正进入求解器，并给三个请求留出在途登记时间。
            await asyncio.to_thread(entered.wait, 10)
            await asyncio.sleep(0.15)
            release.set()
            results = await asyncio.gather(*tasks)

            statuses = [r.status_code for r in results]
            assert statuses == [200, 200, 200]
            assert calls["n"] == 1, "相同输入的并发请求必须只求解一次"
            sources = sorted(
                r.json()["estimate"]["fit_source"] for r in results
            )
            assert sources == sorted(
                [FIT_SOLVED, FIT_INFLIGHT, FIT_INFLIGHT]
            )

            pis = [tuple(r.json()["estimate"]["pi"]) for r in results]
            assert len(set(pis)) == 1


# ---------------------------------------------------------------------
# 3. 迟到拟合不得覆盖较新记录：等待期间发生修改则按新快照重算
# ---------------------------------------------------------------------
async def test_late_fit_does_not_overwrite_newer_snapshot() -> None:
    transport = ASGITransport(app=app)
    entered = threading.Event()
    release = threading.Event()
    real_fit = fit_proportions

    def slow_fit(P, q):
        entered.set()
        assert release.wait(timeout=10)
        return real_fit(P, q)

    async with AsyncClient(transport=transport, base_url="http://t") as c:
        # 先完成满秩登记，再启用慢替身。
        await _upload(c)
        await _make_full_rank(c)

        with _PatchFit(slow_fit):
            async with store.lock:
                store._fit_cache.clear()
                version_at_start = store._version

            older = asyncio.create_task(c.get("/api/batch"))
            await asyncio.to_thread(entered.wait, 10)  # 旧快照拟合已在途

            # 拟合未结束时登记仍可生效（修改在锁内即时落盘）。
            # M5 尚未登记，该次登记会推进快照版本并使已检数 +1。
            inspect_task = asyncio.create_task(
                c.post(
                    "/api/machines/M5/inspect",
                    json={"category": "drive"},
                )
            )
            await _wait_version(version_at_start + 1)
            release.set()  # 放行所有在途拟合

            r = await inspect_task
            newer = r.json()["estimate"]
            old_resp = await older
            old_est = old_resp.json()["estimate"]

            # 迟到请求最终返回的必须是较新快照，而不是发起时的旧快照。
            assert old_est["version"] == newer["version"]
            assert old_est["n_inspected"] == newer["n_inspected"]
            assert old_est["n_inspected"] == 13
            # 明细中 M5 已登记。
            m5 = next(
                m for m in (await c.get("/api/batch")).json()["machines"]
                if m["id"] == "M5"
            )
            assert m5["inspected"] == "drive"

            snap = (await c.get("/api/batch")).json()["estimate"]
            assert snap["version"] == newer["version"]
            assert snap["fit_source"] in {FIT_SOLVED, FIT_CACHE}


# ---------------------------------------------------------------------
# 4. 拟合失败：提示、保留实检、同一输入可重试
# ---------------------------------------------------------------------
async def test_fit_failure_reports_and_retries() -> None:
    transport = ASGITransport(app=app)
    real_fit = fit_proportions
    fail = {"on": True}

    def flaky_fit(P, q):
        if fail["on"]:
            raise FitError("模拟数值失败")
        return real_fit(P, q)

    async with AsyncClient(transport=transport, base_url="http://t") as c:
        # 先完成满秩登记，再启用失败替身。
        await _upload(c)
        await _make_full_rank(c)

        with _PatchFit(flaky_fit):
            async with store.lock:
                store._fit_cache.clear()

            r = await c.get("/api/batch")
            payload = r.json()
            est = payload["estimate"]
            assert r.status_code == 200
            assert est["available"] is False
            assert est["fit_source"] == FIT_FAILED
            assert "模拟数值失败" in est["fit_error"]
            # 已登记实检保留。
            n_insp = sum(
                1 for m in payload["machines"] if m["inspected"] is not None
            )
            assert n_insp == 12
            # 失败不写缓存、不在途：同输入下一次请求重新求解。
            async with store.lock:
                assert store._fit_cache == {}
                assert store._inflight == {}

            # 同一输入重试：恢复成功。
            fail["on"] = False
            r2 = await c.get("/api/batch")
            est2 = r2.json()["estimate"]
            assert est2["available"] is True
            assert est2["fit_source"] == FIT_SOLVED
            assert est2["pi"] is not None


# ---------------------------------------------------------------------
# 5. 撤回：立即重算未检、版本递增；重新登记后按新输入重新拟合
# ---------------------------------------------------------------------
async def test_withdraw_recomputes_and_refits() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        await _upload(c)
        await _make_full_rank(c)  # 12 已检 / 6 未检
        base = (await c.get("/api/batch")).json()["estimate"]
        assert base["fit_source"] in {FIT_SOLVED, FIT_CACHE}
        v0 = base["version"]

        r = await c.post("/api/machines/M1/withdraw")
        est = r.json()["estimate"]
        assert est["version"] == v0 + 1
        assert est["n_inspected"] == 11
        assert est["n_uninspected"] == 7
        # 拟合输入变化 -> 新求解，不允许拿旧 π 套新快照。
        assert est["fit_source"] == FIT_SOLVED
        # 未检折算按新快照（7 台）计算，台数守恒。
        assert sum(est["estimated_uninspected"].values()) == pytest.approx(
            7.0, abs=1e-9
        )
        assert sum(est["estimated_total"].values()) == pytest.approx(
            18.0, abs=1e-9
        )

        # 再轮询一次：同快照命中缓存。
        again = (await c.get("/api/batch")).json()["estimate"]
        assert again["fit_source"] == FIT_CACHE
        assert again["pi"] == est["pi"]

        # 继续撤回至某类实检样本归零（control 共 4 台：M3/M14/M15/M16）。
        for mid in ("M3", "M14", "M15", "M16"):
            r = await c.post(f"/api/machines/{mid}/withdraw")
        gated = r.json()["estimate"]
        assert gated["available"] is False
        assert gated["fit_source"] == FIT_GATED
        assert "控制" in gated["reason"]


# ---------------------------------------------------------------------
# 6. 换批：世代隔离，旧缓存/在途不影响新批次
# ---------------------------------------------------------------------
async def test_batch_reset_clears_cache_and_generation() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        await _upload(c)
        await _make_full_rank(c)
        est = (await c.get("/api/batch")).json()["estimate"]
        assert est["fit_source"] in {FIT_SOLVED, FIT_CACHE}
        gen_before = store._generation
        assert len(store._fit_cache) >= 1

        # 重新上传即换批。
        await _upload(c)
        assert store._generation == gen_before + 1
        assert store._fit_cache == {}
        assert store._inflight == {}
        fresh = (await c.get("/api/batch")).json()["estimate"]
        # 新批次零实检 -> 门控，版本从零开始。
        assert fresh["version"] == 0
        assert fresh["fit_source"] == FIT_GATED


# ---------------------------------------------------------------------
# 7. 并发复用 + 并发修改：求解只一次，两笔修改都生效且计数自洽
# ---------------------------------------------------------------------
async def test_concurrent_poll_reuse_and_mutations() -> None:
    transport = ASGITransport(app=app)
    calls = {"n": 0}
    real_fit = fit_proportions

    def counting_fit(P, q):
        calls["n"] += 1
        return real_fit(P, q)

    with _PatchFit(counting_fit):
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            await _upload(c)
            await _make_full_rank(c)
            async with store.lock:
                store._fit_cache.clear()
            calls["n"] = 0

            polls = await asyncio.gather(
                c.get("/api/batch"), c.get("/api/batch")
            )
            assert calls["n"] == 1
            assert {
                p.json()["estimate"]["fit_source"] for p in polls
            } == {FIT_SOLVED, FIT_INFLIGHT}

            # 并发修改不同机器：两笔都生效。
            await asyncio.gather(
                c.post(
                    "/api/machines/M4/inspect",
                    json={"category": "power"},
                ),
                c.post("/api/machines/M7/withdraw"),
            )
            snap = (await c.get("/api/batch")).json()
            by_id = {m["id"]: m["inspected"] for m in snap["machines"]}
            assert by_id["M4"] == "power" and by_id["M7"] is None
            counts = snap["estimate"]["inspected_counts"]
            actual = {k: 0 for k in CATS}
            for m in snap["machines"]:
                if m["inspected"]:
                    actual[m["inspected"]] += 1
            assert counts == actual


# ---------------------------------------------------------------------
# 8. 换批期间旧拟合失败：失败提示不得污染新批次，按新快照重新处理
# ---------------------------------------------------------------------
async def test_failed_fit_across_batch_reset_is_isolated() -> None:
    transport = ASGITransport(app=app)
    entered = threading.Event()
    release = threading.Event()
    real_fit = fit_proportions

    def slow_fail(P, q):
        entered.set()
        assert release.wait(timeout=10)
        raise FitError("旧批次模拟失败")

    async with AsyncClient(transport=transport, base_url="http://t") as c:
        await _upload(c)
        await _make_full_rank(c)

        with _PatchFit(slow_fail):
            async with store.lock:
                store._fit_cache.clear()
            older = asyncio.create_task(c.get("/api/batch"))
            await asyncio.to_thread(entered.wait, 10)  # 旧批次拟合在途

            # 拟合未结束时换批仍可生效。
            await _upload(c)
            release.set()
            resp = await older

            est = resp.json()["estimate"]
            # 返回的必须是新批次快照：零实检门控，而不是旧批次的失败提示。
            assert est["fit_source"] == FIT_GATED
            assert est["version"] == 0
            assert est["n_inspected"] == 0
            assert est["fit_error"] is None
