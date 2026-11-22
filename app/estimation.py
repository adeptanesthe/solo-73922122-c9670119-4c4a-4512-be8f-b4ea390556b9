"""故障占比估算核心逻辑（仅依赖 NumPy）。

业务口径
========
* 初判列 ``initial`` 与实检列 ``inspected`` 均取三类:
  ``power``（电源）、``drive``（驱动）、``control``（控制）。
* 对已拆检机器，按实检类别分组，统计每类实检样本中各初判类别的经验
  概率，组成 3x3 混淆概率矩阵 ``P``，行=实检类别，列=初判类别，每行和为 1。
* 未检件的初判经验分布为向量 ``q``（和为 1）。
* 在单纯形 ``pi >= 0 且 sum(pi) = 1`` 上最小化平方误差
  ``||P^T pi - q||_2^2``，得到未检件的故障占比估计 ``pi``。
* 折算台数 = ``pi * 未检台数``，再加上各类实检数得到全批估计。
* 门控：仍有未检件时，任一类实检样本缺失或混淆矩阵不满秩，则撤下估数，
  提示补检；全部拆完时直接给实数，不再做拟合。

缓存与复用
==========
最小二乘求解只依赖归一化后的拟合输入 ``(P, q)``。``fit_input_key`` 将其
固化为可哈希的浮点元组：同一批次内只要最近一次成功拟合的输入与当前快照
相同，即可直接复用 ``pi`` 与 ``sse``；实检数、未检折算等仍按各自快照
另行计算。并发请求遇到相同输入时由存储层做单飞合并，求解只执行一次。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CATEGORIES = ("power", "drive", "control")
CATEGORY_LABELS = {
    "power": "电源",
    "drive": "驱动",
    "control": "控制",
}
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

TOL = 1e-10

# 拟合结果来源（写入响应 fit_source 字段，前端据此提示）。
FIT_ACTUALS = "actuals"              # 全检，直接给实数
FIT_GATED = "gated"                  # 门控撤下（缺样本/不满秩）
FIT_SOLVED = "solved"                # 本次新求解
FIT_CACHE = "cache"                  # 复用同批最近一次成功拟合
FIT_INFLIGHT = "inflight_reuse"      # 并入同批在途拟合，只求解一次
FIT_FAILED = "failed"                # 拟合抛错，可同输入重试


class FitError(RuntimeError):
    """最小二乘拟合失败（数值异常等）。不改变任何拆检记录，可重试。"""


@dataclass(frozen=True)
class SnapshotStats:
    """一次快照下与拟合相关的全部统计量（持锁期间计算，纯只读）。"""

    version: int
    n_total: int
    n_inspected: int
    n_uninspected: int
    inspected_counts: dict[str, int]
    uninspected_initial_counts: dict[str, int]
    gate_reason: str | None
    P: np.ndarray | None
    q: np.ndarray | None


@dataclass(frozen=True)
class FitOutput:
    """一次成功拟合的可复用结果（与快照无关，只由 P、q 决定）。"""

    pi: tuple[float, ...]
    sse: float

    def as_array(self) -> np.ndarray:
        return np.asarray(self.pi, dtype=float)


@dataclass
class EstimateResult:
    """一次估算的完整结果。

    ``available`` 为 False 时（需要补检或拟合失败），``pi`` 等拟合字段为 None。
    """

    available: bool
    reason: str | None
    n_total: int
    n_inspected: int
    n_uninspected: int
    inspected_counts: dict[str, int]
    uninspected_initial_counts: dict[str, int]
    P: list[list[float]] | None
    q: list[float] | None
    pi: list[float] | None
    sse: float | None
    estimated_uninspected: dict[str, float] | None
    estimated_total: dict[str, float] | None
    actual_total: dict[str, int] | None
    version: int = 0
    fit_source: str | None = None
    fit_error: str | None = None

    def to_payload(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "categories": list(CATEGORIES),
            "labels": dict(CATEGORY_LABELS),
            "n_total": self.n_total,
            "n_inspected": self.n_inspected,
            "n_uninspected": self.n_uninspected,
            "inspected_counts": self.inspected_counts,
            "uninspected_initial_counts": self.uninspected_initial_counts,
            "confusion_matrix": self.P,
            "q": self.q,
            "pi": self.pi,
            "sse": self.sse,
            "estimated_uninspected": self.estimated_uninspected,
            "estimated_total": self.estimated_total,
            "actual_total": self.actual_total,
            "version": self.version,
            "fit_source": self.fit_source,
            "fit_error": self.fit_error,
        }


def _solve_simplex(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """在单纯形 x>=0、sum(x)=1 上最小化 ||A x - b||_2^2。

    三维单纯形的最优解只可能落在：内部（无不等式活动）、某条边
    （一个分量为 0）或顶点（两个分量为 0）。因此枚举：

    1. 内部点：解等式约束 KKT，可行则采纳；
    2. 三条边：退化为标量二次函数，解析求最小点并截断到 [0,1]；
    3. 三个顶点：恒可行，保证算法在矩阵亏秩时仍有候选。

    纯 NumPy 实现，不依赖优化库。返回 ``(x, sse)``。
    """
    n = A.shape[1]
    candidates: list[np.ndarray] = []

    # 1) 内部点：KKT
    #    [2A^T A  1] [x]   [2 A^T b]
    #    [1^T     0] [lam] = [1      ]
    H = 2.0 * A.T @ A
    g = 2.0 * A.T @ b
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = H
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    rhs = np.concatenate([g, [1.0]])
    try:
        sol = np.linalg.solve(K, rhs)
        x = sol[:n]
        if np.all(x >= -TOL) and abs(float(np.sum(x)) - 1.0) <= 1e-8:
            candidates.append(np.clip(x, 0.0, None))
    except np.linalg.LinAlgError:
        pass

    # 2) 边：x_i = t, x_j = 1-t，其余为 0；解析最小化二次函数
    for i in range(n):
        for j in range(i + 1, n):
            u = A[:, i] - A[:, j]
            v = A[:, j] - b
            denom = float(u @ u)
            t_star = -float(v @ u) / denom if denom > TOL else 0.0
            t_star = min(1.0, max(0.0, t_star))
            x = np.zeros(n)
            x[i] = t_star
            x[j] = 1.0 - t_star
            candidates.append(x)

    # 3) 顶点
    for k in range(n):
        x = np.zeros(n)
        x[k] = 1.0
        candidates.append(x)

    best_x, best_sse = None, np.inf
    for x in candidates:
        sse = float(np.sum((A @ x - b) ** 2))
        if sse < best_sse:
            best_sse, best_x = sse, x
    return best_x, best_sse


def build_snapshot_stats(machines: list[dict], version: int = 0) -> SnapshotStats:
    """持锁期间可快速完成的快照统计：计数、门控、P 与 q（满秩时）。

    不做最小二乘求解；求解被单独放到 ``fit_proportions`` 以便缓存与
    线程池卸载。门控口径与历史版本完全一致。
    """
    n_total = len(machines)
    inspected = [m for m in machines if m["inspected"] is not None]
    uninspected = [m for m in machines if m["inspected"] is None]
    n_insp, n_un = len(inspected), len(uninspected)

    inspected_counts = {c: 0 for c in CATEGORIES}
    for m in inspected:
        inspected_counts[m["inspected"]] += 1

    un_initial_counts = {c: 0 for c in CATEGORIES}
    for m in uninspected:
        un_initial_counts[m["initial"]] += 1

    # 全部拆完：不做拟合，无门控。
    if n_un == 0:
        return SnapshotStats(
            version=version,
            n_total=n_total,
            n_inspected=n_insp,
            n_uninspected=0,
            inspected_counts=inspected_counts,
            uninspected_initial_counts=un_initial_counts,
            gate_reason=None,
            P=None,
            q=None,
        )

    # 门控 1：任一类实检样本缺失。
    missing = [
        CATEGORY_LABELS[c] for c in CATEGORIES if inspected_counts[c] == 0
    ]
    if missing:
        return SnapshotStats(
            version=version,
            n_total=n_total,
            n_inspected=n_insp,
            n_uninspected=n_un,
            inspected_counts=inspected_counts,
            uninspected_initial_counts=un_initial_counts,
            gate_reason=(
                "仍有未检件，且缺少实检类别样本："
                + "、".join(missing)
                + "。请补检后再估算。"
            ),
            P=None,
            q=None,
        )

    # 经验混淆概率矩阵 P：行=实检类别，列=初判类别。
    count = np.zeros((3, 3), dtype=float)
    for m in inspected:
        count[CAT_INDEX[m["inspected"]], CAT_INDEX[m["initial"]]] += 1.0
    row_sum = count.sum(axis=1, keepdims=True)
    P = count / row_sum

    # 门控 2：混淆矩阵不满秩。
    if np.linalg.matrix_rank(P, tol=1e-10) < 3:
        return SnapshotStats(
            version=version,
            n_total=n_total,
            n_inspected=n_insp,
            n_uninspected=n_un,
            inspected_counts=inspected_counts,
            uninspected_initial_counts=un_initial_counts,
            gate_reason=(
                "仍有未检件，但实检样本得到的混淆概率矩阵不满秩"
                "（各类实检样本的初判分布线性相关），无法唯一拟合，"
                "请补充更多拆检样本。"
            ),
            P=P,
            q=None,
        )

    # 未检件初判经验分布 q。
    q = (
        np.array([un_initial_counts[c] for c in CATEGORIES], dtype=float) / n_un
    )
    return SnapshotStats(
        version=version,
        n_total=n_total,
        n_inspected=n_insp,
        n_uninspected=n_un,
        inspected_counts=inspected_counts,
        uninspected_initial_counts=un_initial_counts,
        gate_reason=None,
        P=P,
        q=q,
    )


def fit_input_key(P: np.ndarray, q: np.ndarray) -> tuple:
    """把归一化拟合输入固化为可哈希键：P（行优先 9 元素）+ q（3 元素）。

    相同键意味着最小二乘问题完全相同，``pi`` 与 ``sse`` 可直接复用。
    """
    return (
        tuple(float(x) for x in np.asarray(P, dtype=float).flat),
        tuple(float(x) for x in np.asarray(q, dtype=float).flat),
    )


def fit_proportions(P: np.ndarray, q: np.ndarray) -> FitOutput:
    """纯 CPU 最小二乘拟合（无锁、无 IO），失败抛 :class:`FitError`。

    仅依赖 ``P``、``q``，与批次快照解耦，便于在线程池中执行并按
    ``fit_input_key`` 缓存/单飞复用。
    """
    try:
        pi, sse = _solve_simplex(P.T, q)
    except np.linalg.LinAlgError as exc:
        raise FitError(
            "最小二乘求解出现数值异常（矩阵方程无法求解），请稍后重试"
            "或补充更多拆检样本。"
        ) from exc
    pi = np.clip(pi, 0.0, 1.0)
    total = float(pi.sum())
    if total <= 0.0 or not np.all(np.isfinite(pi)) or not np.isfinite(sse):
        raise FitError(
            "拟合结果异常（出现非数值或退化占比），请稍后重试或补充"
            "更多拆检样本。"
        )
    pi = pi / total
    return FitOutput(pi=tuple(float(x) for x in pi), sse=float(sse))


def assemble_result(
    stats: SnapshotStats,
    fit: FitOutput | None = None,
    fit_source: str | None = None,
    fit_error: str | None = None,
) -> EstimateResult:
    """把快照统计与拟合结果（或门控/失败原因）装配为响应对象。

    折算台数始终按 *当前快照* 的未检台数计算，即便 ``fit`` 复用自
    更早一次同输入拟合。
    """
    base = dict(
        n_total=stats.n_total,
        n_inspected=stats.n_inspected,
        n_uninspected=stats.n_uninspected,
        inspected_counts=dict(stats.inspected_counts),
        uninspected_initial_counts=dict(stats.uninspected_initial_counts),
    )

    # 全部拆完：直接给实数，不做拟合。
    if fit_source == FIT_ACTUALS:
        return EstimateResult(
            available=True,
            reason=None,
            P=None,
            q=None,
            pi=None,
            sse=None,
            estimated_uninspected=None,
            estimated_total={
                c: float(stats.inspected_counts[c]) for c in CATEGORIES
            },
            actual_total=dict(stats.inspected_counts),
            version=stats.version,
            fit_source=FIT_ACTUALS,
            **base,
        )

    # 门控撤下（缺类 / 不满秩），沿用原口径：不满秩时仍回传 P。
    if fit_source == FIT_GATED:
        return EstimateResult(
            available=False,
            reason=stats.gate_reason,
            P=stats.P.tolist() if stats.P is not None else None,
            q=None,
            pi=None,
            sse=None,
            estimated_uninspected=None,
            estimated_total=None,
            actual_total=None,
            version=stats.version,
            fit_source=FIT_GATED,
            **base,
        )

    # 拟合失败：撤下估数并提示；实检记录由调用方保留，同输入可重试。
    if fit_source == FIT_FAILED:
        return EstimateResult(
            available=False,
            reason=fit_error,
            P=stats.P.tolist() if stats.P is not None else None,
            q=stats.q.tolist() if stats.q is not None else None,
            pi=None,
            sse=None,
            estimated_uninspected=None,
            estimated_total=None,
            actual_total=None,
            version=stats.version,
            fit_source=FIT_FAILED,
            fit_error=fit_error,
            **base,
        )

    # 成功路径：solved / cache / inflight_reuse。
    if fit is None:
        raise FitError("内部错误：缺少拟合结果")
    pi_arr = fit.as_array()
    n_un = stats.n_uninspected
    estimated_uninspected = {
        c: float(pi_arr[i] * n_un) for i, c in enumerate(CATEGORIES)
    }
    estimated_total = {
        c: stats.inspected_counts[c] + estimated_uninspected[c]
        for c in CATEGORIES
    }
    return EstimateResult(
        available=True,
        reason=None,
        P=stats.P.tolist(),
        q=stats.q.tolist(),
        pi=list(fit.pi),
        sse=fit.sse,
        estimated_uninspected=estimated_uninspected,
        estimated_total=estimated_total,
        actual_total=None,
        version=stats.version,
        fit_source=fit_source,
        **base,
    )


def estimate(machines: list[dict]) -> EstimateResult:
    """同步估算（保留给单元测试与离线使用），口径与异步接口完全一致。

    ``machines`` 为字典列表，每项含 ``id``、``initial``、
    ``inspected``（None 表示未检）。
    """
    stats = build_snapshot_stats(machines)
    if stats.n_uninspected == 0:
        return assemble_result(stats, fit_source=FIT_ACTUALS)
    if stats.gate_reason is not None:
        return assemble_result(stats, fit_source=FIT_GATED)
    fit = fit_proportions(stats.P, stats.q)
    return assemble_result(stats, fit=fit, fit_source=FIT_SOLVED)
