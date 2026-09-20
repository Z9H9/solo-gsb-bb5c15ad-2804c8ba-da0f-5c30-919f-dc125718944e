"""下一对优先级推荐：模拟每个待比较组合的三种结果并评估排名影响。

对每个尚未比较的组合，分别假设「左胜 / 右胜 / 平局」三种判定结果并重算
Schulze 排名；对每一对候选（共 C(n,2) 对），考察其在三种结果下的相对次序
状态（前 / 后 / 并列），若三种结果之间状态不完全相同则计 1 分。所有候选对
的计分之和即为该组合的**影响分**。

推荐影响分最高的组合；同分时按现有待比较顺序（候选录入顺序的组合枚举序）
选择。全程仅在内存中模拟，不写入任何模拟结果，也不影响进度统计。
"""
from __future__ import annotations

from .schulze import rank_groups, strongest_paths

OUTCOMES = ("a", "b", "tie")


def _pair_states(p: list[list[float]], n: int) -> list[int]:
    """由最强路径矩阵提取全部候选对（按录入顺序枚举）的相对次序状态。

    状态：-1 = 前者名次在前，0 = 并列，1 = 前者名次在后。
    Schulze 关系是全预序，候选对状态即 sign(p[i][j] - p[j][i])。
    """
    states = []
    for i in range(n):
        row_i = p[i]
        for j in range(i + 1, n):
            diff = row_i[j] - p[j][i]
            states.append(0 if diff == 0 else (-1 if diff < 0 else 1))
    return states


def _impact_score(
    candidates: list[str], history: list[dict], pair: dict, n: int
) -> int:
    """模拟某待比较组合的三种判定结果，返回其影响分。"""
    a, b = pair["a"], pair["b"]
    states_per_outcome = []
    for result in OUTCOMES:
        simulated = history + [{"a": a, "b": b, "result": result}]
        p = strongest_paths(candidates, simulated)
        states_per_outcome.append(_pair_states(p, n))
    impact = 0
    for k in range(n * (n - 1) // 2):
        s = states_per_outcome[0][k]
        if states_per_outcome[1][k] != s or states_per_outcome[2][k] != s:
            impact += 1
    return impact


def recommend_next_pair(
    candidates: list[str], history: list[dict], pending: list[dict]
) -> dict | None:
    """返回「下一对推荐」；无待比较组合时返回 None。

    返回字段：
    - pair：推荐组合 {"a", "b"}（按候选录入顺序）
    - impact / total_pairs：影响分与候选对总数
    - reason：推荐原因（人类可读）
    - previews：三种结果（"a" 左胜 / "b" 右胜 / "tie" 平局）各自的 Schulze 排名预览
    """
    if not pending:
        return None
    n = len(candidates)
    total_pairs = n * (n - 1) // 2

    # 按待比较顺序遍历，仅在严格更大时替换：同分自然取顺序靠前者
    best_pos, best_impact, tied = 0, -1, 0
    for pos, pair in enumerate(pending):
        impact = _impact_score(candidates, history, pair, n)
        if impact > best_impact:
            best_pos, best_impact, tied = pos, impact, 1
        elif impact == best_impact:
            tied += 1

    pair = pending[best_pos]
    a, b = pair["a"], pair["b"]
    previews = {}
    for result in OUTCOMES:
        simulated = history + [{"a": a, "b": b, "result": result}]
        previews[result] = rank_groups(
            candidates, strongest_paths(candidates, simulated)
        )

    if best_impact > 0:
        reason = (
            f"三种判定结果会改变 {best_impact} / {total_pairs} 对候选的相对名次"
            "（前 / 后 / 并列），影响分为当前待比较组合中最高"
        )
        if tied > 1:
            reason += f"；另有 {tied - 1} 个组合同分，按待比较顺序取前者"
    else:
        reason = "剩余组合的判定结果均不再改变候选间的相对名次，按待比较顺序推荐"

    return {
        "pair": {"a": a, "b": b},
        "impact": best_impact,
        "total_pairs": total_pairs,
        "reason": reason,
        "previews": previews,
    }
