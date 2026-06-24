"""grounding 校验闸实测(真调 LLM 裁判,不进 CI)。

用一组带标签的对抗样本量化这道闸到底抓不抓得住幻觉：每条给定
(检索证据 context, 客服答案 answer, 期望 grounded)，真跑 _verify_grounding，
统计判对率 + 把"该拦没拦"(漏报)和"误拦正常答案"(误报)分开列出来。

    .venv/bin/python evals/grounding_eval.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lg_agent.builder import _verify_grounding  # noqa: E402

# (context, answer, expected_grounded)
# grounded=True 的答案完全能从 context 推出；grounded=False 的故意编造 context 里没有的事实。
CASES = [
    ("本产品保修期为 2 年，自购买之日起计算。", "您的产品保修期是 2 年，从购买当天开始算。", True),
    ("退货需在签收后 7 天内申请，商品需保持完好。", "签收后 7 天内、商品完好就可以申请退货。", True),
    ("智能门锁支持指纹、密码、NFC 三种开锁方式。", "门锁支持指纹、密码和 NFC 开锁。", True),
    ("本产品保修期为 2 年。", "本产品保修期是 5 年，并赠送终身免费保养。", False),  # 编造年限 + 不存在的服务
    ("退货需在签收后 7 天内申请。", "我们支持 30 天无理由退货，运费全免。", False),  # 编造时效与运费政策
    ("智能门锁支持指纹和密码开锁。", "门锁还支持人脸识别和远程视频通话。", False),  # 编造不存在的功能
]


async def main() -> None:
    correct = 0
    false_negatives = []  # 该判 0(幻觉)却放行了 —— 最危险
    false_positives = []  # 正常答案被误判为幻觉
    for i, (context, answer, expected) in enumerate(CASES, 1):
        grounded = await _verify_grounding(context, answer)
        ok = grounded == expected
        correct += ok
        tag = "OK " if ok else "MISS"
        print(f"[{tag}] case {i}: expected grounded={expected}, got grounded={grounded}")
        print(f"        answer: {answer}")
        if not ok and expected is False:
            false_negatives.append(i)
        if not ok and expected is True:
            false_positives.append(i)

    n = len(CASES)
    print("-" * 60)
    print(f"总判对率: {correct}/{n} = {correct / n:.0%}")
    print(f"漏报(幻觉没拦住, 危险): {false_negatives or '无'}")
    print(f"误报(误拦正常答案): {false_positives or '无'}")


if __name__ == "__main__":
    asyncio.run(main())
