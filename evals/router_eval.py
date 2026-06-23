"""Router classification eval.

Measures how often ``analyze_and_route_query`` puts a query into the right
route. This calls the real LLM (DeepSeek), so it is intentionally NOT part of
the service-free CI — run it on demand:

    .venv/bin/python evals/router_eval.py

Add/adjust cases over time; a drop in accuracy after a prompt change is the
signal this is meant to catch.
"""
import asyncio
import sys
from pathlib import Path

# make the repo root importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from app.lg_agent.builder import analyze_and_route_query  # noqa: E402
from app.lg_agent.state import AgentState  # noqa: E402

# (query, expected_route) — labelled to match ROUTER_SYSTEM_PROMPT's categories
CASES = [
    ("你好呀", "general-query"),
    ("你们客服是真人还是机器人？", "general-query"),
    ("今天天气真不错", "general-query"),
    ("帮我推荐一款智能音箱", "additional-query"),       # vague, no model/spec
    ("我想买个摄像头", "additional-query"),
    ("我的订单到哪了", "additional-query"),             # no order number
    ("你们有哪些智能门锁？", "graphrag-query"),
    ("扫地机器人的退换货政策是什么", "graphrag-query"),
    ("智能灯泡怎么连接wifi", "graphrag-query"),
    ("订单号12345的物流状态", "graphrag-query"),
    ("我发张图片你帮我看看这是什么产品", "image-query"),
    ("这张照片里的门锁是哪个型号", "image-query"),
    ("我上传了一个文件，帮我总结一下", "file-query"),
    ("帮我分析这个PDF文档的内容", "file-query"),
]


async def classify(query: str) -> str:
    state = AgentState(messages=[HumanMessage(content=query)])
    result = await analyze_and_route_query(state, config={})
    return result["router"]["type"]


async def main() -> None:
    correct = 0
    by_type: dict[str, list[int]] = {}
    print(f"{'query':<32}{'expected':<18}{'predicted':<18}ok")
    print("-" * 74)
    for query, expected in CASES:
        try:
            predicted = await classify(query)
        except Exception as exc:  # noqa: BLE001
            predicted = f"ERROR:{exc}"
        ok = predicted == expected
        correct += ok
        hit, total = by_type.setdefault(expected, [0, 0])
        by_type[expected] = [hit + ok, total + 1]
        print(f"{query[:30]:<32}{expected:<18}{predicted:<18}{'OK' if ok else 'XX'}")

    print("-" * 74)
    for route, (hit, total) in by_type.items():
        print(f"  {route:<18} {hit}/{total}")
    print(f"\noverall accuracy: {correct}/{len(CASES)} = {correct / len(CASES):.0%}")


if __name__ == "__main__":
    asyncio.run(main())
