"""create_file_query 端到端实测(真跑,不进 CI)。

真建 FAISS 索引 + 真检索 + 真调 LLM,验证"上传文档问问题能基于文件内容回答"。
模型下载 + torch 很重,所以放 evals/ 手动跑:

    .venv/bin/python evals/file_query_eval.py [可选的 PDF 路径]
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from app.lg_agent.builder import create_file_query  # noqa: E402
from app.lg_agent.state import AgentState  # noqa: E402

DEFAULT_PDF = "app/graphrag/data/DA68-04798A-03_RF8500_CN.pdf"
QUESTIONS = [
    "这个产品是什么？",
    "怎么清洁或保养？",
]


async def main() -> None:
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    if not Path(pdf).exists():
        print(f"找不到 PDF: {pdf}（传一个真实 PDF 路径作参数）")
        return
    print(f"PDF: {pdf}\n")
    for q in QUESTIONS:
        state = AgentState(
            messages=[HumanMessage(content=q)],
            router={"type": "file-query", "logic": ""},
        )
        result = await create_file_query(
            state, config={"configurable": {"file_path": pdf}}
        )
        print(f"Q: {q}\nA: {result['messages'][0].content[:200]}\n{'-' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
