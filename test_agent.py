"""
AssistGen Router 测试脚本
========================================
目的：验证 LangGraph 的路由节点能否正确分类不同类型的问题
运行方式：
  cd ~/projects/assitgen/llm_backend
  source .venv/bin/activate
  python test_agent.py

看这个脚本时要理解三件事：
  1. 每个测试用例为什么发这条消息（预期触发哪条路由）
  2. SSE 流式响应怎么解析
  3. 终端日志里 router 输出了什么分类结果（脚本跑完再回去看服务日志）
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

# ─────────────────────────────────────────────
# Step 1: 登录，拿 token
# ─────────────────────────────────────────────

def login(email: str, password: str) -> tuple[str, int]:
    """登录并返回 (access_token, user_id)"""
    resp = requests.post(
        f"{BASE_URL}/api/token",
        json={"email": email, "password": password}
    )
    if resp.status_code != 200:
        raise RuntimeError(f"登录失败: {resp.status_code} {resp.text}")

    data = resp.json()
    token = data["access_token"]
    print(f"✅ 登录成功，token 前20位: {token[:20]}...")
    return token, 1


# ─────────────────────────────────────────────
# Step 2: 发消息给 agent，解析 SSE 流
# ─────────────────────────────────────────────

def send_query(query: str, user_id: int, conversation_id: str = None) -> tuple[str, str]:
    """
    发送查询到 /api/langgraph/query
    返回 (完整响应文本, conversation_id)

    关键点：这个接口用 Form-data，不是 JSON body
    返回的是 SSE 流：每行格式为 data: "content"\n\n
    """
    data = {
        "query": query,
        "user_id": str(user_id),
    }
    if conversation_id:
        data["conversation_id"] = conversation_id

    # stream=True 让 requests 不要一次性读完，逐块读取
    resp = requests.post(
        f"{BASE_URL}/api/langgraph/query",
        data=data,
        stream=True,
        timeout=60
    )

    if resp.status_code != 200:
        return f"[ERROR] {resp.status_code}: {resp.text}", ""

    conv_id = resp.headers.get("X-Conversation-ID", "")

    # 解析 SSE 流
    # 每个事件格式：data: "json_encoded_string"\n\n
    full_response = []
    for line in resp.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            raw = line[6:]
            try:
                chunk = json.loads(raw)
                if isinstance(chunk, str):
                    full_response.append(chunk)
                elif isinstance(chunk, dict) and "interruption" in chunk:
                    full_response.append(f"[INTERRUPT] conversation_id={chunk.get('conversation_id')}")
            except json.JSONDecodeError:
                full_response.append(raw)

    return "".join(full_response), conv_id


# ─────────────────────────────────────────────
# Step 3: 测试用例
# ─────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "general-query：闲聊",
        "query": "你好，请介绍一下你自己",
        "expected_route": "general-query",
        "why": "不涉及任何电商业务，应走 general 分支，直接 LLM 回答",
    },
    {
        "name": "general-query：常识问题",
        "query": "Python 和 Java 有什么区别？",
        "expected_route": "general-query",
        "why": "技术常识，与电商无关，应走 general-query",
    },
    {
        "name": "additional-query：范围内商品咨询",
        "query": "你们有没有卖智能灯泡的？大概什么价位？",
        "expected_route": "additional-query → guardrails PASS",
        "why": "电商相关 + 在经营范围内（智能家居），guardrails 应放行",
        "note": "⚠️ Neo4j 未启动，会报错，但能看到 router 正确分类",
    },
    {
        "name": "additional-query：超出范围的商品",
        "query": "你们有没有卖耐克运动鞋的？",
        "expected_route": "additional-query → guardrails END",
        "why": "超出经营范围，guardrails 应拦截，返回'暂时没有这方面的商品'",
        "note": "⚠️ Neo4j 未启动，guardrails 检查会失败",
    },
    {
        "name": "graphrag-query：复杂知识图谱查询",
        "query": "帮我分析一下你们智能家居产品有哪些品类，各品类价格区间是多少",
        "expected_route": "graphrag-query",
        "why": "需要跨节点关系查询，应走 GraphRAG 分支",
        "note": "⚠️ Neo4j 未启动，必然报错，但验证 router 能识别这类问题",
    },
]


# ─────────────────────────────────────────────
# Step 4: 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AssistGen Router 测试")
    print("=" * 60)
    print()
    print("⚠️  运行前确认：")
    print("   1. FastAPI 服务已启动（端口 8000）")
    print("   2. 能看到服务终端日志")
    print("   3. 测试完成后去服务日志里看 router 分类结果")
    print()

    try:
        token, user_id = login("test@test.com", "test123")
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    print()

    for i, case in enumerate(TEST_CASES, 1):
        print(f"{'─' * 60}")
        print(f"[{i}/{len(TEST_CASES)}] {case['name']}")
        print(f"  发送消息  : {case['query']}")
        print(f"  预期路由  : {case['expected_route']}")
        print(f"  原因      : {case['why']}")
        if "note" in case:
            print(f"  注意      : {case['note']}")
        print()

        start = time.time()
        response, conv_id = send_query(case["query"], user_id)
        elapsed = time.time() - start

        print(f"  Agent 回复（{elapsed:.1f}s）:")
        for line in response.split("\n"):
            if line.strip():
                print(f"    {line}")
        if conv_id:
            print(f"  conversation_id: {conv_id}")
        print()

        time.sleep(2)

    print("=" * 60)
    print("测试完成 —— 现在去 FastAPI 终端往上翻日志")
    print("找这行：'Analyze user query type completed, result: ...'")
    print("看每条消息被分到了哪个路由分支")
    print("=" * 60)


if __name__ == "__main__":
    main()
