# AssistGen

基于 LangGraph 构建的智能客服 Agent 系统，后端使用 FastAPI，前端使用 Vue 3。支持多路由分发、知识图谱问答、GraphRAG 检索、图片理解等能力。

## 系统架构

```
用户输入
    ↓
analyze_and_route_query  ← Pydantic 结构化输出，对问题意图分类
    ↓
┌──────────────────────────────────────────────────────────────┐
│ general-query    → respond_to_general_query   纯 LLM 问答    │
│ additional-query → get_additional_info        Neo4j Schema   │
│ graphrag-query   → create_research_plan       GraphRAG 检索  │
│ image-query      → create_image_query         Vision 模型    │
│ file-query       → create_file_query          文档解析问答   │
└──────────────────────────────────────────────────────────────┘
```

- **路由节点**：LLM 输出 Pydantic 结构体，自动分发到对应子图
- **Guardrails**：additional-query 分支带安全过滤，拦截无关问题
- **幻觉检测**：graphrag-query 分支输出经二次校验，避免无中生有
- **流式响应**：全链路 SSE 输出，前端实时渲染

## 技术栈

| 层级 | 技术 |
|------|------|
| Agent 框架 | LangGraph |
| LLM | DeepSeek V3 / Ollama（可切换） |
| 后端 | FastAPI + SQLAlchemy |
| 知识图谱 | Neo4j + GraphRAG |
| 缓存 | Redis（上下文语义缓存） |
| 数据库 | MySQL（用户、会话、消息持久化） |
| 认证 | JWT |
| 前端 | Vue 3 + Element Plus + TypeScript |

## 快速启动

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并按实际情况填写：

```env
# LLM（二选一）
CHAT_SERVICE=DEEPSEEK            # 或 OLLAMA
DEEPSEEK_API_KEY=your-api-key

# 数据库
DB_HOST=localhost
DB_PASSWORD=your-password

# Redis
REDIS_HOST=localhost
```

### 3. 初始化数据库

```bash
python scripts/init_db.py
```

### 4. 启动服务

```bash
python run.py
```

启动后访问：
- 前端界面：http://localhost:8000
- API 文档：http://localhost:8000/docs

## 项目结构

```
├── main.py                  # FastAPI 入口
├── app/
│   ├── api/                 # 路由：认证、Agent 接口
│   ├── core/                # 配置、数据库、JWT
│   ├── lg_agent/            # LangGraph Agent 核心
│   │   ├── lg_builder.py    # 图构建、节点定义
│   │   ├── lg_states.py     # 状态类型定义
│   │   └── kg_sub_graph/    # Neo4j 知识图谱子图
│   ├── services/            # LLM 工厂、Redis 缓存、搜索
│   └── models/              # SQLAlchemy 数据模型
└── scripts/
    └── init_db.py           # 数据库初始化
```

## License

MIT
