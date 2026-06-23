import base64
import io
import json
from pathlib import Path
from typing import cast, Literal, List, Dict

import aiohttp
from PIL import Image
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from app.core.config import settings
from app.core.logger import get_logger
from app.services.llm_factory import LLMFactory
from app.services.embedding_service import EmbeddingService
from app.lg_agent.state import AgentState, InputState, Router
from app.lg_agent.kg_sub_graph.kg_neo4j_conn import get_neo4j_graph
from app.lg_agent.kg_sub_graph.agentic_rag_agents.retrievers.cypher_examples.northwind_retriever import NorthwindCypherRetriever
from app.lg_agent.kg_sub_graph.agentic_rag_agents.workflows.multi_agent.multi_tool import create_multi_tool_workflow
from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.utils.utils import retrieve_and_parse_schema_from_graph_for_prompts
from app.lg_agent.prompts import (
    ROUTER_SYSTEM_PROMPT,
    GET_ADDITIONAL_SYSTEM_PROMPT,
    GENERAL_QUERY_SYSTEM_PROMPT,
    GET_IMAGE_SYSTEM_PROMPT,
    GUARDRAILS_SYSTEM_PROMPT,
    IMAGE_ANALYSIS_SYSTEM_PROMPT,
    FILE_QUERY_SYSTEM_PROMPT,
)


class AdditionalGuardrailsOutput(BaseModel):
    """
    格式化输出，用于判断用户的问题是否与图谱内容相关
    """
    decision: Literal["end", "continue"] = Field(
        description="Decision on whether the question is related to the graph contents."
    )


# 构建日志记录器
logger = get_logger(service="builder")

# 电商经营范围描述，供 guardrails 判断与研究计划节点共用
ECOMMERCE_SCOPE_DESCRIPTION = """
    个人电商经营范围：智能家居产品，包括但不限于：
    - 智能照明（灯泡、灯带、开关）
    - 智能安防（摄像头、门锁、传感器）
    - 智能控制（温控器、遥控器、集线器）
    - 智能音箱（语音助手、音响）
    - 智能厨电（电饭煲、冰箱、洗碗机）
    - 智能清洁（扫地机器人、洗衣机）

    不包含：服装、鞋类、体育用品、化妆品、食品等非智能家居产品。
    """

# 文件问答的兜底话术
FILE_REUPLOAD_MSG = "抱歉，我没有收到文件，请上传文件后再提问。"
FILE_UNREADABLE_MSG = "抱歉，这个文件我暂时无法读取，请确认是 PDF 后重试。"
FILE_NO_MATCH_MSG = "抱歉，我在文件里没有找到相关信息。"

# 懒加载的 EmbeddingService 单例（模型很重，只在第一次 file-query 时实例化）
_embedding_service = None


def _get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    # 选择模型实例，通过.env文件中的AGENT_SERVICE参数选择
    model = LLMFactory.create_agent_model(tags=["router"])
    logger.info(f"Using agent model service: {settings.AGENT_SERVICE}")

    # 拼接提示模版 + 用户的实时问题（包含历史上下文对话） 
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ] + state.messages
    logger.info("-----Analyze user query type-----")
    logger.info(f"History messages: {state.messages}")
    
    # 使用结构化输出，输出问题类型
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    logger.info(f"Analyze user query type completed, result: {response}")
    return {"router": response}

def route_query(
    state: AgentState,
) -> Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"]:
    """根据查询分类确定下一步操作。

    Args:
        state (AgentState): 当前代理状态，包括路由器的分类。

    Returns:
        Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query"]: 下一步操作。
    """
    _type = state.router["type"]

    # 分类器不确定、且倾向于知识库类查询时，安全降级为"反问要信息"，避免没把握时硬答错
    if state.router.get("confidence") == "low" and _type in ("graphrag-query", "additional-query"):
        logger.info(f"Low-confidence '{_type}' -> falling back to get_additional_info")
        return "get_additional_info"

    if _type == "general-query":
        return "respond_to_general_query"
    elif _type == "additional-query":
        return "get_additional_info"
    elif _type == "graphrag-query":
        return "create_research_plan"
    elif _type == "image-query":
        return "create_image_query"
    elif _type == "file-query":
        return "create_file_query"
    else:
        raise ValueError(f"Unknown router type {_type}")
    
async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成对一般查询的响应，完全基于大模型，不会触发任何外部服务的调用，包括自定义工具、知识库查询等。

    当路由器将查询分类为一般问题时，将调用此节点。

    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("-----generate general-query response-----")
    
    # 使用大模型生成回复
    model = LLMFactory.create_agent_model(tags=["general_query"])
    
    system_prompt = GENERAL_QUERY_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def get_additional_info(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """生成一个响应，要求用户提供更多信息。

    当路由确定需要从用户那里获取更多信息时，将调用此函数。

    Args:
        state (AgentState): 当前代理状态，包括对话历史和路由逻辑。
        config (RunnableConfig): 用于配置响应生成的模型。

    Returns:
        Dict[str, List[BaseMessage]]: 包含'messages'键的字典，其中包含生成的响应。
    """
    logger.info("------continue to get additional info------")
    
    # 使用大模型生成回复
    model = LLMFactory.create_agent_model(tags=["additional_info"])

    # 如果用户的问题是电商相关，但与自己的业务无关，则需要返回"无关问题"

    # 首先连接 Neo4j 图数据库
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")
        neo4j_graph = None

    # 定义电商经营范围
    scope_description = ECOMMERCE_SCOPE_DESCRIPTION

    scope_context = (
        f"参考此范围描述来决策:\n{scope_description}"
        if scope_description is not None
        else ""
    )

    # 动态从 Neo4j 图表中获取图表结构
    graph_context = (
        f"\n参考图表结构来回答:\n{retrieve_and_parse_schema_from_graph_for_prompts(neo4j_graph)}"
        if neo4j_graph is not None
        else ""
    )

    message = scope_context + graph_context + "\nQuestion: {question}"

    # 拼接提示模版
    full_system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                GUARDRAILS_SYSTEM_PROMPT,
            ),
            (
                "human",
                (message),
            ),
        ]
    )

    # 构建格式化输出的 Chain， 如果匹配，返回 continue，否则返回 end
    guardrails_chain = full_system_prompt | model.with_structured_output(AdditionalGuardrailsOutput)
    guardrails_output = await guardrails_chain.ainvoke(
            {"question": state.messages[-1].content if state.messages else ""}
        )

    # 根据格式化输出的结果，返回不同的响应
    if guardrails_output.decision == "end":
        logger.info("-----Fail to pass guardrails check-----")
        return {"messages": [AIMessage(content="抱歉，我家暂时没有这方面的商品，可以在别家看看哦~")]}
    else:
        logger.info("-----Pass guardrails check-----")
        system_prompt = GET_ADDITIONAL_SYSTEM_PROMPT.format(
            logic=state.router["logic"]
        )
        messages = [{"role": "system", "content": system_prompt}] + state.messages
        response = await model.ainvoke(messages)
        return {"messages": [response]}


# ---- image-query helpers ----

MAX_IMAGE_SIZE = 1024          # 长边最大像素，超过则等比缩小
JPEG_QUALITY = 85             # 压缩后 JPEG 质量
VISION_MAX_TOKENS = 4000      # 视觉模型单次返回上限

# 图片相关的兜底话术：区分"图片本身的问题"和"服务端的问题"
IMAGE_REUPLOAD_MSG = "抱歉，我无法查看这张图片，请重新上传。"
IMAGE_SERVICE_ERROR_MSG = "抱歉，图片识别服务暂时不可用，请稍后再试~"


def _encode_image(image_path: str) -> str:
    """读取图片，等比压缩到长边 <= MAX_IMAGE_SIZE，转 JPEG 后做 base64 编码。"""
    with Image.open(image_path) as img:
        width, height = img.size
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            ratio = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height)
            img = img.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        logger.info(f"Image compressed from {width}x{height} to {img.width}x{img.height}")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def _describe_image(image_data: str) -> str:
    """调用视觉模型，返回对图片的文字描述；HTTP 失败时抛异常。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.VISION_API_KEY}",
    }
    payload = {
        "model": settings.VISION_MODEL,
        "messages": [
            {"role": "system", "content": IMAGE_ANALYSIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    }
                ],
            },
        ],
        "max_tokens": VISION_MAX_TOKENS,
        "temperature": 0.7,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{settings.VISION_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"vision API returned {response.status}: {error_text}")
            result = await response.json()
            return result["choices"][0]["message"]["content"]


async def create_image_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """处理图片查询：视觉模型先描述图片，再由客服模型组织成回复。

    错误分两类，不再把所有失败都甩成"请重新上传"：
    - 图片本身的问题（没传 / 损坏）   -> 让用户重传
    - 视觉服务的问题（未配置 / 调用失败）-> 让用户稍后再试
    """
    logger.info("-----Found User Upload Image-----")
    image_path = config.get("configurable", {}).get("image_path", None)

    # 1. 图片本身的问题 -> 让用户重传
    if not image_path or not Path(image_path).exists():
        logger.warning(f"User upload image not found: {image_path}")
        return {"messages": [AIMessage(content=IMAGE_REUPLOAD_MSG)]}

    # 2. 视觉服务未配置 -> 服务端问题，不该怪用户
    if not (settings.VISION_API_KEY and settings.VISION_BASE_URL and settings.VISION_MODEL):
        logger.error("Vision model configuration is incomplete")
        return {"messages": [AIMessage(content=IMAGE_SERVICE_ERROR_MSG)]}

    # 3. 读图 / 压缩失败 -> 多半是图片损坏，让用户重传
    try:
        image_data = _encode_image(image_path)
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        return {"messages": [AIMessage(content=IMAGE_REUPLOAD_MSG)]}

    # 4. 调用视觉模型失败 -> 服务端问题，让用户稍后再试
    try:
        image_description = await _describe_image(image_data)
    except Exception as e:
        logger.error(f"Vision API call failed: {e}")
        return {"messages": [AIMessage(content=IMAGE_SERVICE_ERROR_MSG)]}

    # 5. 用图片描述 + 历史，生成电商客服口吻的回复
    logger.info("Vision model produced a description; generating the reply")
    model = LLMFactory.create_agent_model(tags=["image_query"])
    system_prompt = GET_IMAGE_SYSTEM_PROMPT.format(image_description=image_description)
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def create_file_query(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[BaseMessage]]:
    """用 RAG 回答关于用户上传文件(PDF)的问题。

    错误分两类:文件本身的问题(让用户重传)vs. 检索为空(如实说没找到)。
    """
    logger.info("-----Found User Upload File-----")
    file_path = config.get("configurable", {}).get("file_path", None)
    if not file_path or not Path(file_path).exists():
        logger.warning(f"User upload file not found: {file_path}")
        return {"messages": [AIMessage(content=FILE_REUPLOAD_MSG)]}

    question = state.messages[-1].content if state.messages else ""
    try:
        results = await _get_embedding_service().query_file(file_path, question, top_k=3)
    except Exception as e:
        logger.error(f"Failed to read/index file {file_path}: {e}")
        return {"messages": [AIMessage(content=FILE_UNREADABLE_MSG)]}

    if not results:
        return {"messages": [AIMessage(content=FILE_NO_MATCH_MSG)]}

    context = "\n\n".join(r["content"] for r in results)
    model = LLMFactory.create_agent_model(tags=["file_query"])
    system_prompt = FILE_QUERY_SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}
    
    # TODO

async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> Dict[str, List[str] | str]:
    """通过查询本地知识库回答客户问题，执行任务分解，创建分布查询计划。

    Args:
        state (AgentState): 当前代理状态，包括对话历史。
        config (RunnableConfig): 用于配置计划生成的模型。

    Returns:
        Dict[str, List[str] | str]: 包含'steps'键的字典，其中包含研究步骤列表。
    """
    logger.info("------execute local knowledge base query------")

    # 使用大模型生成查询/多跳、并行查询计划
    model = LLMFactory.create_agent_model(tags=["research_plan"])
    
    # 初始化必要参数
    # 1. Neo4j图数据库连接 - 使用配置中的连接信息
    try:
        neo4j_graph = get_neo4j_graph()
        logger.info("success to get Neo4j graph database connection")
    except Exception as e:
        logger.error(f"failed to get Neo4j graph database connection: {e}")
        neo4j_graph = None

    # 2. 创建自定义检索器实例，根据 Graph Schema 创建 Cypher 示例，用来引导大模型生成正确的Cypher 查询语句
    cypher_retriever = NorthwindCypherRetriever()

    # step 3. 定义工具模式列表    
    from app.lg_agent.kg_sub_graph.kg_tools_list import cypher_query, predefined_cypher, microsoft_graphrag_query
    tool_schemas: List[type[BaseModel]] = [cypher_query, predefined_cypher, microsoft_graphrag_query]

    # 3. 预定义的Cypher查询 - 为电商场景定义有用的查询
    from app.lg_agent.kg_sub_graph.agentic_rag_agents.components.predefined_cypher.cypher_dict import predefined_cypher_dict

    # 定义电商经营范围
    scope_description = ECOMMERCE_SCOPE_DESCRIPTION

    # 创建多工具工作流
    multi_tool_workflow = create_multi_tool_workflow(
        llm=model,
        graph=neo4j_graph,
        tool_schemas=tool_schemas,
        predefined_cypher_dict=predefined_cypher_dict,
        cypher_example_retriever=cypher_retriever,
        scope_description=scope_description,
        llm_cypher_validation=True,
    )
    
    # 准备输入状态
    last_message = state.messages[-1].content if state.messages else ""
    input_state = {
        "question": last_message,
        "data": [],
        "history": []
    }
    
    # 执行工作流
    response = await multi_tool_workflow.ainvoke(input_state)
    return {"messages": [AIMessage(content=response["answer"])]}

# 定义持久化存储，也可以使用SQLiteSaver()、PostgresSaver()等
# LangGraph官方地址：https://langchain-ai.github.io/langgraph/how-tos/persistence/
checkpointer = MemorySaver()

# 定义状态图
builder = StateGraph(AgentState, input=InputState)
# 添加节点
builder.add_node(analyze_and_route_query)
builder.add_node(respond_to_general_query)
builder.add_node(get_additional_info)
builder.add_node("create_research_plan", create_research_plan)  # 这里是子图
builder.add_node(create_image_query)
builder.add_node(create_file_query)

# 添加边
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)


graph = builder.compile(checkpointer=checkpointer)