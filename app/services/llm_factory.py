from typing import Union
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from app.core.config import settings, ServiceType
from app.services.deepseek_service import DeepseekService
from app.services.ollama_service import OllamaService
from app.services.search_service import SearchService


class LLMFactory:
    @staticmethod
    def create_agent_model(tags: list[str] | None = None, temperature: float = 0.7):
        """创建 LangGraph 节点使用的 LangChain chat 模型。

        按 .env 的 AGENT_SERVICE 在 DeepSeek / Ollama 间选择，统一所有节点的模型创建逻辑，
        替代原先在每个节点里复制粘贴的 if/else。
        """
        if settings.AGENT_SERVICE == ServiceType.DEEPSEEK:
            return ChatDeepSeek(
                api_key=settings.DEEPSEEK_API_KEY,
                model_name=settings.DEEPSEEK_MODEL,
                temperature=temperature,
                tags=tags,
            )
        return ChatOllama(
            model=settings.OLLAMA_AGENT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
            tags=tags,
        )

    @staticmethod
    def create_chat_service():
        """创建聊天服务实例"""
        if settings.CHAT_SERVICE == ServiceType.DEEPSEEK:
            # 如果.env文件中CHAT_SERVICE设置为DEEPSEEK，则使用DeepseekService
            return DeepseekService()
        else:
            # 否则使用OllamaService
            return OllamaService()

    @staticmethod
    def create_reasoner_service():
        """创建推理服务实例"""
        # 如果.env文件中REASON_SERVICE设置为DEEPSEEK，则使用DeepseekService
        if settings.REASON_SERVICE == ServiceType.DEEPSEEK:
            return DeepseekService()
        else:
            # 否则使用OllamaService
            return OllamaService()
    
    @staticmethod
    def create_search_service():
        """创建搜索服务实例"""
        return SearchService()