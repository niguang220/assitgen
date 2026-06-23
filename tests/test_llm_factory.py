"""Unit tests for LLMFactory.create_agent_model — model selection by config.

Verifies the factory that replaced the six inline DeepSeek/Ollama blocks:
it picks the right provider from settings.AGENT_SERVICE and forwards tags.
Constructing the chat model does not open a connection, so no service is needed.
"""
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

import app.services.llm_factory as factory_mod
from app.core.config import ServiceType
from app.services.llm_factory import LLMFactory


def test_returns_deepseek_when_configured(monkeypatch):
    monkeypatch.setattr(factory_mod.settings, "AGENT_SERVICE", ServiceType.DEEPSEEK)
    assert isinstance(LLMFactory.create_agent_model(tags=["t"]), ChatDeepSeek)


def test_returns_ollama_when_configured(monkeypatch):
    monkeypatch.setattr(factory_mod.settings, "AGENT_SERVICE", ServiceType.OLLAMA)
    assert isinstance(LLMFactory.create_agent_model(tags=["t"]), ChatOllama)


def test_forwards_tags(monkeypatch):
    monkeypatch.setattr(factory_mod.settings, "AGENT_SERVICE", ServiceType.DEEPSEEK)
    model = LLMFactory.create_agent_model(tags=["router"])
    assert model.tags == ["router"]
