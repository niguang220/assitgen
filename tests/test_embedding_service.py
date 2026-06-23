import hashlib
from unittest.mock import AsyncMock, MagicMock

from app.services.embedding_service import EmbeddingService


def _service_without_model(tmp_path):
    # __new__ 跳过 __init__(否则会加载 SentenceTransformer 模型,很慢)
    svc = EmbeddingService.__new__(EmbeddingService)
    svc.index_dir = tmp_path
    svc.create_embeddings = AsyncMock()
    svc._load_index = MagicMock()
    svc.search = AsyncMock(return_value=[{"content": "chunk"}])
    return svc


async def test_query_file_builds_index_when_missing(tmp_path):
    svc = _service_without_model(tmp_path)
    result = await svc.query_file("/some/file.pdf", "q", top_k=2)
    svc.create_embeddings.assert_awaited_once()      # 没索引 -> 建
    svc.search.assert_awaited_once_with("q", 2)
    assert result == [{"content": "chunk"}]


async def test_query_file_reuses_existing_index(tmp_path):
    svc = _service_without_model(tmp_path)
    h = hashlib.md5("/some/file.pdf".encode()).hexdigest()
    (tmp_path / f"index_{h}.bin").write_bytes(b"x")   # 假装索引已存在
    await svc.query_file("/some/file.pdf", "q")
    svc.create_embeddings.assert_not_awaited()        # 复用,不重建
    svc._load_index.assert_called_once_with(f"index_{h}")
