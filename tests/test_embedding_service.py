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
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"some content")
    result = await svc.query_file(str(f), "q", top_k=2)
    svc.create_embeddings.assert_awaited_once()      # 没索引 -> 建
    svc.search.assert_awaited_once_with("q", 2)
    assert result == [{"content": "chunk"}]


async def test_query_file_reuses_existing_index(tmp_path):
    svc = _service_without_model(tmp_path)
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"some content")
    content_hash = hashlib.md5(b"some content").hexdigest()
    (tmp_path / f"index_{content_hash}.bin").write_bytes(b"x")   # 假装索引已存在
    await svc.query_file(str(f), "q")
    svc.create_embeddings.assert_not_awaited()        # 复用,不重建
    svc._load_index.assert_called_once_with(f"index_{content_hash}")


async def test_query_file_keys_by_content_not_filename(tmp_path):
    # 同一份内容、不同文件名 -> 只索引一次（按内容哈希，不按路径）
    svc = _service_without_model(tmp_path)
    a = tmp_path / "manual.pdf"
    a.write_bytes(b"same bytes")
    b = tmp_path / "manual_20260624.pdf"
    b.write_bytes(b"same bytes")
    content_hash = hashlib.md5(b"same bytes").hexdigest()
    (tmp_path / f"index_{content_hash}.bin").write_bytes(b"x")
    await svc.query_file(str(a), "q")
    await svc.query_file(str(b), "q")
    svc.create_embeddings.assert_not_awaited()  # 两次都复用同一索引，都不重建
