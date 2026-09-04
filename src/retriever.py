"""주의사항(data/visa_safety.md)·관광지·맛집(data/attractions.md) 문서를 임베딩·검색하는 RAG 파이프라인."""

import hashlib
import os
from pathlib import Path

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_PERSIST_DIR = _DATA_DIR / "chroma_db"

_SOURCE_PATH = _DATA_DIR / "visa_safety.md"
_HASH_PATH = _PERSIST_DIR / ".source_hash"
_COLLECTION_NAME = "visa_safety"

_ATTRACTIONS_SOURCE_PATH = _DATA_DIR / "attractions.md"
_ATTRACTIONS_HASH_PATH = _PERSIST_DIR / ".attractions_source_hash"
_ATTRACTIONS_COLLECTION_NAME = "attractions"

_vectorstore: Chroma | None = None
_attractions_vectorstore: Chroma | None = None


def _extract_disclaimer(raw_text: str) -> str:
    """문서 상단의 인용구(더미 데이터 경고 문구)를 추출한다."""
    lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            lines.append(stripped.lstrip(">").strip())
        elif lines:
            break
    return " ".join(lines)


def _load_and_split_document(path: Path, key_name: str) -> list[Document]:
    """마크다운 문서를 key_name(##)·주제(###) 단위로 쪼개 Document 리스트로 만든다.

    key_name은 visa_safety.md면 "country", attractions.md면 "city"를 쓴다 —
    두 문서 다 같은 구조(H2 + H3 소제목)라 이 함수를 그대로 재사용한다.
    """
    raw_text = path.read_text(encoding="utf-8")
    disclaimer = _extract_disclaimer(raw_text)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", key_name), ("###", "topic")],
        strip_headers=True,
    )
    chunks = splitter.split_text(raw_text)

    documents: list[Document] = []
    for chunk in chunks:
        key_value = chunk.metadata.get(key_name)
        topic = chunk.metadata.get("topic")
        if not key_value or not topic:
            # 문서 최상단 안내문처럼 키/주제가 없는 조각은 검색 대상에서 제외한다
            continue
        content = chunk.page_content.strip()
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={
                    key_name: key_value,
                    "topic": topic,
                    "source": f"data/{path.name}",
                    "disclaimer": disclaimer,
                },
            )
        )
    return documents


def _source_hash(path: Path) -> str:
    """원본 문서 내용의 md5 해시를 계산한다 (재임베딩 필요 여부 판단용)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _get_embeddings() -> BedrockEmbeddings:
    """.env 설정으로 Bedrock 임베딩 클라이언트를 만든다."""
    model_id = os.environ["BEDROCK_EMBEDDING_MODEL_ID"]
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return BedrockEmbeddings(model_id=model_id, region_name=region)


def _build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    """원본 문서 해시를 비교해 기존 Chroma 컬렉션을 재사용하거나 새로 구축한다."""
    _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    current_hash = _source_hash(_SOURCE_PATH)
    previous_hash = (
        _HASH_PATH.read_text(encoding="utf-8").strip() if _HASH_PATH.exists() else None
    )

    store = Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=str(_PERSIST_DIR),
    )

    existing_ids = store.get()["ids"]
    needs_rebuild = force_rebuild or previous_hash != current_hash or not existing_ids
    if needs_rebuild:
        if existing_ids:
            store.delete(ids=existing_ids)
        store.add_documents(_load_and_split_document(_SOURCE_PATH, key_name="country"))
        _HASH_PATH.write_text(current_hash, encoding="utf-8")

    return store


def _get_vectorstore() -> Chroma:
    """프로세스 내에서 벡터스토어를 한 번만 준비해 재사용한다."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_or_load_vectorstore()
    return _vectorstore


def _build_or_load_attractions_vectorstore(force_rebuild: bool = False) -> Chroma:
    """attractions.md 해시를 비교해 기존 Chroma 컬렉션을 재사용하거나 새로 구축한다."""
    _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    current_hash = _source_hash(_ATTRACTIONS_SOURCE_PATH)
    previous_hash = (
        _ATTRACTIONS_HASH_PATH.read_text(encoding="utf-8").strip()
        if _ATTRACTIONS_HASH_PATH.exists()
        else None
    )

    store = Chroma(
        collection_name=_ATTRACTIONS_COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=str(_PERSIST_DIR),
    )

    existing_ids = store.get()["ids"]
    needs_rebuild = force_rebuild or previous_hash != current_hash or not existing_ids
    if needs_rebuild:
        if existing_ids:
            store.delete(ids=existing_ids)
        store.add_documents(_load_and_split_document(_ATTRACTIONS_SOURCE_PATH, key_name="city"))
        _ATTRACTIONS_HASH_PATH.write_text(current_hash, encoding="utf-8")

    return store


def _get_attractions_vectorstore() -> Chroma:
    """프로세스 내에서 관광지·맛집 벡터스토어를 한 번만 준비해 재사용한다."""
    global _attractions_vectorstore
    if _attractions_vectorstore is None:
        _attractions_vectorstore = _build_or_load_attractions_vectorstore()
    return _attractions_vectorstore


def retrieve(query: str, country: str | None = None, k: int = 3) -> list[dict]:
    """주의사항 문서에서 질의와 관련된 근거 조각을 검색한다.

    결과가 없으면 빈 리스트를 반환한다. 이는 "근거 없음"을 뜻하며,
    호출한 쪽(도구)은 이 경우 내용을 지어내지 말고 근거 없음으로 처리해야 한다.

    score는 0~1로 정규화된 유사도가 아니라 Chroma가 계산한 원시 거리값이다
    (작을수록 더 유사함). Titan 임베딩 벡터가 정규화돼 있지 않아
    similarity_search_with_relevance_scores의 기본 정규화 공식이 맞지 않는
    문제가 있어, 원시 거리값을 그대로 쓰는 similarity_search_with_score를 쓴다.
    """
    store = _get_vectorstore()
    filter_ = {"country": country} if country else None
    results = store.similarity_search_with_score(query, k=k, filter=filter_)
    return [
        {
            "content": document.page_content,
            "country": document.metadata.get("country"),
            "topic": document.metadata.get("topic"),
            "source": document.metadata.get("source"),
            "disclaimer": document.metadata.get("disclaimer"),
            "score": score,
        }
        for document, score in results
    ]


def retrieve_attractions(query: str, city: str, k: int = 3) -> list[dict]:
    """관광지·맛집 문서에서 질의와 관련된 근거 조각을 검색한다.

    retrieve()와 동작 방식은 같지만(빈 리스트 = 근거 없음), 국가가 아니라
    도시 단위로 색인된 별도 컬렉션(data/attractions.md)을 검색한다.
    """
    store = _get_attractions_vectorstore()
    results = store.similarity_search_with_score(query, k=k, filter={"city": city})
    return [
        {
            "content": document.page_content,
            "city": document.metadata.get("city"),
            "topic": document.metadata.get("topic"),
            "source": document.metadata.get("source"),
            "disclaimer": document.metadata.get("disclaimer"),
            "score": score,
        }
        for document, score in results
    ]
