import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


# Hugging Face / Sentence Transformers embedding models
EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,

    "sentence-transformers/all-mpnet-base-v2": 768,
    "all-mpnet-base-v2": 768,

    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,

    "sentence-transformers/multi-qa-mpnet-base-dot-v1": 768,
    "multi-qa-mpnet-base-dot-v1": 768,

    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384,

    "sentence-transformers/paraphrase-mpnet-base-v2": 768,
    "paraphrase-mpnet-base-v2": 768,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()

    # Exact match
    if normalized in {
        key.lower(): value
        for key, value in EMBEDDING_DIMENSIONS.items()
    }:
        return {
            key.lower(): value
            for key, value in EMBEDDING_DIMENSIONS.items()
        }[normalized]

    # Fallback checks
    if "all-minilm-l6-v2" in normalized:
        return 384

    if "all-mpnet-base-v2" in normalized:
        return 768

    if "multi-qa-minilm-l6-cos-v1" in normalized:
        return 384

    if "multi-qa-mpnet-base-dot-v1" in normalized:
        return 768

    if "paraphrase-minilm-l6-v2" in normalized:
        return 384

    if "paraphrase-mpnet-base-v2" in normalized:
        return 768

    raise ValueError(
        f"Unsupported Hugging Face embedding model "
        f"'{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(api_key=settings.pinecone_api_key)

    names = [x["name"] for x in pc.list_indexes()]

    # If index already exists, check its dimension
    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)

        current_dimension = getattr(index_info, "dimension", None)

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        # Delete index if dimensions don't match
        if (
            current_dimension is not None
            and current_dimension != desired_dimension
        ):
            print(
                f"Dimension mismatch: existing={current_dimension}, "
                f"required={desired_dimension}"
            )

            print(
                f"Deleting Pinecone index: "
                f"{settings.pinecone_index_name}"
            )

            pc.delete_index(name=settings.pinecone_index_name)

            while settings.pinecone_index_name in [
                x["name"] for x in pc.list_indexes()
            ]:
                time.sleep(1)

    # Create index if it doesn't exist
    if settings.pinecone_index_name not in [
        x["name"] for x in pc.list_indexes()
    ]:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while not pc.describe_index(
            settings.pinecone_index_name
        ).status["ready"]:
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)