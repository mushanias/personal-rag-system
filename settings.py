import os
from dotenv import load_dotenv

load_dotenv()

# 12-Factor App（十二要素应用）了解一下即可

class Settings:
    # HuggingFace 镜像
    HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "knowledge_base")

    # Embedding Models
    DENSE_MODEL_NAME: str = os.getenv(
        "DENSE_MODEL_NAME",
        "BAAI/bge-small-zh-v1.5"
    )
    SPARSE_MODEL_NAME: str = os.getenv(
        "SPARSE_MODEL_NAME",
        "Qdrant/bm25"
    )
    # Reranker
    RERANKER_MODEL_NAME: str = os.getenv(
        "RERANKER_MODEL_NAME",
        "BAAI/bge-reranker-base"
    )

    # LLM
    MOONSHOT_API_KEY: str | None = os.getenv("MOONSHOT_API_KEY")
    MOONSHOT_BASE_URL: str = os.getenv(
        "MOONSHOT_BASE_URL",
        "https://api.moonshot.cn/v1"
    )
    MOONSHOT_MODEL: str = os.getenv(
        "MOONSHOT_MODEL",
        "moonshot-v1-8k"
    )

    # Retrieval
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.05"))


settings = Settings()