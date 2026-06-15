from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)
from settings import settings

client = AsyncQdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
)

COLLECTION_NAME = settings.COLLECTION_NAME

COLLECTION_CONFIG = {
    "vectors_config": {
        "dense": VectorParams(size=512, distance=Distance.COSINE)
    },
    "sparse_vectors_config": {
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        # 内存储存，快
        )
    },
}