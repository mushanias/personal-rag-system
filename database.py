from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)


client = AsyncQdrantClient(
    host="localhost",
    port=6333
)
# 定义表的结构
COLLECTION_NAME = "knowledge_base"

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