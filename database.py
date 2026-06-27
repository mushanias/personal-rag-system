from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

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