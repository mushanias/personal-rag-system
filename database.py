from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

# ===== Qdrant 连接配置 =====
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "knowledge_base"
# ===== Collection 初始化 =====
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=512, distance=Distance.COSINE)
        },
        # 稠密  向量化语义
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            # 全放内存检索快
            )
        },
        # 稀疏  向量化权重（至于语义或者权重标准怎么来的，就不在应用的领域了）
    )
    print(f"成功创建 Collection: {COLLECTION_NAME}")
# else:
#     print(f"Collection 已存在: {COLLECTION_NAME}")