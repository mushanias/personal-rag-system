#我们在这个文件中实现文本的切块，向量化储存
from settings import settings
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector
from database import  COLLECTION_CONFIG
import uuid
import asyncio
from qdrant_client import AsyncQdrantClient
from chunker import chunk_text

client = AsyncQdrantClient(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
)
def load_docs(data_dir: str = "test") -> dict[str, str]:
    docs = {
        f.stem: f.read_text(encoding="utf-8")
        for f in Path(data_dir).glob("*.txt")
    }
    return docs
# 现在 docs 就是一个字典

# 拿到了我们开始切块

# 切块了然后我们需要干什么？向量化储存对吧，记得组装PointStruct，还有载荷里面放隔离字段哦
# dense_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
# sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def build_points(docs, dense_model, sparse_model):
    points = []

    for category, raw_text in docs.items():
        chunks = chunk_text(raw_text)

        for index, chunk_content in enumerate(chunks):
            unique_str = f"{category}_{index}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
            # 这里为什么要这样写，详细看文档
            # Dense 向量
            dense_vector = dense_model.encode(chunk_content).tolist()

            # Sparse 向量
            sparse_result = list(sparse_model.embed([chunk_content]))[0]
            sparse_vector = SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist(),
            )

            # 组装 Point
            # 注意了qdrant只接受PointStruct对象哦
            point = PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                },
                payload={
                    "content": chunk_content,
                    "category": category,
                    "source": f"{category}.txt",
                    "chunk_id": f"{category}_{index}",
                },
            )
            points.append(point)
    return points

async def ensure_collection_exists():
    if not await client.collection_exists(settings.COLLECTION_NAME):
        await client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            **COLLECTION_CONFIG
        )
async def main():
    docs = load_docs()

    dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)

    points = build_points(
        docs=docs,
        dense_model=dense_model,
        sparse_model=sparse_model,
    )
    await ensure_collection_exists()

    await client.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=points,
    )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
# 为什么会做这个决策呢？因为我需要在启动之前，就把文本处理好，
# 用户是无法接触到这部分逻辑的