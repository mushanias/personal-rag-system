#我们在这个文件中实现文本的切块，向量化储存
import os
from settings import settings
os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

from pathlib import Path
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client.models import PointStruct, SparseVector
from database import client, COLLECTION_NAME, COLLECTION_CONFIG
import uuid
import asyncio
# 首先我们拿文档，把所有 txt 内容一次性载入内存
# 这里应该不应该直接进去下次想下怎么办
def load_docs(data_dir: str = "test") -> dict[str, str]:
    docs = {
        f.stem: f.read_text(encoding="utf-8")
        for f in Path(data_dir).glob("*.txt")
    }
    return docs
# 现在 docs 就是一个字典

# 拿到了我们开始切块
def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 40) -> list[str]:

    chunks = []
    start = 0
    # 清洗掉过多的换行符，让文本更紧凑
    text = "".join([line.strip() for line in text.splitlines() if line.strip()])

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 每次向前移动 (窗口大小 - 重叠大小)
        start += (chunk_size - chunk_overlap)

        # 如果剩下的一点点不够一个全窗口，且已经到末尾了，就跳出
        if start >= len(text) or end >= len(text):
            break
    return chunks

# 切块了然后我们需要干什么？向量化储存对吧，记得组装PointStruct，还有载荷里面放隔离字段哦
# dense_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
# sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def build_points(docs, dense_model, sparse_model):
    points = []

    for category, raw_text in docs.items():
        chunks = chunk_text(raw_text,150,40)

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
                    "is_private": False,
                },
            )
            points.append(point)
    return points

async def ensure_collection_exists():
    if not await client.collection_exists(COLLECTION_NAME):
        await client.create_collection(
            collection_name=COLLECTION_NAME,
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
        collection_name=COLLECTION_NAME,
        points=points,
    )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())