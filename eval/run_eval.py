import asyncio
import json
from pathlib import Path
from settings import settings
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient
from query import search_knowledge_base


def get_project_root() -> Path:
    """
    当前文件位置：
        project_root/eval/run_eval.py

    所以 parents[1] 就是项目根目录。
    """
    return Path(__file__).resolve().parents[1]


def load_eval_data() -> list[dict]:
    project_root = get_project_root()
    eval_path = project_root / "eval" / "eval_data.json"

    if not eval_path.exists():
        raise FileNotFoundError(f"评估集不存在: {eval_path}")

    return json.loads(eval_path.read_text(encoding="utf-8"))


def is_hit(expected_chunk_id: str, results: list[dict]) -> bool:
    """
    Strict hit:
    只看 expected_chunk_id 是否出现在检索结果里。
    不做文本模糊匹配。
    不做语义近似判断。
    """
    returned_chunk_ids = [
        item.get("chunk_id")
        for item in results
    ]

    return expected_chunk_id in returned_chunk_ids


async def run_eval() -> None:
    eval_data = load_eval_data()

    dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)

    qdrant_client = AsyncQdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
    )

    total = 0
    hit_count = 0
    details = []

    try:
        for item in eval_data:
            query = item["query"]
            category = item["category"]
            expected_chunk_id = item["expected_chunk_id"]

            results = await search_knowledge_base(
                query_text=query,
                category=category,
                dense_model=dense_model,
                sparse_model=sparse_model,
                qdrant_client=qdrant_client,
            )

            returned_chunk_ids = [
                result.get("chunk_id")
                for result in results
            ]

            hit = is_hit(expected_chunk_id, results)

            total += 1
            if hit:
                hit_count += 1

            details.append({
                "id": item["id"],
                "query": query,
                "category": category,
                "expected_chunk_id": expected_chunk_id,
                "returned_chunk_ids": returned_chunk_ids,
                "hit": hit,
            })

    finally:
        await qdrant_client.close()

    recall_at_3 = hit_count / total if total else 0

    print("\n===== STRICT RETRIEVAL EVAL =====")
    print(f"Total: {total}")
    print(f"Hit: {hit_count}")
    print(f"Strict Recall@3: {recall_at_3:.3f}")

    project_root = get_project_root()
    output_path = project_root / "eval" / "eval_result.json"

    output = {
        "metric": "Strict Recall@3",
        "total": total,
        "hit": hit_count,
        "recall_at_3": recall_at_3,
        "details": details,
    }

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n详细结果已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(run_eval())