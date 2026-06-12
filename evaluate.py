# evaluate.py
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from query import search_knowledge_base

# ==========================================
# 1. 构造黄金测试集 (基于你实际的 txt 内容)
# ==========================================
# 每一个测试用例包含：用户可能问的问题、所属分类、预期必须召回的源文件名
TEST_BENCHMARK = [
    # 针对 study.txt (原 learning.txt)
    {
        "query": "Python的异步编程是怎么实现的？CPU在干嘛？",
        "category": "study",
        "expected_source": "study.txt"
    },
    {
        "query": "混合检索(Hybrid Search)和传统的全网检索有什么不一样？",
        "category": "study",
        "expected_source": "study.txt"
    },
    # 针对 life.txt
    {
        "query": "我经常写代码脖子酸痛僵硬，怎么办？",
        "category": "life",
        "expected_source": "life.txt"
    },
    {
        "query": "自己做饭真的比点外卖省钱吗？",
        "category": "life",
        "expected_source": "life.txt"
    },
    # 针对 thinking.txt
    {
        "query": "什么是费曼技巧？怎么判断我自己懂没懂？",
        "category": "thinking",
        "expected_source": "thinking.txt"
    },
    {
        "query": "遇到想不通的高难度概念时，应该怎么排查？",
        "category": "thinking",
        "expected_source": "thinking.txt"
    }
]

# ==========================================
# 2. 初始化评估环境 (离线加载模型)
# ==========================================
print("🔄 正在初始化评估模型...")
dense_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
print("✅ 模型加载成功，开始跑自动化评估...\n")

# ==========================================
# 3. 执行评估循环
# ==========================================
total_cases = len(TEST_BENCHMARK)
hit_count = 0
failed_cases = []

print("=" * 60)
print(f"{'用户提问':<25} | {'预期来源':<10} | {'实际召回 Top3 来源'}")
print("-" * 60)

for case in TEST_BENCHMARK:
    # 模拟路由调用服务层检索
    chunks = search_knowledge_base(
        query_text=case["query"],
        category=case["category"],
        dense_model=dense_model,
        sparse_model=sparse_model,
        score_threshold=0.01  # 评估时阈值可以放低，主要看排名前三有没有它
    )

    # 提取实际召回的所有 source
    retrieved_sources = [c["source"] for c in chunks]

    # 判断预期的源文件是否在召回列表中
    is_hit = case["expected_source"] in retrieved_sources

    if is_hit:
        hit_count += 1
        status = "✅ HIT"
    else:
        status = "❌ MISS"
        failed_cases.append({
            "query": case["query"],
            "expected": case["expected_source"],
            "got": retrieved_sources
        })

    print(f"[{status}] {case['query'][:12]}... | {case['expected_source']} -> {retrieved_sources}")

# ==========================================
# 4. 输出评估报告
# ==========================================
hit_rate = (hit_count / total_cases) * 100
print("=" * 60)
print(f"📊 评估结束！最终召回命中率 (Hit Rate @Top3): {hit_rate:.2f}%")
print(f"成功: {hit_count} 条 / 失败: {total_cases - hit_count} 条")
print("=" * 60)

if failed_cases:
    print("\n🔍 失败 Case 深度剖析：")
    for idx, fc in enumerate(failed_cases):
        print(f"  错误 {idx + 1}:")
        print(f"    - 提问: '{fc['query']}'")
        print(f"    - 应该召回: {fc['expected']}")
        print(f"    - 实际召回: {fc['got'] if fc['got'] else '[] (可能被 score_threshold 拦截或压根没匹配到)'}")