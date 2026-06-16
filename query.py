from openai import AsyncOpenAI
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Prefetch,
    SparseVector,
    Fusion,
    FusionQuery
)
from database import  client
from settings import settings
from exceptions import RetrievalError, LLMServiceError
from logger import logger
# 1 初始化 Moonshot 客户端
moonshot_client = AsyncOpenAI(
    api_key=settings.MOONSHOT_API_KEY,
    base_url=settings.MOONSHOT_BASE_URL,
)

# 2 统一的 Prompt 模版配置
BASE_SYSTEM_PROMPT = (
    "你是一个高度可靠的个人知识库助手。请严格根据用户提供的参考内容（Context）回答问题。\n"
    "【铁律守则】\n"
    "1. 严格受限于参考内容：如果参考内容中没有相关信息，或者无法从中推导出来，请直接、诚实地回答：'抱歉，知识库中没有包含相关内容。' 绝对不允许胡编乱造或调用你原本的外部常识。\n"
    "2. 忠实于事实：回答不得与参考内容发生事实上的冲突。\n"
    "3. 标明出处：在回答的结尾，必须明确列出你参考了哪些源文件（如：[参考来源: study.txt]）。\n\n"
)

# 我们这里针对三个不同的场景做不同的细节优化
CATEGORY_EXTENSIONS = {
    "study": (
        "【当前场景：学术与技术研讨】\n"
        "用户正在向你请教硬核的知识或技术问题。你的语气应当【严谨、专业、极具条理】。\n"
        "请多使用结构化的段落（如：核心定义、实现原理、应用场景），或者使用 Markdown 列表和代码块，帮助用户建立严密的知识体系。"
    ),
    "life": (
        "【当前场景：生活起居与身心健康】\n"
        "用户正在咨询你关于日常生活、健康或习惯的问题。你的语气应当【亲切、温暖、极具生活气息】。\n"
        "请多提供通俗易懂、具备可操作性（Actionable）的具体步骤、避坑小贴士，注重人文关怀，避免冷冰冰的学术说教。"
    ),
    "thinking": (
        "【当前场景：深度思维与认知复盘】\n"
        "用户正在与你探讨底层的逻辑、思维模型或方法论。你的语气应当【理性、深刻、富有启发性】。\n"
        "请着重剖析事物的底层逻辑、推导过程、潜在的思维误区，可以适当提出反思性问题，引导用户看透问题本质。"
    )
}

def encode_query(
    query_text: str,
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
):
    # 将用户的提问转化为双通道向量
    # 1 稠密向量化 (Dense)
    dense_vector = dense_model.encode(query_text).tolist()

    # 2 稀疏向量化 (Sparse)
    sparse_result = list(sparse_model.embed([query_text]))[0]

    sparse_vector = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    return dense_vector, sparse_vector
# 记住了，召回步骤怎么说呢，可以看作是用一个东西把文档文本，
# 还有用户提问两个东西向量化变成两个可以比较的东西，然后比较
# 所以最重要的是什么？就是一样的模型，一样的对比参数。
# 你向量化储存格式怎么写，这里就要对应！


async def search_knowledge_base(
    query_text: str,#这个是用户要传的东西，记住了
    category: str,
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    score_threshold: float | None = None,
) -> list[dict]:
    if score_threshold is None:
        score_threshold = settings.SCORE_THRESHOLD

    # 1 先把问题向量化
    dense_vector, sparse_vector = encode_query(
        query_text, dense_model, sparse_model
    )

    # 2 构造分类隔离过滤器
    category_filter = Filter(
        must=[
            FieldCondition(
                key="category", match=MatchValue(value=category)
            )
        ]
    )

    # 3 配置双通道预检索 (Prefetch)
    # Dense 通道
    prefetch_dense = Prefetch(
        query=dense_vector,
        using="dense",
        filter=category_filter,
        limit=10
    )
    # Sparse 通道
    prefetch_sparse = Prefetch(
        query=sparse_vector,
        using="sparse",
        filter=category_filter,
        limit=10,
    )

    # 4 执行 RRF 融合检索
    try:
        search_results = await client.query_points(
            collection_name=settings.COLLECTION_NAME,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=3,
        )
    except Exception as e:
        raise RetrievalError(f"知识库检索失败: {str(e)}")

    # 5 解析结果
    valid_chunks = []
    for hit in search_results.points:
        # 使用 RRF 融合时，hit.score 通常是一个融合分数（如 0.0~1.0 之间）
        if hit.score >= score_threshold:
            # valid_chunks.append({
            #     "content": hit.payload["content"],
            #     "source": hit.payload["source"],
            #     "score": hit.score,
            #     "chunk_id": hit.payload.get("chunk_id"),
            # })
            payload = hit.payload or {}

            content = payload.get("content")
            source = payload.get("source")

            if not content or not source:
                continue

            valid_chunks.append({
                "content": content,
                "source": source,
                "score": hit.score,
                "chunk_id": payload.get("chunk_id"),
            })
    #所有的数据都是心怀不轨的
    logger.info(
        "检索完成 category=%s chunk_count=%s top_score=%s chunk_ids=%s sources=%s",
        category,
        len(valid_chunks),
        valid_chunks[0]["score"] if valid_chunks else None,
        [c["chunk_id"] for c in valid_chunks],
        [c["source"] for c in valid_chunks],
    )

    return valid_chunks


async def generate_answer(question: str, chunks: list[dict], category: str) -> dict:


    # 1 动态动态组装专属的 System Prompt
    extension = CATEGORY_EXTENSIONS.get(category, "")
    full_system_prompt = f"{BASE_SYSTEM_PROMPT}{extension}"

    # 2 将 Top3 检索出的片段合并为一段上下文（带上文件名和分数，让 LLM 心里有数）
    context_list = []
    for idx, c in enumerate(chunks, 1):
        context_list.append(f"--- 片段 {idx} [来源: {c['source']}] ---\n{c['content']}\n")
    context = "\n".join(context_list)

    # 3. 构造对话消息
    messages = [
        {"role": "system", "content": full_system_prompt},
        {
            "role": "user",
            "content": f"以下是为你准备的参考内容：\n{context}\n\n根据上述参考内容，请回答问题：{question}"
        }
    ]

    # 4. 请求 Moonshot
    try:
        logger.info(
            "开始调用LLM category=%s chunk_count=%s",
            category,
            len(chunks),
        )
        response = await moonshot_client.chat.completions.create(
            model=settings.MOONSHOT_MODEL,
            messages=messages,
            temperature=0.3,
        )
        answer_text = response.choices[0].message.content

        logger.info(
            "LLM调用成功 category=%s answer_length=%s",
            category,
            len(answer_text) if answer_text else 0,
        )
    except Exception as e:
        raise LLMServiceError(f"大模型服务调用失败: {str(e)}")

    return {
        "answer": answer_text,
        "sources": chunks
    }


async def ask(query_text: str, category: str, dense_model, sparse_model) -> dict:

    # 一路召回
    chunks = await search_knowledge_base(query_text, category, dense_model, sparse_model)

    # 如果连第一层检索都空空如也，直接熔断，保底机制
    if not chunks:
        return {
            "answer": "抱歉，我的知识库里暂时没有找到相关的记录，无法为您解答。",
            "sources": []
        }

    # 喂给大模型（顺便把分类带过去，用于选择 Prompt）
    return await generate_answer(query_text, chunks, category)