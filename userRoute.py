from fastapi import APIRouter, Request
from userSchemas import UserRequest, UserResponse
from query import ask
from logger import logger

# 创建一个专属于 QA 的路由组，带上 /ask 前缀
router = APIRouter(prefix="/ask", tags=["知识库问答"])

async def _ask_handler(req: UserRequest, request: Request, category: str) -> dict:
    logger.info(
        "收到用户请求 category=%s query=%s",
        category,
        req.query,
    )
    llm_client = request.app.state.llm_client
    dense_model = request.app.state.dense_model
    sparse_model = request.app.state.sparse_model
    qdrant_client = request.app.state.qdrant_client
    # 调用服务层的一体化接口 (检索 + 生成)
    result = await ask(
        query_text=req.query,
        category=category,
        dense_model=dense_model,
        sparse_model=sparse_model,
        llm_client=llm_client,
        qdrant_client=qdrant_client,
    )

    return result

@router.post("/public", response_model=UserResponse, summary="公开电商规则问答")
async def ask_public(req: UserRequest, request: Request):
    return await _ask_handler(req, request, "public")


@router.post("/private", response_model=UserResponse, summary="私密订单信息问答")
async def ask_private(req: UserRequest, request: Request):
    return await _ask_handler(req, request, "private")