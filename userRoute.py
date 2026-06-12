# userRoute.py
from fastapi import APIRouter, Request
from userSchemas import UserRequest, UserResponse
from query import ask

# 创建一个专属于 QA 的路由组，带上 /ask 前缀
router = APIRouter(prefix="/ask", tags=["知识库问答"])


async def _ask_handler(req: UserRequest, request: Request, category: str) -> UserResponse:

    dense_model = request.app.state.dense_model
    sparse_model = request.app.state.sparse_model

    # 调用服务层的一体化接口 (检索 + 生成)
    result = await ask(
        query_text=req.query,
        category=category,
        dense_model=dense_model,
        sparse_model=sparse_model,
    )

    # 将服务层返回的 dict (包含 answer 和 sources) 解包并转化为 Pydantic 响应模型
    return UserResponse(**result)

@router.post("/study", response_model=UserResponse, summary="学术与技术研讨问答")
async def ask_study(req: UserRequest, request: Request):
    return await _ask_handler(req, request, "study")


@router.post("/life", response_model=UserResponse, summary="生活起居与健康问答")
async def ask_life(req: UserRequest, request: Request):
    return await _ask_handler(req, request, "life")


@router.post("/thinking", response_model=UserResponse, summary="深度思维与认知问答")
async def ask_thinking(req: UserRequest, request: Request):
    return await _ask_handler(req, request, "thinking")