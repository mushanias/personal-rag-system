from settings import settings
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from userRoute import router as user
from database import client, COLLECTION_CONFIG
from fastapi import Request
from fastapi.responses import JSONResponse
from exceptions import AppException
# 什么时候准备，什么时候运行，什么时候打扫
@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("正在加载模型.")
    app.state.dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME)
    app.state.sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
    # ✅ 初始化 Collection
    if not await client.collection_exists(settings.COLLECTION_NAME):
        await client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            **COLLECTION_CONFIG
        )
    print("模型加载完成 ")
    yield
    await client.close()
    print("服务关闭 ")
#任何占用资源（内存、网络端口、文件句柄、显存）的东西，都有生命周期
#注意关闭,给我记好了啊

app = FastAPI(lifespan=lifespan)

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "path": request.url.path,
        },
    )
app.include_router(user)

@app.get("/")
async def root():
    return {"message": "知识库服务运行中 "}