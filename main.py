
from settings import settings


from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from userRoute import router as user
from database import client, COLLECTION_NAME, COLLECTION_CONFIG


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("正在加载模型.")
    app.state.dense_model = SentenceTransformer(settings.DENSE_MODEL_NAME)
    app.state.sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL_NAME)
    # ✅ 初始化 Collection
    if not await client.collection_exists(COLLECTION_NAME):
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            **COLLECTION_CONFIG
        )
    print("模型加载完成 ")
    yield
    await client.close()
    print("服务关闭 ")

app = FastAPI(lifespan=lifespan)
app.include_router(user)

@app.get("/")
async def root():
    return {"message": "知识库服务运行中 "}