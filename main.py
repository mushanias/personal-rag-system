import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 记得写在加载模型前面！
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from userRoute import router as user



@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("正在加载模型.")
    app.state.dense_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    app.state.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("模型加载完成 ")
    yield
    print("服务关闭 ")

app = FastAPI(lifespan=lifespan)
app.include_router(user)

@app.get("/")
async def root():
    return {"message": "知识库服务运行中 "}