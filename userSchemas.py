from pydantic import BaseModel, Field, field_validator


class UserRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户提问内容"
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        return value


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float
    chunk_id: str | None = None


class UserResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]

