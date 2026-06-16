class AppException(Exception):
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(message)


class RetrievalError(AppException):
    def __init__(self, message: str = "知识库检索失败"):
        super().__init__(
            message=message,
            error_code="RETRIEVAL_ERROR",
            status_code=500,
        )


class LLMServiceError(AppException):
    def __init__(self, message: str = "大模型服务调用失败"):
        super().__init__(
            message=message,
            error_code="LLM_SERVICE_ERROR",
            status_code=502,
        )