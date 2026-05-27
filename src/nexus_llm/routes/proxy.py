from fastapi import APIRouter

from nexus_llm.models.schemas import ChatCompletionRequest

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, str]:
    return {"message": "Skeleton for /v1/chat/completions"}
