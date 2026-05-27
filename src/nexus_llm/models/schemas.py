from typing import Literal

from pydantic import BaseModel, ConfigDict


class ImageUrl(BaseModel):
    url: str


class ContentBlock(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ContentBlock]
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    keep_alive: int | str | None = None
