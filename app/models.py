from typing import List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    top_k: int = 6
    provider: Optional[str] = None  # "ollama" | "gemini" | "anthropic"; None = server default
    model: Optional[str] = None     # model name within the provider; None = provider default


class PassageResult(BaseModel):
    reference: str
    text: str
    book: str
    chapter: int
    verse_start: Optional[int] = None
    verse_end: Optional[int] = None


class ChapterVerse(BaseModel):
    verse: int
    text: str


class ChapterResponse(BaseModel):
    book: str
    chapter: int
    verses: List[ChapterVerse]


class ChatResponse(BaseModel):
    answer: str
    passages: List[PassageResult]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8


class SearchResponse(BaseModel):
    results: List[PassageResult]
