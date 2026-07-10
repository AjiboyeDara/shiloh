from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    top_k: int = Field(6, ge=1, le=20)
    provider: Optional[str] = None  # "ollama" | "gemini" | "anthropic"; None = server default
    model: Optional[str] = None     # model name within the provider; None = provider default


class PassageResult(BaseModel):
    reference: str
    text: str
    book: str
    chapter: int
    verse_start: Optional[int] = None
    verse_end: Optional[int] = None
    cross_references: List[str] = []


class ChapterVerse(BaseModel):
    verse: int
    text: str


class ChapterResponse(BaseModel):
    book: str
    chapter: int
    verses: List[ChapterVerse]


class QuoteCheck(BaseModel):
    quote: str
    status: str  # "verified" | "mismatch" | "not_found"
    reference: Optional[str] = None
    actual: Optional[str] = None  # KJV wording, set when status is "mismatch"


class ChatResponse(BaseModel):
    answer: str
    passages: List[PassageResult]
    quote_checks: List[QuoteCheck] = []


class StrongsEntry(BaseModel):
    number: str  # e.g. "G26" or "H430"
    lemma: str = ""
    translit: str = ""
    pron: str = ""
    definition: str = ""
    kjv_def: str = ""


class WordOccurrence(BaseModel):
    reference: str
    text: str


class WordStudyResponse(BaseModel):
    word: str
    strongs: List[StrongsEntry] = []
    count: int
    occurrences: List[WordOccurrence] = []


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(8, ge=1, le=20)


class SearchResponse(BaseModel):
    results: List[PassageResult]
