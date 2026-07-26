from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: str | None = None


class TranscriptResult(BaseModel):
    source_name: str
    language: str | None = None
    model: str
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return "\n".join(segment.text.strip() for segment in self.segments if segment.text)


class Chapter(BaseModel):
    start_time: str = ""
    title: str
    summary: str


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    deadline: str | None = None


class SummaryResult(BaseModel):
    title: str
    overview: str
    key_points: list[str] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

