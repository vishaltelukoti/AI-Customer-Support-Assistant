from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


TicketText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)]


class TicketCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str | None = Field(default=None, max_length=300)
    body: str | None = Field(default=None, max_length=10000)
    ticket_text: str | None = Field(default=None, max_length=10000)
    top_k: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def validate_ticket_content(self):
        if self.ticket_text is not None:
            text = self.ticket_text.strip()
            if not text:
                raise ValueError("ticket_text must be nonempty.")
            self.ticket_text = text
            self.subject = self.subject or "Customer support request"
            self.body = self.body or text
            return self

        subject = (self.subject or "").strip()
        body = (self.body or "").strip()
        if not subject or not body:
            raise ValueError("subject and body are required when ticket_text is not supplied.")
        combined = f"{subject}\n\n{body}"
        if len(combined) > 10000:
            raise ValueError("combined subject and body must be at most 10,000 characters.")
        self.subject = subject
        self.body = body
        self.ticket_text = combined
        return self


class TicketView(BaseModel):
    ticket_id: UUID
    subject: str
    body: str
    ticket_text: TicketText


class ClassificationView(BaseModel):
    category: str | None = None
    category_confidence: float | None = Field(default=None, ge=0, le=1)
    priority: str | None = None
    priority_confidence: float | None = Field(default=None, ge=0, le=1)


class SimilarTicketView(BaseModel):
    ticket_id: str
    score: float
    category: str
    priority: str
    ticket_text: str


class SourceView(BaseModel):
    ticket_id: str
    similarity_score: float
    category: str
    priority: str
    excerpt: str | None = None


class WorkflowView(BaseModel):
    complexity: Literal["simple", "complex"]
    type: Literal["simple_rag", "complex_multi_agent", "blocked", "error"]
    trace: list[str] = []


class ResponseView(BaseModel):
    answer: str
    sources: list[SourceView] = []
    retrieval_status: str


class ExplanationView(BaseModel):
    predicted_category: str | None = None
    confidence: float | None = None
    top_features: list[dict[str, float | str]] = []


class SecurityView(BaseModel):
    allowed: bool
    status: str
    reason: str
    category: str
    matched_rule: str


class TimingView(BaseModel):
    total_ms: float
    classification_ms: float = 0.0
    retrieval_ms: float = 0.0
    workflow_ms: float = 0.0


class TicketResponse(BaseModel):
    ticket: TicketView
    classification: ClassificationView
    similar_tickets: list[SimilarTicketView]
    workflow: WorkflowView
    response: ResponseView
    explanation: ExplanationView
    security: SecurityView
    timings: TimingView
    status: Literal["completed", "blocked", "error"]
