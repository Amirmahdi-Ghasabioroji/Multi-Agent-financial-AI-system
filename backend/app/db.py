"""SQLAlchemy models and database lifecycle helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


def new_id() -> str:
    """Return a storage-friendly UUID."""
    return str(uuid4())


class Base(DeclarativeBase):
    """Declarative model base."""


class Conversation(Base):
    """A dashboard conversation containing ordered user/assistant messages."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(255), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """A persisted conversation message."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata_json", JSON, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Job(Base):
    """A durable background job and its JSON result."""

    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_status_created", "status", "created_at"),
        Index("ix_jobs_kind_created", "kind", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    job_metadata: Mapped[dict[str, Any]] = mapped_column(
        "job_metadata", JSON, default=dict, nullable=False
    )
    result: Mapped[dict[str, Any] | list[Any] | None] = mapped_column(
        JSON, nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    events: Mapped[list["JobEvent"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobEvent.sequence",
    )


class JobEvent(Base):
    """A replayable event emitted during job execution."""

    __tablename__ = "job_events"
    __table_args__ = (
        UniqueConstraint("job_id", "sequence", name="uq_job_event_sequence"),
        Index("ix_job_events_job_sequence", "job_id", "sequence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    job: Mapped[Job] = relationship(back_populates="events")


class Database:
    """Own an engine and a thread-safe SQLAlchemy session factory."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        engine_kwargs: dict[str, Any] = {
            "pool_pre_ping": True,
            "echo": echo,
        }
        if url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                engine_kwargs["poolclass"] = StaticPool

        self.engine: Engine = create_engine(url, **engine_kwargs)
        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)
        self.sessions = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
        )

    @staticmethod
    def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    def create_all(self) -> None:
        """Create tables for reliable local startup."""
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        """Release pooled database connections."""
        self.engine.dispose()
