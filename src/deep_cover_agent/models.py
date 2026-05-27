from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentEventType(StrEnum):
    ROOM_STARTED = "ROOM_STARTED"
    CHAT_MESSAGE = "CHAT_MESSAGE"
    VOTING_STARTED = "VOTING_STARTED"
    PLAYER_ELIMINATED = "PLAYER_ELIMINATED"
    ROUND_STARTED = "ROUND_STARTED"
    GAME_ENDED = "GAME_ENDED"
    ROOM_DESTROYED = "ROOM_DESTROYED"


class JavaModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class AgentEvent(JavaModel):
    event_id: str = Field(alias="eventId")
    type: AgentEventType
    room_code: str = Field(alias="roomCode")
    created_at: datetime = Field(alias="createdAt")
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_java_timestamp(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            parts = list(value)
            if len(parts) < 3:
                return value
            year, month, day = (int(part) for part in parts[:3])
            hour = int(parts[3]) if len(parts) > 3 else 0
            minute = int(parts[4]) if len(parts) > 4 else 0
            second = int(parts[5]) if len(parts) > 5 else 0
            nano = int(parts[6]) if len(parts) > 6 else 0
            return datetime(
                year,
                month,
                day,
                hour,
                minute,
                second,
                nano // 1000,
                tzinfo=timezone.utc,
            )
        if isinstance(value, dict) and "epochSecond" in value:
            seconds = int(value["epochSecond"])
            nano = int(value.get("nano", 0))
            return datetime.fromtimestamp(seconds + nano / 1_000_000_000, timezone.utc)
        return value


class AgentPlayerView(JavaModel):
    player_id: str = Field(alias="playerId")
    type: str
    alive: bool
    host: bool


class Topic(JavaModel):
    id: str
    content: str


class AgentRoomStateResponse(JavaModel):
    room_code: str = Field(alias="roomCode")
    status: str
    round_number: int = Field(alias="roundNumber")
    alive_human_count: int = Field(alias="aliveHumanCount")
    alive_ai_count: int = Field(alias="aliveAiCount")
    topic: Topic | None = None
    players: list[AgentPlayerView] = Field(default_factory=list)


class AgentMessageView(JavaModel):
    message_id: str = Field(alias="messageId")
    sender_player_id: str = Field(alias="senderPlayerId")
    content: str
    created_at: datetime = Field(alias="createdAt")


class AgentRecentMessagesResponse(JavaModel):
    room_code: str = Field(alias="roomCode")
    messages: list[AgentMessageView] = Field(default_factory=list)


class AgentVoteStateResponse(JavaModel):
    room_code: str = Field(alias="roomCode")
    round_number: int = Field(alias="roundNumber")
    submitted_vote_count: int = Field(alias="submittedVoteCount")
    required_vote_count: int = Field(alias="requiredVoteCount")
    candidate_player_ids: list[str] = Field(default_factory=list, alias="candidatePlayerIds")
