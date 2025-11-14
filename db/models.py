# db/models.py
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid
from sqlalchemy.dialects.sqlite import JSON

Base = declarative_base()

def mk_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    # use string UUID as primary key (consistent with your other tables)
    id = sa.Column(sa.String, primary_key=True, default=mk_uuid)
    email = sa.Column(sa.String, nullable=True, unique=True, index=True)
    name = sa.Column(sa.String, nullable=True)
    password_hash = sa.Column(sa.String, nullable=True)  # optional for your auth method
    is_active = sa.Column(sa.Boolean, nullable=False, default=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=datetime.utcnow)

class Conversation(Base):
    __tablename__ = "conversations"
    id = sa.Column(sa.String, primary_key=True, default=mk_uuid)
    session_id = sa.Column(sa.String, nullable=True, index=True)
    # now a foreign key to users.id (nullable if conversation can be anonymous)
    user_id = sa.Column(sa.String, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    name = sa.Column(sa.String, nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = sa.Column(sa.String, primary_key=True, default=mk_uuid)
    conversation_id = sa.Column(sa.String, sa.ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    session_id = sa.Column(sa.String, nullable=True, index=True)
    # link message to user (nullable: assistant messages might have user_id NULL)
    user_id = sa.Column(sa.String, sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    role = sa.Column(sa.String, nullable=False)  # 'user'|'assistant'|'system'
    content = sa.Column(sa.Text, nullable=False)
    created_at = sa.Column(sa.DateTime(timezone=True), default=datetime.utcnow)

class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    id = sa.Column(sa.String, primary_key=True, default=mk_uuid)
    message_id = sa.Column(sa.String, sa.ForeignKey("messages.id", ondelete="CASCADE"), index=True)
    helpful = sa.Column(sa.Boolean, nullable=True)
    comment = sa.Column(sa.Text, nullable=True)
    created_at = sa.Column(sa.DateTime(timezone=True), default=datetime.utcnow)

class Session(Base):
    __tablename__ = "sessions"
    id = sa.Column(sa.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = sa.Column(sa.String, sa.ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    created_at = sa.Column(sa.DateTime(timezone=True), default=datetime.utcnow)
    expires_at = sa.Column(sa.DateTime(timezone=True), nullable=True)
    meta = sa.Column(JSON, nullable=True)
