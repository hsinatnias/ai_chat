BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "conversations" (
	"id"	VARCHAR(36) NOT NULL,
	"session_id"	VARCHAR(64),
	"user_id"	VARCHAR(64),
	"name"	VARCHAR(255),
	"created_at"	DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "ingested_files" (
	"id"	INTEGER,
	"module"	TEXT NOT NULL,
	"filename"	TEXT NOT NULL,
	"file_path"	TEXT NOT NULL,
	"content_hash"	TEXT NOT NULL,
	"file_size"	INTEGER,
	"uploaded_at"	TEXT DEFAULT (datetime('now')),
	"ingested_at"	TEXT DEFAULT NULL,
	"status"	TEXT DEFAULT 'pending',
	"qdrant_collection"	TEXT DEFAULT NULL,
	"points_count"	INTEGER DEFAULT 0,
	"error"	TEXT DEFAULT NULL,
	UNIQUE("content_hash","module"),
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "message_feedback" (
	"id"	VARCHAR(36) NOT NULL,
	"message_id"	VARCHAR(36) NOT NULL,
	"helpful"	BOOLEAN,
	"comment"	TEXT,
	"created_at"	DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY("id"),
	FOREIGN KEY("message_id") REFERENCES "messages"("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "messages" (
	"id"	VARCHAR(36) NOT NULL,
	"conversation_id"	VARCHAR(36) NOT NULL,
	"session_id"	VARCHAR(64),
	"user_id"	VARCHAR(64),
	"role"	VARCHAR(32) NOT NULL,
	"content"	TEXT NOT NULL,
	"created_at"	DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY("id"),
	FOREIGN KEY("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "sessions" (
	"id"	VARCHAR NOT NULL,
	"user_id"	VARCHAR NOT NULL,
	"created_at"	DATETIME,
	"expires_at"	DATETIME,
	"meta"	JSON,
	PRIMARY KEY("id"),
	FOREIGN KEY("user_id") REFERENCES "users"("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "users" (
	"id"	VARCHAR NOT NULL,
	"email"	VARCHAR,
	"name"	VARCHAR,
	"password_hash"	VARCHAR,
	"is_active"	BOOLEAN NOT NULL,
	"created_at"	DATETIME,
	PRIMARY KEY("id")
);
CREATE INDEX IF NOT EXISTS "ix_conversations_session_id" ON "conversations" (
	"session_id"
);
CREATE INDEX IF NOT EXISTS "ix_conversations_user_id" ON "conversations" (
	"user_id"
);
CREATE INDEX IF NOT EXISTS "ix_message_feedback_message_id" ON "message_feedback" (
	"message_id"
);
CREATE INDEX IF NOT EXISTS "ix_messages_conversation_id" ON "messages" (
	"conversation_id"
);
CREATE INDEX IF NOT EXISTS "ix_messages_session_id" ON "messages" (
	"session_id"
);
CREATE INDEX IF NOT EXISTS "ix_messages_user_id" ON "messages" (
	"user_id"
);
CREATE INDEX IF NOT EXISTS "ix_sessions_user_id" ON "sessions" (
	"user_id"
);
CREATE UNIQUE INDEX IF NOT EXISTS "ix_users_email" ON "users" (
	"email"
);
COMMIT;
