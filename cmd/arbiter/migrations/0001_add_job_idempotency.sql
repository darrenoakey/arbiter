CREATE TABLE job_idempotency (
    idempotency_key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL UNIQUE,
    request_hash TEXT NOT NULL
);
