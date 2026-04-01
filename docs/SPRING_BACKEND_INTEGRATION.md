# Spring Backend Integration (chat logs and feedback)

Skinme_AI can **forward** chat logs and feedback to your Spring Boot app so you persist them in your database. Set the base URL in the environment.

---

## Environment

```env
# Base URL only (no trailing path). Example production:
SPRING_BACKEND_URL=https://backend.skinme.store

# Local Spring:
# SPRING_BACKEND_URL=http://localhost:8080
```

Optional alias: `BACKEND_WEBHOOK_URL` (same purpose as `SPRING_BACKEND_URL`).

---

## Endpoints your Spring app should expose

The Python API **POSTs JSON** to these paths **on your backend host**:

| Method | Path on your backend | Purpose |
|--------|----------------------|---------|
| POST | `/api/v1/chat/log` | Persist a chat turn (payload from Skinme_AI `ChatLogRequest`). |
| POST | `/api/v1/feedback` | Persist user feedback (payload from `FeedbackRequest`). |

Skinme_AI exposes the same logical paths under its own prefix (e.g. `POST /v1/chat/log`, `POST /v1/feedback`); when `SPRING_BACKEND_URL` is set, it **also** forwards to `{SPRING_BACKEND_URL}/api/v1/chat/log` and `{SPRING_BACKEND_URL}/api/v1/feedback`.

See `skin_assistant/api/routes.py` (`_forward_to_backend`, `chat_log`, `feedback`) and Pydantic models in `skin_assistant/domain/schemas.py` for exact JSON fields.

---

## Optional: MySQL shared with Spring

If the chatbot reads products from the same database as Spring, set `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` (see `.env.example`). Use **TLS** for remote MySQL when possible.

---

## See also

- [CHATBOT_WITH_BACKEND.md](CHATBOT_WITH_BACKEND.md) — product sync and high-level flow.
- [FRONTEND_CHATBOT_API.md](FRONTEND_CHATBOT_API.md) — frontend-facing chat endpoints.
