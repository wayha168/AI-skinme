# Spring Backend Integration

Use these route paths to integrate the Skin Assistant API with your Spring Boot app and save data to your database.

## 1. Route paths (Skin Assistant API)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat` | Get assistant reply (request: `message`, `history`, `use_llm`) |
| POST | `/v1/chat/log` | **Save chat turn to DB** — call from your app or let Skin Assistant forward to Spring |
| POST | `/v1/feedback` | **Save user feedback to DB** |
| GET | `/v1/ingredients/search?q=...` | Search ingredients |
| GET | `/v1/ingredients/{name}` | Get ingredient by name |
| GET | `/v1/products?concern=...` or `?ingredient=...` | Search products |
| GET | `/v1/intent?q=...` | Predict message intent |
| GET | `/v1/routes` | List all routes (for reference) |
| GET | `/v1/health` | Health check |

Base URL: `http://localhost:8000` when running `python main.py`.

## 2. Save-to-database endpoints (implement in Spring)

If you set `SPRING_BACKEND_URL=http://localhost:8080`, the Skin Assistant API will **POST** these payloads to your Spring app. Implement the following in Spring to persist to your database.

### POST /api/v1/chat/log

**Request body (JSON):**

```json
{
  "session_id": "uuid-or-session-key",
  "user_id": "optional-user-id",
  "message": "What is niacinamide?",
  "reply": "Niacinamide is...",
  "timestamp": "2025-02-23T12:00:00Z"
}
```

**Spring example (Java):**

```java
@PostMapping("/api/v1/chat/log")
public ResponseEntity<?> chatLog(@RequestBody ChatLogRequest dto) {
    // Save dto to your DB (e.g. chat_logs table)
    chatLogRepository.save(ChatLogEntity.from(dto));
    return ResponseEntity.ok().build();
}
```

### POST /api/v1/feedback

**Request body (JSON):**

```json
{
  "session_id": "uuid-or-session-key",
  "message_id": "optional-id",
  "rating": 5,
  "thumbs_up": true,
  "comment": "Very helpful"
}
```

**Spring example (Java):**

```java
@PostMapping("/api/v1/feedback")
public ResponseEntity<?> feedback(@RequestBody FeedbackRequest dto) {
    feedbackRepository.save(FeedbackEntity.from(dto));
    return ResponseEntity.ok().build();
}
```

## 3. Calling Skin Assistant from Spring

From your Spring app you can:

- **GET** ingredients/products for syncing to your DB or for search.
- **POST** `/v1/chat` with `{"message": "...", "history": []}` to get a reply, then save the turn via your own service that calls your `POST /api/v1/chat/log` or the client can call Skin Assistant’s **POST /v1/chat/log** with the same payload so it gets forwarded to Spring (if `SPRING_BACKEND_URL` is set).

## 4. Environment variable

```bash
# Optional: Skin Assistant will forward POST /v1/chat/log and POST /v1/feedback to your Spring app
export SPRING_BACKEND_URL=http://localhost:8080
python main.py
```

Use `BACKEND_WEBHOOK_URL` instead if you prefer that name.
