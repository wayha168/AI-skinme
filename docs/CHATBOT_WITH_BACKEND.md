# Chatbot with Backend (backend.skinme.store)

## Should you connect the AI to the backend or scrape and then connect FE to the chatbot?

**Recommendation: Connect the AI to the backend.** Do not scrape.

Your Spring Boot backend at **backend.skinme.store** is the source of truth for products and users. The Skinme_AI chatbot should use that backend so recommendations always reflect **products that are available** in your system.

---

## How it works today (already “connected”)

| What | How |
|------|-----|
| **Product data** | Skinme_AI already uses your backend **REST API**: `GET https://backend.skinme.store/api/v1/products/all`. Run `python main.py sync` to pull products into CSV. Optionally use the **same MySQL** as Spring (`MYSQL_*` env) so the chatbot reads live from your DB. |
| **Chat logs per user** | Set `SPRING_BACKEND_URL=https://backend.skinme.store`. The AI will **POST** each chat turn to your Spring app (`/api/v1/chat/log`) and feedback to `/api/v1/feedback`. Implement those endpoints in Spring to store in your DB. |

So: **no scraping**. Use the backend API (and optionally shared MySQL).

---

## Two ways to get “available products” into the chatbot

### Option A: Sync from backend (current, recommended to start)

1. **Sync products** from backend to CSV:  
   `python main.py sync`  
   (calls `https://backend.skinme.store/api/v1/products/all` and writes `data/skinme_products.csv`).
2. Run sync **periodically** (cron or after deploy) so the chatbot’s product list stays close to what’s on the backend.
3. **Frontend:** User logs in on your main app (Spring). When they open chat, the FE calls your **Skinme_AI API** (see below) with `session_id` (and optionally `user_id`). The chatbot answers using the synced product data.

**Pros:** Simple, no extra load on backend per message.  
**Cons:** Product list can be a few minutes/hours old until next sync.

### Option B: Same MySQL as Spring (live products)

1. Point Skinme_AI at the **same MySQL** your Spring backend uses (e.g. `skinme_db`).
2. Set `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` (and optionally `MYSQL_PRODUCTS_TABLE`).
3. In chat (and product search), use **“check with database”** / `use_database=true`. The chatbot will query that DB for product recommendations.

**Pros:** Recommendations always match what’s in the backend DB.  
**Cons:** Requires DB access from where Skinme_AI runs; you must keep product table schema compatible (see `skinme_db.py`).

You can use **both**: sync for fast local search and MySQL when `use_database=true` for live data.

---

## Frontend integration (user logged in)

1. **User logs in** on your main app (Spring) at backend.skinme.store (or your frontend domain).
2. **When the user opens the chat**, your FE calls the **Skinme_AI API** (deploy it on a URL you control, e.g. `https://ai.skinme.store` or same server):
   - **POST /v1/chat**  
     Body: `{ "message": "...", "session_id": "<from Spring session or user id>", "use_llm": true, "use_database": true }`  
     - Use `session_id` so the AI can load/save history (and so you can log to Spring with the same id).
   - **POST /v1/chat/with-image** for skin photo + message (multipart: `message`, `image`, `session_id`).
3. **Optional:** Send `user_id` in your own payload to **POST /v1/chat/log** (or extend the AI’s `ChatLogRequest` to accept `user_id` from the FE so Spring can store which user the chat belongs to).

So: **FE talks to your backend for login; FE talks to Skinme_AI for chat; AI uses backend (API sync and/or MySQL) for products and can forward chat logs to backend.**

---

## Summary

| Question | Answer |
|----------|--------|
| Connect AI to backend or scrape? | **Connect.** Use backend API (sync) and optionally shared MySQL. No scraping. |
| Where do “available products” come from? | Backend: sync from `backend.skinme.store` and/or query same MySQL as Spring. |
| How does the chatbot “check with backend” for products? | Either synced CSV (from backend API) or `use_database=true` with MySQL (same DB as Spring). |
| How does FE use the chatbot for logged-in users? | FE calls Skinme_AI **POST /v1/chat** (and optionally /v1/chat/with-image) with `session_id`; set `SPRING_BACKEND_URL=https://backend.skinme.store` so chat logs are sent to Spring. |

See **.env.example** and **docs/SPRING_BACKEND_INTEGRATION.md** for `SPRING_BACKEND_URL` and MySQL settings.
