# Encryption, Transport Security, and Data Handling (Skinme_AI)

This document explains **encryption-related concepts** and how they apply to the Skinme_AI stack so you can reuse the wording and structure in a **technical or security chapter** of your report.

---

## 1. Core concepts (for your report)

### 1.1 What “encryption” means in web systems

- **Encryption** turns readable data (plaintext) into ciphertext using a key. Only someone with the right key (or the right mathematical relationship, as in public-key crypto) can recover the original data.
- **Symmetric encryption** uses one shared secret key for both encrypt and decrypt (fast; used for bulk data, e.g. AES).
- **Asymmetric (public-key) encryption** uses a **public key** to encrypt and a **private key** to decrypt (or the reverse for signatures). It solves “how do we share a secret over an untrusted network?”—often used to establish a session key, then symmetric crypto carries the traffic.

### 1.2 TLS / HTTPS (data in transit)

- **TLS** (Transport Layer Security) protects data **on the wire** between client and server. When the URL starts with `https://`, the browser and server perform a **TLS handshake**: they agree on algorithms, authenticate the server (usually with **X.509 certificates**), and derive session keys. After that, HTTP requests and responses are **encrypted** and **integrity-protected** (tampering is detected).
- **Why it matters for Skinme_AI:** User messages, images uploaded to `/v1/chat/with-image`, and API responses should travel over **HTTPS** in production so they are not readable or modifiable by eavesdroppers on the path (Wi‑Fi, ISPs, etc.).

### 1.3 Data at rest

- **At rest** means data stored on disk or in a database (CSV under `data/`, MySQL tables, server logs).
- Full **database encryption** (transparent disk encryption, column-level encryption) is typically a **deployment and DBA** choice, not something the Python app alone guarantees.
- **Passwords and API keys** in configuration must **not** be committed to Git; they belong in environment variables or a secrets manager (see §3).

### 1.4 Hashing vs encryption (do not confuse them)

- **Hashing** (e.g. SHA-256, bcrypt for passwords) is usually **one-way**: you cannot “decrypt” a hash back to the password. It is used for **integrity checks** or **password verification**, not for reversible storage of messages.
- **Encryption** is **reversible** with the key. Your chat content is not “hashed for privacy” in the app—it is either stored as application data or sent over TLS.

---

## 2. How this maps to the Skinme_AI architecture

### 2.1 Production URLs (TLS expected)

| Role | Example base | Security note |
|------|----------------|---------------|
| Chatbot API | `https://chatbot.skinme.store` | Browser/FE should use HTTPS; TLS terminates at your reverse proxy or host. |
| Spring / product API | `https://backend.skinme.store` | Same; sync and webhooks use HTTPS in production settings. |
| Storefront / assets | `https://skinme.store` | Public pages and image URLs. |

Local development often uses `http://localhost:8000` without TLS; that is acceptable **only on the same machine** and should not be used for real user data over a network.

### 2.2 Data flows (high level)

1. **User → Frontend (browser)**  
   Login and session handling are your Spring/frontend responsibility (e.g. cookies, JWT—document what **you** implemented there).

2. **Frontend → Skinme_AI API**  
   `POST /v1/chat`, `POST /v1/chat/with-image`: should use **HTTPS** in production so message and image bytes are protected by TLS.

3. **Skinme_AI → Google Generative AI (Gemini)**  
   The official client uses **HTTPS** to Google’s APIs; API keys must stay server-side (environment variables), never in frontend code.

4. **Skinme_AI → Spring backend**  
   When `SPRING_BACKEND_URL` is set, the app forwards payloads to `{SPRING_BACKEND_URL}/api/v1/chat/log` and `/api/v1/feedback` via HTTP(S) from the server. **Use HTTPS** for that base URL in production so logs are not sent in cleartext over the internet.

5. **Skinme_AI ↔ MySQL**  
   Connection uses credentials from `MYSQL_*` env vars. Prefer **TLS to MySQL** if the database is remote (configure MySQL server and connection string / SSL flags as per your host’s docs).

### 2.3 What the Python service does *not* do by itself

- It does **not** replace TLS: you rely on **deployment** (nginx, Caddy, cloud load balancer, PaaS) for HTTPS.
- It uses **CORS** allowing all origins in code; tightening CORS to your real frontend origin is a **recommended hardening step** for production (see `skin_assistant/api/app.py`).

---

## 3. Secrets and configuration (report checklist)

| Secret / sensitive item | Typical storage | Risk if leaked |
|-------------------------|-----------------|----------------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `.env`, host env, secret manager | Abuse of your Google AI quota; billing. |
| `MYSQL_PASSWORD` | Same | Full DB access as that user. |
| `SPRING_BACKEND_URL` | Same | Misrouting or probing; combine with network controls. |

**Best practices to mention in your report:**

- Keep `.env` out of version control (use `.env.example` without real secrets).
- Rotate keys if exposed.
- Run the API only on trusted networks or behind authentication if the service must not be public.

---

## 4. Suggested outline for your report (“Security & encryption” section)

You can adapt this to your faculty’s template:

1. **Threat model (brief):** eavesdropping on user messages/images, credential theft, unauthorized API use.
2. **Controls implemented:**
   - TLS (HTTPS) for user-facing and service-to-service calls in production.
   - Secrets only in environment variables / deployment secrets.
   - Separation: LLM keys on server; frontend never holds Gemini keys.
3. **Residual risks / future work:** stricter CORS, API authentication between FE and chatbot, MySQL SSL, rate limiting, audit logging.
4. **Diagram:** Browser → (HTTPS) → Chatbot API → (HTTPS) → Gemini; Chatbot → (HTTPS) → Spring; optional MySQL with TLS.

---

## 5. Related docs in this repo

| Document | Content |
|----------|---------|
| [FRONTEND_CHATBOT_API.md](FRONTEND_CHATBOT_API.md) | Public HTTP API for the frontend. |
| [CHATBOT_WITH_BACKEND.md](CHATBOT_WITH_BACKEND.md) | How the chatbot ties to `backend.skinme.store` and MySQL. |
| [SPRING_BACKEND_INTEGRATION.md](SPRING_BACKEND_INTEGRATION.md) | Chat log and feedback forwarding to Spring. |
| [TRAINING_WITH_BACKEND.md](TRAINING_WITH_BACKEND.md) | Training pipeline vs runtime API. |

---

## 6. Glossary (one-liners for appendices)

| Term | Short definition |
|------|------------------|
| TLS | Protocol that provides confidentiality and integrity for TCP connections; used by HTTPS. |
| Certificate | Binds a public key to a domain; issued by a CA trusted by browsers. |
| Plaintext / ciphertext | Readable data vs encrypted form. |
| Secret / API key | High-entropy string proving authorization to a service; must stay private. |
