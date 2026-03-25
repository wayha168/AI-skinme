# Deploy Chatbot at chatbot.skinme.store

This guide covers deploying the Skinme AI chatbot API so it is available at **https://chatbot.skinme.store**.

## Overview

- **Subdomain:** `chatbot.skinme.store` → your server running the FastAPI app (uvicorn).
- **HTTPS:** Use a reverse proxy (nginx or Caddy) with Let's Encrypt.
- **Same DB:** App uses the same MySQL (skinme_db) as your backend; set `MYSQL_*` in `.env` on the server.

---

## Option 1: Server with nginx (recommended)

### 1. DNS

Add an **A record** (or CNAME) for the subdomain:

- **Name:** `chatbot` (or `chatbot.skinme.store` depending on your DNS provider)
- **Value:** IP address of the server where you will run the app (same as backend or a new VPS)
- **TTL:** 300 or default

Wait until `chatbot.skinme.store` resolves (e.g. `ping chatbot.skinme.store`).

### 2. Install and run the app on the server

```bash
# On the server (Ubuntu/Debian example)
cd /opt  # or your preferred path
git clone <your-repo> Skinme_AI
cd Skinme_AI

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Copy and edit .env (MYSQL_*, SPRING_BACKEND_URL, GEMINI_API_KEY, etc.)
cp .env.example .env
nano .env
```

Run the API (listen on localhost so nginx can proxy):

```bash
# Bind to 127.0.0.1 so only nginx can reach it
python main.py serve --host 127.0.0.1 --port 8000
```

Or run with **no sync on start** (if you sync separately):

```bash
python main.py --no-sync-on-start serve --host 127.0.0.1 --port 8000
```

For production, run under **systemd** or **supervisor** so it restarts on failure (see section below).

### 3. nginx reverse proxy and HTTPS

Install nginx and certbot:

```bash
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx -y
```

Create a server block for the subdomain:

```bash
sudo nano /etc/nginx/sites-available/chatbot.skinme.store
```

Paste (replace `chatbot.skinme.store` if different):

```nginx
server {
    listen 80;
    server_name chatbot.skinme.store;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site and get SSL:

```bash
sudo ln -s /etc/nginx/sites-available/chatbot.skinme.store /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
sudo certbot --nginx -d chatbot.skinme.store
```

Certbot will add HTTPS and redirect HTTP → HTTPS. After that, **https://chatbot.skinme.store** will serve your API (e.g. https://chatbot.skinme.store/v1/health, https://chatbot.skinme.store/docs).

### 4. systemd service (keep app running)

```bash
sudo nano /etc/systemd/system/skinme-chatbot.service
```

```ini
[Unit]
Description=Skinme AI Chatbot API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/Skinme_AI
Environment="PATH=/opt/Skinme_AI/venv/bin"
ExecStart=/opt/Skinme_AI/venv/bin/python main.py --no-sync-on-start serve --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable skinme-chatbot
sudo systemctl start skinme-chatbot
sudo systemctl status skinme-chatbot
```

Adjust `User` and `WorkingDirectory` to match your setup. Put `.env` in `WorkingDirectory`.

---

## Option 2: Docker

A **Dockerfile** is provided in the project root. Build and run:

```bash
docker build -t skinme-chatbot .
docker run -d --name chatbot -p 8000:8000 --env-file .env skinme-chatbot
```

Then point nginx at `http://127.0.0.1:8000` as in Option 1. For Docker, ensure `.env` is on the server and passed with `--env-file .env`.

---

## Option 3: Caddy (alternative to nginx)

If you use Caddy, it can handle HTTPS automatically:

```bash
# Caddyfile
chatbot.skinme.store {
    reverse_proxy 127.0.0.1:8000
}
```

Run: `caddy run --config Caddyfile`

---

## Environment on the server

Ensure `.env` on the server includes at least:

- `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` (same as backend)
- `SPRING_BACKEND_URL=https://backend.skinme.store` (optional, for forwarding)
- `GEMINI_API_KEY=...` or `GOOGLE_API_KEY=...` (optional, for Gemini LLM replies)

Optional for docs/links:

- `CHATBOT_URL=https://chatbot.skinme.store`

---

## Frontend integration

After deploy, give the frontend team:

- **Chatbot API base URL:** `https://chatbot.skinme.store`
- **Full endpoint reference:** see **docs/FRONTEND_CHATBOT_API.md** (base URL, request/response for `/v1/chat`, `/v1/chat/with-image`, and examples).

Summary:

- **Health:** `GET https://chatbot.skinme.store/v1/health`
- **Chat (text):** `POST https://chatbot.skinme.store/v1/chat` — body: `message`, `session_id`, optional `user_id`, `user_email`, `user_name`, `use_llm`, `use_database`
- **Chat (with image):** `POST https://chatbot.skinme.store/v1/chat/with-image` — multipart: `message`, `image`, `session_id`, optional user fields

The API allows all origins by default. To restrict CORS to your domains, set `CORS_ORIGINS` in `.env` and update the FastAPI app to use it.

---

## Checklist

- [ ] DNS A/CNAME for `chatbot.skinme.store` → server IP
- [ ] App runs on server (e.g. port 8000, bound to 127.0.0.1)
- [ ] `.env` with MySQL and optional GEMINI_API_KEY, SPRING_BACKEND_URL
- [ ] nginx (or Caddy) reverse proxy with SSL for `chatbot.skinme.store`
- [ ] systemd (or Docker) so the app restarts on failure
- [ ] Frontend calls `https://chatbot.skinme.store` for chat
