# Skinme AI Chatbot — deploy at chatbot.skinme.store
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e . 2>/dev/null || true

COPY . .

# Default: run API on 8000 (use --no-sync-on-start in CMD if you sync elsewhere)
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "main.py", "--no-sync-on-start", "serve", "--host", "0.0.0.0", "--port", "8000"]
