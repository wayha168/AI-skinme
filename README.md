# Skin Assistant

AI skincare assistant: REST API, product data sync from [SkinMe API](https://backend.skinme.store/api/v1/products/all), CSV export, optional scraping (bs4), and ML training (intent + product text + image).

## Clean architecture (grouped folders)

```
Data-Mining/
├── data/                      # CSVs (ingredients, skinme_products, etc.)
│   ├── product_images/        # Downloaded product images (sync)
│   ├── skin_disease_images/   # Optional: images for skin condition training
│   └── skin_disease_labels.csv.example  # Optional: image_name, condition for training
├── models/
│   └── artifacts/             # intent_model.joblib, product_type_model.joblib, product_image_model.pt
├── scripts/
│   ├── train.py               # Train intent model
│   ├── sync_products.py       # Fetch API -> CSV, download images, cleanup unused
│   └── train_products.py      # Train product text + optional image model
├── src/skin_assistant/
│   ├── api/                   # FastAPI app & routes
│   ├── config/                # Settings & paths
│   ├── domain/                # Schemas (DTOs)
│   ├── infrastructure/        # KnowledgeRepository, skinme_client, scraper (bs4)
│   ├── models/                # ML: intent trainer, product_trainer (text + image)
│   └── services/              # Chat service
├── main.py                    # serve | sync | train | train-products | train-skin-condition
└── pyproject.toml
```

## Run the API (backend integration)

From project root:

```bash
pip install -r requirements.txt
python main.py
```

API base: `http://localhost:8000`

- **OpenAPI docs:** http://localhost:8000/docs  
- **POST /v1/chat** — Chat with the assistant (body: `{"message": "...", "history": [], "use_llm": true, "session_id": "optional-for-DB"}`). If `session_id` is set and MySQL is configured, the turn is saved to `chat_messages`.  
- **POST /v1/chat/with-image** — Send a skin photo + optional message (multipart: `message`, `image`, `session_id?`). We analyze the image and reply with recommendations; turn is saved to DB when `session_id` is set.  
- **GET /v1/ingredients/search?q=niacinamide** — Search ingredients  
- **GET /v1/ingredients/{name}** — Get ingredient by name  
- **GET /v1/products?concern=dry** or **?ingredient=hyaluronic+acid** — Product search  
- **GET /v1/intent?q=...** — Predict intent (needs trained model)  
- **GET /v1/routes** — List all route paths for backend integration  
- **GET /v1/health** — Health check  

**Save-to-database (Spring backend integration):**

- **POST /v1/chat/log** — Log a chat turn (session_id, user_id, message, reply) so your backend can persist to DB.  
- **POST /v1/feedback** — Save user feedback (session_id, rating/thumbs_up, comment) for DB.  

If you set `SPRING_BACKEND_URL` (e.g. `http://localhost:8080`), the API forwards these payloads to your Spring app (`POST {url}/api/v1/chat/log` and `POST {url}/api/v1/feedback`). Implement those endpoints in Spring to save to your database.

## Optional: MySQL (skinme_db) for product recommendations

You can **check with the live database** when chatting or searching products. Set these **environment variables** (never commit real credentials; use `.env` and keep it out of git):

- `MYSQL_HOST` — e.g. your MySQL server host  
- `MYSQL_PORT` — default `3306`  
- `MYSQL_USER` — DB user  
- `MYSQL_PASSWORD` — DB password  
- `MYSQL_DATABASE` — e.g. `skinme_db`  
- `MYSQL_PRODUCTS_TABLE` — optional; default `product`

Copy `.env.example` to `.env`, fill in the MySQL section, and load it (e.g. `python-dotenv` or export in shell). Then:

- **Chat:** Send `POST /v1/chat` with `"use_database": true` in the body to get product suggestions from MySQL.
- **Product search:** `GET /v1/products?concern=dry&use_database=true` to search the database.
- **Streamlit:** Use the sidebar checkbox “Check with database (skinme_db)”.

Example:

```bash
curl -X POST http://localhost:8000/v1/chat -H "Content-Type: application/json" -d "{\"message\": \"What is niacinamide?\"}"
```

## Sync product data (SkinMe API → CSV, images, cleanup)

Fetches from `https://backend.skinme.store/api/v1/products/all`, writes `data/skinme_products.csv`, downloads images to `data/product_images/`, and **deletes image files for products no longer in the API**.

```bash
python main.py sync
# Options:
python main.py sync --no-download    # only update CSV, skip image download
python main.py sync --no-cleanup     # do not delete unused image files
# Or run script directly:
python -m scripts.sync_products [--no-download] [--no-cleanup] [--no-sync]
```

Run sync periodically to auto-update when new products are added and to clean up removed products.

## Scraping (BeautifulSoup)

For **JSON APIs** (e.g. SkinMe) the sync uses `requests`; no bs4 needed. For **HTML pages**, use the optional scraper:

```python
from skin_assistant.infrastructure.scraper import fetch_html, parse_html, scrape_table_to_rows, save_scraped_to_csv
rows = scrape_table_to_rows("https://example.com/products")
save_scraped_to_csv(rows, Path("data/scraped.csv"))
```

Install: `pip install beautifulsoup4` (included in requirements).

## Train the intent model (scikit-learn)

Uses a small built-in dataset; you can add a CSV with columns `text` and `intent` for better accuracy.

```bash
python main.py train
# Or with custom data/output:
python -m scripts.train --data data/intent_training.csv --output models/artifacts
```

Intents: `greeting`, `ingredient_info`, `products_with_ingredient`, `product_recommendation`, `general_ingredient_search`, `other`.  
Model is saved to `models/artifacts/intent_model.joblib` and used by **GET /v1/intent**.

## Train product models (text + image)

Uses **synced data** (`data/skinme_products.csv` and optionally `data/product_images/`). Run `python main.py sync` first.

- **Text model:** predicts `productType` (or `category_name`) from product name + description (sklearn).
- **Image model:** predicts `productType` from product image (PyTorch/torchvision; optional).

```bash
python main.py train-products              # text model only
python main.py train-products --image       # text + image model (requires torch, torchvision)
python main.py train-products --image --epochs 10 --batch-size 8
# Or:
python -m scripts.train_products [--image] [--label productType|category_name]
```

Artifacts: `models/artifacts/product_type_model.joblib`, `models/artifacts/product_image_model.pt`.  
For image training: `pip install torch torchvision`.

## Product recommendations from SkinMe

The chatbot and **GET /v1/products?concern=...** use the **SkinMe** product database first (`data/skinme_products.csv`). Run `python main.py sync` to fetch the latest products. For “recommend products for dry skin” or “I have acne”, results come from SkinMe; “products containing [ingredient]” still uses `data/skincare_products_clean.csv` when ingredient lists are available.

## Train skin condition classifier (ingredients + skin disease images)

Train a model from your **ingredients** data (already used by intent training) and **skin disease/condition images** so you can recommend SkinMe products by predicted condition.

**Data options:**

1. **Folder structure:** put images in `data/skin_disease_images/<condition>/` (e.g. `acne/`, `dryness/`), one image per file.
2. **CSV:** create `data/skin_disease_labels.csv` with columns `image_name`, `condition` and place image files in `data/skin_disease_images/`. See `data/skin_disease_labels.csv.example`.

```bash
python main.py train-skin-condition
# Or with custom paths:
python -m scripts.train_skin_condition --images data/skin_disease_images --csv data/skin_disease_labels.csv --epochs 10
```

Model is saved to `models/artifacts/skin_condition_model.pt`. Use it to predict condition from a user image and then recommend products from the SkinMe database by mapping condition → concern (e.g. acne → “acne” concern search).

## Optional: Streamlit chat UI

```bash
pip install streamlit
streamlit run app.py
```

(Requires the legacy `skin_chatbot`/`skin_knowledge` modules at project root, or point the app to the API.)

## Optional: LLM replies

Set `OPENAI_API_KEY` so **POST /v1/chat** uses GPT with your ingredients/products as context:

```bash
set OPENAI_API_KEY=sk-...
python main.py
```
#   s k i n m e - r e c o m m e n d a t i o n 
 
 #   s k i n m e - r e c o m m e n d a t i o n 
 
 