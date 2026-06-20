# 🚀 AI CV Analyzer — Setup & Deployment Guide

> Step-by-step guide for local development, production deployment, and troubleshooting.

---

## Table of Contents

- [Local Development Setup](#local-development-setup)
- [Getting a Groq API Key](#getting-a-groq-api-key)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10+ | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |

### Step 1: Clone the Repository

```bash
git clone https://github.com/FahadMustafa001/AI_CV_Analyzer.git
cd AI_CV_Analyzer
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Full dependency list:**

| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `werkzeug` | File upload security |
| `pdfplumber` | PDF text extraction |
| `python-docx` | DOCX file parsing |
| `scikit-learn` | TF-IDF vectorization |
| `groq` | Groq AI SDK |
| `instructor` | Structured LLM outputs |
| `pydantic` | Data validation |
| `reportlab` | PDF generation |
| `python-dotenv` | Environment variable management |
| `gunicorn` | Production WSGI server (Linux) |

### Step 4: Configure API Key

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=gsk_your_api_key_here
```

### Step 5: Run the Application

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

The app runs in debug mode with auto-reload. Any file changes will automatically restart the server.

---

## Getting a Groq API Key

1. Visit **https://console.groq.com/**
2. Sign up (Google login supported — **completely FREE**)
3. Go to **API Keys** → https://console.groq.com/keys
4. Click **"Create API Key"**
5. Copy the key (starts with `gsk_`)
6. Paste it in your `.env` file

### Free Tier Limits

| Resource | Limit |
|----------|-------|
| Requests/minute | 30 |
| Requests/day | 14,400 |
| Tokens/minute | 6,000 |
| Tokens/day | 500,000 |

The system uses two models:
- **Primary:** `llama-3.3-70b-versatile` (highest quality)
- **Fallback:** `llama-3.1-8b-instant` (fast, for rate-limit situations)

---

## Docker Deployment

### Build & Run

```bash
# Build the image
docker build -t ai-cv-analyzer .

# Run with API key
docker run -d \
  -p 5000:5000 \
  -e GROQ_API_KEY=gsk_your_key_here \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/candidates.db:/app/candidates.db \
  --name cv-analyzer \
  ai-cv-analyzer
```

### Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  cv-analyzer:
    build: .
    ports:
      - "5000:5000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./uploads:/app/uploads
      - ./candidates.db:/app/candidates.db
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

---

## Production Deployment

### Option A: Gunicorn (Linux)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
```

| Flag | Purpose |
|------|---------|
| `-w 4` | 4 worker processes |
| `-b 0.0.0.0:5000` | Bind to all interfaces |
| `--timeout 120` | 120s timeout (AI analysis takes time) |

### Option B: Waitress (Windows)

```bash
pip install waitress
python -c "from waitress import serve; from app import app; serve(app, host='0.0.0.0', port=5000, threads=8)"
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    client_max_body_size 50M;  # Large CV uploads

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;  # AI analysis timeout
    }
}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ | — | Groq API key for AI analysis |
| `FLASK_ENV` | ❌ | `development` | `production` for prod |
| `FLASK_DEBUG` | ❌ | `1` | Set to `0` in production |

---

## Troubleshooting

### Common Issues

#### ❌ "Connection error" when analyzing CVs
**Cause:** Groq API key missing or invalid.  
**Fix:**
```bash
# Check your .env file
cat .env

# Make sure it has:
GROQ_API_KEY=gsk_your_actual_key_here

# Restart the server after editing .env
```

#### ❌ "table candidates has no column named X"
**Cause:** Database schema is outdated.  
**Fix:** Delete the database and restart — it auto-creates:
```bash
rm candidates.db
python app.py
```

Or run the migration manually:
```python
import sqlite3
conn = sqlite3.connect("candidates.db")
conn.execute("ALTER TABLE candidates ADD COLUMN column_name TEXT DEFAULT ''")
conn.commit()
```

#### ❌ "No module named 'groq'"
**Cause:** Dependencies not installed in the virtual environment.  
**Fix:**
```bash
# Make sure venv is activated
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux

pip install -r requirements.txt
```

#### ❌ "Error generating optimized resume"
**Cause:** Groq API temporarily down or rate-limited.  
**Fix:** The system has an offline fallback. Wait a few minutes and retry. The system auto-retries with exponential backoff.

#### ❌ PDF export shows blank or errors
**Cause:** `reportlab` not installed or candidate has no analysis data.  
**Fix:**
```bash
pip install reportlab
```

#### ❌ File upload fails
**Cause:** File type not supported.  
**Fix:** Only `.pdf` and `.docx` files are accepted. Ensure files are not corrupted.

### Checking Logs

Application logs are saved to `app.log`:

```bash
# View recent logs
tail -50 app.log

# Search for errors
grep ERROR app.log

# Watch live logs
tail -f app.log
```

### Database Inspection

```bash
# Open SQLite CLI
sqlite3 candidates.db

# View all tables
.tables

# View schema
.schema candidates

# Count candidates
SELECT COUNT(*) FROM candidates;

# View top 5 candidates
SELECT name, final_score, job_role FROM candidates ORDER BY final_score DESC LIMIT 5;

# Exit
.quit
```

### API Testing with cURL

```bash
# Test homepage
curl http://127.0.0.1:5000/

# Test analytics
curl http://127.0.0.1:5000/api/analytics

# Test pipeline board
curl http://127.0.0.1:5000/api/pipeline/board

# Test skill heatmap
curl http://127.0.0.1:5000/api/skill_heatmap

# Bookmark a candidate
curl -X POST http://127.0.0.1:5000/api/pipeline/bookmark \
  -H "Content-Type: application/json" \
  -d '{"name": "Candidate Name"}'
```

---

## File Size Reference

| File | Size | Purpose |
|------|------|---------|
| `app.py` | ~65 KB | Main application |
| `cv_analyzer.py` | ~91 KB | AI engine |
| `pdf_generator.py` | ~51 KB | PDF reports |
| `cv_parser.py` | ~17 KB | Document parser |
| `style.css` | ~30 KB | Stylesheet |
| `ranking.html` | ~83 KB | Ranking template |
| `result.html` | ~80 KB | Results template |
| `index.html` | ~22 KB | Homepage template |

---

> 📝 Setup Guide — AI CV Analyzer v2.3
