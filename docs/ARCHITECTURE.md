# 🏗️ AI CV Analyzer — Architecture Documentation

> Deep technical reference for the system's internal architecture.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│                                                                  │
│  index.html    result.html    ranking.html    dashboard.html     │
│  (Upload)      (Results)      (Rankings)      (Analytics)        │
│                                                                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK APPLICATION (app.py)                     │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  Routes   │  │   APIs   │  │ Pipeline │  │  PDF Export   │   │
│  │ (6 pages) │  │(12 REST) │  │  (CRM)   │  │  Generator    │   │
│  └────┬──────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘   │
│       │              │             │                 │           │
└───────┼──────────────┼─────────────┼─────────────────┼───────────┘
        │              │             │                 │
        ▼              ▼             ▼                 ▼
┌───────────────┐ ┌────────────┐ ┌────────┐ ┌─────────────────┐
│  cv_parser.py │ │cv_analyzer │ │ SQLite │ │ pdf_generator   │
│  (Extraction) │ │  (13 AI    │ │  (DB)  │ │   (ReportLab)   │
│               │ │  Agents)   │ │        │ │                 │
└───────────────┘ └─────┬──────┘ └────────┘ └─────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   Groq Cloud AI  │
              │  (LLaMA Models)  │
              └──────────────────┘
```

---

## Component Details

### 1. `app.py` — Flask Web Application

**Responsibility:** HTTP routing, request handling, database management, API orchestration.

**Key Components:**

| Component | Lines | Purpose |
|-----------|-------|---------|
| Database Init | 110-175 | Schema creation, column migrations |
| File Upload | 308-376 | Multi-file CV upload with validation |
| CV Processing | 195-307 | Per-file analysis pipeline |
| Ranking Route | 720-750 | Paginated candidate listing with search |
| Dashboard Route | 1600+ | Analytics data aggregation |
| Pipeline APIs | 1460+ | CRM stage/notes/bookmark management |
| PDF Export | 750+ | ReportLab PDF generation trigger |

**Database Connection Pattern:**
```python
def get_db():
    conn = sqlite3.connect("candidates.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
    return conn
```

---

### 2. `cv_analyzer.py` — Multi-Agent AI Engine

**Responsibility:** All AI-powered analysis. Contains 13 specialized agents + orchestrator.

**Size:** ~1,974 lines, ~91KB

**Key Patterns:**

#### Agent Structure (repeated for all 13)
```python
def run_agent_name(cv_text: str, job_role: str) -> dict:
    # 1. Check cache
    cache_k = _cache_key(cv_text, job_role, "agent_id")
    cached = _cache_get(cache_k)
    if cached: return cached
    
    # 2. Sanitize input
    cv_clean = _sanitize_cv_text(cv_text)
    
    # 3. Construct prompt with JSON schema
    prompt = f"""..."""
    
    # 4. Call Groq with JSON parsing
    data = groq_call(prompt, response_format={"type": "json_object"})
    
    # 5. Extract and validate fields
    result = { ... }
    
    # 6. Cache and return
    _cache_set(cache_k, result)
    return result
```

#### Orchestrator Pattern (get_ai_deep_analysis)
```python
def get_ai_deep_analysis(cv_text, job_role):
    # Phase 1: Run 13 agents in parallel (ThreadPoolExecutor)
    with ThreadPoolExecutor(max_workers=13) as executor:
        future_audit = executor.submit(run_audit_agent, cv_text, job_role)
        future_tech  = executor.submit(run_tech_matcher_agent, cv_text, job_role)
        # ... 11 more agents
    
    # Phase 2: Merge all results
    merged = {}
    merged.update(audit_res)
    # ... 12 more updates
    
    # Phase 3: Supervisor Meta-Agent
    supervisor = run_supervisor_agent(merged, cv_text, job_role)
    merged.update(supervisor)
    
    # Phase 4: Confidence aggregation
    merged["analysis_confidence"] = avg(all_confidence_scores)
    
    return merged
```

#### Groq API Caller
```python
def groq_call(prompt, system_msg, response_format, temperature, parse_json):
    models = [PRIMARY_MODEL, FALLBACK_MODEL]  # Cascade
    for model in models:
        for attempt in range(MAX_RETRIES):  # 3 retries
            try:
                completion = groq_client.chat.completions.create(**kwargs)
                return json.loads(content)  # Auto-parse JSON
            except (rate_limit, connection, timeout):
                time.sleep(RETRY_BASE_DELAY * 2**attempt)  # Exponential backoff
            except other_error:
                break  # Switch to fallback model
    raise last_error
```

---

### 3. `cv_parser.py` — Document Parser

**Responsibility:** Extract text and structured data from PDF and DOCX files.

**Key Features:**

| Feature | Method | Description |
|---------|--------|-------------|
| PDF Extraction | `pdfplumber` | Full text extraction with layout preservation |
| DOCX Extraction | `python-docx` | Paragraph and table text extraction |
| Email Detection | Regex | RFC-compliant email pattern matching |
| Link Extraction | Regex | GitHub, LinkedIn, portfolio URLs |
| Education Parsing | Pydantic | Structured degree, university, year extraction |
| Experience Parsing | Pydantic | Company, role, duration, description |
| Project Parsing | Pydantic | Title, tech stack, description |

**Data Model (Pydantic):**
```python
class ParsedProfile(BaseModel):
    email: Optional[str]
    links: List[str]
    education: List[EducationEntry]
    experience: List[ExperienceEntry]
    projects: List[ProjectEntry]
```

---

### 4. `pdf_generator.py` — PDF Report Generator

**Responsibility:** Generate professional PDF reports using ReportLab.

**Size:** ~51KB, ~1,200 lines

**Key Features:**
- Two-pass rendering (first pass counts pages, second pass renders)
- Custom `NumberedCanvas` class for page numbers
- Color-coded score gauges
- Personality radar visualization
- 15 distinct sections with proper formatting
- Supports both candidate reports and optimized CV exports

**Report Sections Flow:**
```
Header → Score Breakdown → Personality → Skills →
Education → Experience → Projects → AI Analysis →
Executive Summary → Career Trajectory → Red Flags →
Market Benchmark → Skill Verification → Learning Path →
Culture Fit
```

---

## Data Flow

### CV Upload → Analysis Flow

```
1. User uploads CV files via index.html
                    │
2. Flask saves to /uploads/ directory
                    │
3. cv_parser.py extracts text + structured data
                    │
4. TF-IDF similarity scoring vs job description
                    │
5. Personality estimation from text analysis
                    │
6. 13 AI agents analyze in parallel (ThreadPoolExecutor)
                    │
7. Supervisor agent synthesizes all results
                    │
8. Results saved to SQLite database
                    │
9. Rendered on result.html with visualizations
```

### Scoring Formula

```
final_score = (
    job_match_score     × 0.40 +
    soft_skills_score   × 0.08 +
    certificates_score  × 0.08 +
    interests_score     × 0.06 +
    experience_score    × 0.15 +
    personality_score   × 0.08 +
    tech_alignment      × 0.10 +
    trajectory_score    × 0.05
) × 100
```

---

## Caching Strategy

### In-Memory Agent Cache
- **Location:** `cv_analyzer.py` → `_agent_cache` dict
- **Key:** MD5(cv_text[:500]) + job_role + agent_name
- **Size Limit:** 200 entries (LRU eviction)
- **Purpose:** Avoid re-calling Groq for identical inputs

### Database Cache
- **Optimized CV:** Stored in `optimized_cv` column
- **AI Report:** Full JSON in `ai_report` column
- **Parameters:** Cached optimization params for cache-hit detection

---

## Error Handling Strategy

### Groq API Failures
```
Level 1: Retry with exponential backoff (1s, 2s, 4s)
Level 2: Switch to fallback model (llama-3.1-8b-instant)
Level 3: Retry fallback model (1s, 2s, 4s)
Level 4: Return graceful defaults (agent-specific)
Level 5: Offline fallback (resume optimization only)
```

### Database Errors
- Schema auto-migration on startup
- WAL mode for concurrent read/write safety
- Graceful column addition for version upgrades

---

## Concurrency Model

```
Main Thread (Flask)
    │
    ├── ThreadPoolExecutor (5 workers)  ← CV file processing
    │       │
    │       └── ThreadPoolExecutor (13 workers)  ← Agent execution
    │               ├── Agent 1 (Audit)
    │               ├── Agent 2 (Tech)
    │               ├── ...
    │               └── Agent 13 (Salary)
    │
    └── Single Thread  ← Supervisor meta-agent
```

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Backend** | Flask | 3.x |
| **AI Engine** | Groq SDK | Latest |
| **AI Models** | LLaMA 3.3-70B / 3.1-8B | Via Groq Cloud |
| **NLP** | scikit-learn (TF-IDF) | Latest |
| **PDF Parse** | pdfplumber | Latest |
| **DOCX Parse** | python-docx | Latest |
| **Data Validation** | Pydantic | v2 |
| **PDF Generation** | ReportLab | Latest |
| **Database** | SQLite3 | Built-in |
| **Frontend** | Bootstrap 5.3 | CDN |
| **Icons** | Bootstrap Icons | CDN |
| **Fonts** | Google Fonts (Inter) | CDN |
| **Config** | python-dotenv | Latest |

---

> 📝 Architecture Document — AI CV Analyzer v2.3
