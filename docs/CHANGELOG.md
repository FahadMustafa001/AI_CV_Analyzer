# 📋 Changelog — AI CV Analyzer

All notable changes to this project are documented in this file.

---

## [v2.3] — 2026-06-20

### ✨ Added
- **Analytics Dashboard** (`/dashboard`) — Visual analytics page with KPI cards, charts, pipeline board
- **Hiring Pipeline** — 8-stage candidate tracking (New → Screening → Interview → Assessment → Offer → Hired → Rejected → Withdrawn)
- **Recruiter Notes** — Add/edit notes for each candidate via API
- **Candidate Bookmarks** — Star/bookmark favorite candidates
- **Star Ratings** — Rate candidates 1-5 stars
- **Candidate Delete** — Remove candidates from the database
- **Dashboard Nav Link** — Added to main navigation bar
- **Full Documentation** — README.md, API Reference, Architecture docs, Setup Guide
- **`.env` support** — API keys stored in `.env` file via python-dotenv

### 🔧 Fixed
- **Groq Connection Handling** — Connection errors now trigger retries (was only retrying rate-limits)
- **`created_at` column** — Fixed `CURRENT_TIMESTAMP` default not working with SQLite `ALTER TABLE`
- **Resume Optimization Fallback** — Smart offline reformatting when Groq AI is down (instead of bare error)

### 📡 New API Endpoints
- `POST /api/pipeline/stage` — Update pipeline stage
- `POST /api/pipeline/notes` — Update recruiter notes
- `POST /api/pipeline/bookmark` — Toggle bookmark
- `POST /api/pipeline/rate` — Rate candidate (1-5)
- `GET /api/pipeline/board` — Get pipeline board data
- `POST /api/candidate/delete` — Delete candidate

### 🗄️ Database Changes
- Added columns: `pipeline_stage`, `recruiter_notes`, `is_bookmarked`, `last_contacted`, `interview_date`, `rating`

---

## [v2.2] — 2026-06-20

### ✨ Added
- **Agent 12: Onboarding Plan Generator** — 30/60/90 day onboarding plans
- **Agent 13: Salary Negotiation Intelligence** — Market comp analysis, negotiation strategies
- **Auto-Shortlist API** (`/api/shortlist`) — AI selects top N with reasoning
- **Team Builder API** (`/api/team_builder`) — Find complement candidates
- **Skill Heatmap API** (`/api/skill_heatmap`) — Skill frequency analysis

### 🔧 Improved
- Supervisor Meta-Agent synthesizes 13 streams (was 11)
- ThreadPoolExecutor handles 13 concurrent agents
- `deserialize_candidate()` supports both `sqlite3.Row` and tuple

---

## [v2.0] — 2026-06-20

### ✨ Added — Multi-Agent Architecture
- **11 AI Agents** running in parallel via ThreadPoolExecutor
- **Supervisor Meta-Agent** for unified verdict
- Agent 1: Resume Audit
- Agent 2: Tech Matcher
- Agent 3: Psych Profiler
- Agent 4: Interviewer
- Agent 5: Test Generator
- Agent 6: Career Trajectory
- Agent 7: Red Flag Detector
- Agent 8: Market Benchmarker
- Agent 9: Skill Verification
- Agent 10: Culture Fit
- Agent 11: Learning Path
- Executive Summary, Unified Verdict, Analysis Confidence fields
- Career Trajectory, Red Flags, Market Benchmark columns

### 📡 New API Endpoints
- `POST /api/reanalyze/<name>` — Re-run analysis
- `POST /api/cover_letter/<name>` — AI cover letter generation
- `POST /api/compare` — Head-to-head comparison
- `POST /api/email_template/<name>` — Recruiter email templates
- `GET /api/analytics` — Pool analytics

---

## [v1.5] — 2026-06-20

### ✨ Added
- **ATS CV Optimization** — AI rewrites CVs with 4 tone options
- **PDF Report Export** — 15-section professional reports via ReportLab
- **Optimized CV PDF Export** — Download rewritten CV as PDF
- **Caching** — In-memory LRU cache for agent results
- **Database Caching** — Optimized CVs cached in SQLite

### 🔧 Improved
- CV parser extracts education, experience, projects as structured JSON
- Personality estimation uses Big Five model with LLM enhancement
- Score display with color-coded badges

---

## [v1.0] — 2026-06-19

### 🎉 Initial Release
- Flask web application
- PDF/DOCX CV upload
- TF-IDF similarity scoring against job descriptions
- 6 predefined job roles + custom JD input
- Soft skills, interests, certificates extraction
- Basic personality estimation from text
- SQLite database for candidate storage
- Bootstrap 5 responsive UI with dark/light theme
- Global ranking table
- CSV export
- Psychometric personality test

---

> 📝 Maintained by Muhammad Fahad Mustafa
