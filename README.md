# 🧠 AI CV Analyzer v2.3

> **Enterprise-grade AI-powered Resume Analysis & Talent Intelligence Platform**
> 
> Built with 13 specialized AI agents + 1 Supervisor Meta-Agent = **14 AI brains** analyzing every candidate in parallel.

---

## 🌟 Features at a Glance

| Feature | Description |
|---------|-------------|
| 🧠 **13 AI Agents** | Parallel deep analysis covering audit, tech, psychology, career, salary, and more |
| 📊 **Analytics Dashboard** | Visual charts, score distribution, pipeline board, role breakdown |
| 🔄 **Hiring Pipeline** | 8-stage pipeline (New → Screening → Interview → Assessment → Offer → Hired) |
| 📝 **ATS Optimization** | AI rewrites resumes for maximum ATS score with 4 tone options |
| 📄 **PDF Reports** | 15-section professional PDF export with executive summary |
| 🔍 **Semantic Search** | AI-powered candidate search across entire talent pool |
| ⚔️ **Head-to-Head Compare** | Side-by-side candidate comparison with AI verdict |
| 💌 **Auto Email Templates** | AI-generated outreach, interview, rejection, and offer emails |
| 📋 **Auto-Shortlist** | AI selects top N candidates with reasoning |
| 🏗️ **Team Builder** | Finds candidates that complement existing team skills |
| 🔥 **Skill Heatmap** | Skill frequency analysis across entire talent pool |
| ⭐ **CRM Features** | Bookmarks, star ratings, recruiter notes per candidate |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com/keys) (FREE)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/FahadMustafa001/AI_CV_Analyzer.git
cd AI_CV_Analyzer

# 2. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API key
# Create a .env file in the project root:
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 5. Run the application
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
AI_CV_Analyzer/
├── app.py                    # Flask app — routes, APIs, DB management
├── cv_analyzer.py            # 13 AI Agents + Supervisor + analysis engine
├── cv_parser.py              # PDF/DOCX text extraction & structured parsing
├── pdf_generator.py          # ReportLab PDF report generator (15 sections)
├── candidates.db             # SQLite database (auto-created)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker containerization
├── .env                      # API keys (not committed to git)
├── .gitignore                # Git exclusions
│
├── templates/                # Jinja2 HTML templates
│   ├── base.html             # Master layout with nav, footer, theme toggle
│   ├── index.html            # Homepage — upload CVs + job role selection
│   ├── result.html           # Analysis results after upload
│   ├── ranking.html          # Global talent pool ranking table
│   ├── dashboard.html        # Visual analytics dashboard
│   └── test.html             # Psychometric personality test wizard
│
├── static/
│   ├── css/style.css         # Global stylesheet (dark/light themes)
│   └── images/logo.png       # Application logo
│
├── tests/
│   └── test_multi_agent.py   # Multi-agent integration tests
│
└── uploads/                  # Temporary CV file storage
```

---

## 🧠 AI Agent Architecture (14 Total)

### Phase 1: 13 Agents Run in Parallel

| # | Agent | File | Function | Output |
|---|-------|------|----------|--------|
| 1 | **Resume Audit** | `cv_analyzer.py` | `run_audit_agent()` | Formatting score, ATS compatibility, bullet improvements |
| 2 | **Tech Matcher** | `cv_analyzer.py` | `run_tech_matcher_agent()` | Tech alignment %, categorized skills, critical gaps |
| 3 | **Psych Profiler** | `cv_analyzer.py` | `run_psych_agent()` | Big Five personality traits with evidence |
| 4 | **Interviewer** | `cv_analyzer.py` | `run_interviewer_agent()` | Seniority level, hiring recommendation, interview questions |
| 5 | **Test Generator** | `cv_analyzer.py` | `run_test_generator_agent()` | Custom coding test + grading rubric |
| 6 | **Career Trajectory** | `cv_analyzer.py` | `run_trajectory_agent()` | Growth pattern, momentum, promotion frequency |
| 7 | **Red Flag Detector** | `cv_analyzer.py` | `run_red_flag_agent()` | 8-category risk scan with severity levels |
| 8 | **Market Benchmarker** | `cv_analyzer.py` | `run_benchmark_agent()` | Market percentile, salary range, competitiveness |
| 9 | **Skill Verification** | `cv_analyzer.py` | `run_skill_verification_agent()` | Per-skill audit: Verified/Unverified/Inflated |
| 10 | **Culture Fit** | `cv_analyzer.py` | `run_culture_fit_agent()` | Communication style, archetypes, work preferences |
| 11 | **Learning Path** | `cv_analyzer.py` | `run_learning_path_agent()` | 90-day upskilling plan, certifications |
| 12 | **Onboarding Plan** | `cv_analyzer.py` | `run_onboarding_agent()` | 30/60/90 day plan, tooling, buddy matching |
| 13 | **Salary Intel** | `cv_analyzer.py` | `run_salary_agent()` | Comp analysis, negotiation risk, offer strategy |

### Phase 2: Results Merged
All 13 agent outputs are merged into a single dictionary.

### Phase 3: Supervisor Meta-Agent
| | Agent | Function | Output |
|---|-------|----------|--------|
| S | **Supervisor** | `run_supervisor_agent()` | Executive summary, unified verdict, candidate tier (A/B/C/D/F), top 3 strengths & concerns |

### Phase 4: Confidence Aggregation
Weighted average of all 13 agent confidence scores → `analysis_confidence`.

---

## 🔌 API Reference (18 Endpoints)

### Pages

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Homepage — upload CVs |
| `POST` | `/analyze` | Process uploaded CVs with 13-agent analysis |
| `GET` | `/ranking` | Global talent pool ranking table |
| `GET` | `/dashboard` | Visual analytics dashboard |
| `GET` | `/test/<name>` | Psychometric personality test |
| `POST` | `/submit_test/<name>` | Submit personality test answers |

### CV Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/optimize_cv/<name>` | AI-optimized resume generation |
| `GET` | `/export_pdf/<name>` | Download candidate PDF report |
| `GET` | `/export_optimized_pdf/<name>` | Download optimized CV as PDF |

### AI Analysis APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/reanalyze/<name>` | Re-run 13-agent analysis |
| `POST` | `/api/cover_letter/<name>` | AI cover letter (4 tones) |
| `POST` | `/api/compare` | Head-to-head comparison |
| `POST` | `/api/email_template/<name>` | Recruiter email templates |
| `POST` | `/api/shortlist` | AI auto-shortlist top N |
| `POST` | `/api/team_builder` | Team complement finder |

### Analytics & Data APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/analytics` | Pool stats, score distribution, hiring funnel |
| `GET` | `/api/skill_heatmap` | Skill frequency across talent pool |
| `GET` | `/api/pipeline/board` | Pipeline board (all candidates by stage) |

### CRM / Pipeline Management APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/pipeline/stage` | Update candidate pipeline stage |
| `POST` | `/api/pipeline/notes` | Add/update recruiter notes |
| `POST` | `/api/pipeline/bookmark` | Toggle bookmark |
| `POST` | `/api/pipeline/rate` | Rate candidate (1-5 stars) |
| `POST` | `/api/candidate/delete` | Delete candidate |

---

## 📊 Database Schema

**File:** `candidates.db` (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `name` | TEXT | Candidate name |
| `filename` | TEXT | Original CV filename |
| `job_role` | TEXT | Target job role |
| `score` | REAL | Job description match score (TF-IDF) |
| `soft_skills` | TEXT | Extracted soft skills (comma-separated) |
| `interests` | TEXT | Extracted interests |
| `certificates` | TEXT | Extracted certifications |
| `final_score` | REAL | Weighted composite score (0-100%) |
| `exp_years` | REAL | Years of experience |
| `personality_score` | REAL | Personality fit score |
| `personality_openness` | REAL | Big Five: Openness (0-100) |
| `personality_conscientiousness` | REAL | Big Five: Conscientiousness (0-100) |
| `personality_extraversion` | REAL | Big Five: Extraversion (0-100) |
| `personality_agreeableness` | REAL | Big Five: Agreeableness (0-100) |
| `personality_neuroticism` | REAL | Big Five: Neuroticism (0-100) |
| `ai_report` | TEXT | Full AI analysis JSON (all 13 agents) |
| `raw_text` | TEXT | Original CV plain text |
| `email` | TEXT | Extracted email address |
| `links` | TEXT | Extracted links (JSON array) |
| `education` | TEXT | Parsed education (JSON array) |
| `experience` | TEXT | Parsed experience (JSON array) |
| `projects` | TEXT | Parsed projects (JSON array) |
| `optimized_cv` | TEXT | AI-optimized CV (Markdown) |
| `optimized_cv_params` | TEXT | Optimization parameters (JSON) |
| `created_at` | TEXT | Timestamp of record creation |
| `executive_summary` | TEXT | Supervisor's executive summary |
| `unified_verdict` | TEXT | Supervisor's unified verdict |
| `analysis_confidence` | REAL | Aggregate confidence (0-100%) |
| `career_trajectory` | TEXT | Career trajectory analysis (JSON) |
| `red_flags` | TEXT | Red flag report (JSON) |
| `market_benchmark` | TEXT | Market benchmark data (JSON) |
| `tech_alignment_score` | REAL | Technical alignment (0-100%) |
| `pipeline_stage` | TEXT | Hiring pipeline stage |
| `recruiter_notes` | TEXT | Recruiter's notes |
| `is_bookmarked` | INTEGER | Bookmarked flag (0/1) |
| `last_contacted` | TEXT | Last contact date |
| `interview_date` | TEXT | Scheduled interview date |
| `rating` | INTEGER | Star rating (0-5) |

---

## 🎨 Frontend Templates

### `base.html` — Master Layout
- Responsive navbar with theme toggle (dark/light mode)
- Bootstrap 5.3 + Bootstrap Icons
- Google Fonts (Inter)
- Glassmorphism design with CSS variables
- Footer with navigation links

### `index.html` — Homepage
- Drag & drop CV upload with visual feedback
- Job role selector (6 predefined + custom JD)
- Animated feature cards explaining system capabilities
- FAQ accordion section

### `result.html` — Analysis Results
- Candidate cards with score badges, personality radar charts
- Expandable sections for AI deep analysis
- ATS optimized CV viewer with regeneration controls
- Export buttons (CSV, PDF, Copy)

### `ranking.html` — Global Talent Pool
- Full ranking table with sorting and search
- Semantic AI search bar
- Expanded candidate detail panels
- Personality trait radar visualization
- Quick actions (PDF export, Psych test, Re-analyze)

### `dashboard.html` — Analytics Dashboard
- KPI cards (total pool, avg score, top score, bookmarks)
- Score distribution bar chart
- Experience distribution chart
- Hiring pipeline board (8 stages)
- Role breakdown table
- Top 10 candidates leaderboard

### `test.html` — Psychometric Test
- Multi-step personality assessment wizard
- Progress bar navigation
- Likert scale response options
- Loading animation on submit

---

## ⚙️ Configuration

### Scoring Weights (`cv_analyzer.py`)

```python
SCORING_WEIGHTS = {
    "job_match": 0.40,      # TF-IDF similarity to job description
    "soft_skills": 0.08,    # Extracted soft skills count
    "certificates": 0.08,   # Certifications relevance
    "interests": 0.06,      # Interest alignment
    "experience": 0.15,     # Years of experience fit
    "personality": 0.08,    # Big Five personality match
    "tech_alignment": 0.10, # AI tech stack analysis
    "trajectory": 0.05,     # Career growth pattern
}
```

### Supported Job Roles

| Role | Keywords |
|------|----------|
| Software Engineer | python, java, algorithms, git, docker, APIs... |
| Data Analyst | sql, excel, python, tableau, statistics... |
| Flutter Developer | flutter, dart, firebase, REST, widgets... |
| Graphic Designer | figma, adobe, wireframing, typography... |
| HR Manager | recruitment, onboarding, compliance... |
| UI/UX Designer | figma, prototyping, user research... |
| Custom | AI extracts targets from your job description |

### AI Models

| Model | Usage |
|-------|-------|
| `llama-3.3-70b-versatile` | Primary model (high quality) |
| `llama-3.1-8b-instant` | Fallback model (fast, reliable) |

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t ai-cv-analyzer .

# Run
docker run -p 5000:5000 -e GROQ_API_KEY=gsk_your_key ai-cv-analyzer
```

---

## 📄 PDF Report Sections (15 Total)

1. **Header** — Candidate name, role, date, overall score gauge
2. **Score Breakdown** — Component-wise scoring table
3. **Personality Profile** — Big Five traits with visual bars
4. **Skills Analysis** — Categorized technical/soft skills
5. **Education & Experience** — Structured timeline
6. **Projects** — Key projects with descriptions
7. **AI Deep Analysis** — Full multi-agent insights
8. **Interview Kit** — Custom questions + coding test
9. **Executive Summary & Verdict** — Supervisor's unified assessment
10. **Career Trajectory** — Growth pattern & momentum analysis
11. **Red Flag Report** — Risk assessment with severity levels
12. **Market Benchmark** — Percentile ranking & salary range
13. **Skill Verification** — Per-skill credibility audit table
14. **Learning Path** — 90-day upskilling plan
15. **Culture Fit** — Archetypes & work style preferences

---

## 🔒 Security Notes

- `.env` file stores API keys (excluded from Git via `.gitignore`)
- Uploaded files are sanitized via `werkzeug.utils.secure_filename`
- SQLite uses WAL mode for concurrent access safety
- No sensitive data exposed in frontend templates

---

## 📝 License

MIT License — Free for personal and commercial use.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Muhammad Fahad Mustafa**  
GitHub: [@FahadMustafa001](https://github.com/FahadMustafa001)

---

> Built with ❤️ using Flask, Groq AI, and 14 specialized AI agents.
