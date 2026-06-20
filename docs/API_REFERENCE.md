# 🔌 AI CV Analyzer — API Documentation

> Complete reference for all REST API endpoints.  
> Base URL: `http://127.0.0.1:5000`

---

## Table of Contents

- [Authentication](#authentication)
- [Upload & Analysis](#upload--analysis)
- [CV Operations](#cv-operations)
- [AI Analysis APIs](#ai-analysis-apis)
- [Analytics & Data](#analytics--data)
- [Pipeline Management](#pipeline-management)
- [Error Handling](#error-handling)

---

## Authentication

Currently no authentication is required. The app runs locally on `localhost:5000`.

---

## Upload & Analysis

### `POST /analyze`

Upload and analyze CVs against a job description.

**Content-Type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `files` | File[] | ✅ | One or more CV files (.pdf, .docx) |
| `jd_type` | string | ✅ | `predefined` or `custom` |
| `job_role` | string | ✅* | Required if `jd_type=predefined`. Options: `Software Engineer`, `Data Analyst`, `Flutter Developer`, `Graphic Designer`, `HR Manager`, `UI/UX Designer` |
| `custom_jd` | string | ✅* | Required if `jd_type=custom`. Full job description text. |

**Response:** Renders `result.html` with analysis results.

**Example (cURL):**
```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -F "files=@resume.pdf" \
  -F "jd_type=predefined" \
  -F "job_role=Software Engineer"
```

---

## CV Operations

### `GET/POST /optimize_cv/<name>`

Generate an AI-optimized version of a candidate's CV.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `focus_area` | string | `all` | `all`, `achievements`, `keywords`, `formatting` |
| `tone` | string | `professional` | `professional`, `bold`, `academic`, `minimalist` |
| `target_length` | string | `default` | `default`, `concise`, `detailed` |

**Response:**
```json
{
  "optimized_cv": "# John Doe\n## Professional Summary\n...",
  "raw_text": "Original CV text...",
  "ai_report": { ... }
}
```

### `GET /export_pdf/<name>`

Download a comprehensive PDF report for a candidate.

**Response:** PDF file download (application/pdf)

### `GET /export_optimized_pdf/<name>`

Download the AI-optimized CV as a formatted PDF.

**Response:** PDF file download (application/pdf)

---

## AI Analysis APIs

### `POST /api/reanalyze/<name>`

Re-run the full 13-agent analysis on an existing candidate.

**Response:**
```json
{
  "success": true,
  "name": "Candidate Name",
  "new_score": 72.5,
  "agents_executed": 13,
  "analysis_time_seconds": 45.2
}
```

### `POST /api/cover_letter/<name>`

Generate a personalized AI cover letter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tone` | string | `professional` | `professional`, `enthusiastic`, `concise`, `creative` |
| `company` | string | `""` | Target company name |
| `highlights` | string[] | `[]` | Specific achievements to emphasize |

**Response:**
```json
{
  "cover_letter": "Dear Hiring Manager,\n\n...",
  "tone": "professional",
  "word_count": 350
}
```

### `POST /api/compare`

Head-to-head comparison of two candidates.

**Request Body:**
```json
{
  "candidate_a": "Candidate Name 1",
  "candidate_b": "Candidate Name 2"
}
```

**Response:**
```json
{
  "winner": "Candidate Name 1",
  "comparison": {
    "scores": {"a": 72.5, "b": 65.3},
    "strengths_a": ["..."],
    "strengths_b": ["..."],
    "recommendation": "Candidate A is stronger because..."
  }
}
```

### `POST /api/email_template/<name>`

Generate recruiter email templates.

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `template_type` | string | `outreach` | `outreach`, `interview`, `rejection`, `offer` |

**Response:**
```json
{
  "subject": "Exciting Opportunity at [Company]",
  "body": "Dear [Name],\n\n...",
  "template_type": "outreach"
}
```

### `POST /api/shortlist`

AI-powered automatic shortlisting.

**Request Body:**
```json
{
  "job_role": "Software Engineer",
  "top_n": 5
}
```

**Response:**
```json
{
  "success": true,
  "job_role": "Software Engineer",
  "total_pool": 108,
  "shortlist": [
    {"name": "...", "rank": 1, "reason": "Why selected"}
  ],
  "pool_quality_assessment": "Moderate pool with...",
  "hiring_difficulty": "Moderate"
}
```

### `POST /api/team_builder`

Find candidates that complement an existing team.

**Request Body:**
```json
{
  "existing_team_skills": ["Python", "React", "AWS"],
  "job_role": "Software Engineer",
  "team_size": 3
}
```

**Response:**
```json
{
  "success": true,
  "recommended_hires": [
    {
      "name": "...",
      "role_in_team": "Backend Specialist",
      "complementary_skills": ["Java", "Kubernetes"],
      "why": "Fills the DevOps gap..."
    }
  ],
  "team_gaps_filled": ["DevOps", "Database"],
  "remaining_gaps": ["ML/AI"],
  "team_synergy_score": 78
}
```

---

## Analytics & Data

### `GET /api/analytics`

Comprehensive talent pool analytics.

**Response:**
```json
{
  "pool_size": 108,
  "score_stats": {
    "avg": 22.5,
    "max": 56.6,
    "min": 5.2,
    "median": 20.1
  },
  "score_distribution": {
    "0-20": 45,
    "20-40": 50,
    "40-60": 12,
    "60-80": 1,
    "80-100": 0
  },
  "hiring_funnel": {
    "strong_hire_75plus": 0,
    "interview_50_74": 1,
    "below_threshold_sub50": 107
  },
  "experience_stats": {
    "avg_years": 2.3,
    "max_years": 8.0
  }
}
```

### `GET /api/skill_heatmap`

Skill frequency analysis across the talent pool.

**Response:**
```json
{
  "success": true,
  "total_analyzed": 78,
  "top_skills": [
    {"skill": "python", "count": 45, "pct": 57.7},
    {"skill": "javascript", "count": 32, "pct": 41.0}
  ],
  "top_gaps": [
    {"gap": "kubernetes", "count": 30, "pct": 38.5}
  ],
  "unique_skills_found": 46,
  "unique_gaps_found": 23
}
```

### `GET /api/pipeline/board`

Get all candidates grouped by pipeline stage.

**Response:**
```json
{
  "success": true,
  "total": 109,
  "stages": {
    "new": [{"name": "...", "job_role": "...", "score": 56.6, "bookmarked": true, "rating": 5}],
    "screening": [],
    "interview": [{"name": "...", ...}],
    "assessment": [],
    "offer": [],
    "hired": [],
    "rejected": [],
    "withdrawn": []
  }
}
```

---

## Pipeline Management

### `POST /api/pipeline/stage`

Update a candidate's pipeline stage.

**Request Body:**
```json
{
  "name": "Candidate Name",
  "stage": "interview"
}
```

**Valid Stages:** `new`, `screening`, `interview`, `assessment`, `offer`, `hired`, `rejected`, `withdrawn`

**Response:**
```json
{"success": true, "name": "Candidate Name", "stage": "interview"}
```

### `POST /api/pipeline/notes`

Add or update recruiter notes.

**Request Body:**
```json
{
  "name": "Candidate Name",
  "notes": "Great communication skills. Schedule technical round."
}
```

**Response:**
```json
{"success": true, "name": "Candidate Name", "notes": "Great communication skills..."}
```

### `POST /api/pipeline/bookmark`

Toggle bookmark status (no body parameter needed except name).

**Request Body:**
```json
{"name": "Candidate Name"}
```

**Response:**
```json
{"success": true, "name": "Candidate Name", "is_bookmarked": true}
```

### `POST /api/pipeline/rate`

Rate a candidate with 1-5 stars.

**Request Body:**
```json
{
  "name": "Candidate Name",
  "rating": 5
}
```

**Response:**
```json
{"success": true, "name": "Candidate Name", "rating": 5}
```

### `POST /api/candidate/delete`

Permanently delete a candidate from the database.

**Request Body:**
```json
{"name": "Candidate Name"}
```

**Response:**
```json
{"success": true, "deleted": "Candidate Name"}
```

---

## Error Handling

All API errors return JSON with an `error` field:

```json
{
  "error": "Description of what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Bad Request — missing or invalid parameters |
| `404` | Not Found — candidate doesn't exist |
| `500` | Internal Server Error — check `app.log` |

### Groq API Errors

When the Groq AI service is unavailable:
- **CV Optimization** falls back to offline smart reformatting
- **Agent Analysis** uses cached results or graceful defaults
- Connection errors are automatically retried with exponential backoff

---

## Rate Limits

- Groq API: Depends on your plan (free tier: ~30 req/min)
- Local Flask: No limits
- Recommendation: Use caching (built-in) for repeated analyses

---

> 📝 This documentation is auto-generated for AI CV Analyzer v2.3
