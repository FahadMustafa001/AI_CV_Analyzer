# cv_analyzer.py — Multi-Agent AI Analysis Engine v2.0
import os
import re
import json
import time
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

logger = logging.getLogger(__name__)

# ====================== Configuration Constants ======================
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds
MAX_CV_CHARS = 8000  # Truncate very long CVs to prevent token overflow

SCORING_WEIGHTS = {
    "job_match": 0.40,
    "soft_skills": 0.08,
    "certificates": 0.08,
    "interests": 0.06,
    "experience": 0.15,
    "personality": 0.08,
    "tech_alignment": 0.10,
    "trajectory": 0.05,
}

# Initialize Groq Client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# ====================== Agent Result Cache ======================
_agent_cache = {}
_CACHE_MAX_SIZE = 200

def _cache_key(cv_text: str, job_role: str, agent_name: str) -> str:
    text_hash = hashlib.md5(cv_text.encode('utf-8', errors='ignore')).hexdigest()
    return f"{agent_name}:{text_hash}:{job_role}"

def _cache_get(key: str):
    entry = _agent_cache.get(key)
    if entry and (time.time() - entry["ts"]) < 3600:  # 1 hour TTL
        logger.info(f"Cache HIT for {key[:40]}...")
        return entry["data"]
    return None

def _cache_set(key: str, data):
    if len(_agent_cache) >= _CACHE_MAX_SIZE:
        oldest_key = min(_agent_cache, key=lambda k: _agent_cache[k]["ts"])
        del _agent_cache[oldest_key]
    _agent_cache[key] = {"data": data, "ts": time.time()}


# ====================== Enhanced Groq API Helper ======================

def _sanitize_cv_text(text: str) -> str:
    """Clean and truncate CV text to prevent token overflow."""
    if not text:
        return ""
    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{4,}', '  ', text)
    text = text.strip()
    if len(text) > MAX_CV_CHARS:
        text = text[:MAX_CV_CHARS] + "\n... [truncated for processing]"
    return text


def groq_call(prompt: str, system_msg: str = "You are a precise JSON-only response model.",
              response_format: dict = None, temperature: float = 0.1,
              parse_json: bool = True, max_tokens: int = None,
              retry_on_json_parse: bool = True) -> any:
    """
    Unified Groq API caller with:
    - Primary → Fallback model cascade
    - Exponential backoff retry on rate-limit (429) and transient errors
    - Auto-retry on JSON parse failure
    - Timing metrics logging
    - Automatic JSON parsing when parse_json=True
    """
    models = [PRIMARY_MODEL, FALLBACK_MODEL]
    kwargs = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
    }
    if response_format:
        kwargs["response_format"] = response_format
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    last_error = None
    for model in models:
        kwargs["model"] = model
        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                completion = groq_client.chat.completions.create(**kwargs)
                elapsed = round(time.time() - t0, 2)
                content = completion.choices[0].message.content
                logger.info(f"Groq API call OK (model={model}, attempt={attempt+1}, time={elapsed}s)")

                if parse_json:
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as je:
                        if retry_on_json_parse and attempt < MAX_RETRIES - 1:
                            logger.warning(f"JSON parse failed, retrying... (error={je})")
                            time.sleep(0.5)
                            continue
                        # Try to extract JSON from markdown code blocks
                        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group(1))
                        raise je
                return content
            except json.JSONDecodeError:
                raise  # Don't retry JSON errors further after extraction attempt
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "timeout" in err_str or "503" in err_str or "connection" in err_str or "connect" in err_str or "eof" in err_str:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(f"Groq transient error (model={model}, attempt={attempt+1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.warning(f"Groq error (model={model}): {e}. Switching to fallback...")
                    break

    raise last_error


# ====================== Similarity Scoring ======================

def compute_llm_similarity_score(job_desc: str, cv_text: str) -> float:
    """Queries Groq LLM to semantically grade CV relevance against the Job Description."""
    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert technical recruiter with 15+ years of experience.

TASK: Evaluate semantic match between candidate's resume and the Job Description.
Think step-by-step:
1. Identify the core required skills from the JD
2. Check which of those skills appear in the CV (directly or through equivalent experience)
3. Assess seniority alignment
4. Check for relevant project experience
5. Assign a score from 0-100

Be strict and objective. A perfect score requires near-complete overlap.
Return ONLY valid JSON. No markdown, no notes.

JSON Schema:
{{
    "reasoning": "Brief 2-sentence explanation of scoring decision",
    "score": 85.0,
    "confidence": 90
}}

Job Description:
{job_desc}

Candidate Resume:
{cv_clean}"""
    data = groq_call(prompt, response_format={"type": "json_object"})
    return float(data.get("score", 0.0))


def compute_similarity(job_desc: str, cv_text: str) -> float:
    """
    Computes hybrid similarity: TF-IDF (30%) + LLM semantic (70%).
    Falls back to 100% TF-IDF if LLM is unavailable.
    """
    if not cv_text.strip():
        return 0.0

    try:
        vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform([job_desc, cv_text])
        sim_score = cosine_similarity(vect[0:1], vect[1:2])[0][0]
        tfidf_score = min((sim_score / 0.35) * 100.0, 100.0)
    except Exception as e:
        logger.error(f"TF-IDF matching failed: {e}")
        tfidf_score = 0.0

    try:
        llm_score = compute_llm_similarity_score(job_desc, cv_text)
        final_score = 0.30 * tfidf_score + 0.70 * llm_score
        return round(min(max(final_score, 0.0), 100.0), 2)
    except Exception as e:
        logger.warning(f"LLM similarity failed, using TF-IDF only: {e}")
        return round(tfidf_score, 2)


# ====================== Keyword & Experience Extraction ======================

def extract_keywords(cv_text: str, keywords_list: list) -> list:
    detected = []
    for word in keywords_list:
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        if re.search(pattern, cv_text):
            detected.append(word)
    return detected


def extract_experience_years(cv_text: str) -> float:
    years = 0.0
    explicit_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:years|year|yrs|yr|y)\b", cv_text, re.IGNORECASE)
    for m in explicit_matches:
        try:
            years = max(years, float(m))
        except ValueError:
            continue

    current_year = 2026
    date_ranges = re.findall(r"((?:19|20)\d{2})\s*[-––to]+\s*((?:19|20)\d{2}|present|current)", cv_text, re.IGNORECASE)
    calculated_years = 0.0
    for y1, y2 in date_ranges:
        try:
            start = int(y1)
            y2_clean = y2.lower().strip()
            end = current_year if y2_clean in ["present", "current"] else int(y2_clean)
            if end >= start:
                calculated_years += (end - start)
        except Exception:
            pass

    return round(max(years, calculated_years), 1)


# ====================== Scoring Functions ======================

def experience_score(job_role: str, exp_years: float, target_experience_map: dict) -> float:
    target = target_experience_map.get(job_role, 2)
    return round(min(exp_years / max(target, 0.1), 1.0) * 100.0, 2)


def estimate_personality_from_cv(cv_text: str) -> dict:
    """Cognitive psychological profiling via LLM with keyword fallback."""
    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert organizational psychologist with a PhD in Industrial-Organizational Psychology.

TASK: Analyze the candidate's CV and estimate Big Five personality trait scores (0-100 percentiles).

Think step-by-step for each trait:
- **Openness**: Look for creativity keywords, diverse projects, R&D, design work, novel technologies
- **Conscientiousness**: Look for attention to detail, planning, reliability, optimization, testing, documentation
- **Extraversion**: Look for leadership roles, presentations, team management, community involvement
- **Agreeableness**: Look for collaboration, mentoring, stakeholder management, support roles
- **Neuroticism**: Inversely score — high stress resilience, debugging complex problems = LOW neuroticism

Cite specific evidence from the CV for each score.
Return ONLY valid JSON. No markdown.

JSON Schema:
{{
    "openness": 75.0,
    "conscientiousness": 80.0,
    "extraversion": 55.0,
    "agreeableness": 65.0,
    "neuroticism": 35.0,
    "confidence": 75
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        return {
            "openness": float(data.get("openness", 50.0)),
            "conscientiousness": float(data.get("conscientiousness", 50.0)),
            "extraversion": float(data.get("extraversion", 50.0)),
            "agreeableness": float(data.get("agreeableness", 50.0)),
            "neuroticism": float(data.get("neuroticism", 50.0))
        }
    except Exception as e:
        logger.warning(f"LLM personality estimation failed, using keyword heuristics: {e}")
        trait_map = {
            "openness": ["creative", "innovation", "design", "research", "explore", "curious", "architecture", "ui", "ux", "novel", "experiment"],
            "conscientiousness": ["organized", "detail", "plan", "deadline", "reliable", "responsible", "optimization", "clean code", "testing", "documentation", "quality"],
            "extraversion": ["lead", "presented", "team lead", "community", "public", "network", "speaker", "management", "mentored", "coordinated"],
            "agreeableness": ["collaborated", "support", "mentor", "empathy", "customer", "stakeholder", "teamwork", "assisted", "helped", "volunteered"],
            "neuroticism": ["stress", "pressure", "challenge", "resilient", "cope", "troubleshooting", "debugging", "crisis", "emergency"]
        }
        scores = {t: 50.0 for t in trait_map}
        for trait, kws in trait_map.items():
            cnt = sum(1 for k in kws if k in cv_text.lower())
            val = (min(cnt, 6) / 6.0) * 100.0
            if trait == "neuroticism":
                val = max(10.0, 50.0 - val)
            scores[trait] = round(val if cnt > 0 else 50.0, 2)
        return scores


def score_personality_test(form_data: dict, questions_list: list) -> dict:
    sums = {"openness": 0.0, "conscientiousness": 0.0, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 0.0}
    counts = {k: 0 for k in sums}
    for q in questions_list:
        raw = int(form_data.get(q["id"], 3))
        val = 6 - raw if q["reverse"] else raw
        sums[q["trait"]] += val
        counts[q["trait"]] += 1
    scores = {}
    for trait in sums:
        avg = sums[trait] / max(counts[trait], 1)
        scores[trait] = round(((avg - 1.0) / 4.0) * 100.0, 2)
    return scores


def personality_fit_score(job_role: str, trait_scores: dict, target_personality_map: dict) -> float:
    target = target_personality_map.get(job_role)
    if not target:
        return 0.0
    diffs = [abs(trait_scores.get(t, 50.0) - tgt) for t, tgt in target.items()]
    mad = sum(diffs) / len(diffs)
    return round(max(0.0, 100.0 - mad), 2)


def compute_final_score(job_role, job_score, soft_skills, certificates, interests, exp_years,
                        personality_score_val, target_exp_map, total_soft_len, total_cert_len,
                        total_int_len, tech_alignment_score=50.0, trajectory_score=50.0) -> float:
    """Aggregates composite weights into a singular weighted final scoring metric."""
    soft_score = min(len(soft_skills) / max(total_soft_len, 1), 1) * 100
    cert_score = min(len(certificates) / max(total_cert_len, 1), 1) * 100
    interest_score = min(len(interests) / max(total_int_len, 1), 1) * 100
    exp_score = experience_score(job_role, exp_years, target_exp_map)

    w = SCORING_WEIGHTS
    final = (
        w["job_match"] * job_score +
        w["soft_skills"] * soft_score +
        w["certificates"] * cert_score +
        w["interests"] * interest_score +
        w["experience"] * exp_score +
        w["personality"] * personality_score_val +
        w["tech_alignment"] * tech_alignment_score +
        w["trajectory"] * trajectory_score
    )
    return round(final, 2)


# ====================== Helper ======================

def _safe_get(data: dict, key: str, default, cast=None):
    val = data.get(key, default)
    if cast and val is not None:
        try:
            return cast(val)
        except (ValueError, TypeError):
            return default
    return val


# ======================================================================
#  MULTI-AGENT AI DEEP SEMANTIC ENGINE v2.0
#  8 Specialized Agents + 1 Supervisor Meta-Agent
# ======================================================================

# ---------- AGENT 1: Resume Audit Agent ----------
def run_audit_agent(cv_text: str, job_role: str) -> dict:
    """Evaluates CV layout, parser compatibility, spelling, grammar, and ATS readability."""
    cache_k = _cache_key(cv_text, job_role, "audit")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert resume design auditor and ATS compatibility specialist.

TASK: Analyze the layout, structure, grammar, ATS parser compatibility, and spelling of this CV for '{job_role}'.

Think step-by-step:
1. Check section organization (Contact, Summary, Skills, Experience, Education, Projects)
2. Evaluate bullet point quality (do they use action verbs + metrics?)
3. Assess ATS parsing readiness (no tables, no images, standard headers)
4. Check grammar and spelling
5. Score formatting quality 0-100

Cite specific examples from the CV to support your evaluation.
Return ONLY valid JSON.

JSON Schema:
{{
    "formatting_score": 85.0,
    "formatting_feedback": "Detailed critique about structure, grammar, ATS compatibility, and actionable improvements",
    "bullet_points_improvement": ["Up to 3 specific rewrites using Google X-Y-Z formula: Accomplished X as measured by Y by doing Z"],
    "ats_compatibility": "High | Medium | Low",
    "evidence": ["Specific text excerpts that informed the score"],
    "confidence": 85
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "formatting_score": _safe_get(data, "formatting_score", 70.0, float),
            "formatting_feedback": _safe_get(data, "formatting_feedback", "Layout parsed with general compliance.", str),
            "bullet_points_improvement": list(data.get("bullet_points_improvement", ["Add metrics to experience bullets."])),
            "ats_compatibility": _safe_get(data, "ats_compatibility", "Medium", str),
            "audit_confidence": _safe_get(data, "confidence", 70, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentAudit failed: {e}")
        return {
            "formatting_score": 70.0,
            "formatting_feedback": "Standard formatting parsed. Ensure consistent headers and clear section organization for ATS compatibility.",
            "bullet_points_improvement": ["Rewrite experience bullets with measurable outcomes using the X-Y-Z formula."],
            "ats_compatibility": "Medium",
            "audit_confidence": 30,
        }


# ---------- AGENT 2: Technical Matcher Agent ----------
def run_tech_matcher_agent(cv_text: str, job_role: str) -> dict:
    """Evaluates technical competence, categorizes skills, detects critical gaps."""
    cache_k = _cache_key(cv_text, job_role, "tech")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are a senior technical evaluator and engineering hiring manager.

TASK: Perform deep technical competency analysis of this CV against expectations for '{job_role}'.

Think step-by-step:
1. Extract ALL technical skills mentioned (languages, frameworks, databases, tools, cloud platforms)
2. Categorize them into: programming_languages, frameworks_libraries, databases, tools_platforms, soft_skills
3. For '{job_role}', identify the TOP 5 must-have skills
4. Check which must-haves are PRESENT vs MISSING (critical gaps)
5. Score overall technical alignment 0-100
6. Assess skill DEPTH — are they just listed or demonstrated through projects/experience?

Cite specific CV text that demonstrates skill usage vs. just listing.
Return ONLY valid JSON.

JSON Schema:
{{
    "tech_alignment_score": 80.0,
    "skill_depth_assessment": "Surface-level listing | Demonstrated through projects | Deep expertise with metrics",
    "skills_found": ["list", "of", "all", "detected", "skills"],
    "categorized_skills": {{
        "programming_languages": ["Python", "Java"],
        "frameworks_libraries": ["Django", "React"],
        "databases": ["MySQL", "PostgreSQL"],
        "tools_platforms": ["Docker", "Git", "AWS"],
        "soft_skills": ["Leadership", "Communication"]
    }},
    "must_have_skills": ["Top 5 essential skills for this role"],
    "critical_gaps": ["Missing essential skills"],
    "evidence": ["CV excerpts showing skill depth"],
    "confidence": 85
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        cat = data.get("categorized_skills", {})
        for k in ["programming_languages", "frameworks_libraries", "databases", "tools_platforms", "soft_skills"]:
            if k not in cat:
                cat[k] = []
        result = {
            "tech_alignment_score": _safe_get(data, "tech_alignment_score", 50.0, float),
            "skill_depth_assessment": _safe_get(data, "skill_depth_assessment", "Surface-level listing", str),
            "skills_found": list(data.get("skills_found", [])),
            "categorized_skills": cat,
            "must_have_skills": list(data.get("must_have_skills", [])),
            "critical_gaps": list(data.get("critical_gaps", [])),
            "tech_confidence": _safe_get(data, "confidence", 70, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentTechMatcher failed: {e}")
        pl = extract_keywords(cv_text, ["python", "java", "javascript", "c++", "go", "ruby", "c#", "rust", "php", "typescript", "dart", "kotlin", "swift"])
        fl = extract_keywords(cv_text, ["django", "flask", "react", "angular", "vue", "flutter", "spring", "laravel", "express", "next.js", "bootstrap", "tailwind"])
        db = extract_keywords(cv_text, ["mysql", "postgresql", "mongodb", "sqlite", "redis", "oracle", "dynamodb"])
        tp = extract_keywords(cv_text, ["docker", "git", "kubernetes", "aws", "gcp", "azure", "jenkins", "linux", "firebase"])
        ss = extract_keywords(cv_text, ["communication", "teamwork", "leadership", "problem-solving", "adaptability", "creativity"])
        return {
            "tech_alignment_score": 50.0,
            "skill_depth_assessment": "Unable to assess — using keyword fallback",
            "skills_found": pl + fl + db + tp + ss,
            "categorized_skills": {"programming_languages": pl, "frameworks_libraries": fl, "databases": db, "tools_platforms": tp, "soft_skills": ss},
            "must_have_skills": [],
            "critical_gaps": ["Deep AI analysis unavailable. Review manually."],
            "tech_confidence": 25,
        }


# ---------- AGENT 3: Psychometric Profiler Agent ----------
def run_psych_agent(cv_text: str, job_role: str) -> dict:
    """Assesses Big Five personality traits and role alignment."""
    cache_k = _cache_key(cv_text, job_role, "psych")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert organizational psychologist with deep expertise in personality assessment.

TASK: Estimate the candidate's Big Five personality traits (0-100 percentiles) and explain alignment with '{job_role}'.

Think step-by-step for EACH trait:
1. Identify behavioral cues in the CV text (action verbs, project choices, leadership roles)
2. Assess communication style from writing patterns
3. Evaluate risk-taking vs. methodical approaches from project descriptions
4. Compare projected traits against typical '{job_role}' benchmarks

Cite specific CV evidence for your scoring decisions.
Return ONLY valid JSON.

JSON Schema:
{{
    "personality_traits": {{
        "openness": 75.0,
        "conscientiousness": 80.0,
        "extraversion": 55.0,
        "agreeableness": 65.0,
        "neuroticism": 35.0
    }},
    "trait_evidence": {{
        "openness": "Evidence from CV...",
        "conscientiousness": "Evidence from CV...",
        "extraversion": "Evidence from CV...",
        "agreeableness": "Evidence from CV...",
        "neuroticism": "Evidence from CV..."
    }},
    "personality_fit_explanation": "Comprehensive analysis of trait alignment with {job_role} benchmarks",
    "confidence": 80
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        traits = data.get("personality_traits", {})
        result = {
            "personality_traits": {t: float(traits.get(t, 50.0)) for t in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]},
            "trait_evidence": data.get("trait_evidence", {}),
            "personality_fit_explanation": _safe_get(data, "personality_fit_explanation", "Traits show general alignment.", str),
            "psych_confidence": _safe_get(data, "confidence", 70, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentPsychProfiler failed: {e}")
        scores = estimate_personality_from_cv(cv_text)
        return {
            "personality_traits": scores,
            "trait_evidence": {},
            "personality_fit_explanation": "Projected personality traits show general alignment with standard role profiles.",
            "psych_confidence": 25,
        }


# ---------- AGENT 4: Interviewer & Hiring Agent ----------
def run_interviewer_agent(cv_text: str, job_role: str) -> dict:
    """Determines seniority, writes targeted questions, makes hiring recommendation."""
    cache_k = _cache_key(cv_text, job_role, "interviewer")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert technical interviewer and engineering hiring manager at a top tech company.

TASK: Evaluate the candidate for '{job_role}' and produce a hiring recommendation.

Think step-by-step:
1. Assess total years of experience and career progression
2. Determine seniority level (Intern, Junior, Mid-level, Senior, Lead, Principal)
3. Evaluate quality of experience (FAANG/top-tier vs startups vs freelance)
4. Generate 3 targeted technical/behavioral interview questions customized to THIS candidate's gaps
5. Make a hiring decision with detailed justification

Each question should probe a specific weakness or validate a claimed strength.
Return ONLY valid JSON.

JSON Schema:
{{
    "seniority_level": "Junior | Mid-level | Senior | Lead | Principal",
    "experience_quality_feedback": "Professional evaluation of career quality and trajectory",
    "interview_questions": ["3 targeted questions customized to this candidate"],
    "question_rationale": ["Why each question was chosen — what it probes"],
    "hiring_recommendation": "Strong Hire | Hire | Lean Hire | Interview | Lean Reject | Reject",
    "recommendation_rationale": "Comprehensive hiring rationale with specific evidence",
    "confidence": 85
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "seniority_level": _safe_get(data, "seniority_level", "Mid-level", str),
            "experience_quality_feedback": _safe_get(data, "experience_quality_feedback", "Standard career milestones.", str),
            "interview_questions": list(data.get("interview_questions", ["Explain your core project."])),
            "question_rationale": list(data.get("question_rationale", [])),
            "hiring_recommendation": _safe_get(data, "hiring_recommendation", "Interview", str),
            "recommendation_rationale": _safe_get(data, "recommendation_rationale", "Recommend scheduling interview.", str),
            "interviewer_confidence": _safe_get(data, "confidence", 70, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentInterviewer failed: {e}")
        return {
            "seniority_level": "Mid-level",
            "experience_quality_feedback": "Standard career milestones detected.",
            "interview_questions": [
                "Explain your core project architecture and key decisions.",
                "How do you debug complex production issues?",
                "Describe a challenging problem you solved recently."
            ],
            "question_rationale": [],
            "hiring_recommendation": "Interview",
            "recommendation_rationale": "Standard profile. Recommend live technical interview.",
            "interviewer_confidence": 25,
        }


# ---------- AGENT 5: Recruitment Test Generator Agent ----------
def run_recruitment_test_agent(cv_text: str, job_role: str) -> dict:
    """Generates customized take-home assessment and interview evaluation rubric."""
    cache_k = _cache_key(cv_text, job_role, "test")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert technical interviewer and assessment designer.

TASK: Analyze the candidate's CV for '{job_role}' and generate a customized practical assessment.

Think step-by-step:
1. Identify the candidate's strongest technical areas
2. Identify areas that need validation (gaps or unproven claims)
3. Design a take-home challenge that tests BOTH strengths and gaps
4. Create an evaluation rubric with 2-3 specific interview questions

The challenge should be realistic, time-bounded, and directly relevant to '{job_role}'.
Return ONLY valid JSON.

JSON Schema:
{{
    "custom_coding_test": {{
        "title": "Descriptive title",
        "duration": "2-3 hours",
        "difficulty": "Easy | Medium | Hard",
        "problem_statement": "Detailed problem description tailored to this candidate",
        "requirements": ["3-5 specific technical requirements"],
        "evaluation_criteria": ["How to score the submission"],
        "submission_guidelines": "What and how to submit"
    }},
    "interview_evaluation_rubric": [
        {{
            "question": "Targeted interview question",
            "what_to_look_for": "Key signals in their answer",
            "grading_criteria": "Excellent / Good / Poor criteria"
        }}
    ],
    "confidence": 80
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        test_data = data.get("custom_coding_test", {})
        rubric_data = data.get("interview_evaluation_rubric", [])
        if not isinstance(test_data, dict):
            test_data = {}
        if not isinstance(rubric_data, list):
            rubric_data = []
        result = {
            "custom_coding_test": {
                "title": str(test_data.get("title", f"{job_role} Assessment")),
                "duration": str(test_data.get("duration", "3 Hours")),
                "difficulty": str(test_data.get("difficulty", "Medium")),
                "problem_statement": str(test_data.get("problem_statement", "Build a practical project.")),
                "requirements": list(test_data.get("requirements", ["Implement core logic", "Error handling", "Documentation"])),
                "evaluation_criteria": list(test_data.get("evaluation_criteria", [])),
                "submission_guidelines": str(test_data.get("submission_guidelines", "Upload to GitHub."))
            },
            "interview_evaluation_rubric": rubric_data,
            "test_confidence": _safe_get(data, "confidence", 70, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentRecruitmentTest failed: {e}")
        return {
            "custom_coding_test": {
                "title": f"Custom {job_role} Practical Evaluation",
                "duration": "3 Hours", "difficulty": "Medium",
                "problem_statement": f"Complete a targeted project for '{job_role}'.",
                "requirements": ["Implement core logic", "Error handling", "Documentation"],
                "evaluation_criteria": [], "submission_guidelines": "Submit via GitHub."
            },
            "interview_evaluation_rubric": [{
                "question": f"What are the best practices for {job_role}?",
                "what_to_look_for": "Modern standards and business context.",
                "grading_criteria": "Excellent: cites standards. Good: standard techniques. Poor: basic only."
            }],
            "test_confidence": 25,
        }


# ---------- AGENT 6: Career Trajectory Predictor Agent [NEW] ----------
def run_career_trajectory_agent(cv_text: str, job_role: str) -> dict:
    """Analyzes career progression velocity, predicts growth trajectory, detects career gaps."""
    cache_k = _cache_key(cv_text, job_role, "trajectory")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert career strategist and talent development consultant.

TASK: Analyze this candidate's career trajectory and predict their growth potential for '{job_role}'.

Think step-by-step:
1. Map the full career timeline — list every position with dates
2. Calculate average tenure at each company (months)
3. Track title progression — are they moving UP, LATERAL, or DOWN?
4. Identify career gaps (>6 months unexplained)
5. Calculate promotion frequency (promotions per year)
6. Classify trajectory pattern
7. Score career momentum 0-100

Patterns to detect:
- "Rising Star": Rapid promotions, increasing responsibility, <2 years between levels
- "Steady Performer": Consistent growth, 2-4 year tenures, gradual advancement
- "Plateau": Same level for 4+ years, lateral moves only
- "Career Switcher": Changed industries/domains, transferable skills assessment
- "Early Career": <2 years total experience, assess learning velocity

Return ONLY valid JSON.

JSON Schema:
{{
    "trajectory_pattern": "Rising Star | Steady Performer | Plateau | Career Switcher | Early Career",
    "career_momentum_score": 75.0,
    "promotion_frequency": "1 promotion every 2.5 years",
    "average_tenure_months": 24,
    "career_gaps": ["List of detected gaps with dates and duration"],
    "title_progression": ["List of titles in chronological order showing progression"],
    "growth_prediction": "Predicted career trajectory in next 2-3 years",
    "risk_assessment": "Low | Medium | High — risk of early departure or stagnation",
    "evidence": ["Specific CV passages that informed the analysis"],
    "confidence": 80
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "trajectory_pattern": _safe_get(data, "trajectory_pattern", "Steady Performer", str),
            "career_momentum_score": _safe_get(data, "career_momentum_score", 50.0, float),
            "promotion_frequency": _safe_get(data, "promotion_frequency", "Not enough data", str),
            "average_tenure_months": _safe_get(data, "average_tenure_months", 0, int),
            "career_gaps": list(data.get("career_gaps", [])),
            "title_progression": list(data.get("title_progression", [])),
            "growth_prediction": _safe_get(data, "growth_prediction", "Insufficient data for prediction.", str),
            "risk_assessment": _safe_get(data, "risk_assessment", "Medium", str),
            "trajectory_confidence": _safe_get(data, "confidence", 60, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentCareerTrajectory failed: {e}")
        return {
            "trajectory_pattern": "Unknown",
            "career_momentum_score": 50.0,
            "promotion_frequency": "Not enough data",
            "average_tenure_months": 0,
            "career_gaps": [],
            "title_progression": [],
            "growth_prediction": "Insufficient data for career trajectory prediction.",
            "risk_assessment": "Medium",
            "trajectory_confidence": 10,
        }


# ---------- AGENT 7: Red Flag Detector Agent [NEW] ----------
def run_red_flag_agent(cv_text: str, job_role: str) -> dict:
    """Scans for inconsistencies, exaggerations, and warning signs."""
    cache_k = _cache_key(cv_text, job_role, "redflag")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert HR investigator and background verification specialist.

TASK: Scan this CV for red flags, inconsistencies, and warning signs for '{job_role}'.

Think step-by-step, checking EACH category:
1. **Job Hopping**: >3 roles in 3 years without clear progression
2. **Overlapping Dates**: Multiple jobs listed at the same time without explanation
3. **Exaggerated Claims**: Unrealistic metrics (e.g., "saved $10M" as a junior developer), vague superlatives without evidence
4. **Education Gaps**: Degree claims that seem inconsistent with timeline
5. **Skill Inflation**: Listing expert-level skills without demonstrating them in projects/experience
6. **Unexplained Gaps**: Long periods (>1 year) with no activity listed
7. **Inconsistent Seniority**: Title doesn't match described responsibilities
8. **Generic Content**: Copy-pasted job descriptions instead of personalized achievements

For each flag, rate severity: Critical (deal-breaker) | Warning (needs investigation) | Minor (noted)
Be fair — not every gap is a red flag. Context matters.
Return ONLY valid JSON.

JSON Schema:
{{
    "red_flags_found": [
        {{
            "category": "Job Hopping | Overlapping Dates | Exaggeration | Education | Skill Inflation | Gap | Seniority | Generic",
            "severity": "Critical | Warning | Minor",
            "description": "Specific finding with evidence",
            "cv_evidence": "Exact text from CV that triggered this flag"
        }}
    ],
    "overall_risk_level": "Low | Medium | High | Critical",
    "total_flags": 2,
    "flags_summary": "Brief summary of key concerns",
    "positive_signals": ["List of POSITIVE integrity signals detected"],
    "confidence": 80
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "red_flags_found": list(data.get("red_flags_found", [])),
            "overall_risk_level": _safe_get(data, "overall_risk_level", "Low", str),
            "total_flags": _safe_get(data, "total_flags", 0, int),
            "flags_summary": _safe_get(data, "flags_summary", "No significant red flags detected.", str),
            "positive_signals": list(data.get("positive_signals", [])),
            "redflag_confidence": _safe_get(data, "confidence", 70, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentRedFlag failed: {e}")
        return {
            "red_flags_found": [],
            "overall_risk_level": "Unknown",
            "total_flags": 0,
            "flags_summary": "Red flag analysis unavailable. Manual review recommended.",
            "positive_signals": [],
            "redflag_confidence": 10,
        }


# ---------- AGENT 8: Competitive Benchmarker Agent [NEW] ----------
def run_benchmark_agent(cv_text: str, job_role: str) -> dict:
    """Compares candidate against market benchmarks for their role/experience level."""
    cache_k = _cache_key(cv_text, job_role, "benchmark")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are a senior talent market analyst and compensation consultant.

TASK: Compare this candidate against market benchmarks for '{job_role}'.

Think step-by-step:
1. Estimate the candidate's experience level (years + quality)
2. Compare their technical skills against the TOP 25% of '{job_role}' candidates in the market
3. Assess their project portfolio quality against market standards
4. Estimate a market-competitive salary range based on their profile
5. Rank their overall competitiveness (percentile)
6. Identify what would make them stand out MORE

Return ONLY valid JSON.

JSON Schema:
{{
    "market_percentile": 65,
    "market_competitiveness": "Below Average | Average | Above Average | Top Tier | Elite",
    "salary_range_usd": {{
        "low": 60000,
        "mid": 80000,
        "high": 100000,
        "currency_note": "USD annual, adjust for location"
    }},
    "skill_comparison": {{
        "above_market": ["Skills where candidate exceeds market average"],
        "at_market": ["Skills at par with market"],
        "below_market": ["Skills below market expectations"]
    }},
    "differentiators": ["What makes this candidate unique vs. competition"],
    "improvement_areas": ["What would move them to the next percentile bracket"],
    "comparable_titles": ["Job titles this candidate could realistically target"],
    "confidence": 75
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "market_percentile": _safe_get(data, "market_percentile", 50, int),
            "market_competitiveness": _safe_get(data, "market_competitiveness", "Average", str),
            "salary_range_usd": data.get("salary_range_usd", {"low": 0, "mid": 0, "high": 0}),
            "skill_comparison": data.get("skill_comparison", {"above_market": [], "at_market": [], "below_market": []}),
            "differentiators": list(data.get("differentiators", [])),
            "improvement_areas": list(data.get("improvement_areas", [])),
            "comparable_titles": list(data.get("comparable_titles", [])),
            "benchmark_confidence": _safe_get(data, "confidence", 60, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentBenchmark failed: {e}")
        return {
            "market_percentile": 50,
            "market_competitiveness": "Unknown",
            "salary_range_usd": {"low": 0, "mid": 0, "high": 0},
            "skill_comparison": {"above_market": [], "at_market": [], "below_market": []},
            "differentiators": [],
            "improvement_areas": ["Benchmarking unavailable. Manual market analysis recommended."],
            "comparable_titles": [],
            "benchmark_confidence": 10,
        }


# ---------- AGENT 9: Skill Verification Agent [NEW v2.1] ----------
def run_skill_verification_agent(cv_text: str, job_role: str) -> dict:
    """Cross-references claimed skills with demonstrated evidence in projects/experience."""
    cache_k = _cache_key(cv_text, job_role, "skillverify")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are a senior technical due diligence analyst specializing in talent verification.

TASK: Cross-reference EVERY skill claimed in this CV against demonstrated evidence.

Think step-by-step:
1. List ALL skills/technologies claimed anywhere in the CV
2. For EACH skill, search for evidence in projects, work experience, achievements
3. Classify each skill: VERIFIED (demonstrated in project), PARTIALLY VERIFIED (mentioned in context), UNVERIFIED (only listed), INFLATED (claimed expert but evidence shows basic)
4. Calculate verification_score = (verified_count / total_claimed) * 100
5. Identify strongest skill (most evidence) and weakest claim (least evidence)

Return ONLY valid JSON.

JSON Schema:
{{
    "skills_audit": [
        {{"skill": "Python", "status": "Verified", "evidence": "Built 3 Django projects", "depth": "Advanced"}}
    ],
    "verification_score": 72.0,
    "verified_count": 8,
    "total_claimed": 11,
    "strongest_skill": {{"skill": "Python", "evidence": "Built 3 production projects"}},
    "weakest_claim": {{"skill": "Kubernetes", "reason": "Listed but no project uses it"}},
    "overall_credibility": "High | Moderate | Low | Questionable",
    "confidence": 80
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "skills_audit": list(data.get("skills_audit", [])),
            "verification_score": _safe_get(data, "verification_score", 50.0, float),
            "verified_count": _safe_get(data, "verified_count", 0, int),
            "total_claimed": _safe_get(data, "total_claimed", 0, int),
            "strongest_skill": data.get("strongest_skill", {}),
            "weakest_claim": data.get("weakest_claim", {}),
            "overall_credibility": _safe_get(data, "overall_credibility", "Moderate", str),
            "skillverify_confidence": _safe_get(data, "confidence", 60, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentSkillVerify failed: {e}")
        return {
            "skills_audit": [], "verification_score": 50.0, "verified_count": 0,
            "total_claimed": 0, "strongest_skill": {}, "weakest_claim": {},
            "overall_credibility": "Unknown", "skillverify_confidence": 10,
        }


# ---------- AGENT 10: Culture Fit Analyzer [NEW v2.1] ----------
def run_culture_fit_agent(cv_text: str, job_role: str) -> dict:
    """Analyzes communication style, values, and workplace culture alignment."""
    cache_k = _cache_key(cv_text, job_role, "culturefit")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are an expert organizational culture consultant and workplace psychologist.

TASK: Analyze this candidate's communication style, values, and culture fit indicators for '{job_role}'.

Think step-by-step:
1. Analyze WRITING STYLE: formal vs casual, verbose vs concise, technical vs business-oriented
2. Detect VALUE SIGNALS: innovation, stability, autonomy, collaboration, growth, impact
3. Assess WORK PREFERENCES: remote vs office, startup vs corporate, team vs individual
4. Evaluate COMMUNICATION PATTERNS: how they describe achievements (I vs We), leadership style
5. Identify CULTURAL ARCHETYPES they best match

Return ONLY valid JSON.

JSON Schema:
{{
    "communication_style": {{
        "formality": "Formal | Semi-formal | Casual",
        "verbosity": "Concise | Balanced | Detailed",
        "orientation": "Technical | Business | Hybrid"
    }},
    "core_values_detected": ["innovation", "collaboration", "growth"],
    "work_style_indicators": {{
        "autonomy_preference": "High | Medium | Low",
        "team_orientation": "Individual Contributor | Team Player | Leader",
        "pace_preference": "Fast-paced Startup | Balanced | Structured Corporate",
        "risk_tolerance": "Risk Taker | Moderate | Risk Averse"
    }},
    "cultural_archetypes": ["Builder", "Optimizer", "Connector"],
    "ideal_environment": "Description of the ideal work environment for this candidate",
    "culture_fit_score": 75.0,
    "culture_warnings": ["Potential cultural mismatches to watch for"],
    "confidence": 70
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "communication_style": data.get("communication_style", {}),
            "core_values_detected": list(data.get("core_values_detected", [])),
            "work_style_indicators": data.get("work_style_indicators", {}),
            "cultural_archetypes": list(data.get("cultural_archetypes", [])),
            "ideal_environment": _safe_get(data, "ideal_environment", "Not assessed", str),
            "culture_fit_score": _safe_get(data, "culture_fit_score", 50.0, float),
            "culture_warnings": list(data.get("culture_warnings", [])),
            "culturefit_confidence": _safe_get(data, "confidence", 60, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentCultureFit failed: {e}")
        return {
            "communication_style": {}, "core_values_detected": [],
            "work_style_indicators": {}, "cultural_archetypes": [],
            "ideal_environment": "Not assessed", "culture_fit_score": 50.0,
            "culture_warnings": [], "culturefit_confidence": 10,
        }


# ---------- AGENT 11: Learning Path Generator [NEW v2.1] ----------
def run_learning_path_agent(cv_text: str, job_role: str) -> dict:
    """Generates personalized upskilling roadmap based on identified gaps."""
    cache_k = _cache_key(cv_text, job_role, "learnpath")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are a senior career development coach and technical training architect.

TASK: Generate a personalized learning and upskilling roadmap for this candidate targeting '{job_role}'.

Think step-by-step:
1. Identify the candidate's current skill level and experience
2. Compare against the IDEAL profile for '{job_role}'
3. Identify the TOP skill gaps that would have the highest impact if filled
4. For each gap, recommend specific learning resources (courses, certifications, projects)
5. Create a prioritized 90-day learning plan
6. Estimate the career impact of completing the plan

Return ONLY valid JSON.

JSON Schema:
{{
    "current_level": "Junior | Mid | Senior | Lead",
    "target_level": "The level they could reach after upskilling",
    "skill_gaps_prioritized": [
        {{
            "skill": "Kubernetes",
            "priority": "Critical | High | Medium | Low",
            "current_level": "None | Beginner | Intermediate",
            "target_level": "Intermediate | Advanced",
            "recommended_resources": [
                {{"type": "Course", "name": "Specific course name", "platform": "Udemy/Coursera/etc", "duration": "20 hours"}}
            ],
            "estimated_learning_time": "2-3 weeks"
        }}
    ],
    "ninety_day_plan": {{
        "month_1": "Focus area and goals for month 1",
        "month_2": "Focus area and goals for month 2",
        "month_3": "Focus area and goals for month 3"
    }},
    "recommended_certifications": ["AWS Solutions Architect", "PMP"],
    "recommended_projects": [
        {{"title": "Portfolio project title", "description": "What to build and why", "skills_practiced": ["list"]}}
    ],
    "career_impact_prediction": "Expected improvement in marketability after completing the plan",
    "confidence": 75
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "current_level": _safe_get(data, "current_level", "Mid", str),
            "target_level": _safe_get(data, "target_level", "Senior", str),
            "skill_gaps_prioritized": list(data.get("skill_gaps_prioritized", [])),
            "ninety_day_plan": data.get("ninety_day_plan", {}),
            "recommended_certifications": list(data.get("recommended_certifications", [])),
            "recommended_projects": list(data.get("recommended_projects", [])),
            "career_impact_prediction": _safe_get(data, "career_impact_prediction", "Not assessed", str),
            "learnpath_confidence": _safe_get(data, "confidence", 60, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentLearningPath failed: {e}")
        return {
            "current_level": "Unknown", "target_level": "Unknown",
            "skill_gaps_prioritized": [], "ninety_day_plan": {},
            "recommended_certifications": [], "recommended_projects": [],
            "career_impact_prediction": "Not assessed", "learnpath_confidence": 10,
        }


# ---------- AGENT 12: Onboarding Plan Generator [NEW v2.2] ----------
def run_onboarding_agent(cv_text: str, job_role: str) -> dict:
    """Generates a personalized 30/60/90 day onboarding plan."""
    cache_k = _cache_key(cv_text, job_role, "onboard")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are a senior People Operations specialist and onboarding architect.

TASK: Generate a personalized onboarding plan for this candidate if hired for '{job_role}'.

Think step-by-step:
1. Assess the candidate's experience level and skill gaps
2. Design a tailored 30/60/90 day plan
3. Identify what tools and systems they need access to
4. Suggest a buddy/mentor profile that would complement their skills
5. Define clear success metrics for each phase

Return ONLY valid JSON.

JSON Schema:
{{
    "day_30_goals": ["Specific onboarding goals for first 30 days"],
    "day_60_goals": ["Goals for days 31-60"],
    "day_90_goals": ["Goals for days 61-90"],
    "required_tooling": ["List of tools/systems they need access to"],
    "recommended_buddy_profile": "Description of ideal mentor/buddy",
    "training_priorities": ["Specific training needed based on their gaps"],
    "first_project_suggestion": {{"title": "Suggested first project", "why": "Why this project is ideal for ramp-up"}},
    "success_kpis": ["Measurable KPIs to track onboarding success"],
    "risk_areas": ["Areas where they might struggle during onboarding"],
    "estimated_ramp_time": "Estimated weeks to full productivity",
    "confidence": 70
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "day_30_goals": list(data.get("day_30_goals", [])),
            "day_60_goals": list(data.get("day_60_goals", [])),
            "day_90_goals": list(data.get("day_90_goals", [])),
            "required_tooling": list(data.get("required_tooling", [])),
            "recommended_buddy_profile": _safe_get(data, "recommended_buddy_profile", "Senior engineer", str),
            "training_priorities": list(data.get("training_priorities", [])),
            "first_project_suggestion": data.get("first_project_suggestion", {}),
            "success_kpis": list(data.get("success_kpis", [])),
            "risk_areas": list(data.get("risk_areas", [])),
            "estimated_ramp_time": _safe_get(data, "estimated_ramp_time", "8-12 weeks", str),
            "onboard_confidence": _safe_get(data, "confidence", 60, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentOnboarding failed: {e}")
        return {
            "day_30_goals": [], "day_60_goals": [], "day_90_goals": [],
            "required_tooling": [], "recommended_buddy_profile": "Senior team member",
            "training_priorities": [], "first_project_suggestion": {},
            "success_kpis": [], "risk_areas": [],
            "estimated_ramp_time": "Unknown", "onboard_confidence": 10,
        }


# ---------- AGENT 13: Salary Negotiation Intelligence [NEW v2.2] ----------
def run_salary_agent(cv_text: str, job_role: str) -> dict:
    """Provides salary negotiation intelligence for hiring managers."""
    cache_k = _cache_key(cv_text, job_role, "salary")
    cached = _cache_get(cache_k)
    if cached:
        return cached

    cv_clean = _sanitize_cv_text(cv_text)
    prompt = f"""You are a senior compensation analyst and salary negotiation strategist.

TASK: Provide comprehensive salary negotiation intelligence for this '{job_role}' candidate.

Think step-by-step:
1. Estimate the candidate's current market value based on skills and experience
2. Determine the appropriate salary band for their level
3. Calculate counter-offer thresholds
4. Assess total compensation expectations (base + bonus + equity + benefits)
5. Identify negotiation leverage points (both employer and candidate)

Return ONLY valid JSON.

JSON Schema:
{{
    "recommended_base_salary": {{"low": 60000, "mid": 80000, "high": 100000, "currency": "USD"}},
    "total_comp_estimate": {{"base": 80000, "bonus_pct": 15, "equity_value": 20000, "total": 112000}},
    "candidate_leverage_points": ["Strong points the candidate can use in negotiation"],
    "employer_leverage_points": ["Points the employer can use"],
    "counter_offer_threshold": {{"max_acceptable": 95000, "walk_away_above": 110000}},
    "negotiation_risk": "Low | Medium | High",
    "market_demand_for_role": "Low | Moderate | High | Very High",
    "recommended_offer_strategy": "How to structure the offer for best acceptance rate",
    "benefits_to_emphasize": ["Non-salary benefits that would appeal to this candidate"],
    "confidence": 70
}}

CV Text:
{cv_clean}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        result = {
            "recommended_base_salary": data.get("recommended_base_salary", {}),
            "total_comp_estimate": data.get("total_comp_estimate", {}),
            "candidate_leverage_points": list(data.get("candidate_leverage_points", [])),
            "employer_leverage_points": list(data.get("employer_leverage_points", [])),
            "counter_offer_threshold": data.get("counter_offer_threshold", {}),
            "negotiation_risk": _safe_get(data, "negotiation_risk", "Medium", str),
            "market_demand_for_role": _safe_get(data, "market_demand_for_role", "Moderate", str),
            "recommended_offer_strategy": _safe_get(data, "recommended_offer_strategy", "Standard offer", str),
            "benefits_to_emphasize": list(data.get("benefits_to_emphasize", [])),
            "salary_confidence": _safe_get(data, "confidence", 60, int),
        }
        _cache_set(cache_k, result)
        return result
    except Exception as e:
        logger.error(f"AgentSalary failed: {e}")
        return {
            "recommended_base_salary": {}, "total_comp_estimate": {},
            "candidate_leverage_points": [], "employer_leverage_points": [],
            "counter_offer_threshold": {}, "negotiation_risk": "Unknown",
            "market_demand_for_role": "Unknown", "recommended_offer_strategy": "N/A",
            "benefits_to_emphasize": [], "salary_confidence": 10,
        }


# ====================== Auto-Shortlist Engine ======================

def auto_shortlist(candidates: list, job_role: str, top_n: int = 5) -> dict:
    """AI-powered automatic shortlisting with reasoning for top N candidates."""
    if not candidates:
        return {"shortlist": [], "reasoning": "No candidates available."}

    # Build compact summaries
    summaries = []
    for c in candidates[:30]:  # Cap at 30 for token limits
        summaries.append({
            "name": c.get("name", "Unknown"),
            "score": c.get("final_score", 0),
            "exp": c.get("exp_years", 0),
            "skills": c.get("soft_skills", "")[:100],
        })

    prompt = f"""You are a Chief Talent Acquisition Officer selecting the TOP {top_n} candidates for '{job_role}'.

Candidate Pool (sorted by score):
{json.dumps(summaries, indent=None)}

Select the TOP {top_n} candidates and explain why.
Return ONLY valid JSON.

JSON Schema:
{{
    "shortlist": [
        {{"name": "Candidate name", "rank": 1, "reason": "Why this candidate was selected"}}
    ],
    "pool_quality_assessment": "Overall assessment of the candidate pool quality",
    "hiring_difficulty": "Easy | Moderate | Difficult | Very Difficult"
}}"""
    try:
        return groq_call(prompt, response_format={"type": "json_object"}, temperature=0.15)
    except Exception as e:
        logger.error(f"Auto-shortlist failed: {e}")
        sorted_c = sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)[:top_n]
        return {
            "shortlist": [{"name": c.get("name"), "rank": i+1, "reason": f"Score: {c.get('final_score', 0)}%"} for i, c in enumerate(sorted_c)],
            "pool_quality_assessment": "Automated fallback ranking by score.",
            "hiring_difficulty": "Unknown",
        }


# ====================== Team Builder ======================

def find_team_complement(existing_team_skills: list, candidates: list, job_role: str, team_size: int = 3) -> dict:
    """Finds candidates that best complement existing team skills."""
    summaries = []
    for c in candidates[:20]:
        ai_rep = c.get("ai_report", {})
        summaries.append({
            "name": c.get("name", "Unknown"),
            "score": c.get("final_score", 0),
            "skills": c.get("soft_skills", "")[:100],
            "archetypes": ai_rep.get("cultural_archetypes", [])[:2],
        })

    prompt = f"""You are a Team Composition Architect building the optimal team for '{job_role}'.

Existing Team Skills: {json.dumps(existing_team_skills)}

Available Candidates:
{json.dumps(summaries, indent=None)}

Select {team_size} candidates that BEST COMPLEMENT the existing team (not duplicate).
Return ONLY valid JSON.

JSON Schema:
{{
    "recommended_hires": [
        {{"name": "Candidate", "role_in_team": "Their specific role", "complementary_skills": ["Skills they add"], "why": "Why they complement the team"}}
    ],
    "team_gaps_filled": ["Skills gaps that would be filled"],
    "remaining_gaps": ["Skills still missing after these hires"],
    "team_synergy_score": 78
}}"""
    try:
        return groq_call(prompt, response_format={"type": "json_object"}, temperature=0.2)
    except Exception as e:
        logger.error(f"Team builder failed: {e}")
        return {"recommended_hires": [], "team_gaps_filled": [], "remaining_gaps": [], "team_synergy_score": 0}


def run_supervisor_agent(all_results: dict, cv_text: str, job_role: str) -> dict:
    """
    Supervisor Meta-Agent v2.2: Synthesizes ALL 13 agent outputs into a unified executive verdict.
    Resolves conflicts, calculates overall confidence, and produces top strengths/concerns.
    """
    cv_clean = _sanitize_cv_text(cv_text)[:3000]

    agent_summary = json.dumps({
        "audit": {
            "formatting_score": all_results.get("formatting_score"),
            "ats_compatibility": all_results.get("ats_compatibility"),
        },
        "tech": {
            "tech_alignment_score": all_results.get("tech_alignment_score"),
            "skill_depth": all_results.get("skill_depth_assessment"),
            "critical_gaps": all_results.get("critical_gaps", [])[:3],
        },
        "psychology": {
            "personality_fit": all_results.get("personality_fit_explanation", "")[:150],
        },
        "interviewer": {
            "seniority": all_results.get("seniority_level"),
            "recommendation": all_results.get("hiring_recommendation"),
            "rationale": all_results.get("recommendation_rationale", "")[:150],
        },
        "trajectory": {
            "pattern": all_results.get("trajectory_pattern"),
            "momentum": all_results.get("career_momentum_score"),
            "risk": all_results.get("risk_assessment"),
        },
        "red_flags": {
            "risk_level": all_results.get("overall_risk_level"),
            "total_flags": all_results.get("total_flags"),
            "summary": all_results.get("flags_summary", "")[:150],
        },
        "benchmark": {
            "percentile": all_results.get("market_percentile"),
            "competitiveness": all_results.get("market_competitiveness"),
        },
        "skill_verification": {
            "credibility": all_results.get("overall_credibility"),
            "verification_score": all_results.get("verification_score"),
        },
        "culture_fit": {
            "score": all_results.get("culture_fit_score"),
            "archetypes": all_results.get("cultural_archetypes", [])[:3],
        },
        "learning_path": {
            "current_level": all_results.get("current_level"),
            "target_level": all_results.get("target_level"),
            "top_gap": all_results.get("skill_gaps_prioritized", [{}])[0].get("skill", "N/A") if all_results.get("skill_gaps_prioritized") else "N/A",
        },
        "onboarding": {
            "ramp_time": all_results.get("estimated_ramp_time"),
            "risk_areas_count": len(all_results.get("risk_areas", [])),
        },
        "salary": {
            "negotiation_risk": all_results.get("negotiation_risk"),
            "market_demand": all_results.get("market_demand_for_role"),
        },
    }, indent=None)

    prompt = f"""You are the Chief Hiring Officer supervising a panel of 13 AI evaluation agents.

TASK: Synthesize ALL agent evaluations into a unified executive hiring verdict for '{job_role}'.

Agent Panel Results:
{agent_summary}

Brief CV Context:
{cv_clean}

Think step-by-step:
1. Review each agent's findings and confidence levels
2. Identify AGREEMENTS between agents (convergent signals)
3. Identify CONFLICTS between agents (e.g., tech says Hire but red flags say Reject)
4. Resolve conflicts by weighing evidence strength and confidence
5. Determine the TOP 3 strengths and TOP 3 concerns
6. Make a FINAL unified verdict
7. Calculate overall analysis confidence

Return ONLY valid JSON.

JSON Schema:
{{
    "executive_summary": "3-4 sentence professional executive summary synthesizing all 11 agent findings",
    "unified_verdict": "Strong Hire | Hire | Lean Hire | Interview | Lean Reject | Reject",
    "verdict_rationale": "Why this verdict was chosen, citing specific agent findings",
    "top_strengths": ["Top 3 candidate strengths with evidence"],
    "top_concerns": ["Top 3 candidate concerns or risks"],
    "agent_consensus": "Full Agreement | Mostly Aligned | Mixed Signals | Conflicting",
    "conflicts_resolved": ["Description of agent conflicts and resolutions"],
    "overall_confidence": 78,
    "next_steps": ["Recommended next steps for the recruiter"],
    "candidate_tier": "A-Tier (Top 10%) | B-Tier (Top 25%) | C-Tier (Average) | D-Tier (Below Average) | F-Tier (Not Qualified)"
}}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"}, temperature=0.15)
        return {
            "executive_summary": _safe_get(data, "executive_summary", "Analysis completed.", str),
            "unified_verdict": _safe_get(data, "unified_verdict", "Interview", str),
            "verdict_rationale": _safe_get(data, "verdict_rationale", "Standard profile assessment.", str),
            "top_strengths": list(data.get("top_strengths", [])),
            "top_concerns": list(data.get("top_concerns", [])),
            "agent_consensus": _safe_get(data, "agent_consensus", "Mostly Aligned", str),
            "conflicts_resolved": list(data.get("conflicts_resolved", [])),
            "overall_confidence": _safe_get(data, "overall_confidence", 50, int),
            "next_steps": list(data.get("next_steps", ["Conduct live interview"])),
            "candidate_tier": _safe_get(data, "candidate_tier", "C-Tier (Average)", str),
        }
    except Exception as e:
        logger.error(f"SupervisorAgent failed: {e}")
        return {
            "executive_summary": "Multi-agent analysis completed. Review individual agent reports.",
            "unified_verdict": all_results.get("hiring_recommendation", "Interview"),
            "verdict_rationale": all_results.get("recommendation_rationale", "Standard."),
            "top_strengths": [], "top_concerns": [],
            "agent_consensus": "Unknown", "conflicts_resolved": [],
            "overall_confidence": 30,
            "next_steps": ["Conduct live technical interview"],
            "candidate_tier": "C-Tier (Average)",
        }


# ======================================================================
#  ORCHESTRATOR v2.1 — 11 Parallel Agents + Supervisor
# ======================================================================

def get_ai_deep_analysis(cv_text: str, job_role: str) -> dict:
    """
    Parallelized multi-agent orchestration framework v2.1
    Runs 13 specialized agents concurrently, then a Supervisor Meta-Agent to synthesize.
    """
    t_start = time.time()

    # Phase 1: Run all 13 agents in parallel
    with ThreadPoolExecutor(max_workers=13) as executor:
        future_audit = executor.submit(run_audit_agent, cv_text, job_role)
        future_tech = executor.submit(run_tech_matcher_agent, cv_text, job_role)
        future_psych = executor.submit(run_psych_agent, cv_text, job_role)
        future_interviewer = executor.submit(run_interviewer_agent, cv_text, job_role)
        future_test = executor.submit(run_recruitment_test_agent, cv_text, job_role)
        future_trajectory = executor.submit(run_career_trajectory_agent, cv_text, job_role)
        future_redflag = executor.submit(run_red_flag_agent, cv_text, job_role)
        future_benchmark = executor.submit(run_benchmark_agent, cv_text, job_role)
        future_skillverify = executor.submit(run_skill_verification_agent, cv_text, job_role)
        future_culture = executor.submit(run_culture_fit_agent, cv_text, job_role)
        future_learnpath = executor.submit(run_learning_path_agent, cv_text, job_role)
        future_onboard = executor.submit(run_onboarding_agent, cv_text, job_role)
        future_salary = executor.submit(run_salary_agent, cv_text, job_role)

        def safe_result(future, agent_name, default):
            try:
                return future.result(timeout=90)
            except Exception as e:
                logger.error(f"Agent '{agent_name}' future failed: {e}")
                return default

        audit_res = safe_result(future_audit, "Audit", {"formatting_score": 70.0, "formatting_feedback": "Standard.", "bullet_points_improvement": [], "ats_compatibility": "Medium", "audit_confidence": 20})
        tech_res = safe_result(future_tech, "TechMatcher", {"tech_alignment_score": 50.0, "skill_depth_assessment": "Unknown", "skills_found": [], "categorized_skills": {"programming_languages": [], "frameworks_libraries": [], "databases": [], "tools_platforms": [], "soft_skills": []}, "must_have_skills": [], "critical_gaps": [], "tech_confidence": 20})
        psych_res = safe_result(future_psych, "PsychProfiler", {"personality_traits": {"openness": 50, "conscientiousness": 50, "extraversion": 50, "agreeableness": 50, "neuroticism": 50}, "trait_evidence": {}, "personality_fit_explanation": "General alignment.", "psych_confidence": 20})
        interviewer_res = safe_result(future_interviewer, "Interviewer", {"seniority_level": "Mid-level", "experience_quality_feedback": "Standard.", "interview_questions": ["Explain your project."], "question_rationale": [], "hiring_recommendation": "Interview", "recommendation_rationale": "Standard.", "interviewer_confidence": 20})
        test_res = safe_result(future_test, "TestGenerator", {"custom_coding_test": {"title": f"{job_role} Assessment", "duration": "3h", "difficulty": "Medium", "problem_statement": "Build a project.", "requirements": [], "evaluation_criteria": [], "submission_guidelines": "GitHub."}, "interview_evaluation_rubric": [], "test_confidence": 20})
        trajectory_res = safe_result(future_trajectory, "CareerTrajectory", {"trajectory_pattern": "Unknown", "career_momentum_score": 50.0, "promotion_frequency": "N/A", "average_tenure_months": 0, "career_gaps": [], "title_progression": [], "growth_prediction": "Insufficient data.", "risk_assessment": "Medium", "trajectory_confidence": 10})
        redflag_res = safe_result(future_redflag, "RedFlag", {"red_flags_found": [], "overall_risk_level": "Unknown", "total_flags": 0, "flags_summary": "N/A.", "positive_signals": [], "redflag_confidence": 10})
        benchmark_res = safe_result(future_benchmark, "Benchmark", {"market_percentile": 50, "market_competitiveness": "Unknown", "salary_range_usd": {"low": 0, "mid": 0, "high": 0}, "skill_comparison": {"above_market": [], "at_market": [], "below_market": []}, "differentiators": [], "improvement_areas": [], "comparable_titles": [], "benchmark_confidence": 10})
        skillverify_res = safe_result(future_skillverify, "SkillVerify", {"skills_audit": [], "verification_score": 50.0, "verified_count": 0, "total_claimed": 0, "strongest_skill": {}, "weakest_claim": {}, "overall_credibility": "Unknown", "skillverify_confidence": 10})
        culture_res = safe_result(future_culture, "CultureFit", {"communication_style": {}, "core_values_detected": [], "work_style_indicators": {}, "cultural_archetypes": [], "ideal_environment": "N/A", "culture_fit_score": 50.0, "culture_warnings": [], "culturefit_confidence": 10})
        learnpath_res = safe_result(future_learnpath, "LearningPath", {"current_level": "Unknown", "target_level": "Unknown", "skill_gaps_prioritized": [], "ninety_day_plan": {}, "recommended_certifications": [], "recommended_projects": [], "career_impact_prediction": "N/A", "learnpath_confidence": 10})
        onboard_res = safe_result(future_onboard, "Onboarding", {"day_30_goals": [], "day_60_goals": [], "day_90_goals": [], "required_tooling": [], "recommended_buddy_profile": "N/A", "training_priorities": [], "first_project_suggestion": {}, "success_kpis": [], "risk_areas": [], "estimated_ramp_time": "Unknown", "onboard_confidence": 10})
        salary_res = safe_result(future_salary, "Salary", {"recommended_base_salary": {}, "total_comp_estimate": {}, "candidate_leverage_points": [], "employer_leverage_points": [], "counter_offer_threshold": {}, "negotiation_risk": "Unknown", "market_demand_for_role": "Unknown", "recommended_offer_strategy": "N/A", "benefits_to_emphasize": [], "salary_confidence": 10})

    # Phase 2: Merge all 13 agent results
    merged = {}
    merged.update(audit_res)
    merged.update(tech_res)
    merged.update(psych_res)
    merged.update(interviewer_res)
    merged.update(test_res)
    merged.update(trajectory_res)
    merged.update(redflag_res)
    merged.update(benchmark_res)
    merged.update(skillverify_res)
    merged.update(culture_res)
    merged.update(learnpath_res)
    merged.update(onboard_res)
    merged.update(salary_res)

    # Phase 3: Run Supervisor Meta-Agent on merged results
    try:
        supervisor_res = run_supervisor_agent(merged, cv_text, job_role)
        merged.update(supervisor_res)
    except Exception as e:
        logger.error(f"Supervisor agent failed: {e}")
        merged["executive_summary"] = "Multi-agent analysis completed."
        merged["unified_verdict"] = merged.get("hiring_recommendation", "Interview")
        merged["overall_confidence"] = 30
        merged["candidate_tier"] = "C-Tier (Average)"

    # Phase 4: Calculate aggregate confidence from all 11 agents
    confidence_scores = [
        merged.get("audit_confidence", 50),
        merged.get("tech_confidence", 50),
        merged.get("psych_confidence", 50),
        merged.get("interviewer_confidence", 50),
        merged.get("trajectory_confidence", 50),
        merged.get("redflag_confidence", 50),
        merged.get("benchmark_confidence", 50),
        merged.get("skillverify_confidence", 50),
        merged.get("culturefit_confidence", 50),
        merged.get("learnpath_confidence", 50),
        merged.get("onboard_confidence", 50),
        merged.get("salary_confidence", 50),
    ]
    merged["analysis_confidence"] = round(sum(confidence_scores) / len(confidence_scores), 1)

    elapsed = round(time.time() - t_start, 2)
    merged["analysis_time_seconds"] = elapsed
    merged["agents_executed"] = 13
    logger.info(f"Multi-agent analysis completed in {elapsed}s (13 agents + supervisor)")

    return merged


# ====================== Cover Letter Generator ======================

def generate_cover_letter(cv_text: str, job_role: str, company_name: str = "the company",
                          tone: str = "professional") -> str:
    """Generates a tailored cover letter using AI based on CV and target role."""
    cv_clean = _sanitize_cv_text(cv_text)

    tone_map = {
        "professional": "formal, polished corporate tone",
        "enthusiastic": "energetic, passionate, and enthusiastic tone showing genuine excitement",
        "confident": "bold, confident, results-driven executive tone",
        "creative": "creative, unique, and memorable tone that stands out",
    }
    tone_desc = tone_map.get(tone, tone_map["professional"])

    prompt = f"""You are an expert career coach and professional cover letter writer.

TASK: Write a compelling, personalized cover letter for this candidate applying to '{job_role}' at {company_name}.

Requirements:
1. Use a {tone_desc}
2. Reference SPECIFIC projects, skills, and achievements from their CV
3. Explain WHY they are a great fit for '{job_role}'
4. Show genuine interest in the role and company
5. Keep it to 3-4 paragraphs (300-400 words)
6. Include a strong opening hook and confident closing
7. Do NOT use generic filler — every sentence should add value

CV Text:
{cv_clean}

Output ONLY the cover letter text. No notes, no headers, no markdown formatting."""
    try:
        return groq_call(
            prompt,
            system_msg="You are a professional cover letter writer. Output ONLY the letter text.",
            temperature=0.4,
            parse_json=False
        )
    except Exception as e:
        logger.error(f"Cover letter generation failed: {e}")
        return f"Error generating cover letter: {str(e)}"


# ====================== Head-to-Head Comparison ======================

def compare_candidates(candidate_a: dict, candidate_b: dict, job_role: str) -> dict:
    """AI-powered head-to-head comparison of two candidates for the same role."""
    summary_a = (
        f"Name: {candidate_a.get('name', 'A')}\n"
        f"Score: {candidate_a.get('final_score', 0)}% | Job Match: {candidate_a.get('job_score', 0)}%\n"
        f"Experience: {candidate_a.get('exp_years', 0)} yrs\n"
        f"Skills: {candidate_a.get('soft_skills', '')}\n"
        f"Personality Score: {candidate_a.get('personality_score', 0)}%\n"
        f"Seniority: {candidate_a.get('ai_report', {}).get('seniority_level', 'N/A')}\n"
        f"Market Percentile: {candidate_a.get('ai_report', {}).get('market_percentile', 'N/A')}\n"
        f"Trajectory: {candidate_a.get('ai_report', {}).get('trajectory_pattern', 'N/A')}\n"
        f"Risk Level: {candidate_a.get('ai_report', {}).get('overall_risk_level', 'N/A')}\n"
        f"Credibility: {candidate_a.get('ai_report', {}).get('overall_credibility', 'N/A')}"
    )
    summary_b = (
        f"Name: {candidate_b.get('name', 'B')}\n"
        f"Score: {candidate_b.get('final_score', 0)}% | Job Match: {candidate_b.get('job_score', 0)}%\n"
        f"Experience: {candidate_b.get('exp_years', 0)} yrs\n"
        f"Skills: {candidate_b.get('soft_skills', '')}\n"
        f"Personality Score: {candidate_b.get('personality_score', 0)}%\n"
        f"Seniority: {candidate_b.get('ai_report', {}).get('seniority_level', 'N/A')}\n"
        f"Market Percentile: {candidate_b.get('ai_report', {}).get('market_percentile', 'N/A')}\n"
        f"Trajectory: {candidate_b.get('ai_report', {}).get('trajectory_pattern', 'N/A')}\n"
        f"Risk Level: {candidate_b.get('ai_report', {}).get('overall_risk_level', 'N/A')}\n"
        f"Credibility: {candidate_b.get('ai_report', {}).get('overall_credibility', 'N/A')}"
    )

    prompt = f"""You are a Chief Hiring Officer comparing two candidates for '{job_role}'.

Candidate A:
{summary_a}

Candidate B:
{summary_b}

Compare them across ALL dimensions and pick a WINNER.
Return ONLY valid JSON.

JSON Schema:
{{
    "winner": "Name of the winning candidate",
    "winner_rationale": "Why this candidate is the better choice",
    "dimension_comparison": [
        {{
            "dimension": "Technical Skills",
            "candidate_a_score": 75,
            "candidate_b_score": 82,
            "advantage": "Candidate B",
            "explanation": "B has deeper framework experience"
        }}
    ],
    "final_recommendation": "Clear recommendation paragraph",
    "risk_comparison": "Which candidate carries more hiring risk and why"
}}"""
    try:
        return groq_call(prompt, response_format={"type": "json_object"}, temperature=0.2)
    except Exception as e:
        logger.error(f"Head-to-head comparison failed: {e}")
        return {"error": str(e)}



# ====================== Custom JD & Talent RAG Search ======================

def extract_targets_from_custom_jd(jd_text: str) -> dict:
    prompt = f"""You are an expert HR analyst. Analyze the custom job description and extract benchmarks.

Think step-by-step:
1. Identify the exact job title from the description
2. Estimate required years of experience
3. Determine ideal Big Five personality traits for this role

Return ONLY valid JSON.

JSON Schema:
{{
    "job_role": "Professional job title",
    "experience_target": 3.0,
    "personality_target": {{
        "openness": 70.0, "conscientiousness": 80.0, "extraversion": 50.0,
        "agreeableness": 60.0, "neuroticism": 35.0
    }},
    "key_requirements": ["Top 5 must-have skills from the JD"]
}}

Job Description:
{jd_text}"""
    try:
        return groq_call(prompt, response_format={"type": "json_object"})
    except Exception as e:
        logger.error(f"Custom JD extraction failed: {e}")
        return {
            "job_role": "Custom Job Role",
            "experience_target": 2.0,
            "personality_target": {"openness": 60, "conscientiousness": 70, "extraversion": 50, "agreeableness": 60, "neuroticism": 40},
            "key_requirements": []
        }


def semantic_search_candidates(candidates_list: list, query: str) -> list:
    """Ranks candidates semantically against a natural language query."""
    if not candidates_list:
        return []

    candidate_texts = []
    for c in candidates_list:
        proj_text = " ".join([p.get("title", "") + " " + p.get("description", "") for p in (c.get("projects") or []) if isinstance(p, dict)])
        edu_text = " ".join([e.get("degree", "") + " " + e.get("institution", "") for e in (c.get("education") or []) if isinstance(e, dict)])
        exp_text = " ".join([ex.get("role", "") + " " + ex.get("company", "") for ex in (c.get("experience") or []) if isinstance(ex, dict)])
        profile_content = f"{c['name']} {c['job_role']} {c.get('soft_skills','')} {c.get('certificates','')} {c.get('interests','')} {proj_text} {edu_text} {exp_text}"
        candidate_texts.append(profile_content)

    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([query] + candidate_texts)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    except Exception as e:
        logger.error(f"TF-IDF search failed: {e}")
        scores = [1.0] * len(candidates_list)

    scored = list(zip(candidates_list, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [c for c, _ in scored[:15]]

    candidates_summary = []
    for c in top_candidates:
        ai_rpt = c.get("ai_report", {})
        summary = (
            f"Candidate: {c['name']}\n"
            f"Role: {c['job_role']} (Job Match: {c.get('job_score', 0)}%)\n"
            f"Experience: {c.get('exp_years', 0)} years | Seniority: {ai_rpt.get('seniority_level', 'N/A')}\n"
            f"Personality Score: {c.get('personality_score', 0)}%\n"
            f"Skills: {c.get('soft_skills', '')}\n"
            f"Certs: {c.get('certificates', '')}\n"
            f"Market Percentile: {ai_rpt.get('market_percentile', 'N/A')}\n"
            f"Trajectory: {ai_rpt.get('trajectory_pattern', 'N/A')}\n"
            f"Risk: {ai_rpt.get('overall_risk_level', 'N/A')}\n---"
        )
        candidates_summary.append(summary)

    candidates_text = "\n".join(candidates_summary)

    prompt = f"""You are an AI Talent Search agent with access to a candidate database.

TASK: Evaluate candidates against the search query. Score relevance 0-100 with explanation.
Consider: skills match, experience level, personality fit, market competitiveness, and career trajectory.
Only include candidates with relevance > 0.
Return ONLY valid JSON.

JSON Schema:
{{
    "results": [
        {{
            "name": "Candidate Name",
            "relevance_score": 95,
            "match_explanation": "Why this candidate matches the query"
        }}
    ]
}}

Search Query: "{query}"

Candidates:
{candidates_text}"""
    try:
        data = groq_call(prompt, response_format={"type": "json_object"})
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        return []
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        fallback_results = []
        keywords = [k.lower().strip() for k in query.split() if len(k.strip()) > 2]
        for c in candidates_list:
            score = 0
            matches = []
            c_text = f"{c['name']} {c['job_role']} {c.get('soft_skills', '')} {c.get('certificates', '')} {c.get('interests', '')}".lower()
            for kw in keywords:
                if kw in c_text:
                    score += 25
                    matches.append(kw)
            if score > 0:
                fallback_results.append({
                    "name": c["name"],
                    "relevance_score": min(score, 100),
                    "match_explanation": f"Keyword match: {', '.join(matches)} (local fallback)."
                })
        fallback_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return fallback_results


# ====================== Resume Optimizer ======================
_optimizer_cache = {}

def generate_optimized_resume(raw_text: str, job_role: str, ai_report: dict,
                              focus_area: str = "all", tone: str = "professional",
                              target_length: str = "default") -> str:
    text_hash = hashlib.md5(raw_text.encode('utf-8')).hexdigest()
    cache_key = (text_hash, job_role, focus_area, tone, target_length)
    if cache_key in _optimizer_cache:
        logger.info("Serving optimized resume from memory cache.")
        return _optimizer_cache[cache_key]

    gaps = ai_report.get("critical_gaps", [])
    bullet_improvements = ai_report.get("bullet_points_improvement", [])
    improvements_str = "\n".join([f"- {imp}" for imp in bullet_improvements])
    differentiators = ai_report.get("differentiators", [])
    improvement_areas = ai_report.get("improvement_areas", [])

    focus_instructions = {
        "achievements": "Primary focus: rewrite work experience with metric-driven business outcomes using X-Y-Z formula.",
        "keywords": "Primary focus: inject missing critical keywords and technical skills organically throughout.",
        "formatting": "Primary focus: improve structure, formatting, and ATS parsing readability.",
    }
    focus_inst = focus_instructions.get(focus_area, "Optimize achievements, integrate keywords, and improve formatting concurrently.")

    tone_instructions = {
        "professional": "Write in a polished, professional corporate tone.",
        "bold": "Write in a bold, high-impact, tech-driven executive tone emphasizing leadership and innovation.",
        "academic": "Write in a structured, detail-oriented academic tone suitable for research and credentials.",
        "minimalist": "Write in a clean, crisp, minimalist tone with facts, technologies, and clear metrics only.",
    }
    tone_inst = tone_instructions.get(tone, f"Write in a {tone} tone.")

    length_instructions = {"concise": "Aim for tight single-page layout.", "detailed": "Expand fully on all sections."}
    length_inst = length_instructions.get(target_length, "")

    differentiators_str = "\n".join([f"- {d}" for d in differentiators[:3]]) if differentiators else ""
    improvements_areas_str = "\n".join([f"- {a}" for a in improvement_areas[:3]]) if improvement_areas else ""

    prompt = f"""You are an elite professional resume writer and ATS optimization specialist.

TASK: Rewrite this CV into a polished, high-impact professional resume for '{job_role}'.

Requirements:
1. {focus_inst}
2. {tone_inst}
3. {length_inst}
4. Integrate missing skills naturally: {', '.join(gaps[:5])}.
5. Apply metric-driven improvements:
{improvements_str}
6. Use Google X-Y-Z formula for experience: "Accomplished [X], as measured by [Y], by doing [Z]."
7. Highlight these differentiators: {differentiators_str}
8. Address these improvement areas: {improvements_areas_str}
9. Keep factual to original content — do not invent dates or degrees.
10. Output clean Markdown: Contact Info, Professional Summary, Core Skills, Experience, Projects, Education.

Original CV:
{raw_text}

Output ONLY the professional Markdown. No notes or explanations."""
    try:
        optimized = groq_call(
            prompt,
            system_msg="You are a professional resume optimization writer. Output ONLY high-quality Markdown.",
            temperature=0.3,
            parse_json=False
        )
        _optimizer_cache[cache_key] = optimized
        return optimized
    except Exception as e:
        logger.error(f"Resume optimization failed: {e}")
        # ---- OFFLINE FALLBACK: Smart reformat without AI ----
        logger.info("Generating offline-optimized resume as AI fallback...")
        try:
            lines = raw_text.strip().split('\n')
            # Extract sections from raw text
            sections = {'summary': [], 'skills': [], 'experience': [], 'education': [], 'projects': [], 'other': []}
            current = 'other'
            for line in lines:
                ll = line.strip().lower()
                if not ll:
                    continue
                if any(kw in ll for kw in ['summary', 'objective', 'profile', 'about']):
                    current = 'summary'
                    continue
                elif any(kw in ll for kw in ['skill', 'technolog', 'competenc', 'tools']):
                    current = 'skills'
                    continue
                elif any(kw in ll for kw in ['experience', 'employment', 'work history', 'career']):
                    current = 'experience'
                    continue
                elif any(kw in ll for kw in ['education', 'academic', 'degree', 'university', 'college']):
                    current = 'education'
                    continue
                elif any(kw in ll for kw in ['project', 'portfolio']):
                    current = 'projects'
                    continue
                sections[current].append(line.strip())
            
            # Get candidate name from first non-empty line
            cand_name = lines[0].strip() if lines else 'Candidate'
            for ext in ['.pdf', '.docx']:
                if cand_name.lower().endswith(ext):
                    cand_name = cand_name[:-len(ext)]
            cand_name = cand_name.replace('_', ' ').replace('-', ' ').strip().title()
            
            # Build markdown
            md_parts = [f"# {cand_name}", f"**Target Role: {job_role}**", ""]
            
            if sections['summary']:
                md_parts.append("## Professional Summary")
                md_parts.append(' '.join(sections['summary'][:5]))
                md_parts.append("")
            
            if sections['skills'] or gaps:
                md_parts.append("## Core Skills")
                for s in sections['skills'][:10]:
                    md_parts.append(f"- {s}")
                if gaps:
                    md_parts.append(f"\n**Key Areas for Development:** {', '.join(gaps[:5])}")
                md_parts.append("")
            
            if sections['experience']:
                md_parts.append("## Work Experience")
                for exp_line in sections['experience'][:20]:
                    md_parts.append(f"- {exp_line}" if not exp_line.startswith('-') else exp_line)
                md_parts.append("")
            
            if sections['education']:
                md_parts.append("## Education")
                for edu_line in sections['education'][:10]:
                    md_parts.append(f"- {edu_line}" if not edu_line.startswith('-') else edu_line)
                md_parts.append("")
            
            if sections['projects']:
                md_parts.append("## Projects")
                for proj_line in sections['projects'][:10]:
                    md_parts.append(f"- {proj_line}" if not proj_line.startswith('-') else proj_line)
                md_parts.append("")
            
            if sections['other']:
                md_parts.append("## Additional Information")
                for other_line in sections['other'][:10]:
                    md_parts.append(f"- {other_line}" if not other_line.startswith('-') else other_line)
            
            fallback_result = '\n'.join(md_parts)
            _optimizer_cache[cache_key] = fallback_result
            return fallback_result
        except Exception as fallback_err:
            logger.error(f"Offline fallback also failed: {fallback_err}")
            return f"### Error generating optimized resume\n\nAI service temporarily unavailable. Please try again later.\n\nOriginal CV text is preserved and available in the 'Original Plain Text' tab."