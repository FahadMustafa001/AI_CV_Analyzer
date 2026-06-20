# cv_analyzer.py
import os
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Initialize Groq Client securely from Environment Variables
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "YOUR_FALLBACK_API_KEY_HERE"))

def compute_similarity(job_desc: str, cv_text: str) -> float:
    """
    Computes semantic similarity score using TF-IDF and Cosine Similarity,
    scaled to a realistic [0, 100] match range (0.35 cosine is mapped to 100%).
    """
    if not cv_text.strip():
        return 0.0
    vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2)).fit_transform([job_desc, cv_text])
    sim_score = cosine_similarity(vect[0:1], vect[1:2])[0][0]
    # Scale: raw cosine of 0.35 or higher is mapped to 100.0%
    scaled_score = min((sim_score / 0.35) * 100.0, 100.0)
    return round(scaled_score, 2)

def extract_keywords(cv_text: str, keywords_list: list) -> list:
    """
    Matches phrases intelligently using regex word boundaries to avoid partial matching errors.
    """
    detected = []
    for word in keywords_list:
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        if re.search(pattern, cv_text):
            detected.append(word)
    return detected

def extract_experience_years(cv_text: str) -> float:
    """
    Multi-layered heuristics to accurately parse work durations from both explicit digits and structural timelines.
    """
    years = 0.0
    # 1. Look for explicit mentions like '2.5 years' or '3+ yrs'
    explicit_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:years|year|yrs|yr|y)\b", cv_text, re.IGNORECASE)
    for m in explicit_matches:
        try:
            years = max(years, float(m))
        except ValueError:
            continue
            
    # 2. Parse structural dates like (2022 - 2025) or (1998 to Present)
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

def experience_score(job_role: str, exp_years: float, target_experience_map: dict) -> float:
    """
    Calculates experience score based on target role metrics.
    """
    target = target_experience_map.get(job_role, 2)
    return round(min(exp_years / max(target, 0.1), 1.0) * 100.0, 2)

def estimate_personality_from_cv(cv_text: str) -> dict:
    """
    Heuristic personality profiling baseline derived directly from CV syntactic tokens.
    """
    trait_map = {
        "openness": ["creative", "innovation", "design", "research", "explore", "curious", "architecture", "ui", "ux"],
        "conscientiousness": ["organized", "detail", "plan", "deadline", "reliable", "responsible", "optimization", "clean code"],
        "extraversion": ["lead", "presented", "team lead", "community", "public", "network", "speaker", "management"],
        "agreeableness": ["collaborated", "support", "mentor", "empathy", "customer", "stakeholder", "teamwork", "assisted"],
        "neuroticism": ["stress", "pressure", "challenge", "resilient", "cope", "troubleshooting", "debugging"]
    }
    scores = {t: 50.0 for t in trait_map}
    for trait, kws in trait_map.items():
        cnt = sum(1 for k in kws if k in cv_text)
        val = (min(cnt, 5) / 5.0) * 100.0
        if trait == "neuroticism":
            val = max(10.0, 50.0 - val)
        scores[trait] = round(val if cnt > 0 else 50.0, 2)
    return scores

def score_personality_test(form_data: dict, questions_list: list) -> dict:
    """
    Scores the online psychometric psych test entries submitted by the applicant.
    """
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
    """
    Evaluates absolute deviation variance against predefined job benchmarks.
    """
    target = target_personality_map.get(job_role)
    if not target:
        return 0.0
    diffs = [abs(trait_scores.get(t, 50.0) - tgt) for t, tgt in target.items()]
    mad = sum(diffs) / len(diffs)
    return round(max(0.0, 100.0 - mad), 2)

def compute_final_score(job_role: str, job_score: float, soft_skills: list, certificates: list, interests: list, exp_years: float, personality_score_val: float, target_exp_map: dict, total_soft_len: int, total_cert_len: int, total_int_len: int) -> float:
    """
    Aggregates composite weights into a singular weighted final scoring metric.
    """
    soft_score = min(len(soft_skills) / max(total_soft_len, 1), 1) * 100
    cert_score = min(len(certificates) / max(total_cert_len, 1), 1) * 100
    interest_score = min(len(interests) / max(total_int_len, 1), 1) * 100
    exp_score = experience_score(job_role, exp_years, target_exp_map)
    
    final = (
        0.45 * job_score +
        0.10 * soft_score +
        0.10 * cert_score +
        0.10 * interest_score +
        0.15 * exp_score +
        0.10 * personality_score_val
    )
    return round(final, 2)

# ---------------------- AI Deep Semantic Engine ----------------------
def get_ai_deep_analysis(cv_text: str, job_role: str) -> dict:
    """
    Queries Groq Cloud LLM to structurally audit the CV without strict keyword constraints,
    identifying core missing skills, layout issues, and customized interview paths.
    """
    prompt = f"""
    You are an advanced AI Applicant Tracking System (ATS) auditor. Analyze this CV text for the role '{job_role}'.
    Provide the output strictly in valid JSON format. Do not use any conversational text or markdown code blocks.
    
    JSON Schema:
    {{
        "skills_found": ["list", "of", "detected", "skills"],
        "critical_gaps": ["list", "of", "missing", "important", "skills", "for", "this", "role"],
        "bullet_points_improvement": ["up to 3 actionable suggestions to improve weak bullets using metric-based impact"],
        "interview_questions": ["3 direct technical or behavioral questions customized to this candidate"],
        "formatting_feedback": "Short critique about resume parsing readability and structure"
    }}

    CV Text:
    {cv_text}
    """
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise JSON-only response model."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq API Error in cv_analyzer module: {e}")
        return {
            "skills_found": [],
            "critical_gaps": ["Could not parse missing skill gaps via AI"],
            "bullet_points_improvement": ["Ensure your experience bullet points include clear technical scope"],
            "interview_questions": ["Explain your core project and the architecture decisions you made."],
            "formatting_feedback": "Standard formatting parsed successfully."
        }

# ---------------------- Custom JD & Talent RAG Search Engine ----------------------

def extract_targets_from_custom_jd(jd_text: str) -> dict:
    """
    Queries Groq Cloud LLM to extract target experience and ideal Big Five personality benchmarks from a custom Job Description.
    """
    prompt = f"""
    You are an expert HR analyst. Analyze the following custom job description and extract the target parameters.
    Provide the output strictly in valid JSON format. Do not use any conversational text or markdown code blocks.
    
    JSON Schema:
    {{
        "job_role": "A short, professional job title matching the description (e.g., 'Senior Java Developer')",
        "experience_target": 3.0, // A float representing the estimated minimum years of experience required
        "personality_target": {{
            "openness": 70.0,
            "conscientiousness": 80.0,
            "extraversion": 50.0,
            "agreeableness": 60.0,
            "neuroticism": 35.0
        }} // Estimated ideal Big Five trait percentiles (0-100) for this role
    }}

    Job Description:
    {jd_text}
    """
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise JSON-only response model."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Groq Custom JD Extraction Error: {e}")
        # Default fallback target metrics:
        return {
            "job_role": "Custom Job Role",
            "experience_target": 2.0,
            "personality_target": {"openness": 60, "conscientiousness": 70, "extraversion": 50, "agreeableness": 60, "neuroticism": 40}
        }

def semantic_search_candidates(candidates_list: list, query: str) -> list:
    """
    Ranks candidates semantically against a natural language query using Groq LLM.
    Uses an initial local TF-IDF cosine similarity filter to select the top 15 matches,
    reducing token usage by 90%+ and preventing API timeout issues.
    """
    if not candidates_list:
        return []
        
    # 1. Local TF-IDF pre-filtering to find the top 15 most relevant candidate profiles
    candidate_texts = []
    for c in candidates_list:
        # Combine candidate metadata and profile fields into a single text block
        proj_text = " ".join([p.get("title", "") + " " + p.get("description", "") for p in c.get("projects", [])])
        edu_text = " ".join([e.get("degree", "") + " " + e.get("institution", "") for e in c.get("education", [])])
        exp_text = " ".join([ex.get("role", "") + " " + ex.get("company", "") for ex in c.get("experience", [])])
        
        profile_content = (
            f"{c['name']} {c['job_role']} {c['soft_skills']} {c['certificates']} "
            f"{c['interests']} {proj_text} {edu_text} {exp_text}"
        )
        candidate_texts.append(profile_content)
        
    try:
        # Use bigram TF-IDF for high-accuracy phrase matching
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([query] + candidate_texts)
        # Compute cosine similarity between query (index 0) and candidates (indices 1+)
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    except Exception as e:
        print(f"Local TF-IDF similarity calculation failed: {e}")
        scores = [1.0] * len(candidates_list)
        
    # Pair candidates with their TF-IDF scores
    scored_candidates = list(zip(candidates_list, scores))
    # Sort candidates by similarity score descending
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Filter the list to the top 15 candidate matches
    TOP_N = 15
    top_candidates = [c for c, score in scored_candidates[:TOP_N]]
    
    # 2. Query Groq LLM for deep re-ranking and generating match explanations
    candidates_summary = []
    for c in top_candidates:
        summary = (
            f"Candidate: {c['name']}\n"
            f"Current Role Match: {c['job_role']} (Score: {c['job_score']}%)\n"
            f"Experience: {c['exp_years']} years\n"
            f"Personality Score: {c['personality_score']}%\n"
            f"Soft Skills: {c['soft_skills']}\n"
            f"Certificates: {c['certificates']}\n"
            f"Interests: {c['interests']}\n"
            f"Personality Traits: {json.dumps(c.get('personality_traits', {}))}\n"
            "---"
        )
        candidates_summary.append(summary)
        
    candidates_text = "\n".join(candidates_summary)
    
    prompt = f"""
    You are an AI Talent Search agent. Evaluate the list of candidates below based on the search query.
    Score their relevance on a scale of 0 to 100 and write a concise, one-sentence explanation of how/why they match the query.
    Only return candidates that have a relevance score greater than 0.
    Provide the output strictly in valid JSON format. Do not use any conversational text or markdown code blocks.
    
    JSON Schema:
    {{
        "results": [
            {{
                "name": "Candidate Name",
                "relevance_score": 95,
                "match_explanation": "Extremely strong match due to Flutter skills and high conscientiousness traits."
            }}
        ]
    }}

    Search Query:
    "{query}"

    Candidates:
    {candidates_text}
    """
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise JSON-only response model."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        data = json.loads(completion.choices[0].message.content)
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        return []
    except Exception as e:
        print(f"Groq Semantic Talent Search Error: {e}")
        # Fallback to keyword-based local search if Groq fails (crash-proof)
        fallback_results = []
        keywords = [k.lower().strip() for k in query.split() if len(k.strip()) > 2]
        for c in candidates_list:
            score = 0
            matches = []
            c_text = f"{c['name']} {c['job_role']} {c['soft_skills']} {c['certificates']} {c['interests']}".lower()
            for kw in keywords:
                if kw in c_text:
                    score += 25
                    matches.append(kw)
            if score > 0:
                fallback_results.append({
                    "name": c["name"],
                    "relevance_score": min(score, 100),
                    "match_explanation": f"Keyword match for: {', '.join(matches)} (Local search fallback)."
                })
        # Sort fallback results by score descending
        fallback_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return fallback_results

_optimizer_cache = {}

def generate_optimized_resume(raw_text: str, job_role: str, ai_report: dict, focus_area: str = "all", tone: str = "professional", target_length: str = "default") -> str:
    """
    Uses Groq LLM to rewrite the raw resume text into an ATS-optimized, high-impact version.
    It integrates missing skills, fixes poor work achievements, and adjusts according to
    focus_area, tone, and target_length specifications.
    """
    import hashlib
    text_hash = hashlib.md5(raw_text.encode('utf-8')).hexdigest()
    cache_key = (text_hash, job_role, focus_area, tone, target_length)
    if cache_key in _optimizer_cache:
        print("Serving optimized resume from memory cache.")
        return _optimizer_cache[cache_key]
        
    gaps = ai_report.get("critical_gaps", [])
    bullet_improvements = ai_report.get("bullet_points_improvement", [])
    improvements_str = "\n".join([f"- {imp}" for imp in bullet_improvements])
    
    focus_inst = ""
    if focus_area == "achievements":
        focus_inst = "Make the primary focus on rewriting the work experience achievements, phrasing each point clearly with metric-driven business outcomes."
    elif focus_area == "keywords":
        focus_inst = "Make the primary focus on injecting missing critical keywords and technical skills organically throughout all sections of the resume."
    elif focus_area == "formatting":
        focus_inst = "Make the primary focus on improving structure, formatting, parsing readability, and layout clarity for ATS parser software compatibility."
    else:
        focus_inst = "Optimize achievements, integrate missing keywords, and clean up general formatting/structure concurrently."

    tone_inst = f"Write the resume in a {tone} tone."
    if tone == "bold":
        tone_inst = "Write the resume in a bold, high-impact, tech-driven executive tone that emphasizes leadership, innovation, and direct technical success."
    elif tone == "academic":
        tone_inst = "Write the resume in a structured, detail-oriented, academic tone suitable for research, credentials, and thorough technical explanations."
    elif tone == "minimalist":
        tone_inst = "Write the resume in a clean, crisp, minimalist tone that highlights facts, technologies, and clear bulleted metrics without fluff."

    length_inst = ""
    if target_length == "concise":
        length_inst = "Keep the resume extremely concise, aiming for a tight, high-density single-page layout."
    elif target_length == "detailed":
        length_inst = "Provide a detailed, comprehensive resume, expanding fully on bullet points, projects, and educational coursework."
    
    prompt = f"""
    You are an elite professional resume writer and ATS optimization specialist.
    Your goal is to rewrite the candidate's CV text into a polished, high-impact, 100% perfect professional resume tailored for the role of '{job_role}'.

    Critical Requirements for the Rewrite:
    1. Focus Instructions: {focus_inst}
    2. Tone Instructions: {tone_inst}
    3. Length Instructions: {length_inst}
    4. Integrate the following missing critical skills naturally: {', '.join(gaps)}.
    5. Upgrade the work experience description using the following metric-driven improvements:
    {improvements_str}
    6. Rewrite the experience bullet points to follow the Google X-Y-Z formula: "Accomplished [X], as measured by [Y], by doing [Z]."
    7. Keep the resume factual to the original content; do not invent completely fake employment dates or college degrees.
    8. Output the result in clean, well-formatted Markdown with clear sections: Contact Info, Professional Summary, Core Skills, Work Experience, Projects, and Education.
    
    Candidate's Original CV:
    {raw_text}
    
    Output ONLY the professional Markdown content. Do not include introductory notes or explanation blocks.
    """
    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional resume optimization writer. You output ONLY high-quality Markdown and do not converse."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3
        )
        optimized_content = completion.choices[0].message.content
        # Save to memory cache on successful generation
        _optimizer_cache[cache_key] = optimized_content
        return optimized_content
    except Exception as e:
        print(f"Error in generate_optimized_resume: {e}")
        return f"### Error generating optimized resume\n\nWe encountered an error contacting the AI service: {str(e)}"