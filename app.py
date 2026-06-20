from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sqlite3
import json
import logging

# Configure logging to save errors into a file and output to console
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

# File handler
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Console handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

# Import modern components directly from your custom modules
from cv_parser import parse_cv, extract_text
from cv_analyzer import (
    compute_similarity, extract_keywords, extract_experience_years,
    estimate_personality_from_cv, score_personality_test,
    personality_fit_score, compute_final_score, get_ai_deep_analysis,
    extract_targets_from_custom_jd, semantic_search_candidates,
    generate_optimized_resume
)
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "docx"}

# ---------------------- Predefined Job Profiles & Targets ----------------------
JOB_PROFILES = {
    "Software Engineer": "python java c++ software development engineering algorithms debugging data structures oop backend frontend development",
    "Data Analyst": "excel sql python data visualization tableau statistics powerbi analytics machine learning dashboards data cleaning pandas",
    "Flutter Developer": "flutter dart firebase mobile application ui ux api integration development android ios state management bloc provider",
    "Graphic Designer": "photoshop illustrator creativity branding adobe design logo typography UI UX Figma vector layout",
    "HR Manager": "communication recruitment management leadership employee relations hr policies talent acquisition onboarding screening"
}

JOB_EXPERIENCE_TARGET = {
    "Software Engineer": 3,
    "Data Analyst": 2,
    "Flutter Developer": 2,
    "Graphic Designer": 2,
    "HR Manager": 3,
}

JOB_PERSONALITY_TARGET = {
    "Software Engineer": {"openness": 65, "conscientiousness": 80, "extraversion": 45, "agreeableness": 60, "neuroticism": 35},
    "Data Analyst": {"openness": 60, "conscientiousness": 85, "extraversion": 40, "agreeableness": 55, "neuroticism": 35},
    "Flutter Developer": {"openness": 75, "conscientiousness": 75, "extraversion": 55, "agreeableness": 60, "neuroticism": 40},
    "Graphic Designer": {"openness": 85, "conscientiousness": 65, "extraversion": 55, "agreeableness": 65, "neuroticism": 40},
    "HR Manager": {"openness": 60, "conscientiousness": 75, "extraversion": 70, "agreeableness": 75, "neuroticism": 35},
}

SOFT_SKILLS = ["communication", "teamwork", "leadership", "problem-solving", "adaptability", "creativity", "collaboration", "critical thinking", "time management"]
INTERESTS = ["sports", "reading", "traveling", "music", "gaming", "technology", "coding", "blogging", "fitness"]
CERTIFICATES = ["aws certified", "pmp", "cfa", "scrum master", "google analytics", "microsoft office", "ccna", "coursera", "udemy"]

THRESHOLD = 50  # Final score selection filter

QUESTIONS = [
    {"id": "q1", "text": "I am quick to understand abstract ideas.", "trait": "openness", "reverse": False},
    {"id": "q2", "text": "I have a rich vocabulary.", "trait": "openness", "reverse": False},
    {"id": "q3", "text": "I pay attention to details.", "trait": "conscientiousness", "reverse": False},
    {"id": "q4", "text": "I often leave tasks unfinished.", "trait": "conscientiousness", "reverse": True},
    {"id": "q5", "text": "I am the life of the party.", "trait": "extraversion", "reverse": False},
    {"id": "q6", "text": "I prefer to be alone.", "trait": "extraversion", "reverse": True},
    {"id": "q7", "text": "I am considerate and kind to almost everyone.", "trait": "agreeableness", "reverse": False},
    {"id": "q8", "text": "I insult people.", "trait": "agreeableness", "reverse": True},
    {"id": "q9", "text": "I get stressed out easily.", "trait": "neuroticism", "reverse": False},
    {"id": "q10", "text": "I am relaxed most of the time.", "trait": "neuroticism", "reverse": True},
]

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------- DB Initializer ----------------------
def init_db():
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    filename TEXT,
                    job_role TEXT,
                    score REAL,
                    soft_skills TEXT,
                    interests TEXT,
                    certificates TEXT,
                    final_score REAL
                )''')
    cur.execute("PRAGMA table_info(candidates)")
    existing_cols = {row[1] for row in cur.fetchall()}
    
    def add_col(name, decl):
        if name not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE candidates ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError:
                pass
            
    add_col("exp_years", "REAL DEFAULT 0")
    add_col("personality_openness", "REAL")
    add_col("personality_conscientiousness", "REAL")
    add_col("personality_extraversion", "REAL")
    add_col("personality_agreeableness", "REAL")
    add_col("personality_neuroticism", "REAL")
    add_col("personality_score", "REAL DEFAULT 0")
    add_col("ai_report", "TEXT")
    add_col("raw_text", "TEXT")
    add_col("email", "TEXT")
    add_col("links", "TEXT")
    add_col("education", "TEXT")
    add_col("experience", "TEXT")
    add_col("projects", "TEXT")
    conn.commit()
    conn.close()

# Auto-initialize database on application import (required for Docker & Gunicorn)
init_db()

# ---------------------- App Routes ----------------------
@app.route('/')
def index():
    return render_template('index.html', job_profiles=JOB_PROFILES.keys())

def process_single_cv(file_path, filename, job_role, job_desc, exp_target_map, personality_target_map):
    try:
        # 1. Cognitive Parsing via cv_parser (Strict Pydantic Mapping)
        parsed_profile = parse_cv(file_path)
        raw_text = extract_text(file_path)

        # Candidate Name Fallback logic if AI parser skips
        candidate_name = parsed_profile.full_name if parsed_profile else filename.split('.')[0]

        # 2. Mathematical Hybrid Calculations via cv_analyzer
        job_score = compute_similarity(job_desc, raw_text)
        soft_skills = extract_keywords(raw_text, SOFT_SKILLS)
        interests = extract_keywords(raw_text, INTERESTS)
        certificates = extract_keywords(raw_text, CERTIFICATES)
        # Determine experience years using LLM parse value as primary, regex as fallback
        exp_years = parsed_profile.experience_years if (parsed_profile and parsed_profile.experience_years > 0.0) else extract_experience_years(raw_text)

        est_traits = estimate_personality_from_cv(raw_text)
        personality_score_val = personality_fit_score(job_role, est_traits, personality_target_map)

        final_score = compute_final_score(
            job_role, job_score, soft_skills, certificates, interests, exp_years, 
            personality_score_val, exp_target_map, len(SOFT_SKILLS), len(CERTIFICATES), len(INTERESTS)
        )

        # 3. Fetch Deep Semantic AI Audit Report
        ai_report = get_ai_deep_analysis(raw_text, job_role)

        # 4. Serialize parsed profile blocks safely (using Pydantic model_dump)
        email_val = parsed_profile.email if parsed_profile else "Not Extracted"
        links_json = json.dumps(parsed_profile.links) if parsed_profile else "[]"
        if parsed_profile:
            edu_json = json.dumps([edu.model_dump() for edu in parsed_profile.education])
            exp_json = json.dumps([exp.model_dump() for exp in parsed_profile.experience])
            proj_json = json.dumps([proj.model_dump() for proj in parsed_profile.projects])
        else:
            edu_json = "[]"
            exp_json = "[]"
            proj_json = "[]"

        # 5. Save Record to Database cleanly (independent thread connections)
        conn = sqlite3.connect("candidates.db", timeout=30.0)
        try:
            cur = conn.cursor()
            cur.execute('''INSERT INTO candidates 
                           (name, filename, job_role, score, soft_skills, interests, certificates, final_score, exp_years, personality_score,
                            personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism, ai_report, raw_text, email, links, education, experience, projects)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (candidate_name, filename, job_role, job_score,
                         ", ".join(soft_skills), ", ".join(interests), ", ".join(certificates), final_score,
                         exp_years, personality_score_val,
                         est_traits.get('openness'), est_traits.get('conscientiousness'), est_traits.get('extraversion'), est_traits.get('agreeableness'), est_traits.get('neuroticism'),
                         json.dumps(ai_report), raw_text, email_val, links_json, edu_json, exp_json, proj_json))
            conn.commit()
        finally:
            conn.close()

        return {
            "name": candidate_name,
            "final_score": final_score,
            "job_score": job_score,
            "soft_skills": soft_skills,
            "interests": interests,
            "certificates": certificates,
            "exp_years": exp_years,
            "ai_report": ai_report,
            "email": parsed_profile.email if parsed_profile else "Not Extracted",
            "links": parsed_profile.links if parsed_profile else [],
            "skills": parsed_profile.skills if parsed_profile else soft_skills,
            "education": parsed_profile.education if parsed_profile else [],
            "experience": parsed_profile.experience if parsed_profile else [],
            "projects": parsed_profile.projects if parsed_profile else [],
            "personality_traits": est_traits,
            "personality_score": personality_score_val
        }
    except Exception as e:
        logging.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
        return None

@app.route('/analyze', methods=['POST'])
def analyze():
    jd_type = request.form.get('jd_type', 'predefined')
    
    if jd_type == 'custom':
        custom_jd = request.form.get('custom_jd', '')
        if not custom_jd.strip():
            return redirect(url_for('index'))
        
        # Dynamically extract target benchmarks using Groq
        custom_targets = extract_targets_from_custom_jd(custom_jd)
        job_role = custom_targets.get('job_role', 'Custom Job Role')
        job_desc = custom_jd
        
        # Ensure fallback target maps include this custom role
        personality_target_map = {job_role: custom_targets.get('personality_target')}
        experience_target_map = {job_role: custom_targets.get('experience_target', 2.0)}
    else:
        job_role = request.form.get('job_role')
        if not job_role or job_role not in JOB_PROFILES:
            return redirect(url_for('index'))
        job_desc = JOB_PROFILES[job_role]
        personality_target_map = JOB_PERSONALITY_TARGET
        experience_target_map = JOB_EXPERIENCE_TARGET

    files = request.files.getlist('files')
    if not files or not files[0].filename:
        return redirect(url_for('index'))

    # Save uploaded files sequentially in main thread (avoids Flask concurrent file stream read bugs)
    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        if not filename or not allowed_file(filename):
            continue
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        saved_files.append((file_path, filename))

    if not saved_files:
        return redirect(url_for('index'))

    # Process CV files in parallel using ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                process_single_cv,
                file_path,
                filename,
                job_role,
                job_desc,
                experience_target_map,
                personality_target_map
            )
            for file_path, filename in saved_files
        ]
        for future in futures:
            try:
                item = future.result()
                if item:
                    results.append(item)
            except Exception as e:
                print(f"Error resolving concurrent thread: {e}")

    results.sort(key=lambda x: x["final_score"], reverse=True)
    suitable_count = sum(1 for r in results if r["final_score"] >= THRESHOLD)

    return render_template('result.html', job_role=job_role, results=results, total=len(results), suitable=suitable_count, threshold=THRESHOLD)

@app.route('/test/<name>')
def personality_test(name: str):
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("SELECT name, job_role FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return redirect(url_for('ranking'))
    return render_template('test.html', name=row[0], job_role=row[1], questions=QUESTIONS)

@app.route('/submit_test/<name>', methods=['POST'])
def submit_test(name: str):
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("SELECT id, job_role, score, soft_skills, interests, certificates, exp_years FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return redirect(url_for('ranking'))
    cid, job_role, job_score, soft_skills, interests, certificates, exp_years = row

    # Pass the shared QUESTIONS configuration into the analyzer module
    traits = score_personality_test(request.form, QUESTIONS)
    
    # Custom JD fallback maps
    p_map = JOB_PERSONALITY_TARGET
    e_map = JOB_EXPERIENCE_TARGET
    if job_role not in p_map:
        p_map = {job_role: {"openness": 60, "conscientiousness": 70, "extraversion": 50, "agreeableness": 60, "neuroticism": 40}}
    if job_role not in e_map:
        e_map = {job_role: 2.0}

    pscore = personality_fit_score(job_role, traits, p_map)

    soft_list = [s.strip() for s in (soft_skills or "").split(",") if s.strip()]
    interest_list = [s.strip() for s in (interests or "").split(",") if s.strip()]
    cert_list = [s.strip() for s in (certificates or "").split(",") if s.strip()]
    
    final = compute_final_score(
        job_role, float(job_score or 0), soft_list, cert_list, interest_list, float(exp_years or 0), float(pscore),
        e_map, len(SOFT_SKILLS), len(CERTIFICATES), len(INTERESTS)
    )

    cur.execute('''UPDATE candidates SET 
                    personality_openness = ?,
                    personality_conscientiousness = ?,
                    personality_extraversion = ?,
                    personality_agreeableness = ?,
                    personality_neuroticism = ?,
                    personality_score = ?,
                    final_score = ?
                  WHERE id = ?''', (
                    traits.get('openness'), traits.get('conscientiousness'), traits.get('extraversion'), traits.get('agreeableness'), traits.get('neuroticism'), pscore, final, cid
                  ))
    conn.commit()
    conn.close()
    return redirect(url_for('ranking'))

@app.route('/optimize_cv/<name>', methods=['GET', 'POST'])
def optimize_cv(name: str):
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("SELECT raw_text, job_role, ai_report FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"error": "Candidate not found"}, 404
    
    raw_text, job_role, ai_report_str = row
    if not raw_text or not raw_text.strip():
        return {"error": "Original CV text is missing for this older record. Please re-upload the CV to optimize it!"}, 400
        
    try:
        ai_report = json.loads(ai_report_str) if ai_report_str else {}
    except Exception:
        ai_report = {}
        
    focus_area = "all"
    tone = "professional"
    target_length = "default"
    
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        focus_area = data.get('focus_area', 'all')
        tone = data.get('tone', 'professional')
        target_length = data.get('target_length', 'default')
    else:
        focus_area = request.args.get('focus_area', 'all')
        tone = request.args.get('tone', 'professional')
        target_length = request.args.get('target_length', 'default')
        
    optimized_text = generate_optimized_resume(
        raw_text, job_role, ai_report,
        focus_area=focus_area, tone=tone, target_length=target_length
    )
    return {
        "optimized_cv": optimized_text,
        "raw_text": raw_text,
        "ai_report": ai_report
    }

def deserialize_candidate(r):
    try:
        email = r[14] if (len(r) > 14 and r[14]) else "Not Extracted"
        links = json.loads(r[15]) if (len(r) > 15 and r[15]) else []
        education = json.loads(r[16]) if (len(r) > 16 and r[16]) else []
        experience = json.loads(r[17]) if (len(r) > 17 and r[17]) else []
        projects = json.loads(r[18]) if (len(r) > 18 and r[18]) else []
        ai_report = json.loads(r[19]) if (len(r) > 19 and r[19]) else {}
    except Exception:
        email = "Not Extracted"
        links = []
        education = []
        experience = []
        projects = []
        ai_report = {}
        
    return {
        "name": r[0],
        "job_role": r[1],
        "job_score": r[2],
        "exp_years": r[3],
        "personality_score": r[4],
        "soft_skills": r[5],
        "interests": r[6],
        "certificates": r[7],
        "final_score": r[8],
        "personality_traits": {
            "openness": r[9] if len(r) > 9 else 50.0,
            "conscientiousness": r[10] if len(r) > 10 else 50.0,
            "extraversion": r[11] if len(r) > 11 else 50.0,
            "agreeableness": r[12] if len(r) > 12 else 50.0,
            "neuroticism": r[13] if len(r) > 13 else 50.0
        },
        "email": email,
        "links": links,
        "education": education,
        "experience": experience,
        "projects": projects,
        "ai_report": ai_report
    }

@app.route('/ranking')
def ranking():
    query = request.args.get('query', '').strip()
    
    if query:
        # 1. Fetch all candidates from the database (including Big Five traits)
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        cur.execute("""
            SELECT name, job_role, score, exp_years, personality_score, soft_skills, interests, certificates, final_score,
                   personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism,
                   email, links, education, experience, projects, ai_report
            FROM candidates
        """)
        candidates = []
        for r in cur.fetchall():
            candidates.append(deserialize_candidate(r))
        conn.close()

        # 2. Run Groq RAG Semantic Candidate Search
        search_results = semantic_search_candidates(candidates, query)
        
        # 3. Align matched candidates with original detailed records
        candidates_by_name = {c["name"]: c for c in candidates}
        matched_candidates = []
        for res in search_results:
            name = res.get("name")
            if name in candidates_by_name:
                cand_data = candidates_by_name[name].copy()
                cand_data["relevance_score"] = res.get("relevance_score", 0)
                cand_data["match_explanation"] = res.get("match_explanation", "")
                matched_candidates.append(cand_data)
                
        # Sort matched candidates by relevance score descending
        matched_candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return render_template('ranking.html', rows=matched_candidates, query=query)
    else:
        # Default all candidates ranked by final score
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        cur.execute("""
            SELECT name, job_role, score, exp_years, personality_score, soft_skills, interests, certificates, final_score,
                   personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism,
                   email, links, education, experience, projects, ai_report
            FROM candidates ORDER BY final_score DESC
        """)
        rows = []
        for r in cur.fetchall():
            rows.append(deserialize_candidate(r))
        conn.close()
        return render_template('ranking.html', rows=rows)

if __name__ == '__main__':
    app.run(debug=True)