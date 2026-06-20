from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import io
import csv
import sqlite3
import json
import logging
from datetime import datetime
from pdf_generator import generate_candidate_pdf, generate_optimized_cv_pdf

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

logger = logging.getLogger(__name__)

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

# ---------------------- Database Context Manager ----------------------
def get_db(timeout=30.0):
    """Creates a database connection with WAL journal mode for better concurrent access."""
    conn = sqlite3.connect("candidates.db", timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------- Predefined Job Profiles & Targets ----------------------
JOB_PROFILES = {
    "Software Engineer": "python java c++ software development engineering algorithms debugging data structures oop backend frontend development",
    "Data Analyst": "excel sql python data visualization tableau statistics powerbi analytics machine learning dashboards data cleaning pandas",
    "Flutter Developer": "flutter dart firebase mobile application ui ux api integration development android ios state management bloc provider",
    "Graphic Designer": "photoshop illustrator creativity branding adobe design logo typography UI UX Figma vector layout",
    "HR Manager": "communication recruitment management leadership employee relations hr policies talent acquisition onboarding screening",
    "DevOps Engineer": "docker kubernetes ci/cd jenkins aws gcp azure terraform ansible linux bash monitoring infrastructure automation deployment pipeline",
    "Product Manager": "product strategy roadmap agile scrum sprint stakeholder management user stories kpis analytics requirements prioritization",
    "UI/UX Designer": "figma sketch adobe xd wireframing prototyping user research usability testing design systems interaction design accessibility responsive design",
}

JOB_EXPERIENCE_TARGET = {
    "Software Engineer": 3,
    "Data Analyst": 2,
    "Flutter Developer": 2,
    "Graphic Designer": 2,
    "HR Manager": 3,
    "DevOps Engineer": 3,
    "Product Manager": 4,
    "UI/UX Designer": 2,
}

JOB_PERSONALITY_TARGET = {
    "Software Engineer": {"openness": 65, "conscientiousness": 80, "extraversion": 45, "agreeableness": 60, "neuroticism": 35},
    "Data Analyst": {"openness": 60, "conscientiousness": 85, "extraversion": 40, "agreeableness": 55, "neuroticism": 35},
    "Flutter Developer": {"openness": 75, "conscientiousness": 75, "extraversion": 55, "agreeableness": 60, "neuroticism": 40},
    "Graphic Designer": {"openness": 85, "conscientiousness": 65, "extraversion": 55, "agreeableness": 65, "neuroticism": 40},
    "HR Manager": {"openness": 60, "conscientiousness": 75, "extraversion": 70, "agreeableness": 75, "neuroticism": 35},
    "DevOps Engineer": {"openness": 60, "conscientiousness": 85, "extraversion": 40, "agreeableness": 55, "neuroticism": 30},
    "Product Manager": {"openness": 75, "conscientiousness": 80, "extraversion": 70, "agreeableness": 65, "neuroticism": 35},
    "UI/UX Designer": {"openness": 85, "conscientiousness": 70, "extraversion": 60, "agreeableness": 70, "neuroticism": 35},
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
    add_col("optimized_cv", "TEXT")
    add_col("optimized_cv_params", "TEXT")
    add_col("created_at", "DATETIME DEFAULT CURRENT_TIMESTAMP")
    # v2.0 Agent fields
    add_col("executive_summary", "TEXT")
    add_col("unified_verdict", "TEXT")
    add_col("analysis_confidence", "REAL DEFAULT 0")
    add_col("career_trajectory", "TEXT")
    add_col("red_flags", "TEXT")
    add_col("market_benchmark", "TEXT")
    add_col("tech_alignment_score", "REAL DEFAULT 0")
    conn.commit()
    conn.close()

# Auto-initialize database on application import (required for Docker & Gunicorn)
init_db()

# ---------------------- Custom Error Handlers ----------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template('base.html', error_code=404, error_msg="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Internal server error: {e}", exc_info=True)
    return render_template('base.html', error_code=500, error_msg="Internal server error"), 500

# ---------------------- App Routes ----------------------
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                               'logo.png', mimetype='image/png')

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

        # 3. Fetch Deep Semantic AI Audit Report (8 agents + supervisor)
        ai_report = get_ai_deep_analysis(raw_text, job_role)

        # Extract new agent scores for weighted final score
        tech_alignment = float(ai_report.get('tech_alignment_score', 50.0))
        trajectory_score = float(ai_report.get('career_momentum_score', 50.0))

        final_score = compute_final_score(
            job_role, job_score, soft_skills, certificates, interests, exp_years, 
            personality_score_val, exp_target_map, len(SOFT_SKILLS), len(CERTIFICATES), len(INTERESTS),
            tech_alignment_score=tech_alignment, trajectory_score=trajectory_score
        )

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
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.cursor()
            
            # Extract v2.0 agent fields for separate columns
            exec_summary = ai_report.get('executive_summary', '')
            unified_verd = ai_report.get('unified_verdict', '')
            analysis_conf = float(ai_report.get('analysis_confidence', 0))
            career_traj = json.dumps({
                'trajectory_pattern': ai_report.get('trajectory_pattern', ''),
                'career_momentum_score': ai_report.get('career_momentum_score', 0),
                'promotion_frequency': ai_report.get('promotion_frequency', ''),
                'career_gaps': ai_report.get('career_gaps', []),
                'title_progression': ai_report.get('title_progression', []),
                'growth_prediction': ai_report.get('growth_prediction', ''),
                'risk_assessment': ai_report.get('risk_assessment', ''),
            })
            red_flags_data = json.dumps({
                'red_flags_found': ai_report.get('red_flags_found', []),
                'overall_risk_level': ai_report.get('overall_risk_level', ''),
                'total_flags': ai_report.get('total_flags', 0),
                'flags_summary': ai_report.get('flags_summary', ''),
                'positive_signals': ai_report.get('positive_signals', []),
            })
            market_bench = json.dumps({
                'market_percentile': ai_report.get('market_percentile', 0),
                'market_competitiveness': ai_report.get('market_competitiveness', ''),
                'salary_range_usd': ai_report.get('salary_range_usd', {}),
                'skill_comparison': ai_report.get('skill_comparison', {}),
                'differentiators': ai_report.get('differentiators', []),
                'improvement_areas': ai_report.get('improvement_areas', []),
                'comparable_titles': ai_report.get('comparable_titles', []),
            })
            tech_align_score = float(ai_report.get('tech_alignment_score', 0))
            
            cur.execute('''INSERT INTO candidates 
                           (name, filename, job_role, score, soft_skills, interests, certificates, final_score, exp_years, personality_score,
                            personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism,
                            ai_report, raw_text, email, links, education, experience, projects, created_at,
                            executive_summary, unified_verdict, analysis_confidence, career_trajectory, red_flags, market_benchmark, tech_alignment_score)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (candidate_name, filename, job_role, job_score,
                         ", ".join(soft_skills), ", ".join(interests), ", ".join(certificates), final_score,
                         exp_years, personality_score_val,
                         est_traits.get('openness'), est_traits.get('conscientiousness'), est_traits.get('extraversion'), est_traits.get('agreeableness'), est_traits.get('neuroticism'),
                         json.dumps(ai_report), raw_text, email_val, links_json, edu_json, exp_json, proj_json, datetime.utcnow().isoformat(),
                         exec_summary, unified_verd, analysis_conf, career_traj, red_flags_data, market_bench, tech_align_score))
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
        logger.error(f"Error processing file {filename}: {str(e)}", exc_info=True)
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

    # Save uploaded files sequentially in main thread
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
                logger.error(f"Error resolving concurrent thread: {e}")

    results.sort(key=lambda x: x["final_score"], reverse=True)
    suitable_count = sum(1 for r in results if r["final_score"] >= THRESHOLD)

    return render_template('result.html', job_role=job_role, results=results, total=len(results), suitable=suitable_count, threshold=THRESHOLD)

@app.route('/test/<name>')
def personality_test(name: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT name, job_role FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return redirect(url_for('ranking'))
    return render_template('test.html', name=row['name'], job_role=row['job_role'], questions=QUESTIONS)

@app.route('/submit_test/<name>', methods=['POST'])
def submit_test(name: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, job_role, score, soft_skills, interests, certificates, exp_years FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return redirect(url_for('ranking'))
    cid, job_role, job_score, soft_skills, interests, certificates, exp_years = row['id'], row['job_role'], row['score'], row['soft_skills'], row['interests'], row['certificates'], row['exp_years']

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
        e_map, len(SOFT_SKILLS), len(CERTIFICATES), len(INTERESTS),
        tech_alignment_score=50.0, trajectory_score=50.0
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
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, raw_text, job_role, ai_report, optimized_cv, optimized_cv_params, filename, email, links, education, experience, projects, soft_skills, interests, certificates FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Candidate not found"}), 404
    
    cid = row['id']
    raw_text = row['raw_text']
    job_role = row['job_role']
    ai_report_str = row['ai_report']
    cached_cv = row['optimized_cv']
    cached_params_str = row['optimized_cv_params']
    filename = row['filename']
    email = row['email']
    links_str = row['links']
    education_str = row['education']
    experience_str = row['experience']
    projects_str = row['projects']
    soft_skills = row['soft_skills']
    interests = row['interests']
    certificates = row['certificates']
    
    if not raw_text or not raw_text.strip():
        # Stage 1: File-based raw text healing
        if filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                try:
                    extracted = extract_text(file_path)
                    if extracted and extracted.strip():
                        raw_text = extracted
                        conn = get_db()
                        conn.execute("UPDATE candidates SET raw_text = ? WHERE id = ?", (raw_text, cid))
                        conn.commit()
                        conn.close()
                        logger.info(f"Healed missing raw_text for candidate '{name}' from file '{filename}'.")
                except Exception as e:
                    logger.error(f"Error healing raw_text for '{name}' from file: {e}")
        
        # Stage 2: Fallback database-reconstructed raw text healing
        if not raw_text or not raw_text.strip():
            try:
                reconstructed_parts = []
                if name:
                    reconstructed_parts.append(f"Name: {name}")
                if job_role:
                    reconstructed_parts.append(f"Target Job Role: {job_role}")
                if email:
                    reconstructed_parts.append(f"Email: {email}")
                if links_str:
                    try:
                        lnk = json.loads(links_str)
                        if isinstance(lnk, list) and lnk:
                            reconstructed_parts.append("Links: " + ", ".join(lnk))
                    except Exception:
                        reconstructed_parts.append(f"Links: {links_str}")
                if soft_skills:
                    reconstructed_parts.append(f"Skills: {soft_skills}")
                if interests:
                    reconstructed_parts.append(f"Interests: {interests}")
                if certificates:
                    reconstructed_parts.append(f"Certificates: {certificates}")
                
                if education_str:
                    reconstructed_parts.append("\n## Education")
                    try:
                        edu_list = json.loads(education_str)
                        if isinstance(edu_list, list):
                            for edu in edu_list:
                                inst = edu.get("institution", "Unknown")
                                deg = edu.get("degree", "")
                                y = edu.get("passing_year", "")
                                g = edu.get("gpa", "")
                                reconstructed_parts.append(f"- {deg} at {inst} (Year: {y}, GPA: {g})")
                        else:
                            reconstructed_parts.append(education_str)
                    except Exception:
                        reconstructed_parts.append(education_str)
                
                if experience_str:
                    reconstructed_parts.append("\n## Experience")
                    try:
                        exp_list = json.loads(experience_str)
                        if isinstance(exp_list, list):
                            for exp in exp_list:
                                comp = exp.get("company", "Unknown")
                                r_title = exp.get("role", "")
                                d = exp.get("duration", "")
                                achs = exp.get("achievements", [])
                                if isinstance(achs, list):
                                    ach_bullets = "\n  * ".join(achs)
                                else:
                                    ach_bullets = str(achs)
                                reconstructed_parts.append(f"- {r_title} at {comp} ({d})\n  * {ach_bullets}")
                        else:
                            reconstructed_parts.append(experience_str)
                    except Exception:
                        reconstructed_parts.append(experience_str)
                
                if projects_str:
                    reconstructed_parts.append("\n## Projects")
                    try:
                        proj_list = json.loads(projects_str)
                        if isinstance(proj_list, list):
                            for proj in proj_list:
                                title = proj.get("title", "")
                                tech = proj.get("tech_stack", [])
                                tech_str = ", ".join(tech) if isinstance(tech, list) else str(tech)
                                desc = proj.get("description", "")
                                reconstructed_parts.append(f"- {title} (Tech: {tech_str})\n  * {desc}")
                        else:
                            reconstructed_parts.append(projects_str)
                    except Exception:
                        reconstructed_parts.append(projects_str)
                
                reconstructed_text = "\n".join(reconstructed_parts).strip()
                if reconstructed_text:
                    raw_text = reconstructed_text
                    conn = get_db()
                    conn.execute("UPDATE candidates SET raw_text = ? WHERE id = ?", (raw_text, cid))
                    conn.commit()
                    conn.close()
                    logger.info(f"Healed missing raw_text for candidate '{name}' by reconstructing from structured JSON fields.")
            except Exception as e:
                logger.error(f"Error reconstructing raw_text for candidate '{name}': {e}")
                    
    if not raw_text or not raw_text.strip():
        return jsonify({"error": "Original CV text is missing for this older record. Please re-upload the CV to optimize it!"}), 400
        
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
        
    requested_params = {
        "focus_area": focus_area,
        "tone": tone,
        "target_length": target_length
    }
    
    # Check if cached resume exists and matches parameter criteria
    use_cache = False
    if cached_cv and cached_cv.strip():
        if cached_params_str:
            try:
                cached_params = json.loads(cached_params_str)
                if (cached_params.get("focus_area") == focus_area and 
                    cached_params.get("tone") == tone and 
                    cached_params.get("target_length") == target_length):
                    use_cache = True
            except Exception:
                pass
        else:
            if focus_area == "all" and tone == "professional" and target_length == "default":
                use_cache = True
                
    if use_cache:
        logger.info(f"Serving optimized CV for {name} from persistent database cache.")
        return jsonify({
            "optimized_cv": cached_cv,
            "raw_text": raw_text,
            "ai_report": ai_report
        })
        
    # Generate new optimized resume
    optimized_text = generate_optimized_resume(
        raw_text, job_role, ai_report,
        focus_area=focus_area, tone=tone, target_length=target_length
    )
    
    # Save back to database
    if optimized_text and not optimized_text.startswith("### Error"):
        conn = get_db()
        conn.execute(
            "UPDATE candidates SET optimized_cv = ?, optimized_cv_params = ? WHERE id = ?",
            (optimized_text, json.dumps(requested_params), cid)
        )
        conn.commit()
        conn.close()
        
    return jsonify({
        "optimized_cv": optimized_text,
        "raw_text": raw_text,
        "ai_report": ai_report
    })

def deserialize_candidate(r):
    """Deserialize a candidate row from tuple or Row to dict."""
    # Support both sqlite3.Row and tuple
    if isinstance(r, sqlite3.Row):
        r = tuple(r)
    
    email = r[14] if (len(r) > 14 and r[14]) else "Not Extracted"
    
    try:
        links = json.loads(r[15]) if (len(r) > 15 and r[15]) else []
    except Exception:
        links = []
        
    try:
        education = json.loads(r[16]) if (len(r) > 16 and r[16]) else []
    except Exception:
        education = []
        
    try:
        experience = json.loads(r[17]) if (len(r) > 17 and r[17]) else []
    except Exception:
        experience = r[17] if (len(r) > 17 and r[17]) else []
        
    try:
        projects = json.loads(r[18]) if (len(r) > 18 and r[18]) else []
    except Exception:
        projects = []
        
    try:
        ai_report = json.loads(r[19]) if (len(r) > 19 and r[19]) else {}
    except Exception:
        ai_report = {}
        
    filename_val = r[20] if (len(r) > 20 and r[20]) else ""
        
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
        "ai_report": ai_report,
        "filename": filename_val
    }

CANDIDATE_SELECT_COLS = """
    SELECT name, job_role, score, exp_years, personality_score, soft_skills, interests, certificates, final_score,
           personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism,
           email, links, education, experience, projects, ai_report, filename
    FROM candidates
"""

@app.route('/ranking')
def ranking():
    query = request.args.get('query', '').strip()
    
    if query:
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        cur.execute(CANDIDATE_SELECT_COLS)
        candidates = [deserialize_candidate(r) for r in cur.fetchall()]
        conn.close()

        search_results = semantic_search_candidates(candidates, query)
        
        candidates_by_name = {c["name"]: c for c in candidates}
        matched_candidates = []
        for res in search_results:
            cand_name = res.get("name")
            if cand_name in candidates_by_name:
                cand_data = candidates_by_name[cand_name].copy()
                cand_data["relevance_score"] = res.get("relevance_score", 0)
                cand_data["match_explanation"] = res.get("match_explanation", "")
                matched_candidates.append(cand_data)
                
        matched_candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return render_template('ranking.html', rows=matched_candidates, query=query)
    else:
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        cur.execute(CANDIDATE_SELECT_COLS + " ORDER BY final_score DESC")
        rows = [deserialize_candidate(r) for r in cur.fetchall()]
        conn.close()
        return render_template('ranking.html', rows=rows)

@app.route('/export_pdf/<name>')
def export_pdf(name: str):
    try:
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        cur.execute(CANDIDATE_SELECT_COLS + " WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        row = cur.fetchone()
        conn.close()
        
        if not row:
            return "Candidate not found", 404
            
        candidate_data = deserialize_candidate(row)
        pdf_buffer = generate_candidate_pdf(candidate_data)
        
        filename = f"{name.replace(' ', '_')}_AI_Profile_Audit.pdf"
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error exporting PDF for {name}: {e}", exc_info=True)
        return f"Error generating PDF report: {str(e)}", 500

@app.route('/download_cv/<name>')
def download_cv(name: str):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT filename FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        row = cur.fetchone()
        conn.close()
        
        if not row or not row['filename']:
            return "CV file not found", 404
            
        filename = row['filename']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return f"CV file '{filename}' does not exist on server", 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error downloading original CV for {name}: {e}", exc_info=True)
        return f"Error downloading CV: {str(e)}", 500

@app.route('/export_optimized_pdf/<name>')
def export_optimized_pdf(name: str):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT optimized_cv, job_role FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        row = cur.fetchone()
        conn.close()
        
        if not row or not row['optimized_cv']:
            return "Optimized CV not generated yet. Please optimize first!", 400
            
        optimized_text = row['optimized_cv']
        job_role = row['job_role']
        
        pdf_buffer = generate_optimized_cv_pdf(name, job_role, optimized_text)
        
        filename = f"{name.replace(' ', '_')}_Optimized_Resume.pdf"
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Error exporting optimized PDF for {name}: {e}", exc_info=True)
        return f"Error generating optimized PDF: {str(e)}", 500

@app.route('/delete_candidate/<name>', methods=['POST'])
def delete_candidate(name: str):
    try:
        conn = get_db()
        conn.execute("DELETE FROM candidates WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ---------------------- New API Endpoints ----------------------

@app.route('/api/stats')
def api_stats():
    """Dashboard analytics endpoint returning aggregate stats."""
    try:
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM candidates")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT AVG(final_score) FROM candidates")
        avg_score = round(cur.fetchone()[0] or 0, 1)
        
        cur.execute("SELECT AVG(score) FROM candidates")
        avg_job_match = round(cur.fetchone()[0] or 0, 1)
        
        cur.execute("SELECT AVG(exp_years) FROM candidates")
        avg_exp = round(cur.fetchone()[0] or 0, 1)
        
        cur.execute("SELECT job_role, COUNT(*) as cnt FROM candidates GROUP BY job_role ORDER BY cnt DESC")
        role_distribution = {row[0]: row[1] for row in cur.fetchall()}
        
        cur.execute("SELECT name, final_score, job_role FROM candidates ORDER BY final_score DESC LIMIT 5")
        top_performers = [{"name": r[0], "score": r[1], "role": r[2]} for r in cur.fetchall()]
        
        cur.execute("SELECT COUNT(*) FROM candidates WHERE final_score >= ?", (THRESHOLD,))
        qualified = cur.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "total_candidates": total,
            "average_final_score": avg_score,
            "average_job_match": avg_job_match,
            "average_experience_years": avg_exp,
            "qualified_candidates": qualified,
            "threshold": THRESHOLD,
            "role_distribution": role_distribution,
            "top_performers": top_performers
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/export_csv')
def api_export_csv():
    """Bulk CSV export of all candidates with key metrics."""
    try:
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        cur.execute("""
            SELECT name, job_role, score, exp_years, personality_score, soft_skills, 
                   interests, certificates, final_score, email, created_at
            FROM candidates ORDER BY final_score DESC
        """)
        rows = cur.fetchall()
        conn.close()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Candidate Name", "Job Role", "Job Match Score", "Experience (Years)", 
                         "Personality Score", "Soft Skills", "Interests", "Certificates", 
                         "Final Score", "Email", "Analyzed At"])
        for row in rows:
            writer.writerow(row)
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'cognicv_candidates_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/bulk_delete', methods=['POST'])
def api_bulk_delete():
    """Accept list of candidate names for batch deletion."""
    try:
        data = request.get_json(silent=True) or {}
        names = data.get('names', [])
        if not names or not isinstance(names, list):
            return jsonify({"error": "No candidate names provided"}), 400
        
        conn = get_db()
        placeholders = ','.join(['?' for _ in names])
        conn.execute(f"DELETE FROM candidates WHERE name IN ({placeholders})", names)
        conn.commit()
        deleted = conn.total_changes
        conn.close()
        
        return jsonify({"success": True, "deleted_count": deleted})
    except Exception as e:
        logger.error(f"Bulk delete error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/candidate/<name>')
def api_candidate_detail(name: str):
    """Returns full JSON profile for a single candidate including all v2.0 agent data."""
    try:
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        cur.execute(CANDIDATE_SELECT_COLS + " WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        row = cur.fetchone()
        
        if not row:
            conn.close()
            return jsonify({"error": "Candidate not found"}), 404
        
        candidate = deserialize_candidate(row)
        
        # Also fetch v2.0 fields directly
        cur.execute("""
            SELECT executive_summary, unified_verdict, analysis_confidence, 
                   career_trajectory, red_flags, market_benchmark, tech_alignment_score
            FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1
        """, (name,))
        v2_row = cur.fetchone()
        conn.close()
        
        if v2_row:
            candidate["executive_summary"] = v2_row[0] or ""
            candidate["unified_verdict"] = v2_row[1] or ""
            candidate["analysis_confidence"] = v2_row[2] or 0
            try:
                candidate["career_trajectory"] = json.loads(v2_row[3]) if v2_row[3] else {}
            except Exception:
                candidate["career_trajectory"] = {}
            try:
                candidate["red_flags"] = json.loads(v2_row[4]) if v2_row[4] else {}
            except Exception:
                candidate["red_flags"] = {}
            try:
                candidate["market_benchmark"] = json.loads(v2_row[5]) if v2_row[5] else {}
            except Exception:
                candidate["market_benchmark"] = {}
            candidate["tech_alignment_score"] = v2_row[6] or 0
        
        return jsonify(candidate)
    except Exception as e:
        logger.error(f"Candidate detail API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/reanalyze/<name>', methods=['POST'])
def api_reanalyze(name: str):
    """Re-runs all AI agents on an existing candidate without re-uploading."""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id, raw_text, job_role, filename FROM candidates WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        row = cur.fetchone()
        
        if not row:
            conn.close()
            return jsonify({"error": "Candidate not found"}), 404
        
        cid = row['id']
        raw_text = row['raw_text']
        job_role = row['job_role']
        filename = row['filename']
        conn.close()
        
        # If no raw_text, try to re-extract from file
        if not raw_text or not raw_text.strip():
            if filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(file_path):
                    raw_text = extract_text(file_path)
            if not raw_text or not raw_text.strip():
                return jsonify({"error": "No CV text available for re-analysis"}), 400
        
        # Re-run the full 8-agent + supervisor pipeline
        ai_report = get_ai_deep_analysis(raw_text, job_role)
        
        # Extract v2.0 fields
        exec_summary = ai_report.get('executive_summary', '')
        unified_verd = ai_report.get('unified_verdict', '')
        analysis_conf = float(ai_report.get('analysis_confidence', 0))
        career_traj = json.dumps({
            'trajectory_pattern': ai_report.get('trajectory_pattern', ''),
            'career_momentum_score': ai_report.get('career_momentum_score', 0),
            'promotion_frequency': ai_report.get('promotion_frequency', ''),
            'career_gaps': ai_report.get('career_gaps', []),
            'title_progression': ai_report.get('title_progression', []),
            'growth_prediction': ai_report.get('growth_prediction', ''),
            'risk_assessment': ai_report.get('risk_assessment', ''),
        })
        red_flags_data = json.dumps({
            'red_flags_found': ai_report.get('red_flags_found', []),
            'overall_risk_level': ai_report.get('overall_risk_level', ''),
            'total_flags': ai_report.get('total_flags', 0),
            'flags_summary': ai_report.get('flags_summary', ''),
            'positive_signals': ai_report.get('positive_signals', []),
        })
        market_bench = json.dumps({
            'market_percentile': ai_report.get('market_percentile', 0),
            'market_competitiveness': ai_report.get('market_competitiveness', ''),
            'salary_range_usd': ai_report.get('salary_range_usd', {}),
            'skill_comparison': ai_report.get('skill_comparison', {}),
            'differentiators': ai_report.get('differentiators', []),
            'improvement_areas': ai_report.get('improvement_areas', []),
            'comparable_titles': ai_report.get('comparable_titles', []),
        })
        tech_align = float(ai_report.get('tech_alignment_score', 0))
        
        # Update the database
        conn = get_db()
        conn.execute('''UPDATE candidates SET 
                        ai_report = ?, executive_summary = ?, unified_verdict = ?,
                        analysis_confidence = ?, career_trajectory = ?, red_flags = ?,
                        market_benchmark = ?, tech_alignment_score = ?
                       WHERE id = ?''',
                    (json.dumps(ai_report), exec_summary, unified_verd, analysis_conf,
                     career_traj, red_flags_data, market_bench, tech_align, cid))
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "candidate": name,
            "agents_executed": ai_report.get('agents_executed', 8),
            "analysis_time_seconds": ai_report.get('analysis_time_seconds', 0),
            "analysis_confidence": analysis_conf,
            "unified_verdict": unified_verd,
            "executive_summary": exec_summary
        })
    except Exception as e:
        logger.error(f"Reanalyze API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)