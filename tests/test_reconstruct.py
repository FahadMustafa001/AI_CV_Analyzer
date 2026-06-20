# tests/test_reconstruct.py
import sys
import os
import sqlite3
import json

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from app import app, init_db

def test_reconstruction_self_healing():
    print("Testing DB reconstruction self-healing logic...")
    init_db()
    
    # 1. Insert mock candidate with empty raw_text and non-existent file
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM candidates WHERE name = 'Reconstruct Test Candidate'")
    
    edu_mock = [
        {"institution": "COMSATS", "degree": "BSCS", "passing_year": "2028", "gpa": "3.8"},
        {"institution": "Punjab College", "degree": "ICS", "passing_year": "2024", "gpa": "A+"}
    ]
    exp_mock = [
        {
            "company": "TechCorp",
            "role": "Flutter Intern",
            "duration": "3 months",
            "achievements": ["Built a modular chat screen", "Integrated Firebase auth"]
        }
    ]
    proj_mock = [
        {
            "title": "CogniCV",
            "tech_stack": ["Flask", "HTML", "JS"],
            "description": "AI-powered resume parsing and analysis matrix layout."
        }
    ]
    ai_report_mock = {
        "critical_gaps": ["Docker"],
        "bullet_points_improvement": ["Add numbers to metrics."]
    }
    
    cur.execute('''INSERT INTO candidates 
                   (name, filename, job_role, score, soft_skills, interests, certificates, final_score, exp_years, personality_score,
                    personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism, 
                    email, links, education, experience, projects, ai_report, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                ("Reconstruct Test Candidate", "nonexistent_file.pdf", "Flutter Developer", 75.0,
                 "communication, teamwork", "gaming, tech", "flutter certified", 78.0, 1.5, 80.0,
                 75.0, 75.0, 60.0, 70.0, 40.0,
                 "reconstruct@test.com", json.dumps(["github.com/reconstruct"]), 
                 json.dumps(edu_mock), json.dumps(exp_mock), json.dumps(proj_mock), json.dumps(ai_report_mock),
                 "")) # Empty raw_text
    conn.commit()
    conn.close()
    
    # 2. Call the optimize_cv endpoint
    client = app.test_client()
    resp = client.post('/optimize_cv/Reconstruct Test Candidate', json={})
    
    assert resp.status_code == 200, f"Healing route failed: {resp.status_code} - {resp.text}"
    data = json.loads(resp.data.decode('utf-8'))
    assert "optimized_cv" in data, "No optimized_cv in response"
    assert "raw_text" in data, "No raw_text returned"
    
    # Check that raw_text is indeed reconstructed
    recovered_text = data["raw_text"]
    print("Recovered raw text via database reconstruction:")
    print("-" * 50)
    print(recovered_text)
    print("-" * 50)
    
    assert "Name: Reconstruct Test Candidate" in recovered_text, "Name missing from reconstructed text"
    assert "reconstruct@test.com" in recovered_text, "Email missing"
    assert "TechCorp" in recovered_text, "Company missing"
    assert "CogniCV" in recovered_text, "Project title missing"
    
    # 3. Check DB to ensure raw_text was saved
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("SELECT raw_text FROM candidates WHERE name = 'Reconstruct Test Candidate' ORDER BY id DESC LIMIT 1")
    db_text = cur.fetchone()[0]
    conn.close()
    
    assert db_text and db_text.strip(), "Raw text was not persisted to the database"
    print("Database persistent raw_text: PASSED")
    print("Reconstruction Self-Healing Test: PASS")

if __name__ == "__main__":
    test_reconstruction_self_healing()
