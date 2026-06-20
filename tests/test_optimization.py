# tests/test_optimization.py
import sys
import os
import sqlite3
import json

# Ensure project root is on PYTHONPATH
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from app import app, init_db

def test_optimize_endpoint():
    print("Testing /optimize_cv/<name> endpoint...")
    init_db()
    
    # 1. Insert a mock candidate into the database with raw_text
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM candidates WHERE name = 'Optimizer Test Candidate'")
    
    ai_report_mock = {
        "critical_gaps": ["React", "Typescript", "Docker"],
        "bullet_points_improvement": ["Rewrite the bullet about scaling to mention 100k requests/sec using Redis caching."]
    }
    
    cur.execute('''INSERT INTO candidates 
                   (name, filename, job_role, score, soft_skills, interests, certificates, final_score, exp_years, personality_score,
                    personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism, ai_report, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                ("Optimizer Test Candidate", "test.pdf", "Software Engineer", 80.0,
                 "teamwork", "coding", "aws certified", 85.0, 3.0, 90.0,
                 80.0, 80.0, 50.0, 70.0, 30.0,
                 json.dumps(ai_report_mock), 
                 "John Doe\nSoftware Engineer\nExperience:\n- Built some backend web servers.\n- Optimized databases."))
    conn.commit()
    conn.close()
    
    # 2. Query endpoint using Flask test client
    client = app.test_client()
    resp = client.get('/optimize_cv/Optimizer-Test-Candidate') # Flask route is case sensitive or handles it
    # Wait, the route gets the exact name. In SQLite query we did SELECT where name = 'Optimizer Test Candidate'
    # So we should call /optimize_cv/Optimizer Test Candidate
    resp = client.get('/optimize_cv/Optimizer Test Candidate')
    
    assert resp.status_code == 200, f"Endpoint failed: {resp.status_code}"
    data = json.loads(resp.data.decode('utf-8'))
    assert "optimized_cv" in data, "No optimized_cv key returned"
    print("AI Optimized Resume Content Preview:")
    print("-" * 50)
    print(data["optimized_cv"][:300] + "...")
    print("-" * 50)
    print("Resume Optimization Endpoint Test: PASS")

if __name__ == "__main__":
    test_optimize_endpoint()
