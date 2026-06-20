# test_all_pdfs.py
import sqlite3
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from app import deserialize_candidate
from pdf_generator import generate_candidate_pdf

def test_all():
    conn = sqlite3.connect("candidates.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT name, job_role, score, exp_years, personality_score, soft_skills, interests, certificates, final_score,
               personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism,
               email, links, education, experience, projects, ai_report
        FROM candidates
    """)
    rows = cur.fetchall()
    conn.close()
    
    print(f"Total candidates in database: {len(rows)}")
    failed = 0
    passed = 0
    
    for row in rows:
        candidate_data = deserialize_candidate(row)
        try:
            generate_candidate_pdf(candidate_data)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"CRASH on candidate '{candidate_data['name']}': {e}")
            import traceback
            traceback.print_exc()
            print("-" * 50)
            
    print(f"Test completed: Passed: {passed}, Failed: {failed}")

if __name__ == '__main__':
    test_all()
