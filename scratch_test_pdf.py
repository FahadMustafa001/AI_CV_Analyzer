# scratch_test_pdf.py
import sqlite3
import json
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

from app import deserialize_candidate
from pdf_generator import generate_candidate_pdf

def test_pdf():
    try:
        conn = sqlite3.connect("candidates.db")
        cur = conn.cursor()
        cur.execute("""
            SELECT name, job_role, score, exp_years, personality_score, soft_skills, interests, certificates, final_score,
                   personality_openness, personality_conscientiousness, personality_extraversion, personality_agreeableness, personality_neuroticism,
                   email, links, education, experience, projects, ai_report
            FROM candidates WHERE name = 'Jane Smith' ORDER BY id DESC LIMIT 1
        """)
        row = cur.fetchone()
        conn.close()
        
        if not row:
            print("Candidate Jane Smith not found.")
            return
            
        candidate_data = deserialize_candidate(row)
        print(f"Testing PDF generation for: {candidate_data['name']}")
        
        pdf_buffer = generate_candidate_pdf(candidate_data)
        pdf_size = len(pdf_buffer.getvalue())
        print(f"PDF generated successfully! Size: {pdf_size} bytes")
        
        # Save to a temporary file
        out_path = "temp_candidate_report.pdf"
        with open(out_path, "wb") as f:
            f.write(pdf_buffer.getvalue())
        print(f"Saved test PDF to {out_path}")
        
    except Exception as e:
        print(f"PDF Generation failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_pdf()
