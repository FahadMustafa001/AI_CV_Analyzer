import sys
import os
import json

# Ensure project root is on PYTHONPATH
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from cv_analyzer import extract_targets_from_custom_jd, semantic_search_candidates

def test_custom_jd():
    print("Testing custom JD target extraction...")
    jd = "We are seeking a Senior Python Backend Developer with 5+ years of experience. Must be highly detail-oriented and experienced in FastAPI, PostgreSQL, and AWS."
    targets = extract_targets_from_custom_jd(jd)
    print("Extracted Targets:")
    print(json.dumps(targets, indent=2))
    assert "job_role" in targets
    assert "experience_target" in targets
    assert "personality_target" in targets
    print("Custom JD Target Extraction: PASS")

def test_semantic_search():
    print("Testing semantic talent search...")
    # Mock some candidates
    candidates = [
        {
            "name": "Jane Smith",
            "job_role": "Data Analyst",
            "job_score": 90.0,
            "exp_years": 4.0,
            "personality_score": 85.0,
            "soft_skills": "communication, teamwork",
            "certificates": "tableau, google analytics",
            "interests": "technology, gaming",
            "personality_traits": {
                "openness": 70,
                "conscientiousness": 90,
                "extraversion": 50,
                "agreeableness": 60,
                "neuroticism": 30
            }
        },
        {
            "name": "John Doe",
            "job_role": "Software Engineer",
            "job_score": 95.0,
            "exp_years": 3.0,
            "personality_score": 80.0,
            "soft_skills": "problem-solving, critical thinking",
            "certificates": "aws certified",
            "interests": "coding, reading",
            "personality_traits": {
                "openness": 80,
                "conscientiousness": 85,
                "extraversion": 40,
                "agreeableness": 55,
                "neuroticism": 35
            }
        }
    ]
    query = "Find me a detail-oriented analyst who knows Tableau."
    matches = semantic_search_candidates(candidates, query)
    print("Semantic Matches:")
    print(json.dumps(matches, indent=2))
    assert len(matches) > 0
    print("Semantic Search Test: PASS")

if __name__ == "__main__":
    test_custom_jd()
    print("-" * 40)
    test_semantic_search()
