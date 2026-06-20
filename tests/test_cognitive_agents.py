# tests/test_cognitive_agents.py
import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from cv_parser import extract_rule_based_fallback, parse_cv
from cv_analyzer import compute_similarity, estimate_personality_from_cv

def test_cognitive_parser_fallback():
    print("Testing rule-based fallback parser...")
    raw_text = """
    MUHAMMAD FAHAD MUSTAFA
    mustafafahhad29@gmail.com
    +92 3164468643
    https://github.com/MuhammadFahad829
    
    Education:
    BS Computer Science from COMSATS Sahiwal, CGPA: 3.84
    
    Experience:
    Flutter Intern at TechCorp (1.5 years)
    """
    
    parsed = extract_rule_based_fallback(raw_text)
    assert parsed.full_name == "MUHAMMAD FAHAD MUSTAFA", f"Expected name MUHAMMAD FAHAD MUSTAFA, got: {parsed.full_name}"
    assert parsed.email == "mustafafahhad29@gmail.com", f"Expected email mustafafahhad29@gmail.com, got: {parsed.email}"
    assert parsed.phone == "+92 3164468643" or parsed.phone == "3164468643", f"Got phone: {parsed.phone}"
    assert any("github.com/MuhammadFahad829" in link for link in parsed.links), "Missing github link"
    assert parsed.experience_years == 1.5, f"Expected 1.5 years, got: {parsed.experience_years}"
    assert len(parsed.education) > 0, "Expected education list to contain items"
    print("Rule-based fallback parser: PASS")

def test_semantic_similarity():
    print("Testing semantic similarity calculations...")
    jd = "Software Engineer with Python, Flask, SQL, and database optimization experience."
    cv_good = "Senior Developer. Main stack: Python, Flask, PostgreSQL databases, query optimization, backend APIs."
    cv_bad = "Graphic Designer specialized in Photoshop, Adobe Illustrator, and Figma mockups."
    
    # Check that good CV gets a higher score than bad CV
    score_good = compute_similarity(jd, cv_good)
    score_bad = compute_similarity(jd, cv_bad)
    print(f"Good CV Score: {score_good}%, Bad CV Score: {score_bad}%")
    assert score_good > score_bad, f"Expected good CV to score higher, but got good={score_good} and bad={score_bad}"
    print("Semantic similarity matching: PASS")

def test_personality_profiling():
    print("Testing personality profiling heuristics fallback...")
    cv_creative_leader = "Lead developer. Managed a team, creative architect, UI/UX expert, designed frontend structures."
    traits = estimate_personality_from_cv(cv_creative_leader)
    
    print(f"Projected Traits: {traits}")
    assert "openness" in traits
    assert "conscientiousness" in traits
    assert "extraversion" in traits
    assert "agreeableness" in traits
    assert "neuroticism" in traits
    print("Personality profiling assessment: PASS")

if __name__ == "__main__":
    test_cognitive_parser_fallback()
    test_semantic_similarity()
    test_personality_profiling()
    print("All Cognitive Agent Unit Tests: PASS")
