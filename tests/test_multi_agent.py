# tests/test_multi_agent.py
import sys
import os

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

from cv_analyzer import (
    run_audit_agent,
    run_tech_matcher_agent,
    run_psych_agent,
    run_interviewer_agent,
    run_recruitment_test_agent,
    get_ai_deep_analysis
)

def test_sub_agents_fallback():
    print("Testing sub-agents fallback safety...")
    cv_text = "Fahad Mustafa. Software Engineer with 2 years of Flutter experience."
    role = "Software Engineer"
    
    audit = run_audit_agent(cv_text, role)
    assert "formatting_score" in audit
    assert "formatting_feedback" in audit
    assert "bullet_points_improvement" in audit
    
    tech = run_tech_matcher_agent(cv_text, role)
    assert "tech_alignment_score" in tech
    assert "skills_found" in tech
    assert "categorized_skills" in tech
    assert "critical_gaps" in tech
    
    psych = run_psych_agent(cv_text, role)
    assert "personality_traits" in psych
    assert "personality_fit_explanation" in psych
    
    interviewer = run_interviewer_agent(cv_text, role)
    assert "seniority_level" in interviewer
    assert "experience_quality_feedback" in interviewer
    assert "interview_questions" in interviewer
    assert "hiring_recommendation" in interviewer
    assert "recommendation_rationale" in interviewer
    
    recruitment_test = run_recruitment_test_agent(cv_text, role)
    assert "custom_coding_test" in recruitment_test
    assert "interview_evaluation_rubric" in recruitment_test
    assert "title" in recruitment_test["custom_coding_test"]
    assert "problem_statement" in recruitment_test["custom_coding_test"]
    assert "requirements" in recruitment_test["custom_coding_test"]
    
    print("Sub-agents fallback: PASS")

def test_get_ai_deep_analysis_merging():
    print("Testing get_ai_deep_analysis concurrent execution & merging...")
    cv_text = "Fahad Mustafa. Software Engineer with 2 years of Flutter experience."
    role = "Software Engineer"
    
    report = get_ai_deep_analysis(cv_text, role)
    
    # Assert all keys are merged
    required_keys = [
        "formatting_score", "formatting_feedback", "bullet_points_improvement",
        "tech_alignment_score", "skills_found", "categorized_skills", "critical_gaps",
        "personality_traits", "personality_fit_explanation",
        "seniority_level", "experience_quality_feedback", "interview_questions",
        "hiring_recommendation", "recommendation_rationale",
        "custom_coding_test", "interview_evaluation_rubric"
    ]
    
    for key in required_keys:
        assert key in report, f"Expected key '{key}' in merged report, but it was missing."
        
    print("get_ai_deep_analysis concurrent execution & merging: PASS")

if __name__ == "__main__":
    test_sub_agents_fallback()
    test_get_ai_deep_analysis_merging()
    print("All Multi-Agent Orchestration Unit Tests: PASS")

