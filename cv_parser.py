# cv_parser.py — Robust CV Parsing Engine v2.0
import os
import re
import logging
import time
import pdfplumber
import docx
import instructor
from groq import Groq
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

logger = logging.getLogger(__name__)

# ====================== Structural Sub-Schemas ======================

class ExperienceSchema(BaseModel):
    company: str = Field(description="Name of the organization, company, or workplace")
    role: str = Field(description="Job title, role, or designation held")
    duration: str = Field(description="Time period or duration of the job (e.g., Oct 2025 - Present)")
    achievements: List[str] = Field(description="Core responsibilities, tasks performed, or major metrics achieved")

class EducationSchema(BaseModel):
    institution: str = Field(description="Name of the school, college, or university")
    degree: str = Field(description="Exact degree name or certification (e.g., BSCS, ICS, Matriculation)")
    passing_year: Optional[str] = Field(None, description="Graduation year or completion date")
    gpa: Optional[str] = Field(None, description="CGPA, grade, or percentage obtained")

class ProjectSchema(BaseModel):
    title: str = Field(description="Title or name of the project")
    tech_stack: List[str] = Field(description="List of tools, programming languages, or technologies used")
    description: str = Field(description="A brief summary detailing features or goals of the project")

# ====================== Main Composite Schema ======================

class CVAnalysisSchema(BaseModel):
    full_name: str = Field(description="Full legal or professional name of the candidate")
    email: str = Field(description="Primary contact email address")
    phone: Optional[str] = Field(None, description="Contact phone or mobile number if present")
    links: List[str] = Field(description="Social profile links such as GitHub URL, LinkedIn profile, or portfolio sites")
    skills: List[str] = Field(description="Comprehensive list of tools, frameworks, technical skills, and soft skills mentioned")
    education: List[EducationSchema] = Field(description="Structured list of all educational milestones")
    experience: List[ExperienceSchema] = Field(description="Structured list of all professional work experiences")
    projects: List[ProjectSchema] = Field(description="Structured list of all personal or group projects showcased")
    experience_years: float = Field(default=0.0, description="Calculated total years of professional work experience represented as a float")
    certifications: List[str] = Field(default=[], description="List of certifications, licenses, or professional credentials")
    languages: List[str] = Field(default=[], description="Languages spoken by the candidate (e.g., English, Urdu, Spanish)")
    summary: Optional[str] = Field(None, description="Professional or career summary statement from the resume")
    interests: List[str] = Field(default=[], description="List of personal or professional interests and hobbies")

    @model_validator(mode='after')
    def validate_content_quality(self):
        """Post-validation: ensure critical fields aren't placeholders or empty."""
        # Fix placeholder names
        if self.full_name and self.full_name.lower() in ["candidate", "name", "n/a", "unknown", ""]:
            self.full_name = "Candidate"
        # Fix placeholder emails
        if self.email and self.email.lower() in ["not extracted", "n/a", "unknown@email.com", "email@example.com", ""]:
            self.email = "Not Extracted"
        # Deduplicate skills
        if self.skills:
            seen = set()
            unique = []
            for s in self.skills:
                s_lower = s.lower().strip()
                if s_lower and s_lower not in seen:
                    seen.add(s_lower)
                    unique.append(s)
            self.skills = unique
        return self


# ====================== Text Extraction Utilities ======================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts raw text from PDF using pdfplumber with layout engine."""
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=False)
                if text:
                    full_text += text + "\n"
        return full_text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """Extracts raw text from DOCX, including paragraphs, tables, and headers."""
    try:
        doc = docx.Document(docx_path)
        full_text = []
        # Extract from paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        # Extract from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))
        # Extract from headers/footers
        for section in doc.sections:
            if section.header:
                for para in section.header.paragraphs:
                    if para.text.strip():
                        full_text.append(para.text.strip())
        return "\n".join(full_text).strip()
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""

def extract_text(file_path: str) -> str:
    """Unified text extractor with post-processing cleanup."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        return ""
    
    # Post-process: collapse excessive whitespace but preserve structure
    if text:
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]{5,}', '  ', text)
    return text


# ====================== Regex Extraction Helpers ======================

def extract_phone_number(text: str) -> Optional[str]:
    """Robust international phone extraction supporting multiple country formats."""
    patterns = [
        r'(\+?\d{1,3}[\s\-.]?\(?\d{2,4}\)?[\s\-.]?\d{3,4}[\s\-.]?\d{3,4})',
        r'(0\d{3}[\s\-.]?\d{7})',
        r'(\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4})',
        r'(\+?\d[\d\s\-\.]{8,14}\d)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            phone = match.group(0).strip()
            digits_only = re.sub(r'\D', '', phone)
            if 7 <= len(digits_only) <= 15:
                return phone
    return None


def extract_certifications_regex(text: str) -> List[str]:
    """Extract certifications using keyword patterns."""
    cert_patterns = [
        r'(?:certified|certification|certificate|license|credential)[\s\-:]*([^\n,]{5,60})',
        r'(aws\s+certified[^\n,]{0,40})',
        r'(google\s+(?:analytics|cloud|professional)[^\n,]{0,40})',
        r'(microsoft\s+(?:certified|azure|office)[^\n,]{0,40})',
        r'(pmp|cfa|ccna|ccnp|cissp|comptia|scrum\s+master)',
    ]
    certs = []
    for pattern in cert_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cert = m.strip().rstrip('.')
            if cert and len(cert) > 3 and cert.lower() not in [c.lower() for c in certs]:
                certs.append(cert)
    return certs[:10]


def extract_languages_regex(text: str) -> List[str]:
    """Extract spoken languages from CV text."""
    known_languages = [
        "English", "Urdu", "Hindi", "Spanish", "French", "German", "Arabic",
        "Chinese", "Mandarin", "Japanese", "Korean", "Portuguese", "Russian",
        "Italian", "Turkish", "Persian", "Farsi", "Punjabi", "Bengali", "Tamil"
    ]
    found = []
    for lang in known_languages:
        if re.search(r'\b' + re.escape(lang) + r'\b', text, re.IGNORECASE):
            found.append(lang)
    return found


def extract_experience_regex(text: str) -> List[ExperienceSchema]:
    """Extract work experience entries using common resume patterns."""
    experiences = []
    # Pattern: "Role at/– Company | Duration" or "Company • Role • Duration"
    exp_patterns = [
        # "Role at Company (Duration)" or "Role | Company | Duration"
        r'(?:^|\n)\s*(.{5,50}?)\s*(?:at|@|[-–—]|,)\s*(.{3,60}?)\s*(?:\||,|[-–—]|\()\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{4}).{5,40}?(?:Present|Current|[0-9]{4}))\s*\)?',
    ]
    for pattern in exp_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for role, company, duration in matches[:5]:
            role = role.strip().rstrip('|,.-')
            company = company.strip().rstrip('|,.-')
            duration = duration.strip().rstrip(')')
            if len(role) > 2 and len(company) > 2:
                experiences.append(ExperienceSchema(
                    company=company,
                    role=role,
                    duration=duration,
                    achievements=["Extracted via fallback parser"]
                ))
    return experiences


# ====================== Rule-Based Fallback Parser ======================

def extract_rule_based_fallback(raw_text: str) -> CVAnalysisSchema:
    """Comprehensive rule-based regex fallback parser for when LLM is unavailable."""
    # 1. Email
    email_match = re.search(r'[\w\.\+\-]+@[\w\.-]+\.\w+', raw_text)
    email = email_match.group(0) if email_match else "Not Extracted"

    # 2. Phone
    phone = extract_phone_number(raw_text)

    # 3. Links
    links = re.findall(r'(https?://[^\s,\(\)\[\]]+)', raw_text)
    socials = re.findall(r'\b(?:github\.com|linkedin\.com/in|twitter\.com|gitlab\.com)/[a-zA-Z0-9_\-\.\/]+', raw_text, re.IGNORECASE)
    all_links = list(set(links + [f"https://{s}" for s in socials if not s.startswith("http")]))

    # 4. Name (first non-header line)
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    full_name = "Candidate"
    if lines:
        candidate_line = lines[0]
        if len(candidate_line) < 50 and not any(term in candidate_line.lower() for term in ["resume", "cv", "curriculum", "education", "experience", "objective"]):
            full_name = candidate_line

    # 5. Experience Years
    years = 0.0
    explicit_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:years|year|yrs|yr|y)\b", raw_text, re.IGNORECASE)
    for m in explicit_matches:
        try:
            years = max(years, float(m))
        except ValueError:
            continue

    # 6. Education
    education = []
    edu_matches = re.findall(
        r'(?:bachelor|master|bs|ms|phd|b\.s\.|m\.s\.|diploma|degree|university|college|school|intermediate|matriculation|bscs|bsse|mba|bba)\b[^\n]*',
        raw_text, re.IGNORECASE
    )
    for match in edu_matches[:4]:
        education.append(EducationSchema(institution="Extracted Institution", degree=match.strip()[:100], passing_year=None, gpa=None))

    # 7. Experience
    experience = extract_experience_regex(raw_text)

    # 8. Certifications
    certifications = extract_certifications_regex(raw_text)

    # 9. Languages
    languages = extract_languages_regex(raw_text)

    # 10. Skills (basic extraction)
    skill_keywords = [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "dart", "kotlin", "swift", "php", "ruby",
        "django", "flask", "react", "angular", "vue", "flutter", "spring", "laravel", "express", "next.js", "node.js",
        "mysql", "postgresql", "mongodb", "sqlite", "redis", "firebase", "dynamodb",
        "docker", "kubernetes", "aws", "gcp", "azure", "git", "jenkins", "linux", "terraform",
        "html", "css", "bootstrap", "tailwind", "figma", "photoshop", "illustrator",
        "machine learning", "deep learning", "tensorflow", "pytorch", "pandas", "numpy",
    ]
    skills = extract_keywords_case_insensitive(raw_text, skill_keywords)

    return CVAnalysisSchema(
        full_name=full_name,
        email=email,
        phone=phone,
        links=all_links,
        skills=skills,
        education=education,
        experience=experience,
        projects=[],
        experience_years=years,
        certifications=certifications,
        languages=languages,
        summary=None,
        interests=[]
    )


def extract_keywords_case_insensitive(text: str, keywords: list) -> list:
    """Extract keywords with case-insensitive matching and proper casing."""
    found = []
    text_lower = text.lower()
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
            # Return with proper casing
            found.append(kw.title() if kw.islower() else kw)
    return found


# ====================== Primary LLM Parser ======================

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"
MAX_PARSE_RETRIES = 2


def _validate_parsed_result(result: CVAnalysisSchema, raw_text: str) -> CVAnalysisSchema:
    """Post-LLM validation to catch and fix common parsing issues."""
    # If LLM missed the email, try regex
    if not result.email or result.email in ["Not Extracted", "n/a", ""]:
        email_match = re.search(r'[\w\.\+\-]+@[\w\.-]+\.\w+', raw_text)
        if email_match:
            result.email = email_match.group(0)

    # If LLM missed the phone, try regex
    if not result.phone:
        result.phone = extract_phone_number(raw_text)

    # If LLM missed links, try regex
    if not result.links:
        links = re.findall(r'(https?://[^\s,\(\)\[\]]+)', raw_text)
        if links:
            result.links = list(set(links))

    # If LLM missed certifications, try regex
    if not result.certifications:
        result.certifications = extract_certifications_regex(raw_text)

    # If LLM missed languages, try regex
    if not result.languages:
        result.languages = extract_languages_regex(raw_text)

    # Validate experience_years isn't unrealistic
    if result.experience_years > 50:
        result.experience_years = 0.0
    if result.experience_years < 0:
        result.experience_years = 0.0

    return result


def parse_cv(file_path: str) -> Optional[CVAnalysisSchema]:
    """
    Transforms unstructured resume text into validated Pydantic structure.
    Uses Instructor + Groq LLM with retry and validation, falling back to rule-based parser.
    """
    raw_text = extract_text(file_path)
    if not raw_text:
        logger.error("No text extracted from the CV file.")
        return None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY missing. Running rule-based fallback.")
        return extract_rule_based_fallback(raw_text)

    client = instructor.from_groq(Groq(api_key=api_key))

    system_msg = (
        "You are a professional ATS system backend component. "
        "Your task is to process raw text extracts from CVs and structure EVERY piece of information "
        "into the designated JSON model. Do not truncate, omit, or overlook any project, link, certification, "
        "language, or experience detail. Extract experience_years as the TOTAL calculated work experience. "
        "If you cannot find a field, use reasonable defaults rather than placeholder text."
    )
    user_msg = f"Analyze and parse the following resume plain text extract:\n\n{raw_text}"

    models = [PRIMARY_MODEL, FALLBACK_MODEL]
    for model in models:
        for attempt in range(MAX_PARSE_RETRIES):
            try:
                t0 = time.time()
                structured_data = client.chat.completions.create(
                    model=model,
                    response_model=CVAnalysisSchema,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.1,
                    max_retries=2  # Instructor's built-in retry
                )
                elapsed = round(time.time() - t0, 2)
                logger.info(f"CV parsed successfully with {model} in {elapsed}s (attempt {attempt+1})")

                # Post-validation: fill gaps the LLM might have missed
                validated = _validate_parsed_result(structured_data, raw_text)
                return validated

            except Exception as e:
                logger.warning(f"Parser failed (model={model}, attempt={attempt+1}): {e}")
                if attempt < MAX_PARSE_RETRIES - 1:
                    time.sleep(1.0)  # Brief pause before retry

    # All LLM models exhausted — use rule-based fallback
    logger.warning("All LLM models failed for parsing. Using rule-based fallback.")
    try:
        return extract_rule_based_fallback(raw_text)
    except Exception as e:
        logger.error(f"Fatal parser error in rule-based fallback: {e}")
        return None