# cv_parser.py
import os
import pdfplumber
import docx
import instructor
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional

# ---------------------- Structural Sub-Schemas ----------------------

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

# ---------------------- Main Composite Schema ----------------------

class CVAnalysisSchema(BaseModel):
    full_name: str = Field(description="Full legal or professional name of the candidate")
    email: str = Field(description="Primary contact email address")
    phone: Optional[str] = Field(None, description="Contact phone or mobile number if present")
    links: List[str] = Field(description="Social profile links such as GitHub URL, LinkedIn profile, or portfolio sites")
    skills: List[str] = Field(description="Comprehensive list of tools, frameworks, technical skills, and soft skills mentioned")
    education: List[EducationSchema] = Field(description="Structured list of all educational milestones")
    experience: List[ExperienceSchema] = Field(description="Structured list of all professional work experiences")
    projects: List[ProjectSchema] = Field(description="Structured list of all personal or group projects showcased")
    experience_years: float = Field(default=0.0, description="Calculated total years of professional work experience represented as a float (e.g., 2.5 or 5.0).")
    certifications: List[str] = Field(default=[], description="List of certifications, licenses, or professional credentials (e.g., AWS Certified, PMP)")
    languages: List[str] = Field(default=[], description="Languages spoken by the candidate (e.g., English, Urdu, Spanish)")
    summary: Optional[str] = Field(None, description="Professional or career summary statement from the resume")
    interests: List[str] = Field(default=[], description="List of personal or professional interests and hobbies")

# ---------------------- Extraction Utility Functions ----------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Safely extracts raw plain text string from a PDF file using pdfplumber layout engine.
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=False) # layout=False ensures a logical text streaming structure
                if text:
                    full_text += text + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"Extraction Error inside cv_parser module: {e}")
        return ""

def extract_text_from_docx(docx_path: str) -> str:
    """
    Safely extracts raw plain text from a DOCX file, including paragraphs and tables.
    """
    try:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))
        return "\n".join(full_text).strip()
    except Exception as e:
        print(f"Extraction Error inside cv_parser module (docx): {e}")
        return ""

def extract_text(file_path: str) -> str:
    """
    Unified text extractor for PDF and DOCX files.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    return ""

def extract_rule_based_fallback(raw_text: str) -> CVAnalysisSchema:
    """
    Robust rule-based regex fallback parser to extract resume info when LLM parsing is unavailable.
    """
    import re
    
    # 1. Extract Email
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', raw_text)
    email = email_match.group(0) if email_match else "Not Extracted"
    
    # 2. Extract Phone
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', raw_text)
    phone = phone_match.group(0) if phone_match else None
    
    # 3. Extract Links
    links = re.findall(r'(https?://[^\s,\(\)]+)', raw_text)
    socials = re.findall(r'\b(?:github\.com|linkedin\.com/in|twitter\.com)/[a-zA-Z0-9_\-\.\/]+', raw_text, re.IGNORECASE)
    all_links = list(set(links + socials))
    
    # 4. Extract Name
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    full_name = "Candidate"
    if lines:
        candidate_line = lines[0]
        if len(candidate_line) < 50 and not any(term in candidate_line.lower() for term in ["resume", "cv", "curriculum", "education", "experience"]):
            full_name = candidate_line
            
    # 5. Extract Experience Years (Heuristic)
    years = 0.0
    explicit_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:\+\s*)?(?:years|year|yrs|yr|y)\b", raw_text, re.IGNORECASE)
    for m in explicit_matches:
        try:
            years = max(years, float(m))
        except ValueError:
            continue
            
    # 6. Structured Mock Lists
    education = []
    experience = []
    projects = []
    
    edu_matches = re.findall(r'(?:bachelor|master|bs|ms|phd|b\.s\.|m\.s\.|diploma|degree|university|college|school|intermediate|matriculation)\b[^\n]*', raw_text, re.IGNORECASE)
    for match in edu_matches[:3]:
        education.append(EducationSchema(institution="Extracted Institution", degree=match.strip()[:100], passing_year=None, gpa=None))
        
    return CVAnalysisSchema(
        full_name=full_name,
        email=email,
        phone=phone,
        links=all_links,
        skills=[],
        education=education,
        experience=experience,
        projects=projects,
        experience_years=years,
        certifications=[],
        languages=[],
        summary=None,
        interests=[]
    )

def parse_cv(file_path: str) -> Optional[CVAnalysisSchema]:
    """
    Transforms unstructured resume text into a strictly validated Pydantic JSON structure using Groq LLM.
    """
    raw_text = extract_text(file_path)
    if not raw_text:
        print("Error: No text extracted from the CV.")
        return None
        
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is missing. Running rule-based fallback...")
        return extract_rule_based_fallback(raw_text)

    # Initialize Instructor patch over Groq client
    client = instructor.from_groq(Groq(api_key=api_key))
    
    try:
        structured_data = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=CVAnalysisSchema,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional ATS system backend component. Your task is to process raw text extracts from CVs and seamlessly structure every piece of information into the designated target JSON model. Do not truncate, omit, or overlook any project, link, or experience details."
                },
                {
                    "role": "user", 
                    "content": f"Analyze and parse the following resume plain text extract:\n\n{raw_text}"
                }
            ],
            temperature=0.1
        )
        return structured_data
    except Exception as e:
        print(f"Instructor cognitive parsing failed on primary model: {e}. Trying fallback model llama-3.1-8b-instant...")
        try:
            structured_data = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                response_model=CVAnalysisSchema,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional ATS system backend component. Your task is to process raw text extracts from CVs and seamlessly structure every piece of information into the designated target JSON model. Do not truncate, omit, or overlook any project, link, or experience details."
                    },
                    {
                        "role": "user", 
                        "content": f"Analyze and parse the following resume plain text extract:\n\n{raw_text}"
                    }
                ],
                temperature=0.1
            )
            return structured_data
        except Exception as e2:
            print(f"Instructor cognitive parsing failed on fallback model: {e2}. Reverting to rule-based fallback...")
            try:
                return extract_rule_based_fallback(raw_text)
            except Exception as e3:
                print(f"Fatal parser error in rule-based fallback: {e3}")
                return None