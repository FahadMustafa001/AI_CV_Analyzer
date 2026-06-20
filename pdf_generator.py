# pdf_generator.py
import io
import json
import html
import re
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Canvas to perform two-pass rendering for accurate 'Page X of Y' footer generation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Draw header divider line and title on later pages
        if self._pageNumber > 1:
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)
            self.drawString(36, 762, "CogniCV - AI Candidate Profile Audit Report")
        
        # Draw footer divider line
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(36, 45, 576, 45)
        
        # Draw footer info with timestamp
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.drawString(36, 30, f"CONFIDENTIAL - Generated {gen_time}")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 30, page_text)
        self.restoreState()

def clean_text_for_pdf(text):
    """
    Escapes raw XML entities (like &, <, >) to prevent ReportLab crashes
    and converts basic markdown tags to PDF formatting.
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Escape basic XML entities to prevent ReportLab parsing crashes
    escaped = html.escape(text)
    
    # 2. Convert Markdown bold (**text**) to HTML <b>text</b>
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
    escaped = re.sub(r'\*(.*?)\*', r'<i>\1</i>', escaped)
    
    # 3. Convert list items (* or -) at the start of lines to bullets
    escaped = re.sub(r'^\s*[\*\-]\s+', r'• ', escaped, flags=re.MULTILINE)
    
    # 4. Replace newlines with <br/> for ReportLab line breaks
    escaped = escaped.replace("\n", "<br/>")
    
    return escaped


def generate_candidate_pdf(candidate) -> io.BytesIO:
    """
    Generates a high-fidelity, styled candidate audit PDF report using ReportLab
    and returns it as an in-memory BytesIO buffer.
    """
    buffer = io.BytesIO()
    
    # 0.5 inch margins (36pt) to maximize canvas usage
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom high-contrast styles matching our design system
    primary_color = colors.HexColor("#4f46e5")  # Indigo
    dark_slate = colors.HexColor("#0f172a")     # Deep background slate
    text_dark = colors.HexColor("#1e293b")      # Deep charcoal for body text
    text_muted = colors.HexColor("#64748b")     # Grey for notes
    
    # Modify default styles in-place safely or create new ones
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.white,
        alignment=0, # Left-aligned
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#c7d2fe"), # Soft indigo accent
        spaceAfter=0
    )
    
    h1_style = ParagraphStyle(
        'SecH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SecH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#312e81"), # Dark indigo
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=text_dark,
        spaceAfter=6
    )
    
    body_bold = ParagraphStyle(
        'BodyBoldText',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    meta_label_style = ParagraphStyle(
        'MetaLabel',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=colors.HexColor("#475569")
    )
    
    meta_val_style = ParagraphStyle(
        'MetaVal',
        fontName='Helvetica',
        fontSize=9.5,
        textColor=text_dark
    )
    
    # Tag/badge style for skills
    tag_style = ParagraphStyle(
        'TagStyle',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4338ca"),
    )
    
    story = []
    
    # Safe score styling function
    def get_score_color(val):
        try:
            val_float = float(val) if val is not None else 0.0
        except ValueError:
            val_float = 0.0
        if val_float >= 70: return "#10b981" # Green
        if val_float >= 45: return "#f59e0b" # Orange
        return "#f43f5e" # Red
        
    ai_report = candidate.get('ai_report')
    if not isinstance(ai_report, dict):
        ai_report = {}
    
    # --- 1. HEADER BANNER BLOCK ---
    raw_name = candidate.get('name', 'UNKNOWN')
    display_name = raw_name
    for ext in ['.pdf', '.docx', '.PDF', '.DOCX']:
        if display_name.lower().endswith(ext):
            display_name = display_name[:-len(ext)]
    if '_' in display_name or '-' in display_name:
        display_name = display_name.replace('_', ' ').replace('-', ' ')
    display_name = " ".join(display_name.split())
    if display_name.isupper() or display_name.islower():
        display_name = display_name.title()
    candidate_name = clean_text_for_pdf(display_name)
    
    final_score = candidate.get('final_score')
    if final_score is None:
        final_score = 0.0
    job_role_val = clean_text_for_pdf(candidate.get('job_role', 'Not Specified'))
    hiring_rec = clean_text_for_pdf(ai_report.get('hiring_recommendation', 'N/A'))
    
    header_data = [
        [
            Paragraph(candidate_name.upper(), title_style),
            Paragraph(f"FINAL AUDIT SCORE: <b>{final_score}%</b>", ParagraphStyle(
                'HeaderScore', parent=title_style, alignment=2, fontSize=16, textColor=colors.white
            ))
        ],
        [
            Paragraph(f"Target Job Role: {job_role_val}", subtitle_style),
            Paragraph(f"Hiring Match Grade: {hiring_rec}", ParagraphStyle(
                'HeaderMatch', parent=subtitle_style, alignment=2, fontName='Helvetica-Bold'
            ))
        ]
    ]
    header_table = Table(header_data, colWidths=[300, 240])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), primary_color),
        ('ALIGN', (0,0), (0,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 16),
        ('RIGHTPADDING', (0,0), (-1,-1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 12))
    
    # --- 2. SUMMARY METRICS CARD GRID ---
    job_score = candidate.get('job_score')
    if job_score is None:
        job_score = 0.0
    exp_years = candidate.get('exp_years')
    if exp_years is None:
        exp_years = 0.0
    personality_score = candidate.get('personality_score')
    if personality_score is None:
        personality_score = 0.0
    
    seniority = clean_text_for_pdf(ai_report.get('seniority_level', 'N/A'))
    
    metrics_data = [
        [
            Paragraph("<b>Job Match Score</b>", ParagraphStyle('MetLabel', fontName='Helvetica', fontSize=8.5, textColor=text_muted, alignment=1)),
            Paragraph("<b>Experience</b>", ParagraphStyle('MetLabel', fontName='Helvetica', fontSize=8.5, textColor=text_muted, alignment=1)),
            Paragraph("<b>Personality Fit</b>", ParagraphStyle('MetLabel', fontName='Helvetica', fontSize=8.5, textColor=text_muted, alignment=1)),
            Paragraph("<b>Seniority Level</b>", ParagraphStyle('MetLabel', fontName='Helvetica', fontSize=8.5, textColor=text_muted, alignment=1)),
        ],
        [
            Paragraph(f"<font color='{get_score_color(job_score)}'><b>{job_score}%</b></font>", ParagraphStyle('MetVal1', fontName='Helvetica-Bold', fontSize=18, alignment=1)),
            Paragraph(f"<font color='#4f46e5'><b>{exp_years} Years</b></font>", ParagraphStyle('MetVal2', fontName='Helvetica-Bold', fontSize=18, alignment=1)),
            Paragraph(f"<font color='{get_score_color(personality_score)}'><b>{personality_score}%</b></font>", ParagraphStyle('MetVal3', fontName='Helvetica-Bold', fontSize=18, alignment=1)),
            Paragraph(f"<font color='#4f46e5'><b>{seniority}</b></font>", ParagraphStyle('MetVal4', fontName='Helvetica-Bold', fontSize=14, alignment=1)),
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[135, 135, 135, 135])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 14))
    
    # --- 3. CANDIDATE INFO & METADATA TABLE ---
    links_str = ", ".join(candidate.get('links') or []) or "None Extracted"
    contact_data = [
        [Paragraph("Email Address:", meta_label_style), Paragraph(clean_text_for_pdf(candidate.get('email', 'Not Extracted')), meta_val_style)],
        [Paragraph("External Portfolios:", meta_label_style), Paragraph(clean_text_for_pdf(links_str), meta_val_style)],
    ]
    
    # Add soft skills, certs, interests
    soft_str = candidate.get('soft_skills', '')
    if soft_str:
        contact_data.append([Paragraph("Soft Skills:", meta_label_style), Paragraph(clean_text_for_pdf(soft_str), meta_val_style)])
    certs_str = candidate.get('certificates', '')
    if certs_str:
        contact_data.append([Paragraph("Certifications:", meta_label_style), Paragraph(clean_text_for_pdf(certs_str), meta_val_style)])
    interests_str = candidate.get('interests', '')
    if interests_str:
        contact_data.append([Paragraph("Interests:", meta_label_style), Paragraph(clean_text_for_pdf(interests_str), meta_val_style)])
    
    contact_table = Table(contact_data, colWidths=[110, 430])
    contact_table.setStyle(TableStyle([
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#f1f5f9")),
    ]))
    story.append(contact_table)
    story.append(Spacer(1, 10))
    
    # --- 4. COGNITIVE AI RECRUITER AUDIT REPORT ---
    story.append(Paragraph("Cognitive AI Recruiter Report", h1_style))
    
    rec_decision = clean_text_for_pdf(ai_report.get('hiring_recommendation', 'Interview'))
    rec_rationale = clean_text_for_pdf(ai_report.get('recommendation_rationale', 'Meeting basic requirements. Recommend conducting initial phone screen.'))
    
    rec_box_data = [
        [
            Paragraph(f"<b>AI Recommendation Decision:</b> <font color='{get_score_color(final_score)}'><b>{rec_decision}</b></font>", body_bold),
        ],
        [
            Paragraph(f"<b>Recommendation Rationale:</b> {rec_rationale}", body_style)
        ]
    ]
    rec_box_table = Table(rec_box_data, colWidths=[540])
    rec_box_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff")),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#bfdbfe")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(rec_box_table)
    story.append(Spacer(1, 10))
    
    # CV structure & layout feedback
    formatting_score_val = ai_report.get('formatting_score', 70)
    formatting_feedback_val = clean_text_for_pdf(ai_report.get('formatting_feedback', 'Parsed successfully.'))
    story.append(Paragraph("CV Structure & Parser Readability Feedback", h2_style))
    story.append(Paragraph(f"<b>Layout Grade:</b> {formatting_score_val}% | {formatting_feedback_val}", body_style))
    
    # Bullet points upgrades
    bullet_improvements = ai_report.get('bullet_points_improvement') or []
    if isinstance(bullet_improvements, str):
        bullet_improvements = [bullet_improvements]
    if bullet_improvements:
        story.append(Paragraph("<b>Suggested Resume Upgrades (Google X-Y-Z Formula):</b>", body_bold))
        for bullet in bullet_improvements[:3]:
            story.append(Paragraph(f"• {clean_text_for_pdf(bullet)}", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- 5. TECH ALIGNMENT & CRITICAL GAPS ---
    tech_align_score = ai_report.get('tech_alignment_score', 50)
    story.append(Paragraph("Technical Alignment & Competencies Profile", h1_style))
    story.append(Paragraph(f"<b>Core Technical Match Alignment:</b> {tech_align_score}%", body_bold))
    
    # Render categorized skills inventory
    cat_skills = ai_report.get('categorized_skills')
    if not isinstance(cat_skills, dict):
        cat_skills = {}
    if cat_skills:
        cat_map = [
            ('programming_languages', 'Languages'),
            ('frameworks_libraries', 'Frameworks'),
            ('databases', 'Databases'),
            ('tools_platforms', 'Tools/DevOps'),
            ('soft_skills', 'Soft Skills'),
        ]
        skills_summary_parts = []
        for key, category_title in cat_map:
            skills_list = cat_skills.get(key, [])
            if isinstance(skills_list, str):
                skills_list = [skills_list]
            if skills_list:
                cleaned_skills = clean_text_for_pdf(', '.join(skills_list))
                skills_summary_parts.append(f"<b>{category_title}:</b> {cleaned_skills}")
        
        if skills_summary_parts:
            skills_box_data = [[Paragraph("<br/>".join(skills_summary_parts), body_style)]]
            skills_box_table = Table(skills_box_data, colWidths=[540])
            skills_box_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(skills_box_table)
            story.append(Spacer(1, 6))
            
    # Critical Missing Gaps
    gaps = ai_report.get('critical_gaps') or []
    if isinstance(gaps, str):
        gaps = [gaps]
    if gaps:
        gaps_text = clean_text_for_pdf(", ".join(gaps))
        gaps_box_data = [[Paragraph(f"<b><font color='#f43f5e'>WARNING - Critical Gaps Detected:</font></b> {gaps_text}", body_style)]]
        gaps_box_table = Table(gaps_box_data, colWidths=[540])
        gaps_box_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fff1f2")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fecdd3")),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(gaps_box_table)
    
    # Page break to keep the career timeline organized
    story.append(PageBreak())
    
    # --- 6. CAREER HISTORY TIMELINE ---
    story.append(Paragraph("Extracted Career Milestones & Work Experience", h1_style))
    experiences = candidate.get('experience') or []
    if isinstance(experiences, str):
        if experiences.strip():
            story.append(Paragraph(clean_text_for_pdf(experiences), body_style))
        else:
            story.append(Paragraph("No detailed work experience structured in database.", body_style))
    elif not isinstance(experiences, list) or not experiences:
        story.append(Paragraph("No detailed work experience structured in database.", body_style))
    else:
        for idx, exp in enumerate(experiences[:6]):
            if not isinstance(exp, dict):
                story.append(Paragraph(clean_text_for_pdf(str(exp)), body_style))
                continue
            exp_flowables = []
            
            exp_role = clean_text_for_pdf(exp.get('role', 'Developer'))
            exp_company = clean_text_for_pdf(exp.get('company', 'Unknown Organization'))
            exp_duration = clean_text_for_pdf(exp.get('duration', 'N/A'))
            
            exp_text = f"<b>{exp_role}</b> at <b>{exp_company}</b> ({exp_duration})"
            exp_flowables.append(Paragraph(exp_text, ParagraphStyle('ExpH', parent=body_style, fontSize=10, textColor=colors.HexColor("#1e1b4b"))))
            
            achievements = exp.get('achievements', [])
            if isinstance(achievements, list):
                for ach in achievements:
                    exp_flowables.append(Paragraph(f"• {clean_text_for_pdf(ach)}", bullet_style))
            else:
                exp_flowables.append(Paragraph(clean_text_for_pdf(str(achievements)), bullet_style))
                
            exp_flowables.append(Spacer(1, 4))
            story.append(KeepTogether(exp_flowables))
            
    # Add Education Milestones
    edu_list = candidate.get('education') or []
    if isinstance(edu_list, str):
        edu_list = [edu_list]
    if isinstance(edu_list, list) and edu_list:
        edu_flowables = []
        edu_flowables.append(Spacer(1, 6))
        edu_flowables.append(Paragraph("Education Milestones", h2_style))
        for edu in edu_list:
            if isinstance(edu, str):
                edu_flowables.append(Paragraph(f"• {clean_text_for_pdf(edu)}", bullet_style))
            elif isinstance(edu, dict):
                degree_val = clean_text_for_pdf(edu.get('degree', 'Degree'))
                inst_val = clean_text_for_pdf(edu.get('institution', 'Institution'))
                passing = f" (Graduated: {clean_text_for_pdf(edu.get('passing_year'))})" if edu.get('passing_year') else ""
                gpa = f" | GPA: {clean_text_for_pdf(edu.get('gpa'))}" if edu.get('gpa') else ""
                edu_flowables.append(Paragraph(f"• <b>{degree_val}</b> at {inst_val}{passing}{gpa}", bullet_style))
        story.append(KeepTogether(edu_flowables))
    
    # --- 6b. PROJECTS SHOWCASE ---
    proj_list = candidate.get('projects') or []
    if isinstance(proj_list, list) and proj_list:
        proj_flowables = []
        proj_flowables.append(Spacer(1, 8))
        proj_flowables.append(Paragraph("Projects Portfolio", h2_style))
        for proj in proj_list[:5]:
            if isinstance(proj, dict):
                proj_title = clean_text_for_pdf(proj.get('title', 'Untitled Project'))
                proj_tech = proj.get('tech_stack', [])
                if isinstance(proj_tech, list):
                    tech_str = clean_text_for_pdf(', '.join(proj_tech))
                else:
                    tech_str = clean_text_for_pdf(str(proj_tech))
                proj_desc = clean_text_for_pdf(proj.get('description', ''))
                
                proj_flowables.append(Paragraph(f"<b>{proj_title}</b>", ParagraphStyle('ProjH', parent=body_style, fontSize=10, textColor=colors.HexColor("#1e1b4b"))))
                if tech_str:
                    proj_flowables.append(Paragraph(f"<i>Tech Stack:</i> {tech_str}", ParagraphStyle('ProjTech', parent=body_style, fontSize=9, textColor=colors.HexColor("#6366f1"))))
                if proj_desc:
                    proj_flowables.append(Paragraph(f"• {proj_desc}", bullet_style))
                proj_flowables.append(Spacer(1, 4))
            elif isinstance(proj, str):
                proj_flowables.append(Paragraph(f"• {clean_text_for_pdf(proj)}", bullet_style))
        story.append(KeepTogether(proj_flowables))
        
    story.append(Spacer(1, 10))
    
    # --- 7. PSYCHOMETRIC BIG FIVE TRAITS PROFILE ---
    story.append(Paragraph("Psychometric & Personality Traits Profile", h1_style))
    traits = candidate.get('personality_traits') or {}
    
    # Generate traits grid table
    traits_rows = [
        [Paragraph("<b>Big Five Trait</b>", body_bold), Paragraph("<b>Candidate Score</b>", body_bold), Paragraph("<b>Role Benchmark</b>", body_bold)]
    ]
    
    # Predefined role benchmarks fallback mapping
    benchmarks_map = {
        "Software Engineer": {"openness": 65, "conscientiousness": 80, "extraversion": 45, "agreeableness": 60, "neuroticism": 35},
        "Data Analyst": {"openness": 60, "conscientiousness": 85, "extraversion": 40, "agreeableness": 55, "neuroticism": 35},
        "Flutter Developer": {"openness": 75, "conscientiousness": 75, "extraversion": 55, "agreeableness": 60, "neuroticism": 40},
        "Graphic Designer": {"openness": 85, "conscientiousness": 65, "extraversion": 55, "agreeableness": 65, "neuroticism": 40},
        "HR Manager": {"openness": 60, "conscientiousness": 75, "extraversion": 70, "agreeableness": 75, "neuroticism": 35},
        "DevOps Engineer": {"openness": 60, "conscientiousness": 85, "extraversion": 40, "agreeableness": 55, "neuroticism": 30},
        "Product Manager": {"openness": 75, "conscientiousness": 80, "extraversion": 70, "agreeableness": 65, "neuroticism": 35},
        "UI/UX Designer": {"openness": 85, "conscientiousness": 70, "extraversion": 60, "agreeableness": 70, "neuroticism": 35},
    }
    benchmarks = benchmarks_map.get(candidate.get('job_role', 'Software Engineer'), {"openness": 60, "conscientiousness": 70, "extraversion": 50, "agreeableness": 60, "neuroticism": 40})
    
    for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        score_val = traits.get(trait)
        score_str = f"{score_val}%" if score_val is not None else "Not Evaluated"
        
        bench_val = benchmarks.get(trait, 50.0)
        bench_str = f"{bench_val}%" if bench_val is not None else "50.0%"
        
        # Color-code score based on deviation from benchmark
        score_color = "#1e293b"
        if score_val is not None:
            diff = abs(float(score_val) - float(bench_val))
            if diff <= 10:
                score_color = "#10b981"  # Green - close match
            elif diff <= 25:
                score_color = "#f59e0b"  # Orange - moderate deviation
            else:
                score_color = "#f43f5e"  # Red - significant deviation
        
        traits_rows.append([
            Paragraph(trait.capitalize(), body_style),
            Paragraph(f"<font color='{score_color}'><b>{score_str}</b></font>", body_style),
            Paragraph(bench_str, body_style)
        ])
        
    traits_table = Table(traits_rows, colWidths=[180, 180, 180])
    traits_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(traits_table)
    story.append(Spacer(1, 8))
    
    fit_explanation = clean_text_for_pdf(ai_report.get('personality_fit_explanation', 'No detailed psychological fit report generated.'))
    story.append(Paragraph("<b>Personality Alignment Analysis:</b>", body_bold))
    story.append(Paragraph(fit_explanation, body_style))
    
    # --- 8. DYNAMIC RECRUITING KIT (TEST & RUBRIC) ---
    custom_test = ai_report.get('custom_coding_test') or {}
    rubric = ai_report.get('interview_evaluation_rubric') or []
    
    if custom_test or rubric:
        story.append(PageBreak())
        story.append(Paragraph("Custom Candidate Recruitment Kit", h1_style))
        story.append(Paragraph("This assessment and evaluation rubric are custom generated based on the candidate's experience and detected gaps for direct recruiter use.", ParagraphStyle('RecSub', parent=body_style, fontName='Helvetica-Oblique', textColor=text_muted)))
        story.append(Spacer(1, 10))
        
        if custom_test and isinstance(custom_test, dict):
            test_block = []
            test_title = clean_text_for_pdf(custom_test.get('title', 'Custom Evaluation Assignment'))
            test_duration = clean_text_for_pdf(custom_test.get('duration', '3 Hours'))
            test_problem = clean_text_for_pdf(custom_test.get('problem_statement', 'Build a target project validating coding standards.'))
            test_submission = clean_text_for_pdf(custom_test.get('submission_guidelines', 'Host source on GitHub and mail instructions.'))
            
            test_block.append(Paragraph(f"Practical Assessment: {test_title}", h2_style))
            test_block.append(Paragraph(f"<b>Estimated Time Duration:</b> {test_duration}", body_bold))
            test_block.append(Paragraph(f"<b>Problem Statement:</b> {test_problem}", body_style))
            
            reqs = custom_test.get('requirements', [])
            if reqs and isinstance(reqs, list):
                test_block.append(Paragraph("<b>Technical Requirements:</b>", body_bold))
                for req in reqs:
                    test_block.append(Paragraph(f"• {clean_text_for_pdf(req)}", bullet_style))
            
            test_block.append(Spacer(1, 4))
            test_block.append(Paragraph(f"<b>Submission Guidelines:</b> {test_submission}", body_style))
            
            test_table = Table([[test_block]], colWidths=[540])
            test_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#faf5ff")),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e9d5ff")),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 12),
                ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(KeepTogether([test_table]))
            story.append(Spacer(1, 12))
            
        if rubric and isinstance(rubric, list):
            rubric_block = []
            rubric_block.append(Paragraph("Recruiter Live Interview Grading Rubric", h2_style))
            
            for idx, item in enumerate(rubric):
                if isinstance(item, dict):
                    q_text = clean_text_for_pdf(item.get('question', 'Explain your project structure.'))
                    look_text = clean_text_for_pdf(item.get('what_to_look_for', 'Proper naming conventions, dependency injections.'))
                    grad_text = clean_text_for_pdf(item.get('grading_criteria', 'Excellent: details clean practices. Poor: basic explanation.'))
                    
                    rubric_block.append(Paragraph(f"<b>Question {idx+1}:</b> {q_text}", body_bold))
                    rubric_block.append(Paragraph(f"<i>Look for:</i> {look_text}", body_style))
                    rubric_block.append(Paragraph(f"<i>Grading Criteria:</i> {grad_text}", body_style))
                    rubric_block.append(Spacer(1, 4))
                
            rubric_table = Table([[rubric_block]], colWidths=[540])
            rubric_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#bbf7d0")),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 12),
                ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(KeepTogether([rubric_table]))
            
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    
    buffer.seek(0)
    return buffer

def generate_optimized_cv_pdf(name, job_role, optimized_text) -> io.BytesIO:
    """
    Generates a professional, print-ready PDF resume from the AI-optimized
    markdown text and returns it as an in-memory BytesIO buffer.
    """
    buffer = io.BytesIO()
    
    # Standard 0.75 inch margins (54pt) for resume layout
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom color palette matching a sleek corporate theme
    primary_color = colors.HexColor("#1e3a8a")  # Dark Blue
    text_dark = colors.HexColor("#0f172a")     # Slate 900
    text_muted = colors.HexColor("#475569")    # Slate 600
    
    # Define ParagraphStyles
    name_style = ParagraphStyle(
        'CVName',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=1, # Center
        spaceAfter=4
    )
    
    role_style = ParagraphStyle(
        'CVRole',
        fontName='Helvetica-Oblique',
        fontSize=11,
        leading=15,
        textColor=text_muted,
        alignment=1, # Center
        spaceAfter=12
    )
    
    h1_style = ParagraphStyle(
        'CVH1',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'CVH2',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=text_dark,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'CVBody',
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_dark,
        spaceAfter=4
    )
    
    bullet_style = ParagraphStyle(
        'CVBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )
    
    story = []
    
    # Clean name for display header
    display_name = name
    for ext in ['.pdf', '.docx', '.PDF', '.DOCX']:
        if display_name.lower().endswith(ext):
            display_name = display_name[:-len(ext)]
    if '_' in display_name or '-' in display_name:
        display_name = display_name.replace('_', ' ').replace('-', ' ')
    display_name = " ".join(display_name.split())
    if display_name.isupper() or display_name.islower():
        display_name = display_name.title()
        
    story.append(Paragraph(clean_text_for_pdf(display_name), name_style))
    story.append(Paragraph(clean_text_for_pdf(job_role), role_style))
    
    # Simple horizontal rule helper
    def get_hr():
        hr_table = Table([[""]], colWidths=[504])
        hr_table.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 1, primary_color),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        return hr_table
        
    lines = optimized_text.split('\n')
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        # Parse Markdown headers
        if line_str.startswith("# "):
            title_text = line_str[2:].strip()
            if title_text.lower() == name.lower() or title_text.lower() == display_name.lower():
                continue
            story.append(Paragraph(clean_text_for_pdf(title_text), name_style))
        elif line_str.startswith("## "):
            sec_text = line_str[3:].strip()
            story.append(Paragraph(clean_text_for_pdf(sec_text), h1_style))
            story.append(get_hr())
            story.append(Spacer(1, 4))
        elif line_str.startswith("### "):
            subsec_text = line_str[4:].strip()
            story.append(Paragraph(clean_text_for_pdf(subsec_text), h2_style))
        elif line_str.startswith("- ") or line_str.startswith("* ") or line_str.startswith("• "):
            bullet_text = line_str[2:].strip()
            story.append(Paragraph(f"• {clean_text_for_pdf(bullet_text)}", bullet_style))
        else:
            story.append(Paragraph(clean_text_for_pdf(line_str), body_style))
            
    # Build with page numbering for the optimized CV too
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer
