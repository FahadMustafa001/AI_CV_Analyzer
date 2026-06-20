# pdf_generator.py
import io
import json
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
        
        # Draw footer info
        self.drawString(36, 30, "CONFIDENTIAL - Internal Recruitment Review")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 30, page_text)
        self.restoreState()

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
    
    story = []
    
    # --- 1. HEADER BANNER BLOCK ---
    header_data = [
        [
            Paragraph(candidate['name'].upper(), title_style),
            Paragraph(f"FINAL AUDIT SCORE: <b>{candidate['final_score']}%</b>", ParagraphStyle(
                'HeaderScore', parent=title_style, alignment=2, fontSize=16, textColor=colors.white
            ))
        ],
        [
            Paragraph(f"Target Job Role: {candidate['job_role']}", subtitle_style),
            Paragraph(f"Hiring Match Grade: {candidate.get('ai_report', {}).get('hiring_recommendation', 'N/A')}", ParagraphStyle(
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
    # Calculate colors based on metrics
    def get_score_color(val):
        if val >= 70: return "#10b981" # Green
        if val >= 45: return "#f59e0b" # Orange
        return "#f43f5e" # Red
        
    metrics_data = [
        [
            Paragraph("<b>Job Match Similarity</b>", ParagraphStyle('MetLabel', fontName='Helvetica', fontSize=8.5, textColor=text_muted, alignment=1)),
            Paragraph("<b>Professional Experience</b>", ParagraphStyle('MetLabel', fontName='Helvetica', fontSize=8.5, textColor=text_muted, alignment=1)),
            Paragraph("<b>Personality Benchmark Fit</b>", ParagraphStyle('MetLabel', fontName='Helvetica', fontSize=8.5, textColor=text_muted, alignment=1)),
        ],
        [
            Paragraph(f"<font color='{get_score_color(candidate['job_score'])}'><b>{candidate['job_score']}%</b></font>", ParagraphStyle('MetVal', fontName='Helvetica-Bold', fontSize=18, alignment=1)),
            Paragraph(f"<font color='#4f46e5'><b>{candidate['exp_years']} Years</b></font>", ParagraphStyle('MetVal', fontName='Helvetica-Bold', fontSize=18, alignment=1)),
            Paragraph(f"<font color='{get_score_color(candidate['personality_score'])}'><b>{candidate['personality_score']}%</b></font>", ParagraphStyle('MetVal', fontName='Helvetica-Bold', fontSize=18, alignment=1)),
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[180, 180, 180])
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
    links_str = ", ".join(candidate.get('links', [])) or "None Extracted"
    contact_data = [
        [Paragraph("Email Address:", meta_label_style), Paragraph(candidate.get('email', 'Not Extracted'), meta_val_style)],
        [Paragraph("External Portfolios:", meta_label_style), Paragraph(links_str, meta_val_style)],
    ]
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
    ai_report = candidate.get('ai_report', {})
    
    story.append(Paragraph("Cognitive AI Recruiter Report", h1_style))
    
    rec_box_data = [
        [
            Paragraph(f"<b>AI Recommendation Decision:</b> <font color='{get_score_color(candidate['final_score'])}'><b>{ai_report.get('hiring_recommendation', 'Interview')}</b></font>", body_bold),
        ],
        [
            Paragraph(f"<b>Recommendation Rationale:</b> {ai_report.get('recommendation_rationale', 'Meeting basic requirements. Recommend conducting initial phone screen.')}", body_style)
        ]
    ]
    rec_box_table = Table(rec_box_data, colWidths=[540])
    rec_box_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff")), # Light blue tint
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor("#bfdbfe")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(rec_box_table)
    story.append(Spacer(1, 10))
    
    # Career Milestones & Layout critiques
    story.append(Paragraph("CV Structure & Parser Readability Feedback", h2_style))
    story.append(Paragraph(f"<b>Layout Grade:</b> {ai_report.get('formatting_score', 70)}% | {ai_report.get('formatting_feedback', 'Parsed successfully.')}", body_style))
    
    # Bullet points upgrades
    bullet_improvements = ai_report.get('bullet_points_improvement', [])
    if bullet_improvements:
        story.append(Paragraph("<b>Suggested Resume Upgrades (Google X-Y-Z Formula):</b>", body_bold))
        for bullet in bullet_improvements[:3]:
            story.append(Paragraph(f"• {bullet}", bullet_style))
    story.append(Spacer(1, 10))
    
    # --- 5. TECH ALIGNMENT & CRITICAL GAPS ---
    story.append(Paragraph("Technical Alignment & Competencies Profile", h1_style))
    story.append(Paragraph(f"<b>Core Technical Match Alignment:</b> {ai_report.get('tech_alignment_score', 50)}%", body_bold))
    
    # Render categorized skills inventory
    cat_skills = ai_report.get('categorized_skills', {})
    if cat_skills:
        cat_map = [
            ('programming_languages', 'Languages'),
            ('frameworks_libraries', 'Frameworks'),
            ('databases', 'Databases'),
            ('tools_platforms', 'Tools/DevOps'),
        ]
        skills_summary_parts = []
        for key, title in cat_map:
            skills_list = cat_skills.get(key, [])
            if skills_list:
                skills_summary_parts.append(f"<b>{title}:</b> {', '.join(skills_list)}")
        
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
            
    # Critical Missing Gaps (Marked in Red box)
    gaps = ai_report.get('critical_gaps', [])
    if gaps:
        gaps_text = ", ".join(gaps)
        gaps_box_data = [[Paragraph(f"<b><font color='#f43f5e'>⚠️ Critical Gaps Detected:</font></b> {gaps_text}", body_style)]]
        gaps_box_table = Table(gaps_box_data, colWidths=[540])
        gaps_box_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fff1f2")), # Light red tint
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fecdd3")),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(gaps_box_table)
    
    # Page break to keep the career timeline, psych profiling and recruiting kit organized on subsequent pages
    story.append(PageBreak())
    
    # --- 6. CAREER HISTORY TIMELINE ---
    story.append(Paragraph("Extracted Career Milestones & Work Experience", h1_style))
    experiences = candidate.get('experience', [])
    if not experiences:
        # Fallback to display raw experience if json list is empty
        raw_exp = candidate.get('experience')
        if isinstance(raw_exp, str) and raw_exp.strip():
            story.append(Paragraph(raw_exp.replace("\n", "<br/>"), body_style))
        else:
            story.append(Paragraph("No detailed work experience structured in database.", body_style))
    else:
        for idx, exp in enumerate(experiences[:4]):
            exp_text = f"<b>{exp.get('role', 'Developer')}</b> at <b>{exp.get('company', 'Unknown Organization')}</b> ({exp.get('duration', 'N/A')})"
            story.append(Paragraph(exp_text, ParagraphStyle('ExpH', parent=body_style, fontSize=10, textColor=colors.HexColor("#1e1b4b"))))
            achievements = exp.get('achievements', [])
            if isinstance(achievements, list):
                for ach in achievements:
                    story.append(Paragraph(f"• {ach}", bullet_style))
            else:
                story.append(Paragraph(str(achievements), bullet_style))
            story.append(Spacer(1, 4))
            
    # Add Education Milestones
    edu_list = candidate.get('education', [])
    if edu_list:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Education Milestones", h2_style))
        edu_items = []
        for edu in edu_list:
            passing = f" (Graduated: {edu.get('passing_year')})" if edu.get('passing_year') else ""
            gpa = f" | GPA: {edu.get('gpa')}" if edu.get('gpa') else ""
            edu_items.append(Paragraph(f"• <b>{edu.get('degree', 'Degree')}</b> at {edu.get('institution', 'Institution')}{passing}{gpa}", bullet_style))
        story.append(KeepTogether(edu_items))
        
    story.append(Spacer(1, 10))
    
    # --- 7. PSYCHOMETRIC BIG FIVE TRAITS PROFILE ---
    story.append(Paragraph("Psychometric & Personality Traits Profile", h1_style))
    traits = candidate.get('personality_traits', {})
    
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
    }
    benchmarks = benchmarks_map.get(candidate['job_role'], {"openness": 60, "conscientiousness": 70, "extraversion": 50, "agreeableness": 60, "neuroticism": 40})
    
    for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        score_val = traits.get(trait, 50.0)
        bench_val = benchmarks.get(trait, 50.0)
        traits_rows.append([
            Paragraph(trait.capitalize(), body_style),
            Paragraph(f"{score_val}%", body_style),
            Paragraph(f"{bench_val}%", body_style)
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
    
    story.append(Paragraph("<b>Personality Alignment Analysis:</b>", body_bold))
    story.append(Paragraph(ai_report.get('personality_fit_explanation', 'No detailed psychological fit report generated.'), body_style))
    
    # Page break to keep the dynamic recruiting kit on page 3
    story.append(PageBreak())
    
    # --- 8. DYNAMIC RECRUITING KIT (TEST & RUBRIC) ---
    story.append(Paragraph("Custom Candidate Recruitment Kit", h1_style))
    story.append(Paragraph("This assessment and evaluation rubric are custom generated based on the candidate's experience and detected gaps for direct recruiter use.", ParagraphStyle('RecSub', parent=body_style, fontName='Helvetica-Oblique', textColor=text_muted)))
    
    custom_test = ai_report.get('custom_coding_test', {})
    if custom_test:
        test_block = []
        test_block.append(Paragraph(f"🛠️ Practical Assessment: {custom_test.get('title', 'Custom Evaluation Assignment')}", h2_style))
        test_block.append(Paragraph(f"<b>Estimated Time Duration:</b> {custom_test.get('duration', '3 Hours')}", body_bold))
        test_block.append(Paragraph(f"<b>Problem Statement:</b> {custom_test.get('problem_statement', 'Build a target project validating coding standards.')}", body_style))
        
        reqs = custom_test.get('requirements', [])
        if reqs:
            test_block.append(Paragraph("<b>Technical Requirements:</b>", body_bold))
            for req in reqs:
                test_block.append(Paragraph(f"• {req}", bullet_style))
        
        test_block.append(Spacer(1, 4))
        test_block.append(Paragraph(f"<b>Submission Guidelines:</b> {custom_test.get('submission_guidelines', 'Host source on GitHub and mail instructions.')}", body_style))
        
        test_table = Table([[test_block]], colWidths=[540])
        test_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#faf5ff")), # Light purple tint
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e9d5ff")),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(test_table)
        story.append(Spacer(1, 12))
        
    rubric = ai_report.get('interview_evaluation_rubric', [])
    if rubric:
        rubric_block = []
        rubric_block.append(Paragraph("📝 Recruiter Live Interview Grading Rubric", h2_style))
        
        for idx, item in enumerate(rubric):
            rubric_block.append(Paragraph(f"<b>Question {idx+1}:</b> {item.get('question', 'Explain your project structure.')}", body_bold))
            rubric_block.append(Paragraph(f"<i>Look for:</i> {item.get('what_to_look_for', 'Proper naming conventions, dependency injections.')}", body_style))
            rubric_block.append(Paragraph(f"<i>Grading Criteria:</i> {item.get('grading_criteria', 'Excellent: details clean practices. Poor: absolute layouts.')}", body_style))
            rubric_block.append(Spacer(1, 4))
            
        rubric_table = Table([[rubric_block]], colWidths=[540])
        rubric_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")), # Light green tint
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#bbf7d0")),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(rubric_table)
        
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    
    buffer.seek(0)
    return buffer
