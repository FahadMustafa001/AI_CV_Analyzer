import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_resume(filename, name, role, details):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Write text
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, name)
    
    c.setFont("Helvetica-Oblique", 14)
    c.drawString(50, height - 75, role)
    
    c.setFont("Helvetica", 10)
    y = height - 110
    for line in details:
        if line.startswith("## "):
            c.setFont("Helvetica-Bold", 12)
            y -= 15
            c.drawString(50, y, line[3:])
            c.setFont("Helvetica", 10)
            y -= 12
        else:
            c.drawString(50, y, line)
            y -= 15
            
    c.showPage()
    c.save()

def main():
    resume1_details = [
        "Email: john.doe@email.com",
        "Phone: +1 123 456 7890",
        "Links: https://github.com/johndoe, https://linkedin.com/in/johndoe",
        "## Skills",
        "Python, Java, C++, backend development, Django, FastAPI, SQL, Docker, Git, algorithms, data structures",
        "## Experience",
        "Google - Software Engineer (Oct 2022 - Present)",
        "Developed high performance microservices, debugging complex systems, optimizing queries.",
        "## Education",
        "BS Computer Science from MIT (Passing Year: 2022, GPA: 3.8)",
        "## Projects",
        "Project 1: Task Management System using Django and React",
        "Project 2: Machine Learning pipeline for text classification using scikit-learn"
    ]

    resume2_details = [
        "Email: jane.smith@email.com",
        "Phone: +1 987 654 3210",
        "Links: https://github.com/janesmith, https://linkedin.com/in/janesmith",
        "## Skills",
        "Python, SQL, Tableau, Pandas, Excel, data visualization, powerbi, statistics, machine learning",
        "## Experience",
        "Meta - Data Analyst (Mar 2023 - Present)",
        "Created automated dashboards using PowerBI, analyzed large datasets using SQL and Python.",
        "## Education",
        "BS Data Science from Stanford University (Passing Year: 2023, GPA: 3.9)",
        "## Projects",
        "Project 1: Customer Segmentation Analysis using Python",
        "Project 2: Sales Dashboard using Tableau"
    ]

    create_resume("uploads/john_doe.pdf", "John Doe", "Software Engineer", resume1_details)
    create_resume("uploads/jane_smith.pdf", "Jane Smith", "Data Analyst", resume2_details)
    print("Resumes created successfully.")

if __name__ == "__main__":
    main()
