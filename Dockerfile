# 1. Use an official lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for handling text and files
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements.txt first and install python packages (enables Docker layer caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application files
COPY . /app

# 6. Expose the port Flask runs on
EXPOSE 5000

# 7. Define environment variables default placeholders
ENV FLASK_APP=app.py
ENV GROQ_API_KEY="YOUR_KEY_HERE"

# 8. Command to run the application using production style flags
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
