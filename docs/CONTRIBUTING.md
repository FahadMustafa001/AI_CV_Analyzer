# 🤝 Contributing to AI CV Analyzer

Thank you for your interest in contributing! Here's how you can help.

---

## 🛠️ Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/AI_CV_Analyzer.git
   cd AI_CV_Analyzer
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create `.env` with your Groq API key:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
6. Run the app:
   ```bash
   python app.py
   ```

---

## 📐 Code Style

- **Python:** Follow PEP 8 conventions
- **HTML/CSS:** Use Bootstrap 5.3 classes, maintain dark/light theme compatibility
- **JavaScript:** Use vanilla JS (no jQuery), ES6+ syntax
- **Docstrings:** All functions must have docstrings
- **Logging:** Use `logger.info()`, `logger.warning()`, `logger.error()` — no `print()` statements

---

## 🔀 Pull Request Process

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Test locally — ensure the app starts and all pages load
4. Commit with clear messages:
   ```bash
   git commit -m "Add: brief description of the change"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request against `main`

---

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Before submitting a PR, verify:
- [ ] App starts without errors (`python app.py`)
- [ ] Homepage loads (`http://127.0.0.1:5000`)
- [ ] Ranking page loads (`http://127.0.0.1:5000/ranking`)
- [ ] Dashboard loads (`http://127.0.0.1:5000/dashboard`)
- [ ] Analytics API returns data (`http://127.0.0.1:5000/api/analytics`)
- [ ] No new errors in `app.log`

---

## 💡 Areas for Contribution

### High Priority
- [ ] Unit tests for all 13 AI agents
- [ ] Integration tests for API endpoints
- [ ] Email integration (actual sending via SMTP)
- [ ] User authentication & authorization

### Medium Priority
- [ ] Candidate photo/avatar support
- [ ] Batch CV upload via ZIP
- [ ] Interview scheduling with calendar integration
- [ ] WebSocket for real-time analysis progress
- [ ] Mobile-responsive improvements

### Low Priority
- [ ] Internationalization (i18n)
- [ ] Dark/Light theme improvements
- [ ] Additional AI model support (OpenAI, Anthropic)
- [ ] Export to Excel (XLSX)
- [ ] Webhook notifications

---

## 🐛 Reporting Bugs

Open an issue with:
1. **Description** of the bug
2. **Steps to reproduce**
3. **Expected behavior** vs **Actual behavior**
4. **Screenshots** if applicable
5. **Error logs** from `app.log`

---

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

> Thank you for making AI CV Analyzer better! 🚀
