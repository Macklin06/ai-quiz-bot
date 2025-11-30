# LLM Quiz Agent - Project 2

Automated quiz-solving agent using LLMs for data analysis, processing, and visualization tasks.

## 🎯 Project Overview

This system receives quiz tasks via webhook, solves data-related questions autonomously using GPT-4, and submits answers within a 3-minute time limit.

**Capabilities:**
- Web scraping (JavaScript-rendered pages)
- File processing (PDF, CSV, JSON, images)
- Data analysis and transformation
- Chart generation
- Multi-step quiz chains

## 🏗️ Architecture

```
POST /webhook → FastAPI → Background Task → Playwright + GPT-4 → Submit Answer
```

**Components:**
1. **FastAPI Server** (`main.py`) - Handles incoming webhooks
2. **Quiz Agent** (`agent.py`) - Orchestrates quiz solving
3. **Playwright** - Browser automation for page rendering
4. **GPT-4** - Generates Python code to solve questions

## 📋 Prerequisites

- Python 3.13+
- Docker (recommended)
- AIpipe API key or OpenAI API key

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd quiz-bot

# 2. Create .env file
cp .env.example .env
# Edit .env with your API keys

# 3. Build and run
docker build -t quiz-bot .
docker run -p 8000:8000 --env-file .env quiz-bot
```

### Option 2: Local Development

```bash
# 1. Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
uv pip install -e .

# 4. Install Playwright browsers
playwright install chromium

# 5. Set environment variables
export AI_PIPE_KEY="your_key_here"
export QUIZ_SECRET="your_secret_here"

# 6. Run server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AI_PIPE_KEY` | AIpipe or OpenAI API key | Yes |
| `QUIZ_SECRET` | Quiz authentication secret | Yes |
| `PORT` | Server port (default: 8000) | No |
| `LOG_LEVEL` | Logging level (INFO/DEBUG) | No |

### Google Form Submission

Fill out the project Google Form with:
1. **Email**: Your email address
2. **Secret**: A unique string for authentication
3. **System Prompt**: Defense prompt (max 100 chars)
4. **User Prompt**: Attack prompt (max 100 chars)
5. **API Endpoint**: Your deployed URL (e.g., `https://your-app.com/webhook`)
6. **GitHub Repo**: This repository URL (must be public with MIT license)

## 📡 API Endpoints

### POST /webhook
Receives quiz tasks from evaluation server.

**Request:**
```json
{
  "email": "your@email.com",
  "secret": "your_secret",
  "url": "https://example.com/quiz-123"
}
```

**Response:**
```json
{
  "message": "Task accepted",
  "status": "processing",
  "request_id": 12345,
  "timestamp": "2025-11-30T10:00:00"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "playwright": "ready",
  "openai_configured": true,
  "secret_configured": true
}
```

## 🧪 Testing

### Test with Demo Endpoint

```bash
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your@email.com",
    "secret": "your_secret",
    "url": "https://tds-llm-analysis.s-anand.net/demo"
  }'
```

### Manual Testing

1. Start server: `uvicorn main:app --reload`
2. Check health: `curl http://localhost:8000/health`
3. Send test request (see above)
4. Check logs for execution status

## 📊 How It Works

1. **Receive Task**: FastAPI receives webhook POST with quiz URL
2. **Validate**: Check secret matches configuration
3. **Background Processing**: 
   - Launch Playwright browser
   - Navigate to quiz URL
   - Extract visible text + HTML
4. **LLM Generation**:
   - Send question to GPT-4
   - Receive Python code solution
5. **Execute**: Run generated code safely
6. **Submit**: POST answer to submission endpoint
7. **Chain**: If correct, follow next URL; repeat until complete

### Safety Features

- ✅ 3-minute timeout enforcement (stops at 2:50)
- ✅ Error handling at every step
- ✅ Isolated code execution environment
- ✅ Request logging and tracking
- ✅ Graceful degradation on failures

## 🎓 Question Types Supported

- ✅ CSV/Excel data analysis
- ✅ PDF text extraction
- ✅ JSON data processing
- ✅ Image analysis (OCR, vision)
- ✅ Web scraping (static + JavaScript)
- ✅ Statistical calculations
- ✅ Chart generation
- ✅ Geospatial analysis
- ✅ Network graph analysis

## 🐛 Troubleshooting

### Common Issues

**"Invalid credentials" (403 error)**
- Ensure `QUIZ_SECRET` in `.env` matches Google Form submission

**"OpenAI API error"**
- Verify `AI_PIPE_KEY` is set correctly
- Check AIpipe account has credits

**"Browser launch failed"**
- Run `playwright install chromium`
- For Docker: Use playwright base image (already included)

**"Timeout errors"**
- Questions taking >3 minutes will fail
- Optimize LLM prompts for speed
- Consider using faster model (gpt-4o-mini)

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
uvicorn main:app --reload
```

## 🚢 Deployment

### Render.com

1. Connect GitHub repository
2. Create new Web Service
3. Set environment variables in dashboard
4. Deploy automatically on push

**Settings:**
- Build Command: `pip install -r requirements.txt && playwright install chromium`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Railway.app

```bash
railway login
railway init
railway up
```

### Fly.io

```bash
fly launch
fly secrets set AI_PIPE_KEY=xxx QUIZ_SECRET=yyy
fly deploy
```

## 📝 Project Structure

```
quiz-bot/
├── main.py              # FastAPI server
├── agent.py             # Quiz solving logic
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
├── .env.example         # Environment template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 🔐 Security Notes

- Never commit `.env` file
- Use strong, unique secrets
- API keys should be environment variables only
- Code execution is isolated but still review LLM outputs

## 📜 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## ❓ FAQ

**Q: Can I use OpenAI directly instead of AIpipe?**  
A: Yes! Just change `base_url` in `agent.py` to `https://api.openai.com/v1`

**Q: What if a question takes longer than 3 minutes?**  
A: The agent stops at 2:50 to ensure submission before deadline. Optimize prompts if needed.

**Q: Can I test without the evaluation server?**  
A: Yes, use the demo endpoint provided in testing section.

**Q: What LLM model should I use?**  
A: GPT-4o-mini is recommended for speed/cost balance. GPT-4 for complex tasks.

## 📞 Support

- Open GitHub Issue for bugs
- Check logs first: `docker logs <container-id>`
- Review troubleshooting section above

---

**Built for**: IIT Madras Tools in Data Science - Project 2  
**Date**: November 2025  
**Version**: 1.0.0