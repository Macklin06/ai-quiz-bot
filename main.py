import os
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from agent import run_quiz_task

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Quiz Bot", version="1.0")

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.get("/")
def root():
    return {"status": "ok", "service": "Quiz Bot"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "api_key_set": bool(os.getenv("AI_PIPE_KEY")),
        "secret_set": bool(os.getenv("QUIZ_SECRET"))
    }

@app.post("/webhook")
async def webhook(req: QuizRequest, bg: BackgroundTasks):
    logger.info(f"Received: {req.email} -> {req.url}")
    
    # Check secret
    expected = os.getenv("QUIZ_SECRET")
    if not expected:
        raise HTTPException(500, "Server config error")
    
    if req.secret != expected:
        logger.warning(f"Invalid secret from {req.email}")
        raise HTTPException(403, "Invalid credentials")
    
    # Start background task
    bg.add_task(run_quiz_task, req.url, req.email, req.secret, id(req))
    
    return {"status": "processing"}