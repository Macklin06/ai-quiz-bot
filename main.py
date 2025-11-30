import os
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, field_validator
from agent import run_quiz_task
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Quiz Agent",
    description="Automated quiz-solving agent for LLM Analysis Project",
    version="1.0.0"
)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Invalid URL format')
        return v

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "LLM Quiz Agent",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "playwright": "ready",
        "openai_configured": bool(os.getenv("AI_PIPE_KEY")),
        "secret_configured": bool(os.getenv("QUIZ_SECRET"))
    }

@app.post("/webhook")
async def webhook_handler(
    task: QuizRequest, 
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Main webhook endpoint for receiving quiz tasks
    
    Returns immediately with 200 OK, processes quiz in background
    """
    start_time = datetime.utcnow()
    request_id = id(request)
    
    logger.info(f"[{request_id}] Received request from: {task.email}")
    logger.info(f"[{request_id}] Target URL: {task.url}")
    
    # Verify secret token
    expected_secret = os.getenv("QUIZ_SECRET")
    if not expected_secret:
        logger.error(f"[{request_id}] QUIZ_SECRET environment variable not set!")
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error"
        )
    
    if task.secret != expected_secret:
        logger.warning(f"[{request_id}] Invalid secret from {task.email}")
        raise HTTPException(
            status_code=403, 
            detail="Invalid credentials"
        )
    
    # Dispatch background task
    background_tasks.add_task(
        run_quiz_task, 
        task.url, 
        task.email, 
        task.secret,
        request_id
    )
    
    logger.info(f"[{request_id}] Background task queued successfully")
    
    return {
        "message": "Task accepted",
        "status": "processing",
        "request_id": request_id,
        "timestamp": start_time.isoformat()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "error": "Internal server error",
        "message": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )