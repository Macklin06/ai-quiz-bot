from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from agent import run_quiz_task

app = FastAPI()

# Data model to validate the incoming request payload
class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.post("/webhook")
async def webhook_handler(task: QuizRequest, background_tasks: BackgroundTasks):
    print(f"[INFO] Request received from: {task.email}")
    print(f"[INFO] Target URL: {task.url}")
    
    # Verify the secret token
    # Note: Ensure this matches the secret submitted in the Google Form
    MY_SECRET = "12345" 
    if task.secret != MY_SECRET:
        print(f"[ERROR] Invalid secret. Denying access.")
        raise HTTPException(status_code=403, detail="Invalid credentials")

    # Dispatch the quiz agent as a background task
    # This ensures we return a 200 OK immediately to avoid timeouts
    background_tasks.add_task(run_quiz_task, task.url, task.email, task.secret)
    
    print(f"[INFO] Background task started.")
    return {"message": "Task accepted", "status": "processing"}