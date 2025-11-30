import os
import sys
import io
import json
import urllib.parse
import requests
import time
import logging
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("AI_PIPE_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

class QuizTimeout(Exception):
    """Raised when quiz execution exceeds time limit"""
    pass

def run_quiz_task(url: str, email: str, secret: str, request_id: int = None):
    """
    Main quiz task executor
    
    Args:
        url: Initial quiz URL
        email: User email
        secret: Authentication secret
        request_id: Request tracking ID
    """
    log_prefix = f"[{request_id}]" if request_id else ""
    start_time = time.time()
    max_duration = 170  # 2:50 to leave buffer before 3:00 deadline
    
    logger.info(f"{log_prefix} Starting quiz chain at: {url}")
    
    try:
        with sync_playwright() as p:
            logger.info(f"{log_prefix} Launching browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = context.new_page()
            
            current_url = url
            question_count = 0
            max_questions = 50  # Safety limit
            
            while current_url and question_count < max_questions:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    logger.error(f"{log_prefix} Timeout! Elapsed: {elapsed:.1f}s")
                    raise QuizTimeout(f"Exceeded {max_duration}s limit")
                
                question_count += 1
                remaining = max_duration - elapsed
                logger.info(f"{log_prefix} Q{question_count} | Remaining: {remaining:.1f}s | URL: {current_url}")
                
                try:
                    # Fetch and parse question
                    page.goto(current_url, timeout=30000, wait_until="networkidle")
                    page.wait_for_selector("body", timeout=10000)
                    
                    visible_text = page.locator("body").inner_text()
                    html_content = page.content()
                    
                    logger.info(f"{log_prefix} Q{question_count} | Extracted {len(visible_text)} chars")
                    
                    # Generate solution
                    answer_data = solve_question(
                        current_url, 
                        visible_text, 
                        html_content,
                        log_prefix,
                        question_count
                    )
                    
                    if not answer_data:
                        logger.error(f"{log_prefix} Q{question_count} | Failed to generate solution")
                        break
                    
                    # Submit answer
                    next_url = submit_answer(
                        current_url,
                        email,
                        secret,
                        answer_data,
                        log_prefix,
                        question_count
                    )
                    
                    current_url = next_url
                    
                except PlaywrightTimeoutError as e:
                    logger.error(f"{log_prefix} Q{question_count} | Playwright timeout: {e}")
                    break
                except Exception as e:
                    logger.error(f"{log_prefix} Q{question_count} | Error: {e}", exc_info=True)
                    break
            
            browser.close()
            
            total_time = time.time() - start_time
            logger.info(f"{log_prefix} Completed! Questions: {question_count}, Time: {total_time:.1f}s")
            
    except QuizTimeout as e:
        logger.error(f"{log_prefix} {e}")
    except Exception as e:
        logger.error(f"{log_prefix} Fatal error: {e}", exc_info=True)


def solve_question(url: str, text: str, html: str, log_prefix: str, q_num: int):
    """
    Generate solution code using LLM
    
    Returns: dict with 'answer', 'submit_url', 'answer_key'
    """
    try:
        prompt = f"""You are an expert data analyst and Python programmer.

CURRENT PAGE URL: {url}

VISIBLE TEXT:
{text[:8000]}

HTML CONTENT:
{html[:12000]}

TASK:
1. Analyze the question carefully
2. Download any required files (use urllib.parse.urljoin for relative URLs)
3. Process data (CSV, PDF, JSON, images, etc.)
4. Calculate the correct answer
5. Extract the submission URL and answer key from the page

CRITICAL REQUIREMENTS:
- Use try-except blocks for all operations
- Use .get() for dictionary access to avoid KeyErrors
- Print debug info: "STEP 1:", "STEP 2:", etc.
- Convert numpy/pandas types to native Python (int, float, str, list)
- Handle missing data gracefully
- For file downloads, use: urllib.parse.urljoin('{url}', relative_path)

OUTPUT FORMAT (JSON only):
{{
    "answer": <your_calculated_answer>,
    "submit_url": "<extracted_submission_url>",
    "answer_key": "answer"
}}

Generate clean Python code that prints only the final JSON object."""

        logger.info(f"{log_prefix} Q{q_num} | Calling LLM...")
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a Python code generator. Output ONLY executable Python code, no explanations."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3000
        )
        
        generated_code = completion.choices[0].message.content
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        
        logger.info(f"{log_prefix} Q{q_num} | Code generated ({len(generated_code)} chars)")
        logger.debug(f"{log_prefix} Generated code:\n{generated_code[:500]}...")
        
        # Execute code safely
        result = execute_code(generated_code, log_prefix, q_num)
        return result
        
    except Exception as e:
        logger.error(f"{log_prefix} Q{q_num} | LLM error: {e}", exc_info=True)
        return None


def execute_code(code: str, log_prefix: str, q_num: int):
    """
    Execute generated code in isolated environment
    
    Returns: dict with answer data or None
    """
    try:
        # Redirect stdout
        old_stdout = sys.stdout
        redirected = io.StringIO()
        sys.stdout = redirected
        
        # Create safe globals
        safe_globals = {
            '__builtins__': __builtins__,
            'urllib': __import__('urllib'),
            'requests': __import__('requests'),
            'json': __import__('json'),
            'pandas': __import__('pandas'),
            'numpy': __import__('numpy'),
            'PIL': __import__('PIL'),
            'io': __import__('io'),
            'base64': __import__('base64'),
        }
        
        # Execute with timeout
        exec(code, safe_globals)
        
        # Restore stdout
        sys.stdout = old_stdout
        output = redirected.getvalue()
        
        logger.info(f"{log_prefix} Q{q_num} | Code executed, output length: {len(output)}")
        logger.debug(f"{log_prefix} Raw output:\n{output}")
        
        # Parse JSON from output
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        
        if json_start == -1 or json_end == 0:
            logger.error(f"{log_prefix} Q{q_num} | No JSON found in output")
            return None
        
        json_str = output[json_start:json_end]
        result = json.loads(json_str)
        
        logger.info(f"{log_prefix} Q{q_num} | Parsed result: {result}")
        return result
        
    except Exception as e:
        sys.stdout = old_stdout
        logger.error(f"{log_prefix} Q{q_num} | Execution error: {e}", exc_info=True)
        return None


def submit_answer(current_url: str, email: str, secret: str, answer_data: dict, 
                 log_prefix: str, q_num: int):
    """
    Submit answer to server
    
    Returns: next URL or None
    """
    try:
        answer = answer_data.get("answer")
        submit_url = answer_data.get("submit_url")
        answer_key = answer_data.get("answer_key", "answer")
        
        if not submit_url:
            logger.error(f"{log_prefix} Q{q_num} | No submit URL found")
            return None
        
        # Make URL absolute
        if not submit_url.startswith("http"):
            submit_url = urllib.parse.urljoin(current_url, submit_url)
        
        # Prepare payload
        payload = {
            "email": email,
            "secret": secret,
            "url": current_url,
            answer_key: answer
        }
        
        logger.info(f"{log_prefix} Q{q_num} | Submitting to: {submit_url}")
        logger.debug(f"{log_prefix} Payload: {payload}")
        
        # Submit with timeout
        response = requests.post(submit_url, json=payload, timeout=30)
        
        logger.info(f"{log_prefix} Q{q_num} | Response: {response.status_code}")
        
        try:
            resp_data = response.json()
            logger.info(f"{log_prefix} Q{q_num} | Server response: {resp_data}")
            
            if resp_data.get("correct"):
                logger.info(f"{log_prefix} Q{q_num} | ✓ Correct!")
                return resp_data.get("url")
            else:
                reason = resp_data.get("reason", "Unknown")
                logger.warning(f"{log_prefix} Q{q_num} | ✗ Wrong: {reason}")
                # Return next URL if provided (skip option)
                return resp_data.get("url")
                
        except json.JSONDecodeError:
            logger.error(f"{log_prefix} Q{q_num} | Invalid JSON response: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"{log_prefix} Q{q_num} | Submit error: {e}", exc_info=True)
        return None