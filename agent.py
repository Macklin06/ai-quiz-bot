import os
import sys
import io
import json
import urllib.parse
import requests
import time
import logging
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("AI_PIPE_KEY"),
    base_url="https://aipipe.org/openai/v1"
)

SOLVER_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3


def run_quiz_task(url: str, email: str, secret: str, request_id: int = None):
    """Main quiz executor - handles entire quiz chain"""
    log_prefix = f"[{request_id}]" if request_id else ""
    start_time = time.time()
    max_duration = 170
    
    logger.info(f"{log_prefix} Starting quiz at: {url}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            current_url = url
            question_num = 0
            
            while current_url and question_num < 100:
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    logger.error(f"{log_prefix} Timeout after {elapsed:.1f}s")
                    break
                
                question_num += 1
                remaining = max_duration - elapsed
                logger.info(f"{log_prefix} Q{question_num} | Time left: {remaining:.1f}s | URL: {current_url}")
                
                try:
                    page.goto(current_url, timeout=30000, wait_until="networkidle")
                    page.wait_for_timeout(1000)
                    
                    text = page.locator("body").inner_text()
                    html = page.content()
                    
                    logger.info(f"{log_prefix} Q{question_num} | Extracted {len(text)} chars")
                    
                    answer_obj = None
                    last_error = None
                    
                    for attempt in range(MAX_RETRIES):
                        answer_obj = solve_with_llm(
                            current_url, text, html, email, 
                            log_prefix, question_num, attempt, last_error
                        )
                        if answer_obj:
                            break
                        logger.warning(f"{log_prefix} Q{question_num} | Retry {attempt+1}/{MAX_RETRIES}")
                        time.sleep(1)
                    
                    if not answer_obj:
                        logger.error(f"{log_prefix} Q{question_num} | Failed after {MAX_RETRIES} attempts")
                        break
                    
                    next_url = submit_answer(
                        current_url, email, secret, 
                        answer_obj, log_prefix, question_num
                    )
                    
                    current_url = next_url
                    
                except Exception as e:
                    logger.error(f"{log_prefix} Q{question_num} | Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    break
            
            browser.close()
            logger.info(f"{log_prefix} Completed {question_num} questions in {time.time()-start_time:.1f}s")
            
    except Exception as e:
        logger.error(f"{log_prefix} Fatal error: {e}", exc_info=True)


def solve_with_llm(url: str, text: str, html: str, email: str, 
                   log_prefix: str, q_num: int, attempt: int = 0, last_error: str = None):
    """Use LLM to solve the question"""
    try:
        base_url = url.split('?')[0]
        
        error_context = ""
        if last_error:
            error_context = f"""

⚠️ PREVIOUS ATTEMPT FAILED:
{last_error}

FIX THE ERROR! Common fixes:
- Audio wrong? Try: text.lower(), text.title(), text.upper(), or text.replace(' ', '')
- CSV wrong? Ensure: strip whitespace from values, exact column order, int types
- ZIP wrong? Use pd.read_csv(f) where f = z.open(filename)
- PDF wrong? Use regex to extract quantities and prices from text
- Always strip() ALL data, not just column names
"""

        prompt = f"""Generate Python code to solve this quiz question.

⛔ DO NOT SUBMIT ANSWERS IN YOUR CODE ⛔
Your code calculates the answer and prints JSON. That's it.
Another process will handle the actual submission.

URL: {url}
BASE_URL: {base_url}
EMAIL: {email}
ATTEMPT: {attempt + 1} of {MAX_RETRIES}

PAGE TEXT:
{text[:15000]}

HTML:
{html[:10000]}
{error_context}

🎯 YOUR ONLY JOB:
1. Read the question
2. Calculate/extract the answer
3. Print: print(json.dumps({{"answer": result, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))

❌ NEVER DO THIS:
- requests.post(submit_url, ...)
- Any HTTP POST to submit
- Do not include email/secret in your code
- Just calculate and print JSON

✅ DO THIS:
- Calculate the answer
- Print JSON with answer
- That's all!

CODE EXAMPLES:

**SIMPLE TEXT ANSWER:**
```python
import json
answer = "the_text_answer_from_page"
print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))
```

**COMMAND GENERATION:**
```python
import json
answer = "git add file.txt"
print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))
```

**CSV PROCESSING (MESSY DATA):**
```python
import json
import pandas as pd
import requests
from io import StringIO
import urllib.parse

base_url = '{base_url}'
csv_url = urllib.parse.urljoin(base_url, '/project2/messy.csv')
response = requests.get(csv_url)
df = pd.read_csv(StringIO(response.text))

# Strip ALL whitespace from column names AND values
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower()

# Strip string values too
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Parse dates - handle multiple formats
df['joined'] = pd.to_datetime(df['joined'], errors='coerce').dt.strftime('%Y-%m-%dT%H:%M:%S')
df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int)
df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)

# Sort and reset index
df = df.sort_values('id').reset_index(drop=True)

# Ensure exact column order: id, name, joined, value
df = df[['id', 'name', 'joined', 'value']]

answer = df.to_dict(orient='records')
print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))
```

**JSON WITH GITHUB API (tree structure):**
```python
import json
import requests
import urllib.parse

base_url = '{base_url}'
json_url = urllib.parse.urljoin(base_url, '/project2/gh-tree.json')
response = requests.get(json_url)
config = response.json()

# This has: owner, repo, sha, pathPrefix, extension
# Need to call GitHub API to get tree
gh_url = f"https://api.github.com/repos/{{config['owner']}}/{{config['repo']}}/git/trees/{{config['sha']}}?recursive=1"
tree_response = requests.get(gh_url)
tree_data = tree_response.json()

# Now filter by pathPrefix and extension
path_prefix = config.get('pathPrefix', '')
extension = config.get('extension', '')
count = sum(1 for item in tree_data.get('tree', [])
           if item.get('path', '').startswith(path_prefix) and 
           item.get('path', '').endswith(extension))

email = "{email}"
offset = len(email) % 2
answer = count + offset

print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))
```

**AUDIO TRANSCRIPTION (with retry logic):**
```python
import json
import speech_recognition as sr
from pydub import AudioSegment
import urllib.request
import urllib.parse

base_url = '{base_url}'
file_url = urllib.parse.urljoin(base_url, '/project2/audio-passphrase.opus')
urllib.request.urlretrieve(file_url, "audio.opus")

audio = AudioSegment.from_file("audio.opus")
audio.export("audio.wav", format="wav")

recognizer = sr.Recognizer()
with sr.AudioFile("audio.wav") as source:
    recognizer.adjust_for_ambient_noise(source, duration=0.5)
    audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data)

# ADAPTIVE FORMAT based on attempt number (RETRY STRATEGY)
attempt = {attempt + 1}  # Current attempt number
if attempt == 1:
    answer = text.strip()  # Try as-is first
elif attempt == 2:
    answer = text.strip().lower()  # Try lowercase
elif attempt == 3:
    answer = text.strip().title()  # Try Title Case
else:
    answer = text.strip().upper()  # Try UPPERCASE

print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))
```

**IMAGE PROCESSING:**
```python
import json
import requests
from PIL import Image
from collections import Counter
from io import BytesIO
import urllib.parse

base_url = '{base_url}'
image_url = urllib.parse.urljoin(base_url, '/project2/heatmap.png')
response = requests.get(image_url)

image = Image.open(BytesIO(response.content))
colors = list(image.getdata())
most_common = Counter(colors).most_common(1)[0][0]
answer = '#{{:02x}}{{:02x}}{{:02x}}'.format(*most_common)

print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))
```

**ZIP FILE PROCESSING:**
```python
import json
import pandas as pd
import requests
import zipfile
from io import BytesIO
import urllib.parse

base_url = '{base_url}'
zip_url = urllib.parse.urljoin(base_url, '/project2/logs.zip')
response = requests.get(zip_url)

total = 0
with zipfile.ZipFile(BytesIO(response.content)) as z:
    for filename in z.namelist():
        if filename.endswith('.csv'):
            with z.open(filename) as f:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip().str.lower()
                # Filter and sum based on question
                df_filtered = df[df['event'].str.strip() == 'download']
                total += df_filtered['bytes'].sum()

email = "{email}"
offset = len(email) % 5
answer = int(total + offset)

print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))
```

**PDF PROCESSING:**
```python
import json
import requests
import PyPDF2
from io import BytesIO
import urllib.parse
import re

base_url = '{base_url}'
pdf_url = urllib.parse.urljoin(base_url, '/project2/invoice.pdf')
response = requests.get(pdf_url)

pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Extract line items - look for patterns like "2 x $10.00"
# This depends on PDF structure - might need regex
lines = text.split('\\n')
total = 0.0
for line in lines:
    # Example: match "quantity price" patterns
    match = re.search(r'(\\d+)\\s+x?\\s*\\$?([\\d,.]+)', line)
    if match:
        qty = int(match.group(1))
        price = float(match.group(2).replace(',', ''))
        total += qty * price

answer = round(total, 2)
print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))
```

CRITICAL RULES:
- Always strip() column names for CSV
- For JSON with owner/repo/sha, call GitHub API
- Use base_url without query params for files
- Always json.dumps() the output
- Handle all exceptions

Generate ONLY Python code, no markdown blocks."""

        logger.info(f"{log_prefix} Q{q_num} | Calling {SOLVER_MODEL} (attempt {attempt+1})...")
        
        response = client.chat.completions.create(
            model=SOLVER_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": f"""You are a Python code generator. Your ONLY job is to calculate answers, NOT submit them.

🚨 NEVER EVER make HTTP POST requests in your code. NEVER call requests.post() to submit answers.

Your code should:
1. Calculate/extract the answer
2. Print ONLY: print(json.dumps({{"answer": answer, "submit_url": "https://tds-llm-analysis.s-anand.net/submit"}}))

CSV: Strip whitespace from columns AND values, ensure column order [id, name, joined, value]
GitHub: Call API with owner/repo/sha from JSON
ZIP: Use pd.read_csv(f) where f = z.open(filename)
PDF: Extract text, parse with regex
Audio (attempt {attempt+1}): Try text.strip(), text.lower(), text.title() based on attempt

NEVER submit answers via HTTP POST. Just calculate and print JSON."""
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,
            temperature=0
        )
        
        code = response.choices[0].message.content.strip()
        
        # Clean markdown
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 2:
                code = parts[1].strip()
                if code.startswith(('python', 'py')):
                    code = '\n'.join(code.split('\n')[1:])
        
        logger.info(f"{log_prefix} Q{q_num} | Generated {len(code)} chars of code")
        logger.info(f"{log_prefix} Code:\n{code}\n")
        
        result, error = execute_code(code, log_prefix, q_num)
        
        if error and attempt < MAX_RETRIES - 1:
            last_error = error
        
        return result
        
    except Exception as e:
        logger.error(f"{log_prefix} Q{q_num} | LLM error: {e}")
        return None


def execute_code(code: str, log_prefix: str, q_num: int):
    """Execute LLM-generated code safely"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        safe_env = {
            '__builtins__': __builtins__,
            'json': __import__('json'),
            'requests': __import__('requests'),
            'urllib': __import__('urllib'),
            'pandas': __import__('pandas'),
            'numpy': __import__('numpy'),
            'base64': __import__('base64'),
            'hashlib': __import__('hashlib'),
            're': __import__('re'),
            'io': __import__('io'),
            'sys': sys,
            'StringIO': __import__('io').StringIO,
            'BytesIO': __import__('io').BytesIO,
            'Counter': __import__('collections').Counter,
            'defaultdict': __import__('collections').defaultdict,
            'datetime': __import__('datetime'),
            'time': __import__('time'),
        }
        
        try:
            from PIL import Image
            safe_env['Image'] = Image
        except ImportError:
            pass
        
        try:
            from bs4 import BeautifulSoup
            safe_env['BeautifulSoup'] = BeautifulSoup
        except ImportError:
            pass
        
        try:
            import speech_recognition as sr
            safe_env['sr'] = sr
        except ImportError:
            pass
        
        try:
            from pydub import AudioSegment
            safe_env['AudioSegment'] = AudioSegment
        except ImportError:
            pass
        
        try:
            import PyPDF2
            safe_env['PyPDF2'] = PyPDF2
        except ImportError:
            pass
        
        exec(code, safe_env)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        
        if errors:
            logger.warning(f"{log_prefix} Q{q_num} | Stderr: {errors[:1000]}")
        
        logger.info(f"{log_prefix} Q{q_num} | Output: {output[:500]}")
        
        if not output.strip():
            return None, "No output produced"
        
        try:
            output = output.strip()
            
            # Check if output contains submission response (wrong!)
            if '"correct"' in output.lower() and '"reason"' in output.lower():
                return None, "Code is submitting answers instead of just calculating. DO NOT use requests.post() to submit. Only print the answer JSON."
            
            start = output.find('{')
            end = output.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = output[start:end]
                data = json.loads(json_str)
                
                if 'answer' in data and 'submit_url' in data:
                    answer = data['answer']
                    
                    if hasattr(answer, 'item'):
                        answer = answer.item()
                    elif hasattr(answer, 'tolist'):
                        answer = answer.tolist()
                    
                    data['answer'] = answer
                    logger.info(f"{log_prefix} Q{q_num} | Answer: {answer} (type: {type(answer).__name__})")
                    return data, None
                else:
                    return None, "JSON missing required fields"
            else:
                return None, "No JSON in output"
                
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"
            
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        import traceback
        tb = traceback.format_exc()
        logger.error(f"{log_prefix} Q{q_num} | Error: {e}\n{tb}")
        return None, f"{e}\n{tb}"


def submit_answer(current_url: str, email: str, secret: str, answer_obj: dict, 
                 log_prefix: str, q_num: int):
    """Submit answer to server"""
    try:
        answer = answer_obj['answer']
        submit_url = answer_obj['submit_url']
        
        if not submit_url.startswith('http'):
            submit_url = urllib.parse.urljoin(current_url, submit_url)
        
        payload = {
            "email": email,
            "secret": secret,
            "url": current_url,
            "answer": answer
        }
        
        logger.info(f"{log_prefix} Q{q_num} | Submitting to: {submit_url}")
        logger.info(f"{log_prefix} Q{q_num} | Answer: {answer}")
        
        resp = requests.post(submit_url, json=payload, timeout=30)
        logger.info(f"{log_prefix} Q{q_num} | Response {resp.status_code}: {resp.text[:500]}")
        
        if resp.status_code != 200:
            logger.error(f"{log_prefix} Q{q_num} | HTTP {resp.status_code}")
            return None
        
        try:
            data = resp.json()
            
            if data.get('correct'):
                logger.info(f"{log_prefix} Q{q_num} | ✓ CORRECT!")
            else:
                logger.warning(f"{log_prefix} Q{q_num} | ✗ WRONG: {data.get('reason', '')}")
            
            next_url = data.get('url')
            if next_url:
                logger.info(f"{log_prefix} Q{q_num} | Next: {next_url[:100]}")
            else:
                logger.info(f"{log_prefix} Q{q_num} | Quiz end")
            
            return next_url
            
        except json.JSONDecodeError:
            logger.error(f"{log_prefix} Q{q_num} | Failed to parse response")
            return None
            
    except Exception as e:
        logger.error(f"{log_prefix} Q{q_num} | Submit error: {e}")
        return None