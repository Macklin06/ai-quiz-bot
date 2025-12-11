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
        # Extract base URL properly - scheme + domain only
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        logger.info(f"{log_prefix} Q{q_num} | Base URL: {base_url}")
        
        error_context = ""
        if last_error:
            error_context = f"""

⚠️ PREVIOUS ATTEMPT FAILED WITH ERROR:
{last_error}

FIX THE ERROR! Common solutions:
- Audio: Make sure to use .lower() on the transcribed text, keep spaces between words
- CSV: Strip whitespace from column names AND cell values
- 404: Check the file URL path - use urllib.parse.urljoin(base_url, '/correct/path')
- Import errors: Make sure imports are at the top of the code
- Type errors: Convert numpy/pandas types with int(), float(), or .item()
"""

        prompt = f"""Generate Python code to solve this quiz question.

⛔ DO NOT SUBMIT ANSWERS IN YOUR CODE ⛔
Your code calculates the answer and prints JSON. That's it.

URL: {url}
BASE_URL: {base_url}
EMAIL: {email}
ATTEMPT: {attempt + 1} of {MAX_RETRIES}

PAGE TEXT:
{text[:15000]}

HTML (if needed):
{html[:5000]}
{error_context}

IMPORTANT ANSWER FORMAT RULES:
- For file/link paths: Use RELATIVE paths like "/project2/file.md" (not full URLs)
- For CSV data: Clean whitespace, parse dates as ISO format, sort by id
- For ZIP logs: Sum bytes where event='download', add (len(email) % 5) offset
- For audio: Output lowercase text with spaces

CRITICAL URL CONSTRUCTION RULES:
✅ CORRECT: Use urllib.parse.urljoin(base_url, '/path/to/file.ext')
✅ BASE_URL is: {base_url} (scheme + domain only)
✅ Always use absolute paths starting with /

❌ WRONG: Don't append base_url + '/project2/file' 
❌ WRONG: Don't use f"{{base_url}}/project2/file"

CODE TEMPLATE EXAMPLES:

**1. AUDIO TRANSCRIPTION (Secret Passphrase):**
```python
import json
import speech_recognition as sr
from pydub import AudioSegment
import urllib.request
import urllib.parse

base_url = '{base_url}'
# Look for audio file link in the page - common extensions: .mp3, .wav, .opus, .ogg
file_url = urllib.parse.urljoin(base_url, '/project2/audio-passphrase.mp3')
urllib.request.urlretrieve(file_url, "audio_file")

# Convert to WAV for speech recognition
audio = AudioSegment.from_file("audio_file")
audio.export("audio.wav", format="wav")

recognizer = sr.Recognizer()
with sr.AudioFile("audio.wav") as source:
    recognizer.adjust_for_ambient_noise(source, duration=0.3)
    audio_data = recognizer.record(source)
    # Use Google's speech recognition
    text = recognizer.recognize_google(audio_data)

# Clean up the transcription - passphrase is usually lowercase words + number
# Format: "word word number" like "hushed parrot 219"
answer = text.strip().lower()

print(json.dumps({{"answer": answer}}))
```

**2. CSV PROCESSING (Clean & Robust):**
```python
import json
import pandas as pd
import requests
import urllib.parse
from io import StringIO

base_url = '{base_url}'
csv_url = urllib.parse.urljoin(base_url, '/project2/messy.csv')

response = requests.get(csv_url)
df = pd.read_csv(StringIO(response.text))

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Clean all string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Handle specific columns based on expected structure
if 'id' in df.columns:
    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
if 'value' in df.columns:
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int)
if 'joined' in df.columns:
    # Parse dates with explicit format handling
    df['joined'] = pd.to_datetime(df['joined'], format='mixed', dayfirst=False, errors='coerce')
    df['joined'] = df['joined'].dt.strftime('%Y-%m-%dT%H:%M:%S')

# Sort by id if column exists
if 'id' in df.columns:
    df = df.sort_values('id').reset_index(drop=True)

# Ensure column order matches expected: id, name, joined, value
if set(['id', 'name', 'joined', 'value']).issubset(df.columns):
    df = df[['id', 'name', 'joined', 'value']]

answer = df.to_dict(orient='records')
print(json.dumps({{"answer": answer}}))
```

**3. ZIP FILE PROCESSING (logs with download bytes):**
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
total_bytes = 0

with zipfile.ZipFile(BytesIO(response.content)) as z:
    for filename in z.namelist():
        if filename.endswith('.csv'):
            with z.open(filename) as f:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip().str.lower()
                
                # Clean string columns
                for col in df.select_dtypes(include='object').columns:
                    df[col] = df[col].str.strip()
                
                # Sum bytes for download events
                if 'event' in df.columns and 'bytes' in df.columns:
                    df['bytes'] = pd.to_numeric(df['bytes'], errors='coerce').fillna(0)
                    download_bytes = df[df['event'] == 'download']['bytes'].sum()
                    total_bytes += download_bytes

# IMPORTANT: Add email-length mod 5 offset
email = "{email}"
offset = len(email) % 5
answer = int(total_bytes) + offset

print(json.dumps({{"answer": answer}}))
```

**4. GITHUB API (count files in tree):**
```python
import json
import requests
import urllib.parse

base_url = '{base_url}'
json_url = urllib.parse.urljoin(base_url, '/project2/gh-tree.json')

response = requests.get(json_url)
config = response.json()

# Call GitHub API to get tree
gh_url = f"https://api.github.com/repos/{{config['owner']}}/{{config['repo']}}/git/trees/{{config['sha']}}?recursive=1"
tree_response = requests.get(gh_url)
tree_data = tree_response.json()

# Filter by pathPrefix and extension from config
path_prefix = config.get('pathPrefix', '')
extension = config.get('extension', '')

count = sum(1 for item in tree_data.get('tree', [])
           if item.get('path', '').startswith(path_prefix) and 
           item.get('path', '').endswith(extension))

answer = count
print(json.dumps({{"answer": answer}}))
```

**5. IMAGE PROCESSING (find most common color):**
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

# Format as hex color (lowercase)
answer = '#{{:02x}}{{:02x}}{{:02x}}'.format(*most_common[:3])

print(json.dumps({{"answer": answer}}))
```

**6. PDF PROCESSING (extract and sum values):**
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

# Extract quantities and prices, calculate total
# Look for patterns like "qty x $price" or table rows
total = 0.0
lines = text.split('\\n')
for line in lines:
    # Try to match "quantity x price" or similar patterns
    match = re.search(r'(\\d+)\\s*[xX×]?\\s*\\$?([\\d,]+\\.?\\d*)', line)
    if match:
        qty = int(match.group(1))
        price = float(match.group(2).replace(',', ''))
        total += qty * price

answer = round(total, 2)
print(json.dumps({{"answer": answer}}))
```

**7. MARKDOWN/LINK ANSWERS (use relative paths):**
```python
import json

# For questions asking for file paths or links, use RELATIVE paths
# NOT full URLs like https://domain.com/path
answer = "/project2/data-preparation.md"

print(json.dumps({{"answer": answer}}))
```

RULES:
1. ALWAYS use urllib.parse.urljoin(base_url, '/path') for file URLs
2. ALWAYS strip() whitespace from all text data
3. ALWAYS handle type conversions (str to int/float)
4. For link/path answers: use RELATIVE paths like "/project2/file.md"
5. Only print JSON with answer, NEVER submit via HTTP

Generate ONLY Python code, no markdown blocks."""

        logger.info(f"{log_prefix} Q{q_num} | Calling {SOLVER_MODEL} (attempt {attempt+1})...")
        
        response = client.chat.completions.create(
            model=SOLVER_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a Python code generator that solves quiz questions.

RULES:
1. Generate ONLY Python code - no explanations
2. Print answer as JSON: print(json.dumps({"answer": result}))
3. NEVER submit via HTTP POST - just calculate and print
4. Use urllib.parse.urljoin(base_url, '/path') for file URLs
5. Strip whitespace from ALL data
6. For audio: output lowercase text with spaces (e.g., "hushed parrot 219")
7. For CSV: clean columns AND values, convert types properly
8. Handle all exceptions gracefully"""
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
        
        # Log the generated code for debugging
        logger.debug(f"{log_prefix} Q{q_num} | Code:\n{code[:2000]}\n")
        
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
            'zipfile': __import__('zipfile'),
            'StringIO': __import__('io').StringIO,
            'BytesIO': __import__('io').BytesIO,
            'Counter': __import__('collections').Counter,
            'defaultdict': __import__('collections').defaultdict,
            'datetime': __import__('datetime'),
            'time': __import__('time'),
            'pd': __import__('pandas'),
        }
        
        # Add PIL with proper structure so both 'from PIL import Image' and 'PIL.Image' work
        try:
            from PIL import Image
            import PIL
            safe_env['Image'] = Image
            safe_env['PIL'] = PIL
        except ImportError:
            pass
        
        # Add other optional imports
        try:
            from bs4 import BeautifulSoup
            safe_env['BeautifulSoup'] = BeautifulSoup
        except ImportError:
            pass
        
        try:
            import speech_recognition as sr
            safe_env['sr'] = sr
            safe_env['speech_recognition'] = sr
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
        
        if not output.strip():
            return None, "No output produced"
        
        try:
            output = output.strip()
            
            # Check if code is submitting (wrong!)
            if '"correct"' in output.lower() and '"reason"' in output.lower():
                return None, "Code is submitting answers. DO NOT use requests.post(). Only print answer JSON."
            
            # Extract JSON
            start = output.find('{')
            end = output.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = output[start:end]
                data = json.loads(json_str)
                
                # Accept either 'answer' key or just use the data if it has what we need
                if 'answer' in data:
                    answer = data['answer']
                    
                    # Convert numpy types to Python types
                    if hasattr(answer, 'item'):
                        answer = answer.item()
                    elif hasattr(answer, 'tolist'):
                        answer = answer.tolist()
                    
                    data['answer'] = answer
                    logger.info(f"{log_prefix} Q{q_num} | Answer: {str(answer)[:200]} (type: {type(answer).__name__})")
                    return data, None
                else:
                    return None, "JSON missing 'answer' field"
            else:
                return None, "No JSON in output"
                
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"
            
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        import traceback
        tb = traceback.format_exc()
        logger.error(f"{log_prefix} Q{q_num} | Error: {e}")
        return None, f"{e}"


def submit_answer(current_url: str, email: str, secret: str, answer_obj: dict, 
                 log_prefix: str, q_num: int):
    """Submit answer to server"""
    try:
        answer = answer_obj['answer']
        
        # Always use the correct submit endpoint - don't trust LLM's submit_url
        parsed = urllib.parse.urlparse(current_url)
        submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
        
        payload = {
            "email": email,
            "secret": secret,
            "url": current_url,
            "answer": answer
        }
        
        logger.info(f"{log_prefix} Q{q_num} | Submitting to: {submit_url}")
        
        resp = requests.post(submit_url, json=payload, timeout=30)
        
        if resp.status_code != 200:
            logger.error(f"{log_prefix} Q{q_num} | HTTP {resp.status_code}")
            return None
        
        try:
            data = resp.json()
            
            if data.get('correct'):
                logger.info(f"{log_prefix} Q{q_num} | ✓ CORRECT!")
            else:
                reason = data.get('reason', '')
                logger.warning(f"{log_prefix} Q{q_num} | ✗ WRONG: {reason}")
            
            next_url = data.get('url')
            if next_url:
                logger.info(f"{log_prefix} Q{q_num} | Next URL received")
            
            return next_url
            
        except json.JSONDecodeError:
            logger.error(f"{log_prefix} Q{q_num} | Failed to parse response")
            return None
            
    except Exception as e:
        logger.error(f"{log_prefix} Q{q_num} | Submit error: {e}")
        return None