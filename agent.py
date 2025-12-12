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
RETRY_DELAY = 0.5


def run_quiz_task(url: str, email: str, secret: str, request_id: int = None):
    """Main quiz executor - handles entire quiz chain"""
    log_prefix = f"[{request_id}]" if request_id else ""
    start_time = time.time()
    max_duration = 400  # Increased for 20+ questions
    
    logger.info(f"{log_prefix} Starting quiz at: {url}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            current_url = url
            question_num = 0
            
            while current_url and question_num < 50:  # Support up to 50 questions
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
                    
                    # Retry loop - handles both code errors AND wrong answers
                    last_error = None
                    last_wrong_answer = None
                    last_wrong_reason = None
                    next_url = None
                    
                    for attempt in range(MAX_RETRIES):
                        answer_obj = solve_with_llm(
                            current_url, text, html, email, 
                            log_prefix, question_num, attempt, 
                            last_error, last_wrong_answer, last_wrong_reason
                        )
                        
                        if not answer_obj:
                            logger.warning(f"{log_prefix} Q{question_num} | Code failed, retry {attempt+1}/{MAX_RETRIES}")
                            time.sleep(1)
                            continue
                        
                        # Submit and check result
                        result = submit_answer(
                            current_url, email, secret, 
                            answer_obj, log_prefix, question_num
                        )
                        
                        if result['correct']:
                            next_url = result['next_url']
                            break
                        else:
                            # Wrong answer - retry with feedback
                            last_wrong_answer = str(answer_obj.get('answer', ''))
                            last_wrong_reason = result.get('reason', 'Unknown')
                            logger.info(f"{log_prefix} Q{question_num} | Retrying with feedback: {last_wrong_reason}")
                            
                            # If we got a next_url even with wrong answer, use it after retries exhausted
                            if result.get('next_url') and attempt == MAX_RETRIES - 1:
                                next_url = result['next_url']
                            
                            time.sleep(0.5)
                    
                    if not next_url:
                        logger.error(f"{log_prefix} Q{question_num} | Failed after {MAX_RETRIES} attempts")
                        # Try to continue if we have any next URL from wrong answers
                    
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
                   log_prefix: str, q_num: int, attempt: int = 0, 
                   last_error: str = None, last_wrong_answer: str = None, last_wrong_reason: str = None):
    """Use LLM to solve the question"""
    try:
        # Extract base URL properly - scheme + domain only
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        logger.info(f"{log_prefix} Q{q_num} | Base URL: {base_url}")
        
        error_context = ""
        if last_error:
            error_context = f"""

⚠️ PREVIOUS ATTEMPT FAILED WITH CODE ERROR:
{last_error}

FIX THE ERROR! Common solutions:
- Audio: Make sure to use .lower() on the transcribed text, keep spaces between words
- CSV: Strip whitespace from column names AND cell values
- 404: Check the file URL path - use urllib.parse.urljoin(base_url, '/correct/path')
- Import errors: Make sure imports are at the top of the code
- Type errors: Convert numpy/pandas types with int(), float(), or .item()
"""
        
        if last_wrong_answer and last_wrong_reason:
            error_context += f"""

🚫 PREVIOUS ANSWER WAS WRONG!
Your answer: {last_wrong_answer}
Server feedback: {last_wrong_reason}

TRY A DIFFERENT APPROACH! Common fixes based on feedback:
- "Link should be /path": Return ONLY the relative path, not full URL
- "Normalized JSON does not match": Check column names, data types, date formats, sorting
- "Total line items": Parse the PDF table correctly, multiply qty × price for each row
- Date format: Use ISO format YYYY-MM-DDTHH:MM:SS
- For paths: Extract just the path portion using urlparse(url).path
- "positive integers": Values like 0 are NOT positive! Check min_replicas constraint
- For shards/replicas: replicas MUST be >= min_replicas (NOT 0!)
- "option B/C": Return just the letter like "B", not the full text
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

⚠️ CRITICAL ANSWER FORMAT RULES (READ CAREFULLY):
- **FILE/LINK PATH ANSWERS**: Return ONLY the relative path like "/project2/file.md"
  ❌ WRONG: "https://domain.com/project2/file.md" (full URL)
  ✅ CORRECT: "/project2/file.md" (relative path starting with /)
- For CSV data: Sort by id, dates must be ISO format YYYY-MM-DDTHH:MM:SS
- For ZIP logs: Sum bytes where event='download', add (len(email) % 5) offset
- For audio: Output lowercase text with spaces

CRITICAL URL CONSTRUCTION RULES:
✅ CORRECT: Use urllib.parse.urljoin(base_url, '/path/to/file.ext')
✅ BASE_URL is: {base_url} (scheme + domain only)
✅ Always use absolute paths starting with /

❌ WRONG: Don't append base_url + '/project2/file' 
❌ WRONG: Don't use f"{{base_url}}/project2/file"

CODE TEMPLATE EXAMPLES:
(These show PATTERNS - adapt file paths based on what you find in PAGE TEXT/HTML above!)
(Look for actual file links like .mp3, .csv, .zip, .pdf, .json in the page content)

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

**2. CSV PROCESSING (Clean & Normalize):**
```python
import json
import pandas as pd
import requests
import urllib.parse
from io import StringIO
import dateutil.parser

base_url = '{base_url}'
csv_url = urllib.parse.urljoin(base_url, '/project2/messy.csv')

response = requests.get(csv_url)
df = pd.read_csv(StringIO(response.text))

# 1. Clean column names - strip whitespace only (keep original case)
df.columns = df.columns.str.strip()

# 2. Normalize column names to lowercase for processing
col_map = {{c: c.lower() for c in df.columns}}
df = df.rename(columns=col_map)

# 3. Strip whitespace from ALL string values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# 4. Convert 'id' to integer
if 'id' in df.columns:
    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)

# 5. Convert 'value' to integer
if 'value' in df.columns:
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int)

# 6. Parse and normalize dates to ISO format (YYYY-MM-DDTHH:MM:SS)
if 'joined' in df.columns:
    def normalize_date(x):
        try:
            dt = dateutil.parser.parse(str(x).strip())
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
        except:
            return str(x).strip()
    df['joined'] = df['joined'].apply(normalize_date)

# 7. SORT BY ID (ascending) - critical for matching expected output
df = df.sort_values('id').reset_index(drop=True)

# 8. Ensure exact column order: id, name, joined, value (lowercase)
if set(['id', 'name', 'joined', 'value']).issubset(set(df.columns)):
    df = df[['id', 'name', 'joined', 'value']]

# 9. Convert to list of dicts
answer = df.to_dict(orient='records')
print(json.dumps({{"answer": answer}}))
```

**3. ZIP FILE PROCESSING (logs with download bytes - JSONL format):**
```python
import json
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
        with z.open(filename) as f:
            content = f.read().decode('utf-8')
            # Handle JSONL (one JSON object per line)
            for line in content.strip().split('\\n'):
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get('event') == 'download':
                            total_bytes += record.get('bytes', 0)
                    except json.JSONDecodeError:
                        pass

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

**6. PDF INVOICE PROCESSING (calculate total from table):**
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

# PDF tables often extract as: Item, Quantity, UnitPrice on separate lines
# Parse all numbers from the text and identify qty/price pairs
lines = [l.strip() for l in text.split('\\n') if l.strip()]

# Method 1: Look for lines that are just numbers (qty and price)
numbers = []
for line in lines:
    # Skip header lines
    if any(h in line.lower() for h in ['item', 'quantity', 'unitprice', 'invoice', 'total']):
        continue
    # Check if line is a number
    try:
        num = float(line.replace(',', ''))
        numbers.append(num)
    except ValueError:
        pass

# Numbers come in pairs: quantity, unit_price
total = 0.0
for i in range(0, len(numbers) - 1, 2):
    qty = int(numbers[i])
    price = float(numbers[i + 1])
    total += qty * price

# Round to 2 decimal places
answer = round(total, 2)
print(json.dumps({{"answer": answer}}))
```

**7. MARKDOWN/LINK/PATH ANSWERS (MUST use relative paths):**
```python
import json

# ⚠️ CRITICAL: When a question asks for a file path, link, or URL:
# ALWAYS return the RELATIVE path starting with /
# NEVER return the full URL with domain!

# Example: If the file is at https://tds-llm-analysis.s-anand.net/project2/data-preparation.md
# ❌ WRONG: answer = "https://tds-llm-analysis.s-anand.net/project2/data-preparation.md"
# ✅ CORRECT: answer = "/project2/data-preparation.md"

# Just extract the path portion from any URL
from urllib.parse import urlparse
full_url = "https://example.com/project2/file.md"
path = urlparse(full_url).path  # Returns "/project2/file.md"
answer = path

print(json.dumps({{"answer": answer}}))
```

RULES:
1. ALWAYS use urllib.parse.urljoin(base_url, '/path') for FETCHING files
2. ALWAYS strip() whitespace from all text data
3. ALWAYS handle type conversions (str to int/float)
4. ⚠️ For ANSWER that is a path/link: return ONLY the relative path like "/project2/file.md"
   - Use urlparse(url).path to extract just the path from a full URL
5. Only print JSON with answer, NEVER submit via HTTP
6. For CSV: Sort by 'id' ascending, dates in ISO format YYYY-MM-DDTHH:MM:SS

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
4. Use urllib.parse.urljoin(base_url, '/path') for FETCHING file URLs
5. Find file paths (like .mp3, .csv, .pdf, .zip, .json) from the PAGE TEXT or HTML provided
6. Strip whitespace from ALL data
7. For audio: output lowercase text with spaces
8. For CSV: clean columns AND values, convert types properly, sort by id if present
9. Handle all exceptions gracefully
10. ⚠️ CRITICAL: If the question asks for a file path/link, return ONLY the relative path - NOT the full URL!
11. For optimization questions (shards/replicas): respect min_replicas constraint (replicas cannot be 0 if min_replicas > 0)"""
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
            'dateutil': __import__('dateutil'),
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
                    
                    # POST-PROCESSING: Convert full URLs to relative paths
                    # This fixes cases where LLM returns https://domain.com/path instead of /path
                    if isinstance(answer, str) and answer.startswith('http'):
                        parsed = urllib.parse.urlparse(answer)
                        # Only convert if it looks like a file path (has extension or ends with /)
                        if '.' in parsed.path.split('/')[-1] or parsed.path.endswith('/'):
                            answer = parsed.path
                            logger.info(f"{log_prefix} Q{q_num} | Converted URL to path: {answer}")
                    
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
    """Submit answer to server, returns dict with correct, reason, next_url"""
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
            return {'correct': False, 'reason': f'HTTP {resp.status_code}', 'next_url': None}
        
        try:
            data = resp.json()
            
            is_correct = data.get('correct', False)
            reason = data.get('reason', '')
            next_url = data.get('url')
            
            if is_correct:
                logger.info(f"{log_prefix} Q{q_num} | ✓ CORRECT!")
            else:
                logger.warning(f"{log_prefix} Q{q_num} | ✗ WRONG: {reason}")
            
            if next_url:
                logger.info(f"{log_prefix} Q{q_num} | Next URL received")
            
            return {'correct': is_correct, 'reason': reason, 'next_url': next_url}
            
        except json.JSONDecodeError:
            logger.error(f"{log_prefix} Q{q_num} | Failed to parse response")
            return {'correct': False, 'reason': 'JSON parse error', 'next_url': None}
            
    except Exception as e:
        logger.error(f"{log_prefix} Q{q_num} | Submit error: {e}")
        return {'correct': False, 'reason': str(e), 'next_url': None}