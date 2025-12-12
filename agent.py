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
RETRY_DELAY = 0.3  # Faster retries


def detect_question_type(text: str, html: str) -> str:
    """Detect question type to provide focused guidance"""
    text_lower = text.lower()
    html_lower = html.lower()
    combined = text_lower + html_lower
    
    if 'audio' in combined or '.mp3' in combined or '.wav' in combined or 'passphrase' in combined:
        return 'AUDIO'
    elif 'csv' in combined and ('normalize' in combined or 'json' in combined):
        return 'CSV'
    elif 'pdf' in combined or 'invoice' in combined:
        return 'PDF'
    elif 'heatmap' in combined or 'color' in combined:
        return 'IMAGE'
    elif 'github' in combined or 'tree' in combined:
        return 'GITHUB'
    elif '.zip' in combined or 'logs' in combined:
        return 'ZIP'
    elif 'markdown' in combined or '.md' in combined or ('link' in combined and 'file' in combined):
        return 'LINK'
    elif 'git' in text_lower and ('commit' in text_lower or 'add' in text_lower):
        return 'GIT'
    elif 'chart' in combined or 'visualization' in combined or ('option' in combined and ('a.' in combined or 'b.' in combined)):
        return 'CHART'
    elif 'shard' in combined or 'replica' in combined:
        return 'SHARDS'
    elif 'cache' in combined or 'actions' in combined:
        return 'CACHE'
    elif 'order' in combined and 'customer' in combined:
        return 'ORDERS'
    elif 'uv' in text_lower and 'http' in text_lower:
        return 'UV'
    elif '.xlsx' in combined or 'excel' in combined:
        return 'EXCEL'
    elif 'api' in combined and 'page' in combined:
        return 'PAGINATION'
    elif 'xml' in combined:
        return 'XML'
    else:
        return 'GENERAL'


def run_quiz_task(url: str, email: str, secret: str, request_id: int = None):
    """Main quiz executor - handles entire quiz chain"""
    log_prefix = f"[{request_id}]" if request_id else ""
    start_time = time.time()
    max_duration = 400  # Increased timeout for 20+ questions
    
    logger.info(f"{log_prefix} Starting quiz at: {url}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            current_url = url
            question_num = 0
            
            while current_url and question_num < 50:
                elapsed = time.time() - start_time
                if elapsed > max_duration:
                    logger.error(f"{log_prefix} Timeout after {elapsed:.1f}s")
                    break
                
                question_num += 1
                remaining = max_duration - elapsed
                logger.info(f"{log_prefix} Q{question_num} | Time left: {remaining:.1f}s | URL: {current_url}")
                
                try:
                    page.goto(current_url, timeout=30000, wait_until="networkidle")
                    page.wait_for_timeout(500)  # Reduced for speed
                    
                    text = page.locator("body").inner_text()
                    html = page.content()
                    
                    logger.info(f"{log_prefix} Q{question_num} | Extracted {len(text)} chars")
                    
                    # Detect question type for better prompting
                    q_type = detect_question_type(text, html)
                    logger.info(f"{log_prefix} Q{question_num} | Type: {q_type}")
                    
                    # Retry loop - handles both code errors AND wrong answers
                    last_error = None
                    last_wrong_answer = None
                    last_wrong_reason = None
                    next_url = None
                    
                    for attempt in range(MAX_RETRIES):
                        answer_obj = solve_with_llm(
                            current_url, text, html, email, 
                            log_prefix, question_num, attempt, 
                            last_error, last_wrong_answer, last_wrong_reason, q_type
                        )
                        
                        if not answer_obj:
                            logger.warning(f"{log_prefix} Q{question_num} | Code failed, retry {attempt+1}/{MAX_RETRIES}")
                            time.sleep(RETRY_DELAY)
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
                            
                            time.sleep(RETRY_DELAY)
                    
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
                   last_error: str = None, last_wrong_answer: str = None, 
                   last_wrong_reason: str = None, q_type: str = 'GENERAL'):
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
- "positive integers": Values like 0 are NOT positive! Check min/max constraints in the JSON
- "shards and replicas": Check min_replicas and max_replicas constraints - replicas cannot be 0
- "option B/C/etc": The answer should be just the letter like "B", not the full text
"""

        # Add question-type-specific hints
        type_hints = ""
        if q_type == "CSV":
            type_hints = "\n🎯 CSV QUESTION: Use the CSV PROCESSING template. Sort by id if present, normalize dates to ISO format.\n"
        elif q_type == "AUDIO":
            type_hints = "\n🎯 AUDIO QUESTION: Use the AUDIO TRANSCRIPTION template. Output lowercase text with spaces.\n"
        elif q_type == "PDF":
            type_hints = "\n🎯 PDF QUESTION: Use the PDF PROCESSING template. Parse tables and calculate as needed.\n"
        elif q_type == "ZIP":
            type_hints = "\n🎯 ZIP QUESTION: Use the ZIP FILE PROCESSING template. May contain JSONL, CSV, or other formats.\n"
        elif q_type == "GITHUB":
            type_hints = "\n🎯 GITHUB QUESTION: Use the GITHUB API template. Read config from JSON and call GitHub API.\n"
        elif q_type == "IMAGE":
            type_hints = "\n🎯 IMAGE QUESTION: Use the IMAGE PROCESSING template. Return color in appropriate format.\n"
        elif q_type == "SHARDS":
            type_hints = "\n🎯 OPTIMIZATION QUESTION: Read constraints from config, calculate optimal values respecting min/max limits.\n"
        elif q_type == "CHART":
            type_hints = "\n🎯 MULTIPLE CHOICE: Return just the option letter like 'B' or 'C', NOT the full description.\n"
        elif q_type == "LINK":
            type_hints = "\n🎯 LINK QUESTION: Return ONLY the relative path like '/path/to/file.md', NOT the full URL.\n"
        elif q_type == "EXCEL":
            type_hints = "\n🎯 EXCEL QUESTION: Use pd.read_excel() with BytesIO. May have multiple sheets.\n"
        elif q_type == "PAGINATION":
            type_hints = "\n🎯 PAGINATION QUESTION: Loop through pages until empty response. Aggregate all results.\n"
        elif q_type == "XML":
            type_hints = "\n🎯 XML QUESTION: Use xml.etree.ElementTree or lxml to parse. Use find/findall for elements.\n"

        prompt = f"""Generate Python code to solve this quiz question.

⛔ DO NOT SUBMIT ANSWERS IN YOUR CODE ⛔
Your code calculates the answer and prints JSON. That's it.
{type_hints}
URL: {url}
BASE_URL: {base_url}
EMAIL: {email}
ATTEMPT: {attempt + 1} of {MAX_RETRIES}
DETECTED QUESTION TYPE: {q_type}

PAGE TEXT:
{text[:15000]}

HTML (if needed):
{html[:5000]}
{error_context}

⚠️ CRITICAL ANSWER FORMAT RULES (READ CAREFULLY):
- **FILE/LINK PATH ANSWERS**: Return ONLY the relative path like "/path/to/file.md"
  ❌ WRONG: "https://domain.com/path/file.md" (full URL)
  ✅ CORRECT: "/path/to/file.md" (relative path starting with /)
- For CSV data: Sort by id, dates must be ISO format YYYY-MM-DDTHH:MM:SS
- For ZIP logs: Sum bytes where appropriate event type, add (len(email) % 5) offset if specified
- For audio: Output lowercase text with spaces

CRITICAL URL CONSTRUCTION RULES:
✅ CORRECT: Use urllib.parse.urljoin(base_url, '/path/to/file.ext')
✅ BASE_URL is: {base_url} (scheme + domain only)
✅ Always use absolute paths starting with /
✅ EXTRACT file paths from the page HTML/text - look for links, hrefs, file references

❌ WRONG: Don't append base_url + '/hardcoded/path' 
❌ WRONG: Don't use f"{{base_url}}/hardcoded/path"
❌ WRONG: Don't hardcode file paths - extract them from the page!

CODE TEMPLATE EXAMPLES:

**HOW TO EXTRACT FILE PATHS FROM PAGE:**
```python
import re
from bs4 import BeautifulSoup

# Method 1: Use regex to find file paths in page text
# Find .mp3, .csv, .zip, .pdf, .json, .xlsx, .png files
file_pattern = r'(/[^\s"\'<>]+\.(?:mp3|wav|csv|zip|pdf|json|xlsx|png|md))'
matches = re.findall(file_pattern, page_html)
if matches:
    file_path = matches[0]  # Use first match

# Method 2: Parse HTML for links
soup = BeautifulSoup(page_html, 'html.parser')
for link in soup.find_all('a', href=True):
    href = link['href']
    if href.endswith(('.mp3', '.csv', '.zip', '.pdf')):
        file_path = href
        break
```

**1. AUDIO TRANSCRIPTION (Secret Passphrase):**
```python
import json
import speech_recognition as sr
from pydub import AudioSegment
import urllib.request
import urllib.parse
import re

base_url = '{base_url}'
page_html = '''{html[:2000]}'''

# Extract audio file path from page
audio_pattern = r'(/[^\s"\'<>]+\.(?:mp3|wav|opus|ogg))'
matches = re.findall(audio_pattern, page_html)
audio_path = matches[0] if matches else '/audio.mp3'

file_url = urllib.parse.urljoin(base_url, audio_path)
urllib.request.urlretrieve(file_url, "audio_file")

audio = AudioSegment.from_file("audio_file")
audio.export("audio.wav", format="wav")

recognizer = sr.Recognizer()
with sr.AudioFile("audio.wav") as source:
    recognizer.adjust_for_ambient_noise(source, duration=0.3)
    audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data)

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
import re

base_url = '{base_url}'
page_html = '''{html[:2000]}'''

# Extract CSV file path from page
csv_pattern = r'(/[^\s"\'<>]+\.csv)'
matches = re.findall(csv_pattern, page_html)
csv_path = matches[0] if matches else '/data.csv'

csv_url = urllib.parse.urljoin(base_url, csv_path)
response = requests.get(csv_url)
df = pd.read_csv(StringIO(response.text))

# Clean and normalize
df.columns = df.columns.str.strip().str.lower()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

if 'id' in df.columns:
    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0).astype(int)
if 'value' in df.columns:
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(int)
if 'joined' in df.columns:
    def normalize_date(x):
        try:
            dt = dateutil.parser.parse(str(x).strip())
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
        except:
            return str(x).strip()
    df['joined'] = df['joined'].apply(normalize_date)

if 'id' in df.columns:
    df = df.sort_values('id').reset_index(drop=True)

cols_to_use = [c for c in ['id', 'name', 'joined', 'value'] if c in df.columns]
if cols_to_use:
    df = df[cols_to_use]

records = df.to_dict(orient='records')
answer = json.dumps(records, ensure_ascii=False)
print(json.dumps({{"answer": answer}}, ensure_ascii=False))
```

**3. ZIP FILE PROCESSING:**
```python
import json
import requests
import zipfile
from io import BytesIO
import urllib.parse
import re

base_url = '{base_url}'
page_html = '''{html[:2000]}'''

# Extract ZIP file path from page
zip_pattern = r'(/[^\s"\'<>]+\.zip)'
matches = re.findall(zip_pattern, page_html)
zip_path = matches[0] if matches else '/logs.zip'

zip_url = urllib.parse.urljoin(base_url, zip_path)
response = requests.get(zip_url)
total_bytes = 0

with zipfile.ZipFile(BytesIO(response.content)) as z:
    for filename in z.namelist():
        with z.open(filename) as f:
            content = f.read().decode('utf-8')
            for line in content.strip().split('\\n'):
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get('event') == 'download':
                            total_bytes += record.get('bytes', 0)
                    except json.JSONDecodeError:
                        pass

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
import re

base_url = '{base_url}'
page_html = '''{html[:2000]}'''

# Extract JSON config path from page
json_pattern = r'(/[^\s"\'<>]+\.json)'
matches = re.findall(json_pattern, page_html)
json_path = matches[0] if matches else '/config.json'

json_url = urllib.parse.urljoin(base_url, json_path)
response = requests.get(json_url)
config = response.json()

gh_url = f"https://api.github.com/repos/{{config['owner']}}/{{config['repo']}}/git/trees/{{config['sha']}}?recursive=1"
tree_response = requests.get(gh_url)
tree_data = tree_response.json()

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
import re

base_url = '{base_url}'
page_html = '''{html[:2000]}'''

# Extract image path from page
img_pattern = r'(/[^\s"\'<>]+\.(?:png|jpg|jpeg|gif))'
matches = re.findall(img_pattern, page_html)
img_path = matches[0] if matches else '/image.png'

image_url = urllib.parse.urljoin(base_url, img_path)
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

colors = list(image.getdata())
most_common = Counter(colors).most_common(1)[0][0]
answer = '#{{:02x}}{{:02x}}{{:02x}}'.format(*most_common[:3])
print(json.dumps({{"answer": answer}}))
```

**6. PDF INVOICE PROCESSING:**
```python
import json
import requests
import PyPDF2
from io import BytesIO
import urllib.parse
import re

base_url = '{base_url}'
page_html = '''{html[:2000]}'''

# Extract PDF path from page
pdf_pattern = r'(/[^\s"\'<>]+\.pdf)'
matches = re.findall(pdf_pattern, page_html)
pdf_path = matches[0] if matches else '/invoice.pdf'

pdf_url = urllib.parse.urljoin(base_url, pdf_path)
response = requests.get(pdf_url)
pdf_reader = PyPDF2.PdfReader(BytesIO(response.content))

text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# Extract numbers and calculate total (qty * price pairs)
lines = [l.strip() for l in text.split('\\n') if l.strip()]
numbers = []
for line in lines:
    if any(h in line.lower() for h in ['item', 'quantity', 'unitprice', 'invoice', 'total']):
        continue
    try:
        num = float(line.replace(',', ''))
        numbers.append(num)
    except ValueError:
        pass

total = 0.0
for i in range(0, len(numbers) - 1, 2):
    qty = int(numbers[i])
    price = float(numbers[i + 1])
    total += qty * price

answer = round(total, 2)
print(json.dumps({{"answer": answer}}))
```

**7. PATH/LINK ANSWERS:**
```python
import json
from urllib.parse import urlparse

# When question asks for a file path/link, return ONLY the relative path
# ❌ WRONG: "https://example.com/path/file.md"
# ✅ CORRECT: "/path/file.md"

full_url = "https://example.com/some/path/file.md"
answer = urlparse(full_url).path
print(json.dumps({{"answer": answer}}))
```

**8. OPTIMIZATION (Shards/Resources):**
```python
import json
import requests
import urllib.parse
import math
import re

base_url = '{base_url}'
page_html = '''{html[:2000]}'''

# Extract config JSON path
json_pattern = r'(/[^\s"\'<>]+\.json)'
matches = re.findall(json_pattern, page_html)
json_path = matches[0] if matches else '/config.json'

config_url = urllib.parse.urljoin(base_url, json_path)
response = requests.get(config_url)
config = response.json()

dataset_size = config.get('dataset_size', 0)
max_docs_per_shard = config.get('max_docs_per_shard', 0)
max_shards = config.get('max_shards', 10)
min_replicas = config.get('min_replicas', 1)
max_replicas = config.get('max_replicas', 3)
memory_per_shard_gb = config.get('memory_per_shard_gb', 0)
memory_budget_gb = config.get('memory_budget_gb', 0)

min_shards = math.ceil(dataset_size / max_docs_per_shard)
best_shards, best_replicas = None, None

for shards in range(min_shards, max_shards + 1):
    for replicas in range(min_replicas, max_replicas + 1):
        total_memory = shards * (replicas + 1) * memory_per_shard_gb
        if total_memory <= memory_budget_gb:
            if best_shards is None or shards < best_shards:
                best_shards, best_replicas = shards, replicas
            break

if best_shards is None:
    best_shards, best_replicas = min_shards, min_replicas

answer = {{"shards": best_shards, "replicas": best_replicas}}
print(json.dumps({{"answer": answer}}))
```

RULES:
1. ALWAYS use urllib.parse.urljoin(base_url, '/path') for FETCHING files
2. ALWAYS strip() whitespace from all text data
3. ALWAYS handle type conversions (str to int/float)
4. ⚠️ For ANSWER that is a path/link: return ONLY the relative path like "/path/to/file.md"
   - Use urlparse(url).path to extract just the path from a full URL
5. Only print JSON with answer, NEVER submit via HTTP
6. For CSV: Sort by 'id' ascending, dates in ISO format YYYY-MM-DDTHH:MM:SS
7. Use regex to extract file paths from page HTML - look for patterns like /path/file.ext

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
5. Strip whitespace from ALL data
6. For audio: output lowercase text with spaces
7. For CSV: clean columns AND values, convert types properly, sort by id
8. Handle all exceptions gracefully
9. ⚠️ CRITICAL: If the question asks for a file path/link, return ONLY the relative path - NOT the full URL!
10. EXTRACT file paths from page HTML/text - look for links, hrefs, src attributes. Don't hardcode paths!"""
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