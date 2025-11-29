import os
import sys
import io
import json
import urllib.parse
import requests
import re # Added for better URL extraction
from playwright.sync_api import sync_playwright
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("AI_PIPE_KEY"),
    base_url="https://aipipe.org/openai/v1" 
)

def run_quiz_task(url: str, email: str, secret: str):
    print(f"[INFO] Starting Task Chain at: {url}")
    
    with sync_playwright() as p:
        print("[INFO] Launching headless browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        current_url = url
        
        while current_url:
            print(f"\n[INFO] Navigating to: {current_url}")
            try:
                page.goto(current_url)
                page.wait_for_selector("body")
                
                visible_text = page.locator("body").inner_text()
                content = page.content()
                print(f"[INFO] Text Len: {len(visible_text)}")

                # --- SCORCHED EARTH PROMPT ---
                prompt = f"""
                You are an Autonomous AI Agent in a LIVE EXAM.
                
                CONTEXT:
                Current URL: {current_url}
                Email: {email}
                
                PAGE TEXT:
                {visible_text}
                
                GOAL:
                1. EXTRACT the Submission URL (look for 'https://.../submit' or similar).
                2. SOLVE the question. 
                
                RULES:
                - **FORBIDDEN:** Do NOT output "your_answer_here". You MUST calculate the real value.
                - **REGEX:** Use `re.search` to find the submission URL if simple text search fails.
                - **PRINTING:** You MUST `print(json.dumps(result))` at the end.
                - **SAFETY:** Wrap logic in try/except.
                - **DATA:** Convert numpy types to python int/float.
                
                REQUIRED JSON OUTPUT:
                {{
                    "answer": <calculated_value>,
                    "submit_url": "<extracted_url>",
                    "answer_key": "answer"
                }}
                """

                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a coding assistant. Output ONLY raw python code."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                generated_code = completion.choices[0].message.content
                generated_code = generated_code.replace("```python", "").replace("```", "").strip()
                
                print("[INFO] Executing solution code...")
                
                old_stdout = sys.stdout
                redirected_output = io.StringIO()
                sys.stdout = redirected_output
                
                try:
                    exec(generated_code)
                    sys.stdout = old_stdout 
                    
                    output_str = redirected_output.getvalue().strip()
                    if not output_str:
                        print("[ERROR] No output from script.")
                        break

                    start_idx = output_str.find("{")
                    end_idx = output_str.rfind("}") + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = output_str[start_idx:end_idx]
                        result_json = json.loads(json_str)
                        
                        ans = result_json.get("answer")
                        sub = result_json.get("submit_url")
                        key = result_json.get("answer_key", "answer")
                        
                        # ANTI-LAZY GUARD
                        if str(ans) == "your_answer_here":
                            print("[ERROR] AI returned placeholder. Retrying loop might help.")
                            break
                        
                        print(f"[INFO] Calculated Result: {ans}")
                        print(f"[INFO] Submission Target: {sub}")

                        if sub:
                            if not sub.startswith("http"):
                                sub = urllib.parse.urljoin(current_url, sub)

                            payload = {
                                "email": email,
                                "secret": secret,
                                "url": current_url,
                                key: ans
                            }
                            response = requests.post(sub, json=payload)
                            print(f"[INFO] Server Response: {response.status_code} - {response.text}")
                            
                            try:
                                resp_data = response.json()
                                if resp_data.get("correct") == True:
                                    print("[SUCCESS] Answer correct. Checking next...")
                                    current_url = resp_data.get("url")
                                    if not current_url:
                                        print("[SUCCESS] COMPLETED.")
                                else:
                                    print("[FAILURE] Wrong answer. Stopping.")
                                    break
                            except:
                                print("[ERROR] Bad response format.")
                                break
                        else:
                            print("[ERROR] No submission URL found.")
                            break
                    else:
                        print(f"[ERROR] Invalid JSON. Output: {output_str}")
                        break
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    print(f"[ERROR] Execution Failed: {e}")
                    print(f"[DEBUG] Output: {output_str}")
                    break 

            except Exception as e:
                print(f"[CRITICAL] Error: {e}")
                break

        browser.close()