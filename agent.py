import os
import sys
import io
import json
import urllib.parse  # <--- NEW: For fixing relative links
import requests
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
                print(f"[INFO] Question text extracted: {visible_text[:100]}...")

                # --- UPDATED PROMPT: HANDLES RELATIVE URLS & PRINTING ---
                prompt = f"""
                You are an Autonomous AI Agent.
                
                CONTEXT:
                Current Page URL: {current_url}
                
                PAGE TEXT:
                {visible_text}
                
                HTML CONTENT:
                {content[:15000]}
                
                GOAL:
                1. Solve the question.
                2. Extract the SUBMISSION URL.
                3. Extract the JSON KEY (usually "answer").
                
                CRITICAL INSTRUCTIONS:
                - If the text mentions a file/link (like "/data.csv"), it might be a RELATIVE URL.
                - You MUST use `urllib.parse.urljoin('{current_url}', link)` in your Python code to construct the full URL before downloading.
                - Convert numpy/pandas types to standard Python types.
                - ALWAYS print the JSON object at the end.
                
                Format: {{"answer": <calculated_value>, "submit_url": "<extracted_url>", "answer_key": "<key>"}}
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
                
                print("[INFO] Executing generated solution code...")
                
                old_stdout = sys.stdout
                redirected_output = io.StringIO()
                sys.stdout = redirected_output
                
                try:
                    # Execute the code
                    exec(generated_code)
                    sys.stdout = old_stdout 
                    
                    output_str = redirected_output.getvalue().strip()
                    
                    start_idx = output_str.find("{")
                    end_idx = output_str.rfind("}") + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = output_str[start_idx:end_idx]
                        result_json = json.loads(json_str)
                        
                        calculated_answer = result_json.get("answer")
                        submit_url = result_json.get("submit_url")
                        answer_key = result_json.get("answer_key", "answer")
                        
                        print(f"[INFO] Calculated Result: {calculated_answer}")
                        print(f"[INFO] Submission Target: {submit_url}")

                        # SUBMIT
                        if submit_url:
                            # Fix relative submit URLs too
                            if not submit_url.startswith("http"):
                                submit_url = urllib.parse.urljoin(current_url, submit_url)

                            payload = {
                                "email": email,
                                "secret": secret,
                                "url": current_url,
                                answer_key: calculated_answer
                            }
                            response = requests.post(submit_url, json=payload)
                            print(f"[INFO] Server Response: {response.status_code} - {response.text}")
                            
                            try:
                                resp_data = response.json()
                                if resp_data.get("correct") == True:
                                    print("[SUCCESS] Answer correct. Checking for next task...")
                                    current_url = resp_data.get("url")
                                    if not current_url:
                                        print("[SUCCESS] All tasks completed. Exiting.")
                                else:
                                    print("[FAILURE] Answer rejected. Stopping.")
                                    break
                            except:
                                print("[ERROR] Invalid response format. Stopping.")
                                break
                        else:
                            print("[ERROR] No submission URL. Stopping.")
                            break
                    else:
                        print(f"[ERROR] No valid JSON found. AI Output: {output_str}")
                        break
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    print(f"[ERROR] Code Execution Failed: {e}")
                    print(f"[DEBUG] Failed Output: {output_str}")
                    break 

            except Exception as e:
                print(f"[CRITICAL] Unexpected error: {e}")
                break

        browser.close()