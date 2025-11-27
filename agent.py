import os
import sys
import io
import json
import requests
from playwright.sync_api import sync_playwright
from openai import OpenAI

# Initialize OpenAI client with the AI Pipe configuration
client = OpenAI(
    api_key=os.getenv("AI_PIPE_KEY"),
    base_url="https://aipipe.org/openai/v1" 
)

def run_quiz_task(url: str, email: str, secret: str):
    """
    Executes the autonomous agent loop.
    1. Scrapes the target URL using Playwright.
    2. Uses an LLM to generate Python code for data analysis.
    3. Executes the code locally to obtain a deterministic answer.
    4. Submits the answer and handles the next URL in the chain.
    """
    print(f"[INFO] Starting Task Chain at: {url}")
    
    with sync_playwright() as p:
        print("[INFO] Launching headless browser...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        current_url = url
        
        # Infinite loop to handle multi-step quizzes
        while current_url:
            print(f"\n[INFO] Navigating to: {current_url}")
            try:
                page.goto(current_url)
                # Wait for DOM to load to ensure JS execution (atob decoding) is complete
                page.wait_for_selector("body")
                
                visible_text = page.locator("body").inner_text()
                content = page.content()
                print(f"[INFO] Question text extracted: {visible_text[:100]}...")

                # --- LLM Prompt Construction ---
                # This implements the Code Interpreter pattern:
                # We ask the LLM to write code rather than guess the answer.
                prompt = f"""
                You are an Autonomous AI Agent.
                
                --- PAGE TEXT (Task & Instructions) ---
                {visible_text}
                
                --- HTML CONTENT (For file links) ---
                {content[:15000]}
                
                --- YOUR GOAL ---
                1. Solve the question found in the Page Text. 
                   - If it requires a CSV/PDF, write Python code to download and process it.
                2. Extract the SUBMISSION URL from the Page Text.
                3. Extract the JSON KEY expected for the answer (usually "answer").
                
                --- OUTPUT FORMAT ---
                Write a Python script.
                The script must print a valid JSON object at the very end.
                
                IMPORTANT: 
                - Convert all numpy/pandas types (int64, float64) to standard Python types (int, float) using .item() or int(). 
                - Standard JSON does not support numpy types.
                
                Format: {{"answer": <calculated_value>, "submit_url": "<extracted_url>", "answer_key": "<key>"}}
                """

                # Call the LLM (The "Reasoning" Step)
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
                
                # Capture standard output to read the script's result
                old_stdout = sys.stdout
                redirected_output = io.StringIO()
                sys.stdout = redirected_output
                
                try:
                    # Dynamic Execution (The "Acting" Step)
                    # WARNING: exec() runs arbitrary code. Acceptable here for controlled environments.
                    exec(generated_code)
                    sys.stdout = old_stdout 
                    
                    output_str = redirected_output.getvalue().strip()
                    
                    # Parse JSON output from the generated script
                    # We extract the JSON substring to handle potential debug prints
                    start_idx = output_str.find("{")
                    end_idx = output_str.rfind("}") + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = output_str[start_idx:end_idx]
                        result_json = json.loads(json_str)
                    else:
                        raise ValueError("No valid JSON found in script output")
                    
                    calculated_answer = result_json.get("answer")
                    submit_url = result_json.get("submit_url")
                    answer_key = result_json.get("answer_key", "answer")
                    
                    print(f"[INFO] Calculated Result: {calculated_answer}")
                    print(f"[INFO] Submission Target: {submit_url}")
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    print(f"[ERROR] Code Execution Failed: {e}")
                    print(f"[DEBUG] Failed Output Content:\n{output_str}")
                    break 

                if not submit_url:
                    print("[ERROR] No submission URL identified. Terminating.")
                    break

                # Prepare payload for submission
                payload = {
                    "email": email,
                    "secret": secret,
                    "url": current_url,
                    answer_key: calculated_answer
                }
                
                # Submit Answer via HTTP POST
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
                        print("[FAILURE] Answer rejected by server. Stopping execution.")
                        break
                        
                except:
                    print("[ERROR] Invalid response format from server. Stopping.")
                    break
                    
            except Exception as e:
                print(f"[CRITICAL] Unexpected error in agent loop: {e}")
                break

        browser.close()