# /// script
# dependencies = [
#     "fastapi",
#     "fastapi-cors",
#     "requests",
#     "python-dotenv",
#     "python-multipart",
#     "uvicorn",
#     "numpy",
#     "scikit-learn"
# ]
# ///

from fastapi import FastAPI, Form, Query, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests, os, tempfile, zipfile, csv, io, json, shutil, subprocess, sys, platform
from typing import Optional, Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompts
TDS_SYSTEM_PROMPT = "Provide only the exact answer without any explanations, reasoning, or additional text. Be extremely concise."

CODE_GENERATION_PROMPT = "Generate only the exact code needed to solve the problem. No explanations, comments, or additional text. Return ONLY the executable code."

# API token
API_TOKEN = os.getenv("LLMFOUNDRY_TOKEN")
API_URL = "https://llmfoundry.straive.com/openai/v1/chat/completions"

def load_question_data():
    with open("data.json", "r", encoding="utf-8") as f:
        return json.load(f)

def find_similar_question(query, questions_data):
    questions = [item["question"] for item in questions_data]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    most_similar_idx = np.argmax(similarities)
    return questions_data[most_similar_idx], similarities[most_similar_idx]

def execute_code(code: str, code_type: str, working_dir: str, processed_files: List[str] = None) -> Dict[str, Any]:
    """Execute Python code or Git Bash command and return the result"""
    result = {"success": False, "output": "", "error": ""}
    
    try:
        if code_type.lower() == "python":
            # Save and execute Python code
            script_path = os.path.join(working_dir, "temp_script.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            process = subprocess.run(
                [sys.executable, script_path],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
        else:
            # Execute Git Bash command
            shell = True
            if platform.system() == "Windows":
                git_bash_paths = [
                    r"C:\Program Files\Git\bin\bash.exe",
                    r"C:\Program Files (x86)\Git\bin\bash.exe"
                ]
                git_bash_exe = next((path for path in git_bash_paths if os.path.exists(path)), None)
                shell = [git_bash_exe, "-c"] if git_bash_exe else True
            else:
                shell = ["/bin/bash", "-c"]
            
            process = subprocess.run(
                code if shell is True else shell + [code],
                cwd=working_dir,
                shell=shell is True,
                capture_output=True,
                text=True,
                timeout=30
            )
        
        if process.returncode == 0:
            result["success"] = True
            result["output"] = process.stdout.strip()
        else:
            result["error"] = process.stderr.strip()
            
    except subprocess.TimeoutExpired:
        result["error"] = "Execution timed out after 30 seconds"
    except Exception as e:
        result["error"] = str(e)
    
    return result

def call_llm_api(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Helper function to call the LLM API"""
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_TOKEN}:tds-project2"},
            json={
                "model": model, 
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        )
        
        response_json = response.json()
        
        if "error" in response_json:
            return {
                "success": False,
                "error": f"API Error: {response_json['error'].get('message', 'Unknown API error')}"
            }
        
        content = response_json["choices"][0]["message"]["content"].strip()
        return {"success": True, "content": content}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def predict_code_outcome(code: str, error_message: str, question: str, code_type: str) -> Dict[str, Any]:
    """When code execution fails, ask the LLM to predict the outcome"""
    prompt = f"""
Question that needs to be answered: "{question}"

The following {code_type} code was generated to answer this question:
```
{code}
```

The code execution failed with this error:
```
{error_message}
```

Based on the question and the code, what would the exact output have been if the code had executed successfully? Provide ONLY the raw output that would have been produced, with no explanations or additional text.
"""
    
    result = call_llm_api(
        "Predict only the exact output without any explanations or additional text.",
        prompt
    )
    
    if result["success"]:
        return {"success": True, "predicted_output": result["content"]}
    else:
        return {"success": False, "predicted_output": f"Failed to predict outcome: {result.get('error', 'Unknown error')}"}

def process_file_context(file_path, temp_dir, processed_files):
    """Process uploaded file and extract context"""
    file_context = ""
    filename = os.path.basename(file_path)
    
    # Process based on file type
    if filename.endswith('.zip'):
        # Extract ZIP and process contents
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Add extracted files to processed_files
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f != filename:
                    extracted_path = os.path.join(root, f)
                    processed_files.append(extracted_path)
                    
                    # Process CSV files
                    if f.endswith('.csv'):
                        file_context += process_csv_file(extracted_path, f)
    
    elif filename.endswith('.csv'):
        file_context += process_csv_file(file_path, filename)
    
    else:
        # For other file types, read as text if possible
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_context += f"\nFile: {filename}\nContent (first 1000 chars):\n{content[:1000]}\n"
                if len(content) > 1000:
                    file_context += "...(truncated)...\n"
        except Exception as e:
            file_context += f"\nFile: {filename} (could not read: {str(e)})\n"
    
    return file_context

def process_csv_file(file_path, filename):
    """Process a CSV file and return context"""
    context = ""
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            
            context += f"\nFile: {filename}\nCSV columns: {', '.join(headers)}\n"
            
            # Read sample rows
            rows = []
            for i, row in enumerate(csv_reader):
                if i < 10:
                    rows.append(row)
                else:
                    break
            
            if rows:
                context += "Sample data:\n"
                for i, row in enumerate(rows):
                    context += f"Row {i+1}: {row}\n"
    except Exception as e:
        context += f"\nError reading CSV file {filename}: {str(e)}\n"
    
    return context

def generate_code_for_question(question: str, code_type: str, file_context: str, temp_dir: str, processed_files: List[str] = None) -> Dict[str, Any]:
    """Generate and execute code for a question"""
    prompt = f"""
Question that needs to be answered: "{question}"

Context from uploaded files:
{file_context}

Task: Generate {code_type} code that will solve this question and produce the exact answer.
Requirements:
1. The code must be complete and executable
2. Use standard libraries when possible
3. Handle any file operations correctly
4. Return only the final answer as output
5. Do not include any explanatory text or comments in the output

Operating System: {platform.system()}
Python Version: {platform.python_version()}
Current working directory: {temp_dir}
"""
    
    result = call_llm_api(CODE_GENERATION_PROMPT, prompt)
    
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    
    generated_code = result["content"].strip()
    
    # Extract code from markdown code blocks if present
    if "```" in generated_code:
        code_blocks = []
        lines = generated_code.split("\n")
        in_code_block = False
        current_block = []
        
        for line in lines:
            if line.startswith("```"):
                if in_code_block:
                    code_blocks.append("\n".join(current_block))
                    current_block = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_block.append(line)
        
        if code_blocks:
            generated_code = code_blocks[0]
    
    # Execute the generated code
    result = execute_code(generated_code, code_type, temp_dir, processed_files)
    
    if result["success"]:
        return {
            "success": True,
            "output": result["output"],
            "code": generated_code
        }
    else:
        # Try to predict the outcome
        prediction = predict_code_outcome(generated_code, result["error"], question, code_type)
        
        if prediction["success"]:
            return {
                "needed": True,
                "success": True,
                "code": generated_code,
                "output": prediction["predicted_output"],
                "error": result["error"],
                "is_predicted": True
            }
        else:
            return {
                "needed": True,
                "success": False,
                "code": generated_code,
                "output": "",
                "error": result["error"]
            }

@app.post("/api")
async def answer_question_post(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """Main API endpoint to process questions and files"""
    # Create temporary directory for file processing
    with tempfile.TemporaryDirectory() as temp_dir:
        file_context = ""
        processed_files = []
        
        # Process uploaded file if present
        if file and file.filename:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            processed_files.append(file_path)
            file_context = process_file_context(file_path, temp_dir, processed_files)
        
        try:
            # Find most similar question in data.json
            questions_data = load_question_data()
            similar_question, similarity_score = find_similar_question(question, questions_data)
            
            # Process based on the matched question
            if similar_question['code'] == "yes":
                # Generate and execute code
                code_type = similar_question['type'] or "python"
                result = generate_code_for_question(question, code_type, file_context, temp_dir, processed_files)
                
                if result["success"]:
                    return {
                        "answer": result["output"],
                    }
                else:
                    return {
                        "error": result["error"],
                        "code": result.get("code", ""),
                        "similar_question": similar_question["question"],
                        "similarity_score": float(similarity_score)
                    }
            else:
                # Get direct answer from LLM
                detailed_prompt = f"""
Question that needs to be answered: "{question}"

Context from uploaded files:
{file_context}

Similar question from our database: "{similar_question["question"]}"
Similarity score: {similarity_score}

Provide only the exact answer without any explanations, reasoning, or additional text. Be extremely concise.
"""
                direct_result = call_llm_api(TDS_SYSTEM_PROMPT, detailed_prompt)
                
                if direct_result["success"]:
                    return {"answer": direct_result["content"]}
                else:
                    return {
                        "error": direct_result["error"],
                        "similar_question": similar_question["question"],
                        "similarity_score": float(similarity_score)
                    }
                
        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}

@app.get("/")
def root():
    return {"message": "TDS Project API is running. Use /api endpoint with POST requests."}

if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)