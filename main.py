#!/usr/bin/env python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "python-multipart",
#     "openai>=1.0.0",
#     "pandas",
#     "numpy",
#     "pydantic",
#     "aiofiles",
#     "requests",
#     "beautifulsoup4",
#     "python-dotenv",
#     "matplotlib",
#     "pillow",
#     "base64io",
# ]
# ///

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import tempfile
import zipfile
import csv
import io
import json
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from openai import OpenAI
import base64
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key with fallback
api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI client with custom base URL
client = OpenAI(
    api_key=api_key,
    base_url="https://llmfoundry.straive.com/openai/v1"
)

app = FastAPI(title="IIT Madras Graded Assignment Helper")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Answer(BaseModel):
    answer: str

class QuestionRequest(BaseModel):
    question_text: str
    html_content: Optional[str] = None

class CodeResponse(BaseModel):
    code: str
    language: str
    explanation: str

def execute_python_code(code: str) -> Any:
    """Execute Python code and return the result."""
    try:
        # Create a temporary file to save the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file.write(code.encode())
            temp_path = temp_file.name
        
        # Execute the code and capture the output
        result = {}
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        exec_cmd = f"python {temp_path} > {output_path} 2>&1"
        exit_code = os.system(exec_cmd)
        
        # Read the output
        with open(output_path, 'r') as f:
            output = f.read()
        
        # Clean up temporary files
        os.unlink(temp_path)
        os.unlink(output_path)
        
        if exit_code != 0:
            return {"error": True, "output": output}
        
        return {"error": False, "output": output}
    except Exception as e:
        return {"error": True, "output": str(e)}

def extract_code_from_llm_response(response: str) -> Dict[str, Any]:
    """Extract code from LLM response."""
    # Look for Python code blocks
    python_pattern = "```python\n"
    bash_pattern = "```bash\n"
    end_pattern = "```"
    
    code = ""
    language = "python"  # Default language
    explanation = response
    
    # Extract Python code
    if python_pattern in response:
        start_idx = response.find(python_pattern) + len(python_pattern)
        end_idx = response.find(end_pattern, start_idx)
        if end_idx != -1:
            code = response[start_idx:end_idx].strip()
            explanation = response.replace(python_pattern + code + end_pattern, "").strip()
    
    # Extract Bash code if no Python code found
    elif bash_pattern in response:
        start_idx = response.find(bash_pattern) + len(bash_pattern)
        end_idx = response.find(end_pattern, start_idx)
        if end_idx != -1:
            code = response[start_idx:end_idx].strip()
            language = "bash"
            explanation = response.replace(bash_pattern + code + end_pattern, "").strip()
    
    return {
        "code": code,
        "language": language,
        "explanation": explanation
    }

def execute_code(code: str, language: str) -> Dict[str, Any]:
    """Execute code based on the language."""
    if language == "python":
        return execute_python_code(code)
    elif language == "bash":
        # Create a temporary file to save the code
        with tempfile.NamedTemporaryFile(suffix='.sh', delete=False) as temp_file:
            temp_file.write(code.encode())
            temp_path = temp_file.name
        
        # Make the script executable
        os.chmod(temp_path, 0o755)
        
        # Execute the bash script and capture the output
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as output_file:
            output_path = output_file.name
        
        exit_code = os.system(f"{temp_path} > {output_path} 2>&1")
        
        # Read the output
        with open(output_path, 'r') as f:
            output = f.read()
        
        # Clean up temporary files
        os.unlink(temp_path)
        os.unlink(output_path)
        
        if exit_code != 0:
            return {"error": True, "output": output}
        
        return {"error": False, "output": output}
    else:
        return {"error": True, "output": f"Unsupported language: {language}"}

@app.post("/api", response_model=Answer)
async def answer_question(
    request: Request
):
    """
    Process a question and return the answer.
    This endpoint handles all request formats by parsing them manually.
    """
    try:
        question_text = ""
        html_content_text = None
        file_data = None
        file_name = None
        file_content_type = None
        
        # Determine the content type
        content_type = request.headers.get('content-type', '').lower()
        
        # Handle different content types
        if 'application/json' in content_type:
            # Parse JSON body manually
            body = await request.json()
            if 'question_text' in body:
                question_text = body['question_text']
            if 'html_content' in body:
                html_content_text = body['html_content']
        else:
            # Handle form data (both multipart and url-encoded)
            try:
                form = await request.form()
                print(f"Form data keys: {list(form.keys())}")
                
                # Get question text from form data - be lenient with whitespace in key names
                question_key = next((key for key in form.keys() if key.strip() == 'q1' or key.strip() == 'question'), None)
                if question_key:
                    question_text = str(form[question_key])
                    print(f"Found question text with key '{question_key}': {question_text[:50]}...")
                
                # Get HTML content if available
                html_key = next((key for key in form.keys() if key.strip() == 'html_content'), None)
                if html_key:
                    html_content_text = str(form[html_key])
                
                # Handle file upload if available - check for any file field
                file_fields = [key for key in form.keys() if isinstance(form[key], UploadFile)]
                if file_fields:
                    file_field = file_fields[0]  # Use the first file field found
                    file = form[file_field]
                    file_name = file.filename
                    file_content_type = file.content_type
                    print(f"Found file: {file_name}, content type: {file_content_type}")
                    
                    try:
                        # Read file content
                        file_data = await file.read()
                        print(f"Read {len(file_data)} bytes from file")
                        
                        # Create a temporary file for processing
                        file_ext = os.path.splitext(file_name)[1]
                        if not file_ext:  # If no extension, try to determine from content type
                            if 'pdf' in file_content_type:
                                file_ext = '.pdf'
                            elif 'image' in file_content_type:
                                file_ext = '.png'
                            elif 'excel' in file_content_type or 'spreadsheet' in file_content_type:
                                file_ext = '.xlsx'
                            elif 'csv' in file_content_type:
                                file_ext = '.csv'
                            else:
                                file_ext = '.bin'  # Default extension
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                            temp_file.write(file_data)
                            temp_file_path = temp_file.name
                        
                        print(f"Saved file to temporary path: {temp_file_path}")
                        
                        # If no question was provided but a file was, create a default question based on file type
                        if not question_text:
                            if file_content_type == 'application/pdf' or file_ext.lower() == '.pdf':
                                question_text = f"Extract and analyze the text content from this PDF file."
                            elif 'image/' in file_content_type or file_ext.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                                question_text = f"Analyze this image and describe what you see."
                            elif 'spreadsheet' in file_content_type or file_ext.lower() in ['.xlsx', '.xls', '.csv']:
                                question_text = f"Analyze this spreadsheet and provide a summary of its contents."
                            else:
                                question_text = f"Analyze the contents of this file: {file_name}"
                            
                            print(f"Generated default question: {question_text}")
                        
                        # Add file info to the question
                        file_info = f"\n\nThe uploaded file is named '{file_name}' with content type '{file_content_type}'. "
                        file_info += f"The file is available at '{temp_file_path}' for processing."
                        question_text += file_info
                    except Exception as file_error:
                        print(f"Error processing file: {str(file_error)}")
                        # Continue without the file if there's an error
            except Exception as form_error:
                # If form parsing fails, try to get the raw body and log
                body = await request.body()
                print(f"Form parsing error: {str(form_error)}")
                print(f"Raw body (first 500 chars): {body.decode('utf-8', errors='ignore')[:500]}")
                # See if we can extract the question from the raw body as a last resort
                raw_body = body.decode('utf-8', errors='ignore')
                if 'q1' in raw_body:
                    try:
                        # Very crude extraction attempt for multipart form data
                        q1_start = raw_body.find('name="q1"')
                        if q1_start != -1:
                            content_start = raw_body.find('\r\n\r\n', q1_start) + 4
                            content_end = raw_body.find('\r\n---', content_start)
                            if content_end != -1:
                                question_text = raw_body[content_start:content_end]
                                print(f"Extracted question from raw body: {question_text[:50]}...")
                    except Exception as extract_error:
                        print(f"Error extracting from raw body: {str(extract_error)}")
        
        # If no question was found in any format
        if not question_text:
            raise HTTPException(
                status_code=400, 
                detail="No question provided or could not parse the request. Please check your request format."
            )
            
        # Prepare the prompt for the LLM
        system_prompt = """You are an AI assistant helping a student with IIT Madras Online Degree in Data Science graded assignments.
        Your task is to solve the given question and provide executable Python or bash code that will produce the answer.
        DO NOT provide the answer directly. Instead, provide code that when executed will generate the answer.
        Format your code inside a code block using triple backticks with the language specified (```python or ```bash).
        
        IMPORTANT RULES:
        1. ONLY use libraries that are included in the dependencies list: pandas, numpy, matplotlib, requests, beautifulsoup4, and standard Python libraries.
        2. DO NOT use libraries like PyPDF2, pdfplumber, camelot, or any other PDF library that is not explicitly listed in the dependencies.
        3. When a file path is provided, ALWAYS use the EXACT file path that is provided in the input, not a placeholder like 'path_to_file.pdf'.
        4. For PDF files, since we don't have PDF-specific libraries, you can use subprocess to call system tools or extract text from PDFs using basic file operations.
        5. Keep your code simple and focused only on answering the exact question asked.
        
        For questions involving:
        - Excel/GSheets: Use pandas to read and analyze Excel files. For CSV or Excel files, the code should read the file from the provided path.
        - HTML scraping: Use BeautifulSoup or similar libraries
        - GitHub interactions: Be aware that GitHub search API doesn't return complete user data. For each user from search results, you need to make additional requests to /users/{username} to get full user details like 'created_at'. Always check API rate limits.
        - Image responses: Generate images and return them as base64 encoded data URIs
        - Docker operations: Use the Docker API or subprocess to run docker commands
        - PDF files: NEVER use libraries like PyPDF2, pdfplumber, or camelot. Instead, try to use subprocess to call system tools like 'pdftotext' if available, or handle the file as binary data.
        
        Provide only minimal output - the exact answer to the question and nothing else. No explanations or commentary needed.
        """
        
        user_prompt = question_text
        
        # If HTML content is provided, include it in the prompt
        if html_content_text:
            user_prompt += f"\n\nHTML Content for processing: \n{html_content_text}"
        
        # Call the OpenAI API to get the solution
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or another appropriate model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        # Extract the response content
        llm_response = response.choices[0].message.content
        
        # Extract code from the LLM response
        extracted = extract_code_from_llm_response(llm_response)
        code = extracted["code"]
        language = extracted["language"]
        explanation = extracted["explanation"]
        
        if not code:
            return {"answer": "Could not extract executable code from the LLM response. Please try again with a more specific question."}
        
        # Execute the code
        execution_result = execute_code(code, language)
        
        if execution_result["error"]:
            return {"answer": f"Error executing code: {execution_result['output']}\n\nCode: {code}"}
        
        # Return the output as the answer
        return {"answer": execution_result["output"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8010)
