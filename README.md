# IIT Madras Graded Assignment Helper API

This API helps answer questions from IIT Madras' Online Degree in Data Science graded assignments.

## Features

- Accepts assignment questions via API
- Processes optional file attachments
- Returns answers in JSON format
- Handles various question types from all 5 graded assignments
- Uses a custom LLM endpoint for question processing
- Supports a wide range of assignment questions through intelligent prompt engineering

## Setup Instructions

### Prerequisites

- Python 3.8 or newer
- OpenAI API key
- uv (Python package installer) - optional but recommended

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Set up your environment:

   **Option 1: Using uv (recommended)**
   ```
   pip install uv
   uv pip install -e .
   ```

   **Option 2: Manual installation**
   Create a virtual environment and install the dependencies listed in the inline script at the top of main.py:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install fastapi uvicorn python-multipart openai pandas numpy pydantic aiofiles requests beautifulsoup4
   ```

3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY=your-api-key-here
   ```
   On Windows:
   ```
   set OPENAI_API_KEY=your-api-key-here
   ```

### Running Locally

Run the FastAPI application using Uvicorn:
```
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000/api/`

### Deployment

#### Deploying to Vercel

1. Create a Vercel account if you don't have one.
2. Install Vercel CLI: `npm i -g vercel`
3. Deploy: `vercel --prod`
4. Don't forget to set your OPENAI_API_KEY as an environment variable in the Vercel dashboard.

## API Usage

### Endpoint

`POST /api/`

### Request Format

Send a POST request with `multipart/form-data` containing:
- `question` (required): The assignment question text
- `file` (optional): Any file attachment mentioned in the question

### Example Request

```bash
curl -X POST "https://your-app.vercel.app/api/" \
  -H "Content-Type: multipart/form-data" \
  -F "question=Download and unzip file abcd.zip which has a single extract.csv file inside. What is the value in the 'answer' column of the CSV file?" \
  -F "file=@abcd.zip"
```

### Example Response

```json
{
  "answer": "1234567890"
}
```

## How It Works

The API uses a generalized approach to handle all types of assignment questions:

1. **Question Analysis**: The system analyzes the question to understand what's being asked
2. **File Processing**: If a file is provided, the system extracts relevant information based on the file type
3. **LLM Processing**: The question and file information are sent to a language model with instructions to provide only the direct answer
4. **Response Formatting**: The answer is returned in a simple JSON format

This approach allows the system to handle a wide variety of question types without needing specific handlers for each type.

## Notes

- The API uses a custom LLM endpoint (https://llmfoundry.straive.com/openai/v1/chat/completions) for processing questions
- The system relies on intelligent prompt engineering to handle different question types
- Dependencies are managed using uv inline script dependencies at the top of main.py
- Remember that this tool is for educational purposes to help understand the course material better

## License

MIT 