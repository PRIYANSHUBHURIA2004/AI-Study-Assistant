import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from groq import AsyncGroq

load_dotenv()

app = FastAPI(title="AI Study Buddy")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

QUIZ_PROMPT = """Create a 3-question multiple-choice quiz on the topic: "{topic}".

Format each question exactly like this:

Question 1: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [letter]

Repeat for Questions 2 and 3. Keep questions clear and educational."""

class QuizRequest(BaseModel):
    topic: str

@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)

@app.post("/api/quiz")
async def generate_quiz(request: QuizRequest):
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise HTTPException(
            status_code=500,
            detail="Groq API key is not configured. Set GROQ_API_KEY in environment variables",
        )

    async def stream_quiz():
        try:
            stream = await client.chat.completions.create(
                model="qwen-2.5-32b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful study buddy that creates clear, educational quizzes.",
                    },
                    {
                        "role": "user",
                        "content": QUIZ_PROMPT.format(topic=topic),
                    },
                ],
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            yield f"\n\nError generating quiz: {exc}"

    return StreamingResponse(stream_quiz(), media_type="text/plain")
