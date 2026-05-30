from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BoothRequest(BaseModel):
    size: int
    booth_type: str
    city: str
    country: str
    color: str

@app.get("/")
def home():
    return {"message": "AI Booth Agent is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/generate-image")
async def generate_image(request: BoothRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OpenAI API key missing"}
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"Professional {request.size} sqm {request.booth_type} exhibition booth in {request.city}, {request.country}. Main color: {request.color}. Modern design."
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    
    return {"image_url": response.data[0].url}
