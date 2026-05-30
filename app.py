from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# Fix CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENAI_API_KEY")

class ImageRequest(BaseModel):
    booth_size: int
    booth_type: str
    city: str
    country: str
    color: str

@app.get("/")
def root():
    return {"message": "AI Booth Agent Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    if not api_key:
        return {"error": "OpenAI API key not set"}
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"Professional {request.booth_size} square meter {request.booth_type} exhibition booth in {request.city}, {request.country}. Main brand color: {request.color}. Modern, well-lit, with reception counter and product displays."
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    
    return {"image_url": response.data[0].url}
