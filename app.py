from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Try to import OpenAI only when needed
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "OpenAI module not installed"}
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OpenAI API key not set"}
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"Professional {request.booth_size} sqm {request.booth_type} exhibition booth in {request.city}, {request.country}. Color: {request.color}"
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024"
    )
    
    return {"image_url": response.data[0].url}
