import os
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

# Enable CORS for WordPress to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your WordPress URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Country cost multipliers
COUNTRY_MULTIPLIERS = {
    "usa": 1.0, "germany": 1.2, "uk": 1.15, "france": 1.18,
    "uae": 0.95, "india": 0.45, "china": 0.65, "singapore": 1.1,
    "australia": 1.08, "canada": 1.05, "brazil": 0.7, "mexico": 0.55
}

BASE_MATERIAL_PRICES = {
    "aluminum_frame_per_m": 25, "plywood_per_sheet": 35, "carpet_per_sqm": 18,
    "vinyl_graphics_per_sqm": 45, "led_light_each": 65, "reception_counter_each": 350,
    "shelf_per_m": 85, "tv_screen_55_inch": 450
}

class BoothRequest(BaseModel):
    size_sqm: float
    booth_type: str
    colors: List[str]
    elements: List[str]
    city: str
    country: str
    style: str = "modern minimalist"

@app.post("/api/design-booth")
async def design_booth(request: BoothRequest):
    try:
        # Step 1: Generate design prompt
        prompt = generate_design_prompt(request)
        
        # Step 2: Generate image
        image_url = generate_image(prompt)
        
        # Step 3: Calculate materials and cost
        materials = estimate_materials(prompt, request.size_sqm)
        cost = calculate_cost(materials, request.country)
        
        return {
            "success": True,
            "design_image_url": image_url,
            "materials": materials,
            "cost_estimate": cost,
            "design_prompt": prompt
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def generate_design_prompt(request: BoothRequest):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an exhibition booth designer. Create a detailed prompt for DALL-E."},
            {"role": "user", "content": f"""
                Create a design prompt for a {request.size_sqm} sqm {request.booth_type} exhibition booth.
                Colors: {', '.join(request.colors)}
                Elements: {', '.join(request.elements)}
                Location: {request.city}, {request.country}
                Style: {request.style}
                
                Write a prompt for DALL-E to generate a realistic 3D render.
            """}
        ]
    )
    return response.choices[0].message.content

def generate_image(prompt: str):
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=f"{prompt} Professional exhibition booth, high quality",
        size="1024x1024",
        n=1
    )
    return response.data[0].url

def estimate_materials(design_prompt: str, size_sqm: float):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract materials list as JSON. Return format: [{'name': 'item', 'quantity': number}]"},
            {"role": "user", "content": f"From this design: {design_prompt[:500]}. Estimate materials for {size_sqm} sqm booth."}
        ],
        response_format={"type": "json_object"}
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("materials", [])

def calculate_cost(materials: List[Dict], country: str):
    country = country.lower()
    multiplier = COUNTRY_MULTIPLIERS.get(country, 1.0)
    
    total = 0
    for item in materials:
        price = BASE_MATERIAL_PRICES.get(item.get("name", ""), 50)
        qty = item.get("quantity", 1)
        total += price * qty * multiplier
    
    labor = total * 0.3
    shipping = total * 0.15
    final_total = total + labor + shipping
    
    return {
        "materials_cost_usd": round(total, 2),
        "labor_cost_usd": round(labor, 2),
        "shipping_cost_usd": round(shipping, 2),
        "total_cost_usd": round(final_total, 2),
        "currency": "USD",
        "country": country.title()
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)