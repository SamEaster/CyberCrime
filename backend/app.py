import os
import base64
import tempfile
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from score_model import calculate_importance_score
from data_processing import previous_counts

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(title="CyberCrime Risk Assessment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ─── Constants ───────────────────────────────────────────────────────────────

INCIDENT_TYPES = [
    'Ransomware', 'Data Breach', 'Hacking', 'Malware',
    'Identity Theft', 'Phishing', 'Online Fraud', 'Cyber Bullying', 'Others'
]

CATEGORIES = [
    'Social Media', 'Government', 'Corporate', 'E-commerce',
    'Educational', 'Financial', 'Personal', 'Health'
]

EXTRACTION_PROMPT = '''Task: You are a data extraction specialist. Analyze the provided complaint {input_type} and extract the following fields into a structured string format with each field separated by $.
Fields to Extract: Year$Amount_Lost_INR$City$Category$Type
- Year: The 4-digit year when the incident occurred. If not found, use "NULL".
- Amount_Lost_INR: The total monetary loss in Indian Rupees (number only). If not found, use "0".
- City: The city where the incident took place. If not found, use "NULL".
- Category: Map to exactly one of: Social Media, Government, Corporate, E-commerce, Educational, Financial, Personal, Health. If unclear, use "Personal".
- Type: Map to exactly one of: Ransomware, Data Breach, Hacking, Malware, Identity Theft, Phishing, Online Fraud, Cyber Bullying, Others. If unclear, use "Others".
Constraint: Return ONLY the 5 values separated by $. No extra text. Example: 2024$50000$Mumbai$Financial$Phishing
Content: {content}'''


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_llm_output(raw: str) -> dict:
    """Parse the $-separated LLM output into structured fields."""
    raw = raw.strip().strip('`').strip()
    parts = [p.strip() for p in raw.split('$')]

    result = {
        'Year': 2025, 'Amount_Lost_INR': 0,
        'City': 'Unknown', 'Category': 'Personal', 'Incident_Type': 'Others'
    }

    if len(parts) >= 1 and parts[0] not in ('NULL', ''):
        try:
            result['Year'] = int(parts[0])
        except ValueError:
            pass
    if len(parts) >= 2 and parts[1] not in ('NULL', ''):
        try:
            result['Amount_Lost_INR'] = int(float(parts[1].replace(',', '')))
        except ValueError:
            pass
    if len(parts) >= 3 and parts[2] not in ('NULL', ''):
        result['City'] = parts[2]
    if len(parts) >= 4 and parts[3] not in ('NULL', ''):
        result['Category'] = parts[3]
    if len(parts) >= 5 and parts[4] not in ('NULL', ''):
        result['Incident_Type'] = parts[4]

    return result


def build_risk_row(fields: dict) -> pd.DataFrame:
    """Build a single-row DataFrame compatible with the scoring model."""
    incident_type = fields.get('Incident_Type', 'Others')

    row = {
        'Year': fields.get('Year', 2025),
        'Amount_Lost_INR': fields.get('Amount_Lost_INR', 0),
        'City': fields.get('City', 'Unknown'),
        'Category': fields.get('Category', 'Personal'),
    }

    for t in INCIDENT_TYPES:
        row[t] = 1 if t == incident_type else 0

    # Load historical data to get prev_ counts
    csv_path = os.path.join(DATA_DIR, 'n_cyber_crime.csv')
    if os.path.exists(csv_path):
        hist_df = pd.read_csv(csv_path)
        year_data = hist_df[hist_df['Year'] == row['Year']]
        if not year_data.empty:
            for t in INCIDENT_TYPES:
                row[f'prev_{t}'] = int(year_data.iloc[0].get(f'prev_{t}', 0))
        else:
            latest = hist_df.iloc[0] if len(hist_df) > 0 else None
            for t in INCIDENT_TYPES:
                row[f'prev_{t}'] = int(latest.get(f'prev_{t}', 0)) if latest is not None else 0
    else:
        for t in INCIDENT_TYPES:
            row[f'prev_{t}'] = 0

    df = pd.DataFrame([row])

    return df


def classify_risk(score: float) -> str:
    if score >= 6.5:
        return 'High'
    elif score >= 4.0:
        return 'Medium'
    else:
        return 'Low'


def get_risk_result(fields: dict) -> dict:
    """Calculate risk score and return result."""
    df = build_risk_row(fields)
    scored_df = calculate_importance_score(df)
    score = round(float(scored_df['Risk_Score'].iloc[0]), 2)
    classification = classify_risk(score)

    return {
        'risk_score': score,
        'classification': classification,
        'extracted_fields': {
            'year': fields.get('Year'),
            'amount_lost_inr': fields.get('Amount_Lost_INR'),
            'city': fields.get('City'),
            'category': fields.get('Category'),
            'incident_type': fields.get('Incident_Type'),
        }
    }


# ─── API Endpoints ───────────────────────────────────────────────────────────

class FormInput(BaseModel):
    year: int
    amount_lost_inr: float
    city: str
    category: str
    incident_type: str


@app.post("/api/assess/form")
async def assess_form(data: FormInput):
    """Risk assessment from structured form fields."""
    fields = {
        'Year': data.year,
        'Amount_Lost_INR': data.amount_lost_inr,
        'City': data.city,
        'Category': data.category,
        'Incident_Type': data.incident_type,
    }
    return get_risk_result(fields)


@app.post("/api/assess/text")
async def assess_text(text: str = Form(...)):
    """Risk assessment from free-text complaint."""
    prompt = PromptTemplate.from_template(EXTRACTION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({'input_type': 'text', 'content': text})
    fields = parse_llm_output(raw)
    result = get_risk_result(fields)
    result['raw_llm_output'] = raw
    return result


@app.post("/api/assess/image")
async def assess_image(file: UploadFile = File(...)):
    """Risk assessment from an image of a complaint."""
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode('utf-8')

    mime = file.content_type or 'image/jpeg'
    message = HumanMessage(content=[
        {"type": "text", "text": EXTRACTION_PROMPT.format(input_type='image', content='(see attached image)')},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
    ])
    response = llm.invoke([message])
    fields = parse_llm_output(response.content)
    result = get_risk_result(fields)
    result['raw_llm_output'] = response.content
    return result


@app.post("/api/assess/audio")
async def assess_audio(file: UploadFile = File(...)):
    """Risk assessment from audio complaint."""
    contents = await file.read()
    audio_b64 = base64.b64encode(contents).decode('utf-8')
    mime = file.content_type or 'audio/mpeg'

    message = HumanMessage(content=[
        {"type": "text", "text": EXTRACTION_PROMPT.format(input_type='audio', content='(see attached audio)')},
        {"type": "media", "mime_type": mime, "data": audio_b64}
    ])
    response = llm.invoke([message])
    fields = parse_llm_output(response.content)
    result = get_risk_result(fields)
    result['raw_llm_output'] = response.content
    return result


@app.post("/api/assess/video")
async def assess_video(file: UploadFile = File(...)):
    """Risk assessment from video complaint."""
    contents = await file.read()
    video_b64 = base64.b64encode(contents).decode('utf-8')
    mime = file.content_type or 'video/mp4'

    message = HumanMessage(content=[
        {"type": "text", "text": EXTRACTION_PROMPT.format(input_type='video', content='(see attached video)')},
        {"type": "media", "mime_type": mime, "data": video_b64}
    ])
    response = llm.invoke([message])
    fields = parse_llm_output(response.content)
    result = get_risk_result(fields)
    result['raw_llm_output'] = response.content
    return result


@app.get("/api/analytics/overview")
async def analytics_overview():
    """Overall analytics from the historical dataset."""
    csv_path = os.path.join(DATA_DIR, 'n_cyber_crime.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Processed data not found.")

    df = pd.read_csv(csv_path)

    # Preserve original string columns before scoring mutates them
    original_cities = df['City'].copy()
    original_categories = df['Category'].copy()

    # Score all records
    scored = calculate_importance_score(df.copy())
    scored['Classification'] = scored['Risk_Score'].apply(classify_risk)

    # Summary stats
    total_cases = len(df)
    total_loss = float(pd.to_numeric(df['Amount_Lost_INR'], errors='coerce').fillna(0).sum())
    avg_loss = float(pd.to_numeric(df['Amount_Lost_INR'], errors='coerce').fillna(0).mean())
    avg_risk = float(scored['Risk_Score'].mean())

    # Classification distribution
    class_dist = scored['Classification'].value_counts().to_dict()

    # Year-wise trends
    df_numeric = df.copy()
    df_numeric['Year'] = pd.to_numeric(df_numeric['Year'], errors='coerce').fillna(2025).astype(int)
    df_numeric['Amount_Lost_INR'] = pd.to_numeric(df_numeric['Amount_Lost_INR'], errors='coerce').fillna(0)
    year_trend = df_numeric.groupby('Year').agg(
        cases=('Year', 'count'),
        total_loss=('Amount_Lost_INR', 'sum'),
        avg_loss=('Amount_Lost_INR', 'mean')
    ).reset_index().to_dict(orient='records')

    # Incident type distribution
    incident_counts = {}
    for t in INCIDENT_TYPES:
        if t in df.columns:
            incident_counts[t] = int(pd.to_numeric(df[t], errors='coerce').fillna(0).sum())

    # Category distribution (use original strings)
    cat_counts = original_categories.value_counts().to_dict()

    # City distribution (top 10, use original strings)
    city_counts = original_cities.value_counts().head(10).to_dict()

    # Top 10 highest risk cases — restore readable Category column
    scored['Category'] = original_categories.values
    top_risk = scored.nlargest(10, 'Risk_Score')[
        ['Year', 'Amount_Lost_INR', 'Category', 'Risk_Score', 'Classification']
    ].to_dict(orient='records')

    return {
        'total_cases': total_cases,
        'total_loss': total_loss,
        'avg_loss': round(avg_loss, 2),
        'avg_risk_score': round(avg_risk, 2),
        'classification_distribution': class_dist,
        'year_trend': year_trend,
        'incident_type_distribution': incident_counts,
        'category_distribution': cat_counts,
        'city_distribution': city_counts,
        'top_risk_cases': top_risk,
    }


@app.get("/api/analytics/risk-distribution")
async def risk_distribution():
    """Risk score distribution across all records."""
    csv_path = os.path.join(DATA_DIR, 'n_cyber_crime.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Processed data not found.")

    df = pd.read_csv(csv_path)
    scored = calculate_importance_score(df.copy())

    bins = [0, 2, 4, 6, 8, 10]
    labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
    scored['bin'] = pd.cut(scored['Risk_Score'].astype(float), bins=bins, labels=labels, include_lowest=True)
    distribution = scored['bin'].value_counts().sort_index().to_dict()

    return {'distribution': {str(k): int(v) for k, v in distribution.items()}}


# Serve frontend
@app.get("/")
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CyberCrime Risk Assessment API is running. Frontend not found."}


# Mount static files last
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
