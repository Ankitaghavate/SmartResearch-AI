import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODEL = "meta-llama/llama-3-8b-instruct"


def call_llm(system, user):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return res.choices[0].message.content


def search(query):
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY}
    data = requests.get(url, params=params).json()
    return [r.get("snippet", "") for r in data.get("organic_results", [])]


def create_pdf(text, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    doc.build([Paragraph(text, styles["Normal"])])
    return filename
