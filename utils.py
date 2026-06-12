import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODEL = "meta-llama/llama-3-8b-instruct"


def call_llm(system, user):
    """
    Call LLM with improved error handling
    
    Args:
        system: System prompt defining agent role
        user: User message/prompt
        
    Returns:
        str: LLM response content
    """
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        return res.choices[0].message.content
    
    except Exception as e:
        error_msg = f"LLM Error: {str(e)}"
        print(error_msg)
        return error_msg


def search(query):
    """
    Perform web search using SerpAPI with error handling
    
    Args:
        query: Search query string
        
    Returns:
        list: List of search result snippets
    """
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": 10  # Get more results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        snippets = [r.get("snippet", "") for r in data.get("organic_results", [])]
        
        # Filter out empty snippets
        return [s for s in snippets if s.strip()]
    
    except requests.exceptions.Timeout:
        return ["Search timeout - please retry"]
    except requests.exceptions.RequestException as e:
        return [f"Search error: {str(e)}"]
    except Exception as e:
        return [f"Unexpected error: {str(e)}"]


def create_pdf(text, filename="report.pdf"):
    """
    Create professional PDF with improved formatting
    
    Args:
        text: Report content
        filename: Output filename
        
    Returns:
        str: Path to created PDF file
    """
    try:
        # Create document
        doc = SimpleDocTemplate(
            filename,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
            leftMargin=1*inch,
            rightMargin=1*inch
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom style for title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#0f172a',
            spaceAfter=30,
            alignment=1  # Center
        )
        
        # Create custom style for body
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            alignment=4  # Justify
        )
        
        # Build document content
        content = []
        
        # Add title
        content.append(Paragraph("Research Report", title_style))
        content.append(Spacer(1, 0.2*inch))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"<i>Generated on: {timestamp}</i>", styles['Normal']))
        content.append(Spacer(1, 0.3*inch))
        
        # Add main content - split into paragraphs
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Check if it's a heading (starts with # or ##)
                if para.strip().startswith('#'):
                    para = para.replace('#', '').strip()
                    content.append(Paragraph(para, styles['Heading2']))
                else:
                    content.append(Paragraph(para, body_style))
                content.append(Spacer(1, 0.15*inch))
        
        # Build PDF
        doc.build(content)
        return filename
    
    except Exception as e:
        error_msg = f"PDF creation failed: {str(e)}"
        print(error_msg)
        # Create basic PDF as fallback
        try:
            doc = SimpleDocTemplate(filename)
            styles = getSampleStyleSheet()
            doc.build([Paragraph(text, styles["Normal"])])
            return filename
        except:
            return None
