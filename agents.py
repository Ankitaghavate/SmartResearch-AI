import json
from utils import call_llm


def planner(topic):
    """
    Planner Agent: Breaks down research topic into 3 strategic search queries
    """
    prompt = f"""
You are an AI Research Planner with expertise in information retrieval.

Generate exactly 3 detailed and strategic search queries related to: {topic}

Each query should target different aspects:
1. Overview and fundamentals - broad understanding
2. Applications and real-world use cases - practical implementation
3. Benefits, challenges, and future scope - advantages and limitations

Return ONLY valid JSON array format:
[
    {{"query": "detailed query 1"}},
    {{"query": "detailed query 2"}},
    {{"query": "detailed query 3"}}
]

Do NOT include any text outside the JSON array.
"""

    res = call_llm("Planner", prompt)

    try:
        # Try to parse JSON
        parsed = json.loads(res)
        return parsed if isinstance(parsed, list) else [{"query": str(parsed)}]

    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return [
            {"query": f"{topic} overview fundamentals introduction"},
            {"query": f"{topic} applications real-world use cases examples"},
            {"query": f"{topic} benefits advantages challenges future scope"}
        ]


def critic(report):
    """
    Critic Agent: Analyzes report for errors, gaps, and areas for improvement
    """
    prompt = f"""
You are an expert AI Critic Agent specializing in research evaluation.

Carefully analyze the following report and identify:

1. **Factual Errors**: Any incorrect or misleading information
2. **Gaps & Omissions**: Missing important concepts or explanations
3. **Clarity Issues**: Poorly explained or confusing sections
4. **Grammar & Style**: Language and formatting problems
5. **Structure**: Logical flow and organization weaknesses

Provide detailed, constructive feedback that can be used to improve the report.

Report:
{report}

---

Your Critique:
"""

    return call_llm("Critic", prompt)


def improver(report, critique):
    """
    Improver Agent: Refines report based on critic feedback
    """
    prompt = f"""
You are an expert AI Report Improver with editorial experience.

Your task: Enhance the following report by addressing the critique provided.

Make the report:
- ✓ Factually accurate and verifiable
- ✓ Well-structured with clear sections
- ✓ Professional in tone and language
- ✓ Grammatically correct and polished
- ✓ Comprehensive yet concise
- ✓ Addresses all points from the critique

Original Report:
{report}

---

Critique to Address:
{critique}

---

Improved Report:
"""

    return call_llm("Improve", prompt)


def verifier(report, context):
    """
    Fact Verification Agent: Checks report claims against source context
    Helps reduce hallucinations by verifying against actual data
    """
    prompt = f"""
You are a Fact Verification Specialist with rigorous research standards.

Compare the following report against the provided source material.
Identify any claims that:
- ✗ Contradict the source material
- ✗ Are unsupported by the provided context
- ✗ Appear to be hallucinated or fabricated
- ✓ Are well-supported and verified

Format your response as:

**VERIFIED CLAIMS:**
[List claims that are factually accurate]

**UNVERIFIED/QUESTIONABLE CLAIMS:**
[List claims lacking source support]

**CONTRADICTIONS:**
[List any direct contradictions with source material]

**OVERALL RELIABILITY SCORE:**
[Rate reliability: Excellent/Good/Fair/Poor]

---

Report to Verify:
{report[:3000]}

---

Source Material:
{context[:3000]}

---

Verification Results:
"""

    return call_llm("Fact-Checker", prompt)


def writer(summary):
    """
    Writer Agent: Transforms extracted information into polished report
    """
    prompt = f"""
You are a Professional Technical Report Writer with publishing experience.

Convert the following summary into a well-structured, professional research report.

The report MUST include:
1. **Introduction**: Hook and overview of the topic
2. **Key Concepts**: Detailed explanation of core ideas
3. **Applications & Use Cases**: Real-world implementations
4. **Advantages & Benefits**: Why this matters
5. **Challenges & Limitations**: Realistic considerations
6. **Future Outlook**: Emerging trends and developments
7. **Conclusion**: Summary and key takeaways

Style Requirements:
- Professional academic tone
- Clear section headers
- Logical flow and coherence
- Accessible language (avoid jargon where possible)
- Well-formatted and readable

Summary to Transform:
{summary}

---

Professional Report:
"""

    return call_llm("Writer", prompt)


def summarizer(context):
    """
    Summarization Agent: Creates concise yet comprehensive summaries
    More flexible than T5 model for varied content
    """
    prompt = f"""
You are an Expert Summarization Specialist.

Read the following content carefully and generate a professional executive summary.

Your summary should:
- ✓ Be concise (150-250 words)
- ✓ Capture essential concepts
- ✓ Highlight key findings
- ✓ Note important applications
- ✓ Mention major challenges/benefits
- ✓ Provide actionable insights

Content to Summarize:
{context}

---

Executive Summary:
"""

    return call_llm("Summarize", prompt)
