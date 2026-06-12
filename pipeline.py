import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from utils import search, call_llm, create_pdf
from memory import Memory
from agents import planner, critic, improver, writer, summarizer, verifier
from t5_model import t5_summarize


def run_pipeline(topic):

    memory = Memory()
    progress = st.progress(0)
    original_context = ""  # Store original context for fact verification

    # STEP 1: QUERY DISPLAY
    st.markdown("## 📌 Research Query")
    st.write(f"**Topic:** {topic}")
    progress.progress(10)

    # STEP 2: PLANNING
    st.markdown("## 🧠 Planning Phase")
    st.write("Generating optimized search queries...")
    plan = planner(topic)
    queries = [p["query"] for p in plan]
    
    with st.expander("View Generated Queries"):
        for i, q in enumerate(queries, 1):
            st.write(f"{i}. {q}")
    
    progress.progress(20)

    # STEP 3: WEB SEARCH
    st.markdown("## 🔍 Web Search Results")
    st.write("Searching for relevant information...")

    with ThreadPoolExecutor() as ex:
        results = list(ex.map(search, queries))

    flat = [i for sub in results for i in sub]
    original_context = "\n".join(flat)  # Save for fact verification
    
    st.write(f"Found {len(flat)} relevant snippets")
    
    with st.expander("View Top 5 Search Results"):
        for i, r in enumerate(flat[:5], 1):
            st.markdown(
                f'<div class="card"><b>Result {i}:</b> {r}</div>',
                unsafe_allow_html=True
            )

    progress.progress(40)

    # STEP 4: MEMORY & EMBEDDING
    st.markdown("## 💾 Processing & Embedding")
    st.write("Converting text to embeddings and storing in FAISS...")
    memory.add(flat)
    progress.progress(50)

    # STEP 5: CONTEXT RETRIEVAL
    st.markdown("## 🎯 Context Retrieval")
    st.write("Retrieving most relevant chunks from memory...")
    context = memory.search(topic)
    context_text = "\n".join(context)
    
    with st.expander("View Retrieved Context"):
        st.write(context_text)

    progress.progress(55)

    # STEP 6: DRAFT REPORT GENERATION
    st.markdown("## 📝 Draft Report Generation")
    st.write("Writer agent creating initial report...")
    draft = writer(context_text)

    st.markdown(
        f'<div class="card" style="border-left: 4px solid #4CAF50;">{draft}</div>',
        unsafe_allow_html=True
    )

    progress.progress(65)

    # STEP 7: CRITIC ANALYSIS
    st.markdown("## 🔎 Critical Analysis")
    st.write("Critic agent identifying gaps and errors...")
    critique_text = critic(draft)

    st.markdown(
        f'<div class="card" style="border-left: 4px solid #FF9800;">{critique_text}</div>',
        unsafe_allow_html=True
    )

    progress.progress(75)

    # STEP 8: REPORT IMPROVEMENT
    st.markdown("## ✨ Report Refinement")
    st.write("Improver agent enhancing the report...")
    corrected = improver(draft, critique_text)

    st.markdown(
        f'<div class="card" style="border-left: 4px solid #2196F3;">{corrected}</div>',
        unsafe_allow_html=True
    )

    progress.progress(82)

    # STEP 9: FACT VERIFICATION (NEW)
    st.markdown("## ✅ Fact Verification")
    st.write("Verifying report against original sources...")
    verification = verifier(corrected, original_context)

    st.markdown(
        f'<div class="card" style="border-left: 4px solid #9C27B0;">{verification}</div>',
        unsafe_allow_html=True
    )

    progress.progress(88)

    # STEP 10: FINAL REPORT
    st.markdown("## 🎯 Final Report")
    final = corrected

    st.markdown(
        f'<div class="card" style="border-left: 4px solid #4CAF50; font-weight: bold;">{final}</div>',
        unsafe_allow_html=True
    )

    progress.progress(92)

    # STEP 11: SUMMARIZATION (Using Summarizer Agent instead of T5)
    st.markdown("## 📋 Executive Summary")
    st.write("Generating concise summary...")
    final_summary = summarizer(corrected)  # Use full report, not just 1000 chars

    st.markdown(
        f'<div class="card" style="border-left: 4px solid #FFC107;">{final_summary}</div>',
        unsafe_allow_html=True
    )

    progress.progress(95)

    # STEP 12: PDF GENERATION
    st.markdown("## 📥 PDF Export")
    pdf = create_pdf(final, "research_report.pdf")

    with open(pdf, "rb") as f:
        st.download_button(
            label="⬇️ Download Full Report (PDF)",
            data=f,
            file_name="research_report.pdf",
            mime="application/pdf"
        )

    progress.progress(97)

    # STEP 13: CONFIDENCE SCORING (IMPROVED)
    st.markdown("## 🎓 Quality Assessment")
    confidence_prompt = f"""
    Based on the research report below, provide a confidence score (0-100) and brief explanation.
    
    Report:
    {corrected[:2000]}
    
    Format your response as:
    Score: [0-100]
    Confidence Assessment: [Brief explanation]
    """
    confidence = call_llm(
        "You are a research quality assessor. Evaluate report quality objectively.",
        confidence_prompt
    )

    st.markdown(
        f'<div class="card" style="border-left: 4px solid #2196F3; font-weight: bold;">{confidence}</div>',
        unsafe_allow_html=True
    )

    progress.progress(100)

    # Store original context in memory for Q&A assistant
    memory.original_context = original_context

    return memory
