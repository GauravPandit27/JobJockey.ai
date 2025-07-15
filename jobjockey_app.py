import streamlit as st
import fitz  # PyMuPDF
from constants import groq_key
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -------- INIT --------
llm = ChatGroq(api_key=groq_key, model_name="llama3-8b-8192")
st.set_page_config(page_title="JobJockey.ai | Powered by OpenBook", layout="wide")
st.title("💼 JobJockey.ai – Resume Intelligence Engine")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        padding: 10px 20px;
    }
    .question-block {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 0.5rem;
    }
    .question-spacing {
        margin-bottom: 2rem;
    }
    .ats-score {
        font-size: 24px;
        font-weight: 700;
        color: #0f62fe;
    }
</style>
""", unsafe_allow_html=True)

# -------- Resume Extractor --------
def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    else:
        return file.read().decode("utf-8")

# -------- Prompts --------
section_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
Split this resume into structured sections:

- Education
- Experience
- Projects
- Technical Skills / Tech Stack
- Hobbies (if any)
- Certifications (if any)
- Top 10 keywords

Format using markdown headers. Resume:
{resume}
"""
)

ats_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""
You are an ATS evaluation assistant.

Please analyze this resume and provide:

1. ATS Score (out of 100)
2. Verdict (Good, Average, Needs Improvement)
3. 🔍 Missing Sections
4. 📝 Grammar or Tone Suggestions
5. 🎯 Clarity / Accuracy Issues
6. 💡 Suggestions to Improve

Resume:
{resume}
"""
)

mock_prompt = PromptTemplate(
    input_variables=["resume", "interview_type"],
    template="""
Act as a senior interviewer conducting a {interview_type} round.

Read this resume and generate **10 thoughtful, relevant questions**.

Resume:
{resume}
"""
)

dsa_prompt = PromptTemplate(
    input_variables=["level"],
    template="""
You are a DSA coding interviewer.

Generate 10 questions of level {level} difficulty (1 = easy, 5 = very hard). Include:
- Mix of topics (array, string, tree, graph, dynamic programming)
- One-line titles for each
- Label Q1, Q2... Q10
"""
)

# -------- Sidebar --------
resume_file = st.sidebar.file_uploader("📄 Upload Resume", type=["pdf", "txt"])
page = st.sidebar.radio("🧭 Navigate", ["📑 Resume Breakdown", "📈 ATS Analyzer", "🧠 Mock Interview"])

if resume_file:
    resume_text = extract_text(resume_file)
    st.sidebar.success("✅ Resume uploaded!")

    # ------------------ Resume Breakdown ------------------
    if page == "📑 Resume Breakdown":
        st.subheader("📌 Structured Resume Sections")
        with st.spinner("🧠 Parsing resume..."):
            chain = LLMChain(llm=llm, prompt=section_prompt)
            structured = chain.run({"resume": resume_text})
        st.markdown(structured)

    # ------------------ ATS Analyzer ------------------
    elif page == "📈 ATS Analyzer":
        st.subheader("📋 Automated ATS Report")
        if st.button("📊 Run ATS Evaluation"):
            ats_chain = LLMChain(llm=llm, prompt=ats_prompt)
            with st.spinner("🤖 Analyzing resume..."):
                feedback = ats_chain.run({"resume": resume_text})
            lines = feedback.strip().splitlines()

            score_line = lines[0] if len(lines) > 0 else "ATS Score: Not available"
            verdict_line = lines[1] if len(lines) > 1 else "Verdict: Unknown"
            detail_lines = lines[2:]

            st.markdown(f"<div class='ats-score'>{score_line}</div>", unsafe_allow_html=True)
            st.markdown(f"#### 🧠 Verdict: `{verdict_line.split(':')[-1].strip()}`")
            st.divider()
            st.markdown("### 🔍 Detailed Feedback")
            current_section = ""
            for line in detail_lines:
                if line.startswith("🔍") or line.startswith("📝") or line.startswith("🎯") or line.startswith("💡"):
                    st.markdown(f"#### {line}")
                else:
                    st.markdown(line)

    # ------------------ Mock Interview ------------------
    elif page == "🧠 Mock Interview":
        st.subheader("🧠 Interview Question Generator")
        interview_type = st.selectbox("Select Round Type", ["HR", "Technical", "Coding"])

        if interview_type == "Coding":
            level = st.select_slider("Select Coding Difficulty Level", options=["1", "2", "3", "4", "5"], value="3")

        if st.button("🎤 Generate 10 Questions"):
            with st.spinner("Generating interview questions..."):
                if interview_type == "Coding":
                    dsa_chain = LLMChain(llm=llm, prompt=dsa_prompt)
                    questions = dsa_chain.run({"level": level})
                else:
                    mock_chain = LLMChain(llm=llm, prompt=mock_prompt)
                    questions = mock_chain.run({
                        "resume": resume_text,
                        "interview_type": interview_type
                    })

            st.markdown(f"### 📌 {interview_type} Round Questions")
            for q in questions.split("\n"):
                if q.strip():
                    st.markdown(f"<div class='question-block question-spacing'>{q.strip()}</div>", unsafe_allow_html=True)
else:
    st.info("📥 Upload a resume to get started.")
