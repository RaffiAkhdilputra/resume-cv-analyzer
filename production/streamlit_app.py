import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
from agents import ResumeModels

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import document processing libraries
try:
    import fitz  # PyMuPDF
    import docx
except ImportError:
    st.error("Required libraries not installed. Run: pip install PyMuPDF python-docx")

# Global Configurations
GOOGLE_API_KEY:str

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header-description {
        font-size: 1.2rem;
        color: white;
        text-align: center;
    }
    .score-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .score-value {
        font-size: 3rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #4464ad;
        margin-left: 2rem;
        color: white;
    }
    .assistant-message {
        background-color: #7d4600;
        color: white;
        margin-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None

# Helper Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PyMuPDF"""
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        
        # Extract text using PyMuPDF
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        return text.strip()
    except Exception as e:

        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file using python-docx"""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return None

def analyze_resume(resume_text, job_description=None):
    # TODO: Implement analysis logic
    if job_description is not None:
        pass
    return results

def generate_feedback(overall, skill, exp, fmt, read):
    """Generate detailed feedback based on scores"""
    pass
    return feedback

def generate_ai_response(user_message, resume_text, evaluation_results):
    """
    Generate AI chatbot response
    This is a placeholder - replace with actual LLM integration
    """
    # TODO: Integrate with your LLM (LLM 1 and LLM 2 from the diagram)
    # Use prompt engineering based on the evaluation results

    return "This is a placeholder response from the AI assistant."

# Main App Header
st.markdown('<div class="main-header">📄 Resume Analyzer AI</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header-description">Upload your resume and get AI-powered feedback with detailed scoring</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Navigation")
    
    app_mode = st.radio(
        "Select Mode:",
        ["Upload & Analyze", "Chat Assistant", "View Results"],
        index=0
    )
    
    st.divider()
    
    st.header("ℹ️ About")
    st.info("""
    This AI-powered tool analyzes your resume and provides:
    - Overall score (0-100)
    - Skill match analysis
    - Experience evaluation
    - Format assessment
    - Readability check
    """)
    
    if st.session_state.uploaded_file_name:
        st.success(f"📄 Current file: {st.session_state.uploaded_file_name}")
    
    if st.button("🗑️ Clear All Data"):
        st.session_state.chat_history = []
        st.session_state.resume_text = None
        st.session_state.evaluation_results = None
        st.session_state.uploaded_file_name = None
        st.rerun()

    api_key_input = st.text_input(
        "Google API Key",
        type="password",
        key="google_api_key_input",
    )

    if st.button("💾 Save API Key"):
        if api_key_input.strip():
            st.session_state.GOOGLE_API_KEY = api_key_input.strip()
            st.success("✅ Google API Key saved!")
        else:
            st.error("⚠️ Please enter a valid API key.")

# Main Content Area
if app_mode == "Upload & Analyze":
    st.markdown('<div class="section-header">📤 Upload Resume</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        jd_input = st.text_area(
            "Paste Job Description (optional):",
            height=150,
            help="Provide a job description to tailor the analysis (optional)"
        )

        uploaded_file = st.file_uploader(
            "Choose your resume file (PDF or DOCX)",
            type=['pdf', 'docx'],
            help="Upload your resume in PDF or DOCX format"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_file_name = uploaded_file.name
            
            # Extract text based on file type
            if uploaded_file.name.endswith('.pdf'):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
            
            if resume_text:
                st.session_state.resume_text = resume_text
                st.success("✅ Resume uploaded successfully!")
                
                # Show preview
                with st.expander("📄 Preview Resume Text"):
                    st.text_area("Extracted Text", resume_text, height=200)
                
                # Analyze button
                if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
                    with st.spinner("Analyzing your resume..."):
                        jd = jd_input.strip() if jd_input.strip() else None
                        results = analyze_resume(resume_text, jd)
                        st.session_state.evaluation_results = results
                        st.success("✅ Analysis complete!")
                        st.rerun()
    
    with col2:
        st.info("""
        **Supported formats:**
        - PDF (.pdf)
        - Word Document (.docx)
        
        **What we analyze:**
        - Skill matching
        - Experience level
        - Format quality
        - Readability
        """)

elif app_mode == "Chat Assistant":
    st.markdown('<div class="section-header">💬 Chat with AI Assistant</div>', unsafe_allow_html=True)
    
    if not st.session_state.resume_text:
        st.warning("⚠️ Please upload a resume first in the 'Upload & Analyze' tab.")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message">👤 You: {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message">🤖 Assistant: {message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask about your resume...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate AI response
            ai_response = generate_ai_response(
                user_input,
                st.session_state.resume_text,
                st.session_state.evaluation_results
            )
            
            # Add AI response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            st.rerun()

elif app_mode == "View Results":
    st.markdown('<div class="section-header">📊 Evaluation Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.evaluation_results:
        st.warning("⚠️ No evaluation results yet. Please upload and analyze a resume first.")
    else:
        results = st.session_state.evaluation_results
        
        # Overall Score Display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            score_color = "green" if results['overall_score'] >= 80 else "orange" if results['overall_score'] >= 60 else "red"
            st.markdown(f"""
                <div class="score-card">
                    <div style="font-size: 1.2rem;">Overall Score</div>
                    <div class="score-value" style="color: {score_color};">{results['overall_score']}/100</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Detailed Scores
        st.subheader("📈 Detailed Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Skill Match", f"{results['skill_match']}/40", help="How well your skills match the requirements")
            st.metric("Experience", f"{results['experience']}/30", help="Quality and relevance of experience")
        
        with col2:
            st.metric("Format", f"{results['format']}/20", help="Resume formatting and structure")
            st.metric("Readability", f"{results['readability']}/10", help="How easy your resume is to read")
        
        st.divider()
        
        # Feedback Sections
        feedback = results['feedback']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Strengths")
            if feedback['strengths']:
                for strength in feedback['strengths']:
                    st.success(f"• {strength}")
            else:
                st.info("Keep working to build your strengths!")
        
        with col2:
            st.subheader("⚠️ Areas for Improvement")
            if feedback['improvements']:
                for improvement in feedback['improvements']:
                    st.warning(f"• {improvement}")
        
        st.divider()
        
        st.subheader("💡 Actionable Suggestions")
        if feedback['suggestions']:
            for i, suggestion in enumerate(feedback['suggestions'], 1):
                st.info(f"{i}. {suggestion}")
        
        # Export Results
        st.divider()
        st.subheader("📥 Export Results")
        
        export_data = {
            "file_name": st.session_state.uploaded_file_name,
            "analysis_date": results['timestamp'],
            "scores": {
                "overall": results['overall_score'],
                "skill_match": results['skill_match'],
                "experience": results['experience'],
                "format": results['format'],
                "readability": results['readability']
            },
            "feedback": feedback
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download JSON Report",
                data=json.dumps(export_data, indent=2),
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Text report
            text_report = f"""
RESUME ANALYSIS REPORT
{'='*50}

File: {st.session_state.uploaded_file_name}
Date: {results['timestamp']}

SCORES
{'='*50}
Overall Score: {results['overall_score']}/100
- Skill Match: {results['skill_match']}/40
- Experience: {results['experience']}/30
- Format: {results['format']}/20
- Readability: {results['readability']}/10

STRENGTHS
{'='*50}
{chr(10).join(f"• {s}" for s in feedback['strengths'])}

IMPROVEMENTS NEEDED
{'='*50}
{chr(10).join(f"• {i}" for i in feedback['improvements'])}

SUGGESTIONS
{'='*50}
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(feedback['suggestions']))}
            """
            
            st.download_button(
                label="📝 Download Text Report",
                data=text_report,
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using Streamlit | Resume Analyzer AI v1.0</p>
    </div>
""", unsafe_allow_html=True)