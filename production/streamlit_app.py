import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import document processing libraries
try:
    import fitz  # PyMuPDF
    import docx
except ImportError:
    st.error("Required libraries not installed. Run: pip install PyMuPDF python-docx")

# Global Configurations
GOOGLE_API_KEY: str = None

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
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
        letter-spacing: 1px;
    }
    .main-header-description {
        font-size: 1.35rem;
        color: #e0e0e0;
        text-align: center;
        max-width: 800px;
        margin: 0 auto 3rem auto;
    }
    .score-card {
        padding: 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
        color: white;
        text-align: center;
        margin: 1.5rem auto;
        max-width: 350px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease-in-out;
    }
    .score-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .score-value {
        font-size: 4.5rem;
        font-weight: 900;
        line-height: 1;
        # text-shadow: 0 0 10px rgba(255, 255, 255, 0.6); 
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4a90e2;
        margin-top: 3rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #4a90e2;
        padding-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .chat-message {
        padding: 1.25rem 1.5rem; /* Increased padding for more breathing room */
        border-radius: 18px; /* Softer, more modern rounded corners */
        margin: 1rem 0; /* Increased vertical margin to separate bubbles */
        max-width: 85%; /* Ensures bubbles don't stretch across the entire width */
        line-height: 1.5; /* Improved readability */
        font-size: 1.05rem; /* Slightly larger text */
        /* Added a subtle shadow for depth */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
        transition: transform 0.2s ease-in-out; /* Smooth transition for hover effects */
    }
    .user-message {
        background-color: #3b5998; 
        color: white;
        margin-left: auto; /* Pushes the user message to the right (important for max-width) */
        margin-right: 0;
        border-bottom-right-radius: 4px; /* A slight corner difference for the tail effect */
    }
    .assistant-message {
        background-color: #f0f0f0; 
        color: #333333; /* Darker text for contrast on light background */
        margin-right: auto; /* Pushes the assistant message to the left */
        margin-left: 0;
        border-bottom-left-radius: 4px; /* A slight corner difference for the tail effect */
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
if 'GOOGLE_API_KEY' not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None
if 'ResumeModels' not in st.session_state:
    st.session_state.ResumeModels = None
if 'ResumeAnalyzer' not in st.session_state:
    st.session_state.ResumeAnalyzer = None

# load Resume Analyzer Module
if st.session_state.ResumeModels is None:
    with st.spinner("Loading Resume Analyzer Module..."):
        try:
            import agents as agents
            st.session_state.ResumeModels = "imported"
            st.success("✅ Resume Analyzer Module loaded! Welcome to AI Resume Analyzer")
        except Exception as e:
            st.error(f"Failed to load Resume Analyzer Module: {str(e)}")

# Helper Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PyMuPDF"""
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())  # Fixed: use getvalue() instead of read()
            tmp_path = tmp_file.name
        
        # Extract text using PyMuPDF
        doc = fitz.open(tmp_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        
        # Clean up temp file
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
    """Analyze resume using ResumeAnalyzer"""
    if st.session_state.ResumeAnalyzer is None:
        st.error("⚠️ ResumeAnalyzer not initialized. Please save your API key first.")
        return None
    
    try:
        # Fixed: Handle case when job_description is None or empty
        if job_description is None or job_description.strip() == "":
            job_description = "General software engineering position"
        
        result = st.session_state.ResumeAnalyzer.analyze_resume(resume_text, job_description)
        
        # Format the result to match expected structure
        formatted_result = {
            'overall_score': result.get('overall_score', 0),
            'skill_match': result.get('score_breakdown', {}).get('skill_match', 0),
            'experience': result.get('score_breakdown', {}).get('experience', 0),
            'format': result.get('score_breakdown', {}).get('format', 0),
            'readability': result.get('score_breakdown', {}).get('readability', 0),
            'feedback': {
                'strengths': result.get('strengths', []),
                'improvements': result.get('weaknesses', []),
                'suggestions': result.get('revision_suggestions', [])
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'raw_result': result  # Keep original result for reference
        }
        
        return formatted_result
    except Exception as e:
        st.error(f"Error analyzing resume: {str(e)}")
        return None

def generate_ai_response(user_message, resume_text, evaluation_results):
    """Generate AI chatbot response"""
    if st.session_state.ResumeAnalyzer is None:
        return "⚠️ AI assistant not available. Please save your API key first."
    
    try:
        # Use the LLM to generate a response
        prompt = f"""You are a helpful resume advisor. Based on the following context, answer the user's question.

Resume Text (truncated):
{resume_text[:1500]}

Evaluation Results:
{json.dumps(evaluation_results, indent=2)}

User Question: {user_message}

Provide a helpful, specific answer based on the resume and evaluation results."""
        
        response = st.session_state.ResumeAnalyzer.llm_reviewer.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

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

    with st.expander("🔍 How to use"):
        st.markdown("""
        1. Upload your resume in PDF or DOCX format.
        2. (Optional) Paste a job description to tailor the analysis.
        3. Save your Google API Key in the sidebar.
        4. Click "Analyze Resume" to get detailed feedback.
        5. Use the Chat Assistant to ask questions about your resume.
        6. View and export your evaluation results in the "View Results" tab.
        """)
    
    if st.session_state.uploaded_file_name:
        st.success(f"📄 Current file: {st.session_state.uploaded_file_name}")
    
    if st.button("🗑️ Clear All Data"):
        st.session_state.chat_history = []
        st.session_state.resume_text = None
        st.session_state.evaluation_results = None
        st.session_state.uploaded_file_name = None
        st.rerun()

    st.divider()
    st.header("🔑 API Configuration")
    
    api_key_input = st.text_input(
        "Google API Key ([Get Code](https://aistudio.google.com/api-keys))",
        type="password",
        key="google_api_key_input",
    )

    if st.button("💾 Save API Key"):
        if api_key_input.strip():
            st.session_state.GOOGLE_API_KEY = api_key_input.strip()
            
            try:
                # Import agents module
                import agents
                
                # Initialize ResumeAnalyzer
                st.session_state.ResumeAnalyzer = agents.ResumeModels(
                    api_key=st.session_state.GOOGLE_API_KEY
                )
                
                # Validate API key
                with st.spinner("⚙️ Testing API connection..."):
                    res = st.session_state.ResumeAnalyzer.check_connection()
                
                if not res:
                    raise ValueError("Invalid API Key, connection failed.")
                else: 
                    st.success("✅ Google API Key saved and verified!")
            except Exception as e:
                st.error(f"⚠️ {str(e)}")
                st.session_state.GOOGLE_API_KEY = None
                st.session_state.ResumeAnalyzer = None
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
            with st.spinner("Extracting text from file..."):
                if uploaded_file.name.endswith('.pdf'):
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = extract_text_from_docx(uploaded_file)
            
            if resume_text:
                st.session_state.resume_text = resume_text
                st.success("✅ Resume uploaded successfully!")
                
                # Show preview
                with st.expander("📄 Preview Resume Text"):
                    st.text_area("Extracted Text", resume_text, height=200, disabled=True)
                
                # Analyze button
                analyze_disabled = st.session_state.get("GOOGLE_API_KEY") is None

                if st.button(
                    "🔍 Analyze Resume",
                    type="primary",
                    use_container_width=True,
                    disabled=analyze_disabled
                ):
                    with st.spinner("Analyzing your resume... This may take a minute."):
                        jd = jd_input.strip() if jd_input.strip() else None
                        results = analyze_resume(resume_text, jd)
                        print(json.dumps(results, indent=2))  # Debugging output
                        
                        if results:
                            st.session_state.evaluation_results = results
                            st.success("✅ Analysis complete!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("⚠️ Analysis failed. Please try again.")
                
                if analyze_disabled:
                    st.warning("⚠️ Please save your Google API Key in the sidebar first.")
    
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
    elif st.session_state.ResumeAnalyzer is None:
        st.warning("⚠️ Please save your Google API Key in the sidebar first.")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask about your resume...")
        
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            time.sleep(0.5)
            
            # Generate AI response
            with st.spinner("Thinking..."):
                ai_response = generate_ai_response(
                    user_input,
                    st.session_state.resume_text,
                    st.session_state.evaluation_results
                )
            
            time.sleep(0.5)

            # Add AI response to history
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
            score_color = "yellow" if results['overall_score'] >= 80 else "orange" if results['overall_score'] >= 60 else "red"
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
                    st.success(f"{strength}")
            else:
                st.info("Keep working to build your strengths!")
        
        with col2:
            st.subheader("⚠️ Areas for Improvement")
            if feedback['improvements']:
                for improvement in feedback['improvements']:
                    st.warning(f"{improvement}")
        
        st.divider()
        
        st.subheader("💡 Actionable Suggestions")
        if feedback['suggestions']:
            for suggestion in feedback['suggestions']:
                st.info(f"{suggestion}")
        
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
{chr(10).join(f"• {s}" for s in feedback['strengths']) if feedback['strengths'] else 'None listed'}

IMPROVEMENTS NEEDED
{'='*50}
{chr(10).join(f"• {i}" for i in feedback['improvements']) if feedback['improvements'] else 'None listed'}

SUGGESTIONS
{'='*50}
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(feedback['suggestions'])) if feedback['suggestions'] else 'None listed'}
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