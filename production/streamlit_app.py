import streamlit as st
import sys
from pathlib import Path
import json
from datetime import datetime
import time

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

ROLE_MAPPING = {
    0: "INFORMATION-TECHNOLOGY",
    1: "BUSINESS-DEVELOPMENT",
    2: "FINANCE",
    3: "ADVOCATE",
    4: "ACCOUNTANT",
    5: "ENGINEERING",
    6: "CHEF",
    7: "AVIATION",
    8: "FITNESS",
    9: "SALES",
    10: "BANKING",
    11: "HEALTHCARE",
    12: "CONSULTANT",
    13: "CONSTRUCTION",
    14: "PUBLIC-RELATIONS",
    15: "HR",
    16: "DESIGNER",
    17: "ARTS",
    18: "TEACHER",
    19: "APPAREL",
    20: "DIGITAL-MEDIA",
    21: "AGRICULTURE",
    22: "AUTOMOBILE",
    23: "BPO"
}

ACCEPTANCE_MAPPING = {
    0: "Rejected",
    1: "Accepted"
}

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
if 'role_prediction' not in st.session_state:
    st.session_state.role_prediction = None
if 'acceptance_prediction' not in st.session_state:
    st.session_state.acceptance_prediction = None

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
        
        st.session_state.role_prediction = st.session_state.ResumeAnalyzer.predict_role(resume_text)
        st.session_state.acceptance_prediction = st.session_state.ResumeAnalyzer.predict_acceptance(resume_text)

        # Format the result to match expected structure
        formatted_result = {
            'explanation': result.get('explanation', ''),
            'keyword_recommendations': result.get('keyword_recommendations', []),
            'summary_rewrite': result.get('summary_rewrite', ''),
            'ats_tips': result.get('ats_tips', []),
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
            'confidence_level': result.get('confidence_level', 'unknown'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'raw_result': result
        }

        # Setting up chat context
        st.session_state.ResumeAnalyzer.set_system_message([
            {
            "role": "system",
            "content": f"Resume analysis completed.{formatted_result}. answer the user questions based on this analysis."
            },
            {
                "role": "assistant",
                "content": f"""
✅ Resume analyzed successfully! Overall Score: {formatted_result['overall_score']}/100

{formatted_result['explanation']}

You can now view detailed results in the 'View Results' tab or ask me anything about your resume here

"""
            }
        ])

        # Add chat history entry for analysis
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"""
✅ Resume analyzed successfully! Overall Score: {formatted_result['overall_score']}/100

{formatted_result['explanation']}

You can now view detailed results in the 'View Results' tab or ask me anything about your resume here

"""
        })
        
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
        ["Upload & Analyze", "Chat Assistant", "View Results", "ML Models Descriptions", "AI Models Descriptions"],
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
            
            st.metric("Predicted Role", f"{st.session_state.role_prediction["role_name"]}", help="Predicted job role based on resume content")
            st.info(f"ℹ️ **Confidence: {st.session_state.role_prediction["confidence"]*100:.2f}%**")
        
        with col2:
            st.metric("Format", f"{results['format']}/20", help="Resume formatting and structure")
            st.metric("Readability", f"{results['readability']}/10", help="How easy your resume is to read")
        
            decision = st.session_state.acceptance_prediction["prediction_label"]
            if decision == "Accepted":
                st.metric("Decision", f"✅ {decision}", help="Predicted acceptance based on resume quality")
                st.info(f"ℹ️ **Probability: {st.session_state.acceptance_prediction['probability']*100:.2f}%**")
            else:
                st.metric("Decision", f"❌ {decision}", help="Predicted acceptance based on resume quality")
                st.info(f"ℹ️ **Probability: {st.session_state.acceptance_prediction['probability']*100:.2f}%**")

        print(st.session_state.acceptance_prediction)

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
        
        col1, col2, col3 = st.columns(3)
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

File: {st.session_state.uploaded_file_name}
Date: {results['timestamp']}

SCORES
Overall Score: {results['overall_score']}/100
- Skill Match: {results['skill_match']}/40
- Experience: {results['experience']}/30
- Format: {results['format']}/20
- Readability: {results['readability']}/10
- Predicted Role: {st.session_state.role_prediction['role_name']} (Confidence: {st.session_state.role_prediction['confidence']*100:.2f}%)
- Decision: {st.session_state.acceptance_prediction['prediction_label']} (Probability: {st.session_state.acceptance_prediction['probability']*100:.2f}%)

STRENGTHS
{chr(10).join(f"• {s}" for s in feedback['strengths']) if feedback['strengths'] else 'None listed'}

IMPROVEMENTS NEEDED
{chr(10).join(f"• {i}" for i in feedback['improvements']) if feedback['improvements'] else 'None listed'}

SUGGESTIONS
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(feedback['suggestions'])) if feedback['suggestions'] else 'None listed'}
            """
            
            st.download_button(
                label="📝 Download Text Report",
                data=text_report,
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

        with col3:
            st.download_button(
                label="📊 Download Raw Results",
                data=json.dumps(results['raw_result'], indent=2),
                file_name=f"resume_raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

elif app_mode == "ML Models Descriptions":
    st.markdown('<div class="section-header">🚀 Our ML Models</div>', unsafe_allow_html=True)
    st.info("Our models evaluations and prompts.")

    # --- Model Descriptions ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Role Classification Model")
        st.markdown("""
        - **Purpose:** Classifies resumes into job roles (e.g., Software Engineer, Data Scientist).
        - **Model:** Using TF-IDF vectorizer and Naive Bayes Classifier with SMOTE oversampling.
        - **Dataset:** [kaggle | Snehaan Bhawal/Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
        """)
        st.image("production/images/role_model_wordcloud.png", caption="Role Classification Model Word Cloud")

    with col2:
        st.subheader("📄 Acceptance Classification Model")
        st.markdown("""
        - **Purpose:** Predicts acceptance probability based on resume quality.
        - **Model:** Logistic Regression with TF-IDF features.
        - **Dataset:** [Hugging Face | AzharAli05/Resume-Screening-Dataset](https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset).
        """)
        st.image("production/images/acceptance_model_wordcloud.png", caption="Acceptance Classification Model Word Cloud")

    st.divider()

    st.markdown("### 📈 Model Performance Metrics (Raw Data)")
    
    # Metrics DataFrame
    data = {
        "Model": ["Role Classifier", "Acceptance Classifier"],
        "Accuracy": [0.5895, 0.5660],
        "F1 Score": [0.5895, 0.5664],
        "Precision": [0.6154, 0.5630],
        "Recall": [0.5651, 0.5647]
    }

    df = pd.DataFrame(data).set_index("Model")
    
    # Use a styled dataframe for a prettier look
    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

    st.divider()

    st.markdown("### 📊 Metrics Visualization")
    
    # Function to add data labels to the bars
    def add_labels(ax):
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.3f}', 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', 
                        va='bottom', 
                        xytext=(0, 5), 
                        textcoords='offset points', 
                        fontsize=12,
                        fontweight='bold')

    sns.set_theme(style="white", palette="tab10") 

    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        # Tabs for each metric
        tab_list = st.tabs(metrics)
        
        for i, metric in enumerate(metrics):
            with tab_list[i]:
                # Create a separate, focused plot for each metric
                fig, ax = plt.subplots(figsize=(6, 4))
                
                # Use the DataFrame column directly for the plot
                metric_data = df[metric].reset_index()

                # Create the bar plot
                sns.barplot(data=metric_data, x="Model", y=metric, ax=ax, hue="Model", legend=False)
                
                # Set Y-axis limits from 0 to 1 for standard metric comparison
                ax.set_ylim(0, 1.0)
                
                # Add labels, remove axis titles for a cleaner look, and add a main title
                ax.set_title(f'{metric} Comparison', fontsize=16, fontweight='bold', pad=15)
                ax.set_ylabel("Score", fontsize=12)
                ax.set_xlabel("")
                
                # Add data labels
                add_labels(ax)
                
                # Display the plot
                st.pyplot(fig)
                plt.close(fig)

    role_matrix_data = np.array([[21, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
[0, 7, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 6, 2, 0, 1, 2, 1],
[0, 1, 6, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
[0, 1, 0, 3, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 2, 6, 0],
[0, 0, 0, 0, 3, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 1, 0, 10],
[0, 1, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 15, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 2, 1, 0, 1, 0],
[2, 2, 0, 0, 0, 0, 0, 14, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 1, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0],
[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2],
[0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 16, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
[3, 1, 1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 2, 0, 2, 0, 1, 0, 1, 1, 2, 3, 1, 0],
[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 12, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 0, 0, 0, 0, 2, 2, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 0, 6, 0, 0, 1],
[12, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 1, 0, 1, 0, 0],
[0, 0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 1, 2, 2],
[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 13, 3, 1, 0, 0, 0],
[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 16, 2, 0],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 17]]
    )

    st.divider()
    st.subheader("🔢 Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_cm = pd.DataFrame(role_matrix_data, index=ROLE_MAPPING, columns=ROLE_MAPPING)

        plt.figure(figsize=(10, 8))
        sns.set_context("notebook", font_scale=1.1)

        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='gray')

        plt.title('Confusion Matrix: Role Classification Model', fontsize=16, pad=20)
        plt.ylabel('Actual Role', fontsize=12, labelpad=10)
        plt.xlabel('Predicted Role', fontsize=12, labelpad=10)

        st.pyplot(plt)
        plt.close()

    accuracy_matix_data = np.array([[582, 441], [447, 576]])

    with col2:
        df_cm = pd.DataFrame(accuracy_matix_data, index=ACCEPTANCE_MAPPING, columns=ACCEPTANCE_MAPPING)

        plt.figure(figsize=(10, 8))
        sns.set_context("notebook", font_scale=1.1)

        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='gray')

        plt.title('Confusion Matrix: Role Classification Model', fontsize=16, pad=20)
        plt.ylabel('Actual Decision', fontsize=12, labelpad=10)
        plt.xlabel('Predicted Decision', fontsize=12, labelpad=10)

        st.pyplot(plt)
        plt.close()

if app_mode == "AI Models Descriptions":

    st.markdown('<div class="section-header">🗨️ Our AI Models</div>', unsafe_allow_html=True)
    st.info("Details about the AI models powering the Resume Analyzer — including custom prompts, scoring logic, and verification flow.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 LLM 1 — Resume Reviewer",
        "🛡️ LLM 2 — Verification Layer",
        "📄 Custom Prompts",
        "🔗 Pipeline Workflow"
    ])

    with tab1:
        st.subheader("🔍 LLM 1 | Resume Review & Scoring Engine")

        st.markdown("""
        **Purpose:**  
        LLM 1 performs a *deep evaluation* of the resume using the model outputs and the job description.  
        It generates a professional, hiring-style evaluation in **structured JSON**.

        **Key Responsibilities:**
        - Score resume on a **0–100 scale**
        - Generate detailed **sub-scores**
        - Identify strengths & weaknesses
        - Recommend improvements & missing keywords
        - Produce ATS-focused suggestions
        - Ensure scoring aligns with ML outputs (role, acceptance probability, similarity)

        **Input Sources:**
        - Cleaned resume text  
        - Job description (if provided)
        - Role prediction (ML)  
        - Acceptance prediction (ML)  
        - Similarity model outputs  

        **Output:**  
        A strict JSON object with:
        - overall_score  
        - score_breakdown  
        - strengths, weaknesses  
        - revision suggestions  
        - keyword recommendations  
        - summary rewrite  
        - ats tips  
        - confidence level  
        """)

    with tab2:
        st.subheader("🛡️ LLM 2 | Consistency & Hallucination Verifier")

        st.markdown("""
        **Role:**  
        LLM 2 acts as a **safety and correctness validator** for LLM 1.

        **What this layer checks:**
        - 🧪 *Hallucination detection*: Are claims supported by the resume content?
        - 🔍 *Score consistency*: Do sub-scores and explanation match the final score?
        - ⚠️ *Logical correctness*: Strengths/weaknesses align with scoring.
        - 🔑 *Keyword verification*: Suggested keywords must be truly missing.
        - 🤝 *ML model alignment*: Analysis must not contradict ML predictions.
        - 🧹 *Cleans text*: Removes invalid bullet characters (•, -, etc.)

        **Possible Outcomes:**
        - `APPROVED` — LLM1 is correct  
        - `APPROVED_WITH_MODIFICATIONS` — LLM2 adjusts scoring or rewrites lists  

        **Output:**
        A JSON dict containing:
        - is_valid  
        - hallucinations_found  
        - inconsistencies_found  
        - score_adjustment  
        - recommended_changes  
        - final_verdict  
        """)

    with tab3:
        st.subheader("📄 Custom Prompts Used")

        st.markdown("### 🔍 LLM 1 Prompt (Reviewer & Scoring)")
        with st.expander("Click to view full prompt"):
            st.code("""
You are an expert resume/CV reviewer and career consultant with deep understanding of ATS systems and hiring practices.

I have analyzed a resume using ML models. Your task is to review these outputs and provide a comprehensive assessment.

**RESUME TEXT:**
{resume_text[:3000]}

**JOB DESCRIPTION:**
{jd_text}

**ML MODEL OUTPUTS:**
1. Predicted Role: {role_prediction['role_name']} (confidence: {role_prediction['confidence']:.2%})
2. Acceptance Probability: {acceptance_prediction['probability']:.2%} (Predicted: {'ACCEPTED' if acceptance_prediction['accepted'] else 'REJECTED'})
3. Resume-JD Similarity Score: {similarity_result['similarity_score']:.3f} (0-1 scale, semantic similarity)
4. Missing Keywords: {', '.join(similarity_result['missing_keywords'][:15])}

**YOUR TASK:**
Provide a comprehensive assessment in JSON format with these exact keys:

1. **overall_score** (0-100): Overall match quality score
2. **score_breakdown**: Object with sub-scores:
   - skill_match (0-40): How well skills align with JD
   - experience (0-30): Relevance and quality of experience
   - format (0-20): Resume formatting and ATS compatibility
   - readability (0-10): Clarity and professionalism
3. **explanation**: 2-3 sentences explaining the overall score
4. **strengths**: Array of 3-5 key strengths
5. **weaknesses**: Array of 3-5 key weaknesses or gaps
6. **revision_suggestions**: Array of 5-8 specific, actionable improvements
7. **keyword_recommendations**: Array of 5-10 keywords/skills to add
8. **summary_rewrite**: Suggested 2-3 sentence professional summary optimized for this role
9. **ats_tips**: Array of 3-5 ATS optimization tips
10. **confidence_level**: Your confidence in this assessment (high/medium/low)

**IMPORTANT GUIDELINES:**
- Base your assessment on BOTH the resume content AND the ML model predictions
- If ML models show low confidence or contradictory results, mention this in your explanation
- Be specific and actionable in your suggestions
- Consider the missing keywords when making recommendations
- Score honestly - don't inflate scores if the match is poor

Return ONLY valid JSON with no markdown formatting:
{{
  "overall_score": <number>,
  "score_breakdown": {{
    "skill_match": <number>,
    "experience": <number>,
    "format": <number>,
    "readability": <number>
  }},
  "explanation": "<string>",
  "strengths": ["<string>", ...],
  "weaknesses": ["<string>", ...],
  "revision_suggestions": ["<string>", ...],
  "keyword_recommendations": ["<string>", ...],
  "summary_rewrite": "<string>",
  "ats_tips": ["<string>", ...],
  "confidence_level": "<string>"
}}
""")

        st.markdown("### 🛡️ LLM 2 Prompt (Verifier)")
        with st.expander("Click to view full prompt"):
            st.code("""
You are a fact-checking and verification expert. Your role is to verify another AI's assessment for accuracy and hallucinations.

**ORIGINAL DATA:**

Resume Text (truncated):
{resume_text[:2000]}

Job Description:
{jd_text}

ML Model Outputs:
- Predicted Role: {role_prediction['role_name']} (confidence: {role_prediction['confidence']:.2%})
- Acceptance Probability: {acceptance_prediction['probability']:.2%}
- Similarity Score: {similarity_result['similarity_score']:.3f}
- Missing Keywords: {', '.join(similarity_result['missing_keywords'][:10])}

**LLM 1's ASSESSMENT TO VERIFY:**
{json.dumps(llm1_assessment, indent=2)}

**YOUR VERIFICATION TASKS:**

1. **Check for Hallucinations**: Verify each claim in the assessment can be supported by the resume text or ML outputs
2. **Score Consistency**: Verify the overall_score aligns with score_breakdown
3. **Logic Check**: Ensure strengths/weaknesses align with the score given
4. **Keyword Verification**: Check if suggested keywords are actually missing and relevant
5. **ML Output Alignment**: Verify assessment doesn't contradict ML model predictions
6. **Remove elements**: "•", "\-", and similar bullet characters from lists in the assessment (usually found in strengths, weaknesses, revision_suggestions, keyword_recommendations, ats_tips)

**RESPOND IN JSON FORMAT:**
{{
  "is_valid": <boolean>,
  "confidence": <"high" | "medium" | "low">,
  "hallucinations_found": [<list of specific hallucinations or false claims>],
  "inconsistencies_found": [<list of logical inconsistencies>],
  "score_adjustment": <number, suggested adjustment to overall_score, can be negative, 0, or positive>,
  "verification_notes": "<string explaining verification results>",
  "recommended_changes": {{
    "overall_score": <number or null if no change>,
    "strengths": [<corrected list or null>],
    "weaknesses": [<corrected list or null>],
    "revision_suggestions": [<corrected list or null>]
  }},
  "final_verdict": "<'APPROVED' | 'APPROVED_WITH_MODIFICATIONS'>"
}}

Be strict but fair. Only flag actual errors, not stylistic differences.
""")

        st.markdown("""
        **Notes:**  
        - Both prompts enforce **strict JSON output**  
        - LLM2 evaluates and may adjust results  
        - Prompts are optimized for Gemini 2.5 Flash  
        """)

    with tab4:
        st.subheader("🔗 End-to-End Pipeline Workflow")

        st.markdown("""
        ### **🧬 Complete Resume Evaluation Pipeline**
        1. **Role Prediction (ML)**  
           Predicts most likely job category from resume.

        2. **Acceptance Prediction (ML)**  
           Predicts likelihood of resume acceptance.

        3. **Semantic Similarity (NLP)**  
           Measures how aligned resume is with job description.  
           Also extracts missing keywords.

        4. **LLM 1 (Reviewer & Scorer)**  
           Generates full structured evaluation.

        5. **LLM 2 (Verifier)**  
           Validates LLM1 output for accuracy and logical consistency.

        6. **Final Assessment Engine**  
           Applies adjustments and produces final trusted result.

        ### **🎉 What Users Get**
        - Detailed resume analysis  
        - Skill match score  
        - ATS optimization tips  
        - Strengths & weaknesses  
        - Summary rewrite  
        - Verified final score (0–100)  
        """)



# Footer
st.divider()
st.markdown("""
    <div style="
    text-align: center; 
    padding: 10px;  
    font-size: 0.85em; 
    color: #6c757d; 
    margin-top: 20px;
">
    <p style="margin: 0 0 5px 0;">
        Built with <span style="color: #e74c3c;">❤️</span> using Streamlit | Resume Analyzer AI v1.0
    </p>
    <p style="margin: 0;">
        Authors: Aisyah Naurotul Athifah, Ilham Satria Difta, Muhammad Raffi Akhdilputra
    </p>
</div>
""", unsafe_allow_html=True)