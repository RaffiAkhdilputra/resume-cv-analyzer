import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
from typing import Dict, Any, List, Optional
import numpy as np

# LangChain + Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# TensorFlow tokenizer and sequence processing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# SentenceTransformer for similarity computation
from sentence_transformers import SentenceTransformer

# Joblib for loading sklearn models
import joblib

# Global configuration
GEMINI_MODEL = "gemini-2.5-flash"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Model paths
DEFAULT_ROLE_MODEL_PATH = "production/model/role-classification-model.joblib"
DEFAULT_ROLE_LABEL_ENCODER = "production/model/role_label_encoder.joblib"
DEFAULT_ROLE_TFIDF_VECTORIZER_PATH = "production/model/role_tfidf_vectorizer.joblib"
DEFAULT_ACCEPTANCE_MODEL_PATH = "production/model/acceptance_classification_model.joblib"
DEFAULT_ACCEPTANCE_TFIDF_VECTORIZER_PATH = "production/model/acceptance_tfidf_vectorizer.joblib"
DEFAULT_ACCEPTANCE_LABEL_ENCODER = "production/model/acceptance_label_encoder.joblib"

# Role mapping
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


# ACCEPTANCE CLASSIFIER
class AcceptanceClassifier:
    """
    Acceptance classifier using joblib-saved sklearn model
    and TF-IDF vectorizer.
    """

    def __init__(self,
                 model_path: str = DEFAULT_ACCEPTANCE_MODEL_PATH,
                 vectorizer_path: str = DEFAULT_ACCEPTANCE_TFIDF_VECTORIZER_PATH
):
        
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        self.acceptance_model = None
        self.vectorizer = None
        
        self._init_acceptance_model()

    def _init_acceptance_model(self):
        """Load the joblib model and TF-IDF vectorizer."""
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.acceptance_model = joblib.load(self.model_path)

        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {self.vectorizer_path}")

        self.vectorizer = joblib.load(self.vectorizer_path)

    def _preprocess_text(self, text: str):
        """Transform text into a TF-IDF sparse vector."""
        return self.vectorizer.transform([text])

    def predict_acceptance(self, text: str) -> Dict[str, Any]:
        """Return acceptance probability."""
        if not text or len(text.strip()) == 0:
            raise ValueError("Input text is empty.")

        processed = self._preprocess_text(text)

        try:
            # Sklearn model with probabilities
            if hasattr(self.acceptance_model, "predict_proba"):
                proba = self.acceptance_model.predict_proba(processed)
                prob = float(proba[0, 1])  # class 1 = accepted

            else:
                # Fallback if model has no probas
                pred = self.acceptance_model.predict(processed)[0]
                prob = 1.0 if pred == 1 else 0.0

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

        prediction_class = 1 if prob >= 0.5 else 0

        return {
            "accepted": prediction_class == 1,
            "probability": prob,
            "prediction_class": prediction_class,
            "prediction_label": "Accepted" if prediction_class == 1 else "Rejected"
        }

class RoleClassifier:
    """
    Role classifier using joblib-saved sklearn model, TF-IDF vectorizer,
    and LabelEncoder for class name mapping.
    """

    def __init__(self,
                 model_path: str = DEFAULT_ROLE_MODEL_PATH,
                 encoder_path: str = DEFAULT_ROLE_LABEL_ENCODER,
                 vectorizer_path: str = DEFAULT_ROLE_TFIDF_VECTORIZER_PATH
                 ):
        
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.vectorizer_path = vectorizer_path
        
        self.role_model = None
        self.label_encoder = None
        self.vectorizer = None
        
        self._init_role_model()

    def _init_role_model(self):
        """Load model, label encoder, and TF-IDF vectorizer."""

        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.role_model = joblib.load(self.model_path)

        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {self.encoder_path}")

        self.label_encoder = joblib.load(self.encoder_path)

        # Load TF-IDF vectorizer
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found: {self.vectorizer_path}")

        self.vectorizer = joblib.load(self.vectorizer_path)

    def _preprocess_text(self, text: str):
        """Transform text using the TF-IDF vectorizer."""
        return self.vectorizer.transform([text])  # sparse matrix

    def predict_role(self, text: str) -> Dict[str, Any]:
        """
        Predict job role from resume text.
        """
        if not text or len(text.strip()) == 0:
            raise ValueError("Input text is empty.")

        processed = self._preprocess_text(text)

        try:
            # Predict class index
            class_idx = self.role_model.predict(processed)[0]

            # Get probabilities (if supported)
            if hasattr(self.role_model, "predict_proba"):
                proba = self.role_model.predict_proba(processed)[0]
                confidence = float(np.max(proba))
                all_probs = proba.tolist()
            else:
                # Probabilities not available
                all_probs = [0.0] * len(self.label_encoder.classes_)
                all_probs[class_idx] = 1.0
                confidence = 1.0

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")

        # Convert class index → class label (role name)
        role_name = self.label_encoder.inverse_transform([class_idx])[0]

        return {
            "role_class": int(class_idx),
            "role_name": role_name,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "top_3_roles": self._get_top_k_roles(all_probs, k=3)
        }

    def _get_top_k_roles(self, probabilities: list, k: int = 3) -> list:
        """Return the top-k most likely roles."""
        top_indices = np.argsort(probabilities)[-k:][::-1]

        results = []
        for idx in top_indices:
            role_name = self.label_encoder.inverse_transform([idx])[0]
            results.append({
                "class_id": int(idx),
                "role_name": role_name,
                "probability": float(probabilities[idx])
            })

        return results


# Resume Class containing all models and LLMs
class ResumeModels:
    """
    Container for trained ML models with dual LLM verification system.
    LLM 1: Reviews model outputs and provides detailed assessment
    LLM 2: Verifies LLM 1's response for hallucinations
    """
    
    def __init__(
        self,
        api_key: str,
        embed_model_name: str = EMBED_MODEL_NAME,
        role_mapping: Dict[int, str] = None,
    ):
        """
        Initialize ChatGoogleGenerativeAI + embedding + ML models.
        
        Args:
            api_key: Google Gemini API key
            embed_model_name: SentenceTransformer model name for similarity computation
            role_model_path: Path to role classification model
            role_tokenizer_path: Path to role tokenizer pickle file
            acceptance_model_path: Path to acceptance classification model
            acceptance_tokenizer_path: Path to acceptance tokenizer pickle file
            role_mapping: Optional custom role mapping dictionary
        """
        # Store API key
        self.api_key = api_key
        os.environ['GOOGLE_API_KEY'] = api_key
        
        # Chat models (Gemini)
        self.llm_reviewer = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.1,
            api_key=api_key
        )
        
        self.llm_verifier = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.0,
            api_key=api_key
        )
        
        # Embeddings (HF SentenceTransformer) - only for similarity computation
        self.embedding_model = SentenceTransformer(embed_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Role mapping
        self.role_mapping = role_mapping or ROLE_MAPPING
        
        # Model containers
        self.role_classifier = None
        self.acceptance_classifier = None
        
        # Load models immediately
        self._init_role_model()
        self._init_acceptance_model()
    
    def _init_role_model(self):
        """Initialize role classifier using RoleClassifier class."""
        try:
            self.role_classifier = RoleClassifier()
        except Exception as e:
            print(f"Warning: Role classifier could not be loaded: {e}")
            self.role_classifier = None
    
    def _init_acceptance_model(self):
        """Initialize acceptance classifier using AcceptanceClassifier class."""
        self.acceptance_classifier = AcceptanceClassifier()
    
    def predict_role(self, text: str) -> Dict[str, Any]:
        """
        Predict job role classification from resume text.
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with role_class (int), role_name (str), and confidence (float)
        """
        if self.role_classifier is None:
            # Fallback mode
            return {
                "role_class": 0,
                "role_name": "INFORMATION-TECHNOLOGY",
                "confidence": 0.5,
                "all_probabilities": [],
                "note": "Using fallback prediction - role classifier not loaded"
            }
        
        return self.role_classifier.predict_role(text)
    
    def predict_acceptance(self, text: str) -> Dict[str, Any]:
        """
        Predict acceptance probability from resume text.
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with accepted (bool) and probability (float)
        """
        if self.acceptance_classifier is None:
            raise RuntimeError("Acceptance classifier is not loaded.")
        
        return self.acceptance_classifier.predict_acceptance(text)
    
    def compute_similarity(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """
        Compute semantic similarity between resume and job description.
        
        Args:
            resume_text: Resume text
            jd_text: Job description text
            
        Returns:
            Dictionary with similarity_score and missing_keywords
        """
        # Get embeddings
        texts = [resume_text, jd_text]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        resume_vec = embeddings[0]
        jd_vec = embeddings[1]
        
        # Cosine similarity
        cosine_sim = np.dot(resume_vec, jd_vec) / (
            np.linalg.norm(resume_vec) * np.linalg.norm(jd_vec) + 1e-9
        )
        
        # Find missing keywords
        jd_tokens = set(jd_text.lower().split())
        resume_tokens = set(resume_text.lower().split())
        missing_keywords = list(jd_tokens - resume_tokens)[:20]
        
        return {
            'similarity_score': float(cosine_sim),
            'missing_keywords': missing_keywords
        }
    
    def llm1_review_and_score(
        self,
        resume_text: str,
        jd_text: str,
        role_prediction: Dict[str, Any],
        acceptance_prediction: Dict[str, Any],
        similarity_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM 1: Reviews all model outputs and provides comprehensive assessment with 0-100 score.
        
        Args:
            resume_text: Cleaned resume text
            jd_text: Job description
            role_prediction: Output from predict_role()
            acceptance_prediction: Output from predict_acceptance()
            similarity_result: Output from compute_similarity()
            
        Returns:
            Dictionary with detailed assessment including score, strengths, weaknesses, etc.
        """
        prompt = f"""
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
"""
        
        try:
            response = self.llm_reviewer.invoke(prompt)
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.rsplit('```', 1)[0]
            
            assessment = json.loads(response_text)
            
            # Validate score is in range
            if not (0 <= assessment.get('overall_score', 0) <= 100):
                assessment['overall_score'] = min(100, max(0, assessment['overall_score']))
            
            return assessment
            
        except json.JSONDecodeError as e:
            return {
                'overall_score': 0,
                'explanation': f'Error parsing LLM 1 response: {str(e)}',
                'raw_response': response.content,
                'error': True
            }
        except Exception as e:
            return {
                'overall_score': 0,
                'explanation': f'Error in LLM 1 review: {str(e)}',
                'error': True
            }
    
    def llm2_verify_assessment(
        self,
        resume_text: str,
        jd_text: str,
        llm1_assessment: Dict[str, Any],
        role_prediction: Dict[str, Any],
        acceptance_prediction: Dict[str, Any],
        similarity_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM 2: Verifies LLM 1's assessment for hallucinations and inconsistencies.
        
        Args:
            resume_text: Cleaned resume text
            jd_text: Job description
            llm1_assessment: Output from llm1_review_and_score()
            role_prediction: Output from predict_role()
            acceptance_prediction: Output from predict_acceptance()
            similarity_result: Output from compute_similarity()
            
        Returns:
            Dictionary with verification results and final trusted assessment
        """
        prompt = f"""
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
"""
        
        try:
            response = self.llm_verifier.invoke(prompt)
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.rsplit('```', 1)[0]
            
            verification = json.loads(response_text)
            
            return verification
            
        except json.JSONDecodeError as e:
            return {
                'is_valid': True,  # Default to trusting LLM 1 if verification fails
                'confidence': 'low',
                'verification_notes': f'Error parsing LLM 2 response: {str(e)}',
                'final_verdict': 'APPROVED',
                'error': True
            }
        except Exception as e:
            return {
                'is_valid': True,
                'confidence': 'low',
                'verification_notes': f'Error in LLM 2 verification: {str(e)}',
                'final_verdict': 'APPROVED',
                'error': True
            }
    
    def get_final_assessment(
        self,
        llm1_assessment: Dict[str, Any],
        llm2_verification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine LLM 1's assessment with LLM 2's verification to produce final trusted results.
        
        Args:
            llm1_assessment: Output from llm1_review_and_score()
            llm2_verification: Output from llm2_verify_assessment()
            
        Returns:
            Final trusted assessment dictionary
        """
        final = dict(llm1_assessment)
        
        # Apply score adjustment if recommended
        if llm2_verification.get('score_adjustment', 0) != 0:
            old_score = final['overall_score']
            new_score = old_score + llm2_verification['score_adjustment']
            final['overall_score'] = min(100, max(0, new_score))
            final['score_adjusted'] = True
            final['score_adjustment_reason'] = llm2_verification.get('verification_notes', '')
        
        # Apply recommended changes if verification suggests modifications
        if llm2_verification.get('final_verdict') == 'APPROVED_WITH_MODIFICATIONS':
            changes = llm2_verification.get('recommended_changes', {})
            for key, value in changes.items():
                if value is not None:
                    final[key] = value
        
        # Add verification metadata
        final['verification'] = {
            'is_valid': llm2_verification.get('is_valid', True),
            'confidence': llm2_verification.get('confidence', 'medium'),
            'hallucinations_found': llm2_verification.get('hallucinations_found', []),
            'inconsistencies_found': llm2_verification.get('inconsistencies_found', []),
            'final_verdict': llm2_verification.get('final_verdict', 'APPROVED')
        }
        
        return final

    def analyze_resume(
        self,
        resume_text: str,
        jd_text: str
    ) -> Dict[str, Any]:
        """
        Full analysis pipeline: role prediction, acceptance, similarity, LLM reviews.
        
        Args:
            resume_text: Cleaned resume text
            jd_text: Job description text
        Returns:
            Final trusted assessment dictionary
        """

        # Step 1: ML Model Predictions
        role_pred = self.predict_role(resume_text)
        acceptance_pred = self.predict_acceptance(resume_text)
        similarity_res = self.compute_similarity(resume_text, jd_text)
        
        # Step 2: LLM 1 Review
        llm1_assess = self.llm1_review_and_score(
            resume_text=resume_text,
            jd_text=jd_text,
            role_prediction=role_pred,
            acceptance_prediction=acceptance_pred,
            similarity_result=similarity_res
        )
        
        # Step 3: LLM 2 Verification
        llm2_verify = self.llm2_verify_assessment(
            resume_text=resume_text,
            jd_text=jd_text,
            llm1_assessment=llm1_assess,
            role_prediction=role_pred,
            acceptance_prediction=acceptance_pred,
            similarity_result=similarity_res
        )
        
        # Step 4: Combine for Final Assessment
        final_assessment = self.get_final_assessment(llm1_assess, llm2_verify)
        
        return final_assessment

# DEBUGGING PURPOSES
if __name__ == "__main__":
    print("Resume Analysis Agent with Dual LLM Verification")
    print("="*60)
    
    # Test with sample text
    sample_text = "Experienced backend engineer skilled in Python, APIs, and machine learning." 
    sample_jd = """
    We are seeking a Backend Engineer with 3+ years of experience in Python, Django, REST APIs.
    Experience with PostgreSQL and Docker is required.
    """

    print("\nQuick Test (without file):")
    print("=" * 60)    
    
    # Initialize models
    rm = ResumeModels(
        api_key="AIzaSyAGG090tcKmTAcizYgm_0K0mJAtkpmLblA",
    )
    
    try:
        result = rm.analyze_resume(
            resume_text=sample_text,
            jd_text=sample_jd
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error in acceptance test: {e}")
