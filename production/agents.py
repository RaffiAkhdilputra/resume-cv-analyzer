import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
from typing import Dict, Any, List, Optional
import numpy as np

# LangChain + Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Document processing
import fitz  # PyMuPDF
from docx import Document

# Embedder model from huggingface
from transformers import DistilBertTokenizer, DistilBertTokenizerFast, DistilBertModel
from sentence_transformers import SentenceTransformer

# PyTorch for loading models
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global configuration
GEMINI_MODEL = "gemini-2.5-flash"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Model paths
DEFAULT_ROLE_MODEL_PATH = "train/model/role_model"
DEFAULT_ACCEPTANCE_MODEL_PATH = "production/model/acceptance_model/"

# Role mapping (customize based on your training labels)
ROLE_MAPPING = {
    0: "Backend Engineer",
    1: "Frontend Engineer",
    2: "Full Stack Engineer",
    3: "Data Scientist",
    4: "Machine Learning Engineer",
    5: "DevOps Engineer",
    6: "Mobile Developer",
    7: "QA Engineer",
    8: "Product Manager",
    9: "Designer"
}

ACCEPTANCE_MAPPING = {
    0: "Rejected",
    1: "Accepted"
}

# ACCEPTANCE CLASSIFIER
class AcceptanceClassifier:
    def __init__(self, model_dir: str = DEFAULT_ACCEPTANCE_MODEL_PATH):
        """
        model_dir should contain:
            - backbone.pkl
            - head.pkl
            - tokenizer.json
            - tokenizer_config.json
            - vocab.json
        """
        self.model_dir = model_dir
        self.acceptance_backbone = None
        self.acceptance_head = None
        self.acceptance_tokenizer = None
        self._init_acceptance_model()
    
    def _init_acceptance_model(self):
        """Loads tokenizer, backbone, and classifier head."""
        print("Loading acceptance tokenizer...")
        self.acceptance_tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.model_dir
        )
        
        print("Loading BERT backbone...")
        backbone_state = torch.load(f"{self.model_dir}/backbone.pkl", map_location="cpu")
        
        self.acceptance_backbone = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.acceptance_backbone.load_state_dict(backbone_state, strict=False)
        self.acceptance_backbone.eval()
        
        print("Loading classification head...")
        head_state = torch.load(f"{self.model_dir}/head.pkl", map_location="cpu")
        
        # Get hidden dimension from saved state_dict
        # Position 1 is Linear layer with shape [hidden_dim, 768]
        hidden_dim = head_state["1.weight"].shape[0]
        hidden_size = head_state["1.weight"].shape[1]  # Should be 768 (DistilBERT)
        
        print(f"Detected architecture: Linear({hidden_size}, {hidden_dim}) -> ... -> Linear({hidden_dim}, 2)")
        
        # Build head EXACTLY like during training
        self.acceptance_head = nn.Sequential(
            nn.Dropout(0.3),                # 0
            nn.Linear(hidden_size, hidden_dim),  # 1: (768, 256)
            nn.ReLU(),                      # 2
            nn.BatchNorm1d(hidden_dim),     # 3: (256)
            nn.Dropout(0.15),               # 4
            nn.Linear(hidden_dim, 2)        # 5: (256, 2) - 2 classes!
        )
        
        self.acceptance_head.load_state_dict(head_state)
        self.acceptance_head.eval()
        
        print("✓ Acceptance model fully loaded.\n")
    
    def predict_acceptance(self, text: str) -> Dict[str, Any]:
        """Returns acceptance probability."""
        if not text or len(text.strip()) == 0:
            raise ValueError("Input text is empty.")
        
        inputs = self.acceptance_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.acceptance_backbone(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            cls_vec = outputs.last_hidden_state[:, 0, :]  # CLS embedding
            
            # Get logits for 2 classes
            logits = self.acceptance_head(cls_vec)  # Shape: (1, 2)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)  # Shape: (1, 2)
            
            # Get probability of class 1 (accepted)
            prob = probs[0, 1].item()
        
        return {
            "accepted": prob >= 0.5,
            "probability": float(prob)
        }

class RoleClassifier:
    # TODO: UNFINISHED: Implement role classifier later
    def __init__(self, model_path: str = DEFAULT_ROLE_MODEL_PATH):
        """
        model_path should contain:
            - role_model.h5
        """
        self.model_path = model_path
        self.role_model = None
        self._init_role_model()
    
    def _init_role_model(self):
        pass

    def predict_role(self, text: str) -> Dict[str, Any]:
        pass

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
        role_model_path: str = DEFAULT_ROLE_MODEL_PATH,
        acceptance_model_path: str = DEFAULT_ACCEPTANCE_MODEL_PATH,
        role_mapping: Dict[int, str] = None
    ):
        """
        Initialize ChatGoogleGenerativeAI + embedding + ML models.
        
        Args:
            api_key: Google Gemini API key
            embed_model_name: SentenceTransformer model name
            role_model_path: Path to role classification model
            acceptance_model_path: Path to acceptance classification model directory
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
        
        # Embeddings (HF SentenceTransformer)
        self.embedding_model = SentenceTransformer(embed_model_name)
        self.embedding_dimension = 384
        
        # Role mapping
        self.role_mapping = role_mapping or ROLE_MAPPING
        
        # Paths
        self.role_model_path = role_model_path
        self.acceptance_model_path = acceptance_model_path
        
        # Model containers
        self.role_model = None
        self.acceptance_classifier = None
        
        # Load models immediately
        # TODO: UNFINISHED: Enable role model later
        # self._init_role_model()
        self._init_acceptance_model()
    
    def _init_acceptance_model(self):
        """Initialize acceptance classifier using AcceptanceClassifier class."""
        self.acceptance_classifier = AcceptanceClassifier(model_dir=self.acceptance_model_path)
    
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Convert text into a 384-dim embedding.
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array of shape (1, 384)
        """
        embedding = self.embedding_model.encode(text)
        embedding = np.array(embedding).reshape(1, -1)
        return embedding
    
    def predict_role(self, text: str) -> Dict[str, Any]:
        """
        Predict job role classification from resume text.
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with role_class (int), role_name (str), and confidence (float)
        """
        if self.role_model is None:
            raise RuntimeError("Role model is not loaded.")
        
        emb = self._embed_text(text)
        preds = self.role_model.predict(emb, verbose=0)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        role_name = self.role_mapping.get(class_id, f"Unknown Role {class_id}")
        
        return {
            "role_class": class_id,
            "role_name": role_name,
            "confidence": confidence,
            "all_probabilities": preds[0].tolist()
        }
    
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
        # TODO: UNFINISHED: Enable role model later
        role_pred = {
            "role_class": 4,
            "role_name": "Machine Learning Engineer",
            "confidence": 0.92,
        } 
        # self.predict_role(resume_text)
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
    # Example usage
    print("Resume Analysis Agent with Dual LLM Verification")
    print("="*60)
    
    # Test with sample text
    sample_text = "Experienced backend engineer skilled in Python, APIs, and machine learning."
    sample_jd = """
    We are seeking a Backend Engineer with 3+ years of experience in Python, Django, REST APIs.
    Experience with PostgreSQL and Docker is required.
    """
    
    print(DEFAULT_ACCEPTANCE_MODEL_PATH)
    print("\nQuick Test (without file):")
    print("-" * 60)    
    
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