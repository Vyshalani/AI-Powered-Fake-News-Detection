from flask import Flask, render_template, request, jsonify
from pipeline.detect import detect_claim
import joblib
import os
import time
import re
from langdetect import detect as detect_language, DetectorFactory

# For consistent language detection results
DetectorFactory.seed = 0

app = Flask(__name__)

# Simplified language detection function
def detect_language(text):
    """
    Detect if text is English or Afrikaans
    Returns: 'en', 'af', or 'other'
    """
    try:
        # Simple word-based detection for Afrikaans vs English
        afrikaans_indicators = ['die', 'en', 'van', 'het', 'sy', 'nie', 'om', 'vir', 'is', 'ek', 'jy', 'hy', 'ons', 'hulle']
        english_indicators = ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on']
        
        words = text.lower().split()
        
        af_count = sum(1 for word in words if word in afrikaans_indicators)
        en_count = sum(1 for word in words if word in english_indicators)
        
        # Use langdetect for more accurate detection
        lang_code = detect_language(text)

        if lang_code == 'af' or (af_count > en_count and af_count >= 2):
            return 'af'
        elif lang_code == 'en' or (en_count > af_count and en_count >= 2):
            return 'en'
        else:
            return 'other'
            
    except Exception:
        # Fallback to word counting if langdetect fails
        if af_count > en_count:
            return 'af'
        elif en_count > af_count:
            return 'en'
        else:
            return 'other'

def validate_input_text(text):
    """
    Comprehensive input validation
    """
    errors = []
    
    # Length validation
    if len(text.strip()) < 10:
        errors.append("Claim is too short. Please enter at least 10 characters.")
    
    if len(text.strip()) > 50000:
        errors.append("Claim is too long. Please keep it under 5000 characters.")
    
    # Language validation
    language = detect_language(text)
    if language == 'other':
        errors.append("Please enter text in English or Afrikaans only. Other languages are not supported.")
    
    # Repetitive text detection
    words = text.split()
    if len(words) > 5 and len(set(words)) / len(words) < 0.3:
        errors.append("Text appears to be repetitive. Please enter a meaningful news claim.")
    
    return errors, language

# Load resources once at startup
def load_resources():
    """Loads the model and vectorizer once."""
    model_path = os.path.join("models", "logreg_model.pkl")
    vectorizer_path = os.path.join("models", "logreg_vectorizer.pkl")

    if not all(os.path.exists(p) for p in [model_path, vectorizer_path]):
        print("Error: Model files not found!")
        return None, None

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model resources loaded successfully")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model resources: {e}")
        return None, None

# Load resources when app starts
LOGREG_MODEL, LOGREG_VEC = load_resources()

# Store session data (in production use Redis or database)
user_sessions = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_claim():
    data = request.json
    claim = data.get('claim', '').strip()
    session_id = data.get('session_id', 'default')
    
    # Input validation
    if not claim:
        return jsonify({'error': 'Please enter a claim first.'}), 400
    
    # Comprehensive validation
    validation_errors, detected_language = validate_input_text(claim)
    if validation_errors:
        return jsonify({'error': ' | '.join(validation_errors)}), 400
    
    # Check if model resources are loaded
    if LOGREG_MODEL is None or LOGREG_VEC is None:
        return jsonify({'error': 'System is still initializing. Please try again in a moment.'}), 503
    
    # Initialize session if not exists
    if session_id not in user_sessions:
        user_sessions[session_id] = {'history': [], 'lang': 'English'}
    
    try:
        print(f"Analyzing claim in {detected_language}: {claim}")  # Debug print
        
        # Call your detection pipeline
        res = detect_claim(claim)
        print(f"Detection result: {res}")  # Debug print
        
        # Process response 
        if isinstance(res, (list, tuple)):
            if len(res) >= 4:
                verdict, confidence, evidence, similarity = res[:4]
            elif len(res) == 3:
                verdict, confidence, evidence = res
                similarity = None
            else:
                error_msg = f"Unexpected response shape from detect_claim(): {res}"
                print(f" {error_msg}")
                return jsonify({'error': error_msg}), 500
        elif isinstance(res, dict):
            verdict = res.get("verdict") or res.get("label")
            confidence = res.get("confidence") or res.get("probability")
            evidence = res.get("evidence") or []
            similarity = res.get("similarity")
        else:
            error_msg = f"Unexpected response type from detect_claim(): {type(res)}"
            print(f"{error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # Ensure confidence is float
        try:
            conf_val = float(confidence)
            conf_val = max(0.0, min(1.0, conf_val))
        except:
            conf_val = 0.0
        
        # Ensure similarity is float if exists
        sim_val = None
        if similarity is not None:
            try:
                sim_val = float(similarity)
            except:
                sim_val = None
        
        print(f"Processed - Verdict: {verdict}, Confidence: {conf_val}")  # Debug
        
        # Add to history
        history_entry = {
            "claim": claim,
            "verdict": verdict,
            "confidence": conf_val,
            "similarity": sim_val,
            "evidence": evidence,
            "language": detected_language
        }
        
        user_sessions[session_id]['history'].append(history_entry)
        
        # Prepare response
        response_data = {
            'verdict': verdict,
            'confidence': conf_val,
            'similarity': sim_val,
            'evidence': evidence,
            'history': user_sessions[session_id]['history'][-10:]  # Last 10 entries
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f'Analysis failed: {str(e)}'
        print(f"{error_msg}")  # This will show in your terminal
        import traceback
        traceback.print_exc()  # This will print the full error stack
        return jsonify({'error': 'System temporarily unavailable. Please try again shortly.'}), 500
    
@app.route('/history/<session_id>')
def get_history(session_id):
    """Get user's claim history"""
    if session_id not in user_sessions:
        return jsonify([])
    
    history = user_sessions.get(session_id, {}).get('history', [])
    return jsonify(history)

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'healthy' if LOGREG_MODEL and LOGREG_VEC else 'unhealthy',
        'model_loaded': LOGREG_MODEL is not None,
        'vectorizer_loaded': LOGREG_VEC is not None,
        'timestamp': time.time()
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)