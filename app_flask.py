from flask import Flask, render_template, request, jsonify
import pipeline.detect as detect
import joblib
import os
import time

app = Flask(__name__)

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
        print("✅ Model resources loaded successfully")
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
    
    if not claim:
        return jsonify({'error': 'Please enter a claim first.'}), 400
    
    # Initialize session if not exists
    if session_id not in user_sessions:
        user_sessions[session_id] = {'history': [], 'lang': 'English'}
    
    try:
        print(f"🔍 Analyzing claim: {claim}")  # Debug print
        
        # Call your detection pipeline
        res = detect.detect_claim(claim)
        print(f"✅ Detection result: {res}")  # Debug print
        
        # Process response (same logic as your Streamlit app)
        if isinstance(res, (list, tuple)):
            if len(res) >= 4:
                verdict, confidence, evidence, similarity = res[:4]
            elif len(res) == 3:
                verdict, confidence, evidence = res
                similarity = None
            else:
                error_msg = f"Unexpected response shape from detect_claim(): {res}"
                print(f"❌ {error_msg}")
                return jsonify({'error': error_msg}), 500
        elif isinstance(res, dict):
            verdict = res.get("verdict") or res.get("label")
            confidence = res.get("confidence") or res.get("probability")
            evidence = res.get("evidence") or []
            similarity = res.get("similarity")
        else:
            error_msg = f"Unexpected response type from detect_claim(): {type(res)}"
            print(f"❌ {error_msg}")
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
        
        print(f"✅ Processed - Verdict: {verdict}, Confidence: {conf_val}")  # Debug
        
        # Add to history
        history_entry = {
            "claim": claim,
            "verdict": verdict,
            "confidence": conf_val,
            "similarity": sim_val,
            "evidence": evidence
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
        print(f"❌ {error_msg}")  # This will show in your terminal
        import traceback
        traceback.print_exc()  # This will print the full error stack
        return jsonify({'error': error_msg}), 500
    
@app.route('/history/<session_id>')
def get_history(session_id):
    """Get user's claim history"""
    history = user_sessions.get(session_id, {}).get('history', [])
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True, port=5000)