# pipeline/detect.py
import os
import yaml
import joblib

from retrieval.retriever import fetch_evidence
from retrieval.processor import TextProcessor as EvidenceProcessor

# ---------------- CONFIG ----------------
with open("pipeline/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

LOGREG_MODEL = "models/logreg_model.pkl"
LOGREG_VEC   = "models/logreg_vectorizer.pkl"

FETCH_WEB    = bool(cfg.get("fetch_web", True))
NUM_EVIDENCE = int(cfg.get("num_evidence", 5))
LABEL_MAP    = {int(k): v for k, v in cfg.get("label_mapping", {0: "Fake", 1: "Real"}).items()}

# ---------------- LOAD MODEL ----------------
if not (os.path.exists(LOGREG_MODEL) and os.path.exists(LOGREG_VEC)):
    raise FileNotFoundError("Logistic Regression model or vectorizer not found. Train with train_logreg.py first.")

logreg_clf = joblib.load(LOGREG_MODEL)
logreg_vec = joblib.load(LOGREG_VEC)

# ---------------- EVIDENCE PROCESSOR ----------------
evid_proc = EvidenceProcessor()

# ---------------- DETECT FUNCTION ----------------
def detect_claim(claim_text: str):
    """
    Returns: (verdict:str, confidence:float, evidence_texts:list[str], similarity:float|None)
    """
    X = logreg_vec.transform([claim_text])
    pred_class = int(logreg_clf.predict(X)[0])
    confidence = float(logreg_clf.predict_proba(X)[0][pred_class])
    verdict = LABEL_MAP.get(pred_class, str(pred_class))

    evidence_texts = fetch_evidence(claim_text, num_results=NUM_EVIDENCE)
    similarity = None
    if evidence_texts:
        similarity = evid_proc.compute_similarity(claim_text, evidence_texts)

        if pred_class == 1:  # real news
            if similarity < 0.30:
                confidence = min(1.0, confidence + 0.20)
            elif similarity > 0.55:
                confidence = max(0.0, confidence - 0.20)

    return verdict, confidence, evidence_texts, similarity
