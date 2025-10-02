# pipeline/detect.py
import os
import pickle
import torch
import yaml
import joblib

from classifier.threehan_model import ThreeHANModel
from classifier.utils import TextProcessor as ClfProcessor
from retrieval.retriever import fetch_evidence
from retrieval.processor import TextProcessor as EvidenceProcessor

# ---------------- CONFIG ----------------
with open("pipeline/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MODEL_TYPE   = cfg.get("model_type", "3han")   # <-- new key: choose "3han" or "logreg"
MODEL_PATH   = cfg.get("model_path", "models/final_model.pt")
VOCAB_PATH   = cfg.get("vocab_path", "models/vocab.pkl")
LOGREG_MODEL = "models/logreg_model.pkl"
LOGREG_VEC   = "models/logreg_vectorizer.pkl"

MAX_LEN      = int(cfg.get("max_len", 100))
DEVICE_PREF  = cfg.get("device", "cuda")
DEVICE       = torch.device(DEVICE_PREF if (DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu")
FETCH_WEB    = bool(cfg.get("fetch_web", True))
NUM_EVIDENCE = int(cfg.get("num_evidence", 5))
LABEL_MAP    = {int(k): v for k, v in cfg.get("label_mapping", {0: "Fake", 1: "Real"}).items()}

# ---------------- PROCESSORS ----------------
evid_proc = EvidenceProcessor()

# ---------------- HAN MODEL ----------------
clf_proc, han_model = None, None
if MODEL_TYPE == "3han":
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocab not found at {VOCAB_PATH}. Run training first.")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    clf_proc = ClfProcessor(max_len=MAX_LEN)
    clf_proc.vocab = vocab

    vocab_size = len(clf_proc.vocab)
    han_model = ThreeHANModel(vocab_size=vocab_size)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    han_model.load_state_dict(state)
    han_model.to(DEVICE)
    han_model.eval()

# ---------------- LOGREG MODEL ----------------
logreg_clf, logreg_vec = None, None
if MODEL_TYPE == "logreg":
    if not (os.path.exists(LOGREG_MODEL) and os.path.exists(LOGREG_VEC)):
        raise FileNotFoundError("LogReg model/vectorizer not found. Train with train_logreg.py first.")
    logreg_clf = joblib.load(LOGREG_MODEL)
    logreg_vec = joblib.load(LOGREG_VEC)

# ---------------- DETECTION ----------------
def detect_claim(claim_text: str):
    """
    Returns: (verdict:str, confidence:float, evidence_texts:list[str], similarity:float|None)
    """
    base_conf = 0.0
    pred_class = None

    if MODEL_TYPE == "3han":
        x = clf_proc.texts_to_tensor([claim_text]).to(DEVICE)
        with torch.no_grad():
            logits = han_model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = int(torch.argmax(probs).item())
            base_conf = float(probs[pred_class].item())

    elif MODEL_TYPE == "logreg":
        X = logreg_vec.transform([claim_text])
        pred_class = int(logreg_clf.predict(X)[0])
        base_conf = float(logreg_clf.predict_proba(X)[0][pred_class])

    verdict = LABEL_MAP.get(pred_class, str(pred_class))

    # Always fetch evidence
    evidence_texts = fetch_evidence(claim_text, num_results=NUM_EVIDENCE)
    similarity = None
    if evidence_texts:
        similarity = evid_proc.compute_similarity(claim_text, evidence_texts)

        # confidence adjustment (still applied to both models)
        if pred_class == 1:  
            if similarity < 0.30:
                base_conf = min(1.0, base_conf + 0.20)
            elif similarity > 0.55:
                base_conf = max(0.0, base_conf - 0.20)

    return verdict, base_conf, evidence_texts, similarity
