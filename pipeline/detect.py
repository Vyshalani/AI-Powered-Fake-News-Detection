# pipeline/detect.py
import os
import pickle
import torch
import yaml

from classifier.threehan_model import ThreeHANModel          # ✅ fixed import
from classifier.utils import TextProcessor as ClfProcessor
from retrieval.retriever import fetch_evidence
from retrieval.processor import TextProcessor as EvidenceProcessor

# ---------------- CONFIG ----------------
with open("pipeline/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH   = cfg.get("model_path", "models/final_model.pt")
VOCAB_PATH   = cfg.get("vocab_path", "models/vocab.pkl")
MAX_LEN      = int(cfg.get("max_len", 100))
DEVICE_PREF  = cfg.get("device", "cuda")
DEVICE       = torch.device(DEVICE_PREF if (DEVICE_PREF == "cuda" and torch.cuda.is_available()) else "cpu")
FETCH_WEB    = bool(cfg.get("fetch_web", True))
NUM_EVIDENCE = int(cfg.get("num_evidence", 5))
LABEL_MAP    = {int(k): v for k, v in cfg.get("label_mapping", {0: "Real", 1: "Fake"}).items()}

# ---------------- LOAD VOCAB ----------------
if not os.path.exists(VOCAB_PATH):
    raise FileNotFoundError(
        f"Vocab not found at {VOCAB_PATH}. Make sure you saved it after training:\n"
        f'  with open("models/vocab.pkl","wb") as f: pickle.dump(processor.vocab, f)'
    )

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# ---------------- PROCESSORS ----------------
clf_proc = ClfProcessor(max_len=MAX_LEN)
clf_proc.vocab = vocab  # reuse exact training vocab

evid_proc = EvidenceProcessor()

# ---------------- MODEL ----------------
vocab_size = len(clf_proc.vocab)
model = ThreeHANModel(vocab_size=vocab_size)   # same signature as used in training
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# ---------------- DETECTION ----------------
def detect_claim(claim_text: str):
    """
    Returns: (verdict:str, confidence:float, evidence_texts:list[str], similarity:float|None)
    """
    # Prepare input
    x = clf_proc.texts_to_tensor([claim_text]).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(torch.argmax(probs).item())
        base_conf = float(probs[pred_class].item())

    verdict = LABEL_MAP.get(pred_class, str(pred_class))

    # Always fetch evidence
    evidence_texts = fetch_evidence(claim_text, num_results=NUM_EVIDENCE)
    similarity = None
    if evidence_texts:
        similarity = evid_proc.compute_similarity(claim_text, evidence_texts)

        # confidence adjustment for Fake claims only
        if pred_class == 1:  
            if similarity < 0.30:
                base_conf = min(1.0, base_conf + 0.20)
            elif similarity > 0.55:
                base_conf = max(0.0, base_conf - 0.20)

    return verdict, base_conf, evidence_texts, similarity
