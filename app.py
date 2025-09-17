import streamlit as st
from pipeline.detect import detect_claim
import pandas as pd
import time

# --- Page setup ---
st.set_page_config(
    page_title="Namibian Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# --- Session state for history ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
st.sidebar.title("⚙️ Options")
lang = st.sidebar.radio("Language", ["English", "Afrikaans"])
st.sidebar.markdown("---")
show_history = st.sidebar.checkbox("Show Claim History", True)

# --- Main Title ---
st.title("📰 Namibian Fake News Detector")
st.markdown(
    """
    **Verify news claims in real-time** using AI and trusted Namibian news sources  
    *(The Namibian, Republikein, Kosmos 94.1, Namibian Sun)*  
    """
)

# --- Input Area ---
claim = st.text_area("✍️ Enter a news claim:", height=100, placeholder="e.g. Namibia wins AFCON 2025")

# --- Action Button ---
if st.button("🔍 Verify Claim", use_container_width=True):
    if not claim.strip():
        st.warning("⚠️ Please enter a claim first.")
    else:
        with st.spinner("Analyzing claim with AI..."):
            time.sleep(1)  # simulate loading
            result = detect_claim(claim)
            verdict, confidence, evidence = result[0], result[1], result[2]


        # Save to history
        st.session_state.history.append({
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "evidence": evidence
        })

        # --- Results Section ---
        st.success("Analysis complete ✅")

        # Split layout: Left verdict, right confidence
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Verdict")
            if verdict.lower() == "real":
                st.markdown(f"<h3 style='color:green;'>🟢 {verdict}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:red;'>🔴 {verdict}</h3>", unsafe_allow_html=True)

        with col2:
            st.subheader("Confidence")
            st.progress(int(confidence * 100))
            st.markdown(f"**{confidence:.2f}**")

        # --- Evidence Section ---
        st.subheader("📚 Supporting Evidence")
        if evidence:
            for idx, ev in enumerate(evidence, start=1):
                if "(" in ev and ev.endswith(")"):
                    title, url = ev.rsplit("(", 1)
                    url = url[:-1]  # remove trailing ")"
                    with st.expander(f"{idx}. {title.strip()}"):
                        st.markdown(f"[Read more]({url})")
                else:
                    with st.expander(f"{idx}. Evidence snippet"):
                        st.write(ev)
        else:
            st.info("No supporting evidence retrieved.")

# --- History Section ---
if show_history and st.session_state.history:
    st.markdown("---")
    st.subheader("📖 Claim History (this session)")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df[["claim", "verdict", "confidence"]])
