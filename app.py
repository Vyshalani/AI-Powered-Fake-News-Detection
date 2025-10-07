import streamlit as st
import pipeline.detect as detect
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
# --- Action Button ---
if st.button("🔍 Verify Claim", use_container_width=True):
    if not claim.strip():
        st.warning("⚠️ Please enter a claim first.")
    else:
        with st.spinner("Analyzing claim with AI..."):
            time.sleep(1)  # simulate loading

            import pipeline.detect as detect
            


            # call once and unpack robustly
            res = detect.detect_claim(claim)


            if isinstance(res, (list, tuple)):
                if len(res) >= 4:
                    verdict, confidence, evidence, similarity = res[:4]
                elif len(res) == 3:
                    verdict, confidence, evidence = res
                    similarity = None
                else:
                    st.error("Unexpected response shape from detect_claim().")
                    raise RuntimeError("detect_claim returned unexpected number of values")
            elif isinstance(res, dict):
                verdict = res.get("verdict") or res.get("label")
                confidence = res.get("confidence") or res.get("probability")
                evidence = res.get("evidence") or []
                similarity = res.get("similarity")
            else:
                st.error("Unexpected response type from detect_claim().")
                raise RuntimeError("detect_claim returned unexpected type")



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
            if isinstance(verdict, str) and verdict.lower() == "real":
                st.markdown(f"<h3 style='color:green;'>🟢 {verdict}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:red;'>🔴 {verdict}</h3>", unsafe_allow_html=True)

        with col2:
            st.subheader("Confidence")
            # ensure confidence is a float and in [0,1]
            try:
                conf_val = float(confidence)
                conf_val = max(0.0, min(1.0, conf_val))
            except Exception:
                conf_val = 0.0
            st.progress(int(conf_val * 100))
            st.markdown(f"**{conf_val:.2f}**")

            # show similarity if available, with color
            if similarity is not None:
                try:
                    sim = float(similarity)
                except Exception:
                    sim = None

                if sim is not None:
                    # color scale: green (high), orange (mid), red (low)
                    if sim >= 0.55:
                        color = "#198754"   # green
                    elif sim >= 0.30:
                        color = "#f0ad4e"   # orange
                    else:
                        color = "#d9534f"   # red

                    st.markdown(
                        f"<div style='font-weight:bold;color:{color};'>"
                        f"Evidence similarity: {sim:.2f}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("Evidence similarity: N/A")

                    with col2:
                     st.subheader("Confidence")
                    st.progress(int(confidence * 100))
                    st.markdown(f"**{confidence:.2f}**")

                if similarity is not None:
                    st.markdown(f"**Evidence Similarity:** {similarity:.2f}")


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
            st.info("No supporting evidence retrieved. Verdict is based only on AI model.")

# --- History Section ---
if show_history and st.session_state.history:
    st.markdown("---")
    st.subheader("📖 Claim History (this session)")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df[["claim", "verdict", "confidence"]])


