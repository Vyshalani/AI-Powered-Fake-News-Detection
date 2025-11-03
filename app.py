import streamlit as st
import pipeline.detect as detect
import pandas as pd
import time
import os
import joblib # Required for @st.cache_resource function below

# --- Function to inject custom CSS from local file ---
def inject_custom_css(css_file_path):
    """Reads a local CSS file and injects it into the Streamlit app."""
    try:
        with open(css_file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at: {css_file_path}")

# --- Resource Loading (Necessary to prevent detect.py errors) ---
@st.cache_resource
def load_resources():
    """Loads the model and vectorizer once."""
    model_path = os.path.join("models", "logreg_model.pkl")
    vectorizer_path = os.path.join("models", "logreg_vectorizer.pkl")

    if not all(os.path.exists(p) for p in [model_path, vectorizer_path]):
        st.error("Model files not found! Please ensure 'logreg_model.pkl' and 'logreg_vectorizer.pkl' are in the 'models/' directory.")
        return None, None

    # Load resources
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        return None, None

# Load the resources *before* the app UI starts
LOGREG_MODEL, LOGREG_VEC = load_resources()

# --- Page setup ---
st.set_page_config(
    page_title="Namibian Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# --- Inject Custom CSS ---
inject_custom_css(".streamlit/style.css")



# --- Session state initialization (FIX: MUST BE HERE BEFORE ANY ACCESS) ---
if "history" not in st.session_state:
    st.session_state.history = []
if "lang" not in st.session_state:
    st.session_state.lang = "English" # Default language


# --- Sidebar ---
st.sidebar.title("⚙️ Options")

# Custom Language Selector
st.sidebar.markdown('<p class="sidebar-lang-label">LANGUAGE</p>', unsafe_allow_html=True)

# --- Custom Language Buttons (Functional Click Handlers) ---
# We use st.empty containers to hold the buttons cleanly
lang_buttons = st.sidebar.empty()

with lang_buttons.container():
    # English Button (The click handler)
    if st.button("English", key="btn_en", use_container_width=True):
        st.session_state.lang = "English"
        st.rerun()

    # Afrikaans Button (The click handler)
    if st.button("Afrikaans", key="btn_af", use_container_width=True):
        st.session_state.lang = "Afrikaans"
        st.rerun()


# --- Custom Visual Appearance ---
# This part is purely visual, reading the state set by the buttons.
st.sidebar.markdown('<div class="custom-language-visuals">', unsafe_allow_html=True)

# English Appearance
en_class = "lang-button-active" if st.session_state.lang == "English" else "lang-button-inactive"
st.sidebar.markdown(
    f'<div class="lang-option-wrapper"><div class="{en_class}"></div>'
    f'<span class="lang-text">English</span></div>', 
    unsafe_allow_html=True
)

# Afrikaans Appearance
af_class = "lang-button-active" if st.session_state.lang == "Afrikaans" else "lang-button-inactive"
st.sidebar.markdown(
    f'<div class="lang-option-wrapper"><div class="{af_class}"></div>'
    f'<span class="lang-text">Afrikaans</span></div>', 
    unsafe_allow_html=True
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Separate the elements
st.sidebar.markdown("---")

# Show Claim History Checkbox
show_history = st.sidebar.checkbox("Show Claim History", True)




# --- Main Title and Subtitle ---

# 1. Use columns to force center alignment
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    # Main Heading: Cyan color and centered (using Markdown centering trick)
    st.markdown(
        "<h1 style='color: #00FFFF; text-align: center;'>📰 NAMIBIAN FAKE NEWS DETECTOR</h1>",
        unsafe_allow_html=True
    )
    
    # Subtitle: Green color, centered, and subtitle font
    st.markdown(
        """
        <p style='color: #00FF7F; text-align: center; font-size: 1.2em; margin-top: -15px;'>
            Verify news claims in real-time using AI & trusted Namibian news sources
        </p>
        """,
        unsafe_allow_html=True
    )

    # Source List: Muted Text, centered
    st.markdown(
        """
        <p style='color: #AAAAAA; text-align: center; font-size: 0.9em;'>
            (The Namibian, Republikein, Kosmos 94.1, Namibian Sun)
        </p>
        """,
        unsafe_allow_html=True
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