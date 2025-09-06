import streamlit as st
import joblib
import os
import google.generativeai as genai
import numpy as np

# ----------------------------
# Load TF-IDF + Logistic Regression Model
# ----------------------------
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

model, vectorizer = None, None
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

# ----------------------------
# Configure Gemini API
# ----------------------------
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
if API_KEY:
    genai.configure(api_key=API_KEY)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Fake News & Misinformation Detector", layout="centered")

st.title("Fake News & Misinformation Detector")
st.markdown(
    "Enter a headline or short article and the model will classify as **FAKE** or **REAL** "
    "with a confidence score. The demo also shows the top words that pushed the prediction "
    "and a few simple heuristics (exclamation marks, ALL CAPS, URLs)."
)

# Input fields
news_text = st.text_area("Paste a news headline or short article", placeholder="Type or paste text here...")
source_domain = st.text_input("Source domain (optional)", placeholder="e.g. bbc.com or nytimes.com")
explain = st.checkbox("Show explainability (top contributing words)", value=True)

# ----------------------------
# TF-IDF Model Prediction
# ----------------------------
if st.button("Run Model"):
    if model and vectorizer:
        if news_text.strip():
            # Transform and predict
            X = vectorizer.transform([news_text])
            prediction = model.predict(X)[0]
            probas = model.predict_proba(X)[0]
            confidence = np.max(probas)

            st.subheader("🔎 Prediction Result")
            st.write(f"**Prediction:** {'✅ REAL' if prediction == 'REAL' else '❌ FAKE'}")
            st.write(f"**Confidence:** {confidence:.2f}")

            # Explainability
            if explain:
                st.subheader("📊 Top contributing words")
                feature_names = vectorizer.get_feature_names_out()
                coef = model.coef_[0]
                indices = X.nonzero()[1]

                contrib = sorted(
                    [(feature_names[i], coef[i] * X[0, i]) for i in indices],
                    key=lambda x: -abs(x[1])
                )[:10]

                for word, score in contrib:
                    st.write(f"- **{word}** → {score:.4f}")

            # Simple heuristics
            st.subheader("⚠️ Heuristics")
            if "!" in news_text:
                st.write("- Exclamation marks detected ❗")
            if any(word.isupper() for word in news_text.split()):
                st.write("- ALL CAPS words detected 🔠")
            if "http" in news_text or "www" in news_text:
                st.write("- URL detected 🔗")
            if source_domain:
                trusted_sources = ["bbc.com", "nytimes.com", "reuters.com", "theguardian.com"]
                if any(src in source_domain.lower() for src in trusted_sources):
                    st.write(f"- ✅ Trusted source detected: {source_domain}")
                else:
                    st.write(f"- ⚠️ Source not in trusted list: {source_domain}")
        else:
            st.warning("Please enter some text to analyze.")
    else:
        st.error("❌ Could not load TF-IDF model/vectorizer. Please run `train_model.py` first!")

# ----------------------------
# Gemini Few-Shot Option
# ----------------------------
st.markdown("---")
st.subheader("🤖 Try Gemini Few-Shot")
if st.button("Run Gemini Few-Shot"):
    if not API_KEY:
        st.error("❌ Gemini API key not found. Please set GOOGLE_API_KEY in environment or .streamlit/secrets.toml")
    elif not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        st.info("Querying Gemini... please wait.")
        try:
            model_gemini = genai.GenerativeModel("gemini-1.5-flash")
            few_shot_prompt = f"""
            You are a fake news detection system. Classify the following news as REAL or FAKE.
            Provide a short explanation.

            Example 1:
            Text: "NASA confirms water found on the moon surface"
            Label: REAL
            Explanation: Verified by multiple trusted space agencies.

            Example 2:
            Text: "Celebrity spotted on Mars colony opening"
            Label: FAKE
            Explanation: No evidence or scientific backing exists for Mars colonies.

            Now classify this:
            Text: "{news_text}"
            """
            response = model_gemini.generate_content(few_shot_prompt)
            st.success("Gemini Response:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error using Gemini API: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Model: TF-IDF ➝ LogisticRegression. Explainability: per-word contribution (coef × tfidf). Expand TRUSTED/UNTRUSTED lists in app.py for quick domain checks.")
