import streamlit as st
import numpy as np
from model.utils import load_weights, predict
import google.generativeai as genai

# ✅ Configure Gemini API directly
genai.configure(api_key="AIzaSyA675j-vLk9LBiksdIWV8eaBkGXF8J5RMw")

# ✅ Gemini Suggestion (based only on model prediction)
def gemini_suggestion(condition):
    prompt = f"""
    A patient has been predicted by a machine learning model to have **{condition}** thyroid condition.

    Explain what this condition is, including:
    - Causes
    - Common symptoms
    - Lifestyle precautions
    - General treatment options

    Keep it brief, clear, and understandable for patients.
    """
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 512
        }
    )
    return response.text

# ✅ Streamlit Interface
st.set_page_config(page_title="Thyroid Disease Predictor", layout="centered")
st.title("💉 Thyroid Disease Prediction")
st.markdown("Provide the following 21 inputs to predict your thyroid condition.")

features = [
    "age", "sex_M", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary",
    "psych", "TSH", "T3", "TT4", "T4U", "FTI"
]

user_input = []
for feat in features:
    val = st.number_input(feat, value=0.0, step=0.01)
    user_input.append(val)

if st.button("Predict"):
    input_array = np.array(user_input)
    w1, w2, b1, b2 = load_weights()
    output = predict(input_array, w1, w2, b1, b2)
    pred = np.argmax(output)

    condition = ["Normal", "Hypothyroid", "Hyperthyroid"][pred]
    st.success(f"Predicted Condition: **{condition}**")

    st.markdown("### 🤖 Gemini Health Suggestion")
    with st.spinner("Contacting Gemini for a recommendation..."):
        advice = gemini_suggestion(condition)
    st.info(advice)
