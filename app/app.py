import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Multimodal Clinical AI", layout="wide")
st.title("🩺 Multimodal Disease & Specialty Predictor")
st.write("Enter patient symptoms and clinical notes below to view AI predictions and confidence metrics.")

# --- 2. LOAD THE SAVED MODELS ---
@st.cache_resource
def load_models():
    with open('models/symptom_xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('models/disease_label_encoder.pkl', 'rb') as f:
        disease_encoder = pickle.load(f)
    with open('models/nlp_model.pkl', 'rb') as f:
        nlp_model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/nlp_label_encoder.pkl', 'rb') as f:
        nlp_encoder = pickle.load(f)
    return xgb_model, disease_encoder, nlp_model, vectorizer, nlp_encoder

xgb_model, disease_encoder, nlp_model, vectorizer, nlp_encoder = load_models()

# --- 3. BUILD THE USER INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Clinical Report (Unstructured)")
    clinical_text = st.text_area("Paste the doctor's notes or clinical report here:", height=200)
    
    if st.button("Analyze Report"):
        if clinical_text:
            cleaned = str(clinical_text).lower()
            cleaned = re.sub(r'\[.*?\]', '', cleaned)
            cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            vec_text = vectorizer.transform([cleaned])
            
            # Get probabilities for ALL specialties
            prob_scores = nlp_model.predict_proba(vec_text)[0]
            categories = nlp_encoder.classes_
            
            # Sort them from highest confidence to lowest
            sorted_indices = np.argsort(prob_scores)[::-1]
            
            st.success(f"**Top Prediction:** {categories[sorted_indices[0]]}")
            st.write("### Confidence Breakdown:")
            for idx in sorted_indices[:3]:  # Display top 3 likely specialties
                score = prob_scores[idx]
                st.write(f"**{categories[idx]}** ({score * 100:.1f}%)")
                st.progress(float(score))
        else:
            st.warning("Please enter some text first.")

with col2:
    st.header("2. Symptoms (Structured)")
    symptom_list = xgb_model.feature_names_in_
    
    selected_symptoms = st.multiselect(
        "Search and select patient symptoms:",
        options=symptom_list,
        help="Type to search for symptoms like 'fever', 'cough', etc."
    )
    
    if st.button("Predict Disease"):
        if selected_symptoms:
            patient_profile = np.zeros(len(symptom_list))
            for symptom in selected_symptoms:
                index = np.where(symptom_list == symptom)[0][0]
                patient_profile[index] = 1
                
            patient_df = pd.DataFrame([patient_profile], columns=symptom_list)
            
            # Get probabilities for ALL diseases
            prob_scores = xgb_model.predict_proba(patient_df)[0]
            diseases = disease_encoder.classes_
            
            # Sort them from highest confidence to lowest
            sorted_indices = np.argsort(prob_scores)[::-1]
            
            st.success(f"**Top Prediction:** {diseases[sorted_indices[0]]}")
            st.write("### Confidence Breakdown:")
            for idx in sorted_indices[:3]:  # Display top 3 likely diseases
                score = prob_scores[idx]
                st.write(f"**{diseases[idx]}** ({score * 100:.1f}%)")
                st.progress(float(score))
        else:
            st.warning("Please select at least one symptom.")