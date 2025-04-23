import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from streamlit_chat import message

# Load model and preprocessing files
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #D6336C;'>❤️ Heart Disease Prediction App</h1>
        <h4 style='color: gray;'>Powered by Machine Learning</h4>
    </div>
""", unsafe_allow_html=True)

# Sidebar chatbot
st.sidebar.title("💬 HeartBot Assistant")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.sidebar.text_input("Ask a question:")
if user_input:
    # Basic mock chatbot
    if "oldpeak" in user_input.lower():
        answer = "Oldpeak is ST depression induced by exercise relative to rest."
    elif "chest pain" in user_input.lower():
        answer = "Chest pain type indicates the type of angina experienced."
    else:
        answer = "I'm here to help with heart-related input questions."

    st.session_state.chat_history.append((user_input, answer))

for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    message(a, key=f"bot_{i}")

st.markdown("---")

# Patient Input Form
with st.form("predict_form"):
    st.subheader("📝 Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 100, 45)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar", ["≤ 120 mg/dl (False)", "> 120 mg/dl (True)"])

    with col2:
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("# Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"])

    submit_btn = st.form_submit_button("Predict")

if submit_btn:
    input_dict = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    for col, encoder in encoders.items():
        input_dict[col] = encoder.transform([input_dict[col]])[0]

    input_array = np.array([list(input_dict.values())])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    if prediction == 0:
        st.success("😊 No signs of heart disease detected.")
    else:
        st.error("⚠️ Warning: Signs of possible heart disease.")

    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(input_scaled)
    st.subheader("📊 Why this prediction?")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.bar(shap_values[0])
    st.pyplot(bbox_inches='tight')
    st.markdown("---")
    st.markdown("<h5 style='text-align: center; color: gray;'>This is a predictive tool. Always consult a doctor for medical advice.</h5>", unsafe_allow_html=True)
