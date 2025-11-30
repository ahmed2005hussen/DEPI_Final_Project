import streamlit as st
import joblib
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load model
@st.cache_resource
def load_model():
    return joblib.load('heart_disease_model.pkl')

# Function to convert age to category
def age_to_category(age):
    """Convert actual age to age category (1-13)"""
    if age < 18:
        return 1
    elif age <= 24:
        return 1
    elif age <= 29:
        return 2
    elif age <= 34:
        return 3
    elif age <= 39:
        return 4
    elif age <= 44:
        return 5
    elif age <= 49:
        return 6
    elif age <= 54:
        return 7
    elif age <= 59:
        return 8
    elif age <= 64:
        return 9
    elif age <= 69:
        return 10
    elif age <= 74:
        return 11
    elif age <= 79:
        return 12
    else:
        return 13

model = load_model()

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'patient_context' not in st.session_state:
    st.session_state.patient_context = {}

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

# App title
st.title('🫀 Heart Disease Prediction System')
st.write('Enter patient information to predict heart disease risk')
st.divider()

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader('📋 Medical History')
    high_bp = st.selectbox('High Blood Pressure', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    high_chol = st.selectbox('High Cholesterol', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    stroke = st.selectbox('History of Stroke', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    diabetes = st.selectbox('Diabetes', [0, 1, 2], format_func=lambda x: 'No' if x == 0 else 'Pre-diabetes' if x == 1 else 'Yes')

with col2:
    st.subheader('👤 Lifestyle & Demographics')
    smoker = st.selectbox('Smoker', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    phys_activity = st.selectbox('Physical Activity (Last 30 days)', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    age_input = st.number_input('Age (years)', min_value=18, max_value=120, value=50, step=1)
    age_category = age_to_category(age_input)
    st.caption(f'Age category: {age_category} (Used for prediction)')

st.divider()

st.subheader('💪 Health Status')
gen_hlth = st.slider('General Health', 1, 5, 3, help='1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor')
diff_walk = st.selectbox('Difficulty Walking or Climbing Stairs', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

st.divider()

# Predict button
if st.button('🔍 Predict Heart Disease Risk', type='primary', use_container_width=True):
    # Prepare data (only the features used in training)
    patient_data = pd.DataFrame([{
        'HighBP': high_bp,
        'HighChol': high_chol,
        'Smoker': smoker,
        'Stroke': stroke,
        'Diabetes': diabetes,
        'PhysActivity': phys_activity,
        'GenHlth': gen_hlth,
        'DiffWalk': diff_walk,
        'Sex': sex,
        'Age': age_category
    }])

    # Make prediction
    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0, 1]
    prob_float = float(probability)
    
    # Store prediction context
    st.session_state.prediction_made = True
    st.session_state.patient_context = {
        'age': age_input,
        'sex': 'Male' if sex == 1 else 'Female',
        'high_bp': 'Yes' if high_bp == 1 else 'No',
        'high_chol': 'Yes' if high_chol == 1 else 'No',
        'smoker': 'Yes' if smoker == 1 else 'No',
        'diabetes': 'Yes' if diabetes == 2 else ('Pre-diabetes' if diabetes == 1 else 'No'),
        'stroke': 'Yes' if stroke == 1 else 'No',
        'phys_activity': 'Yes' if phys_activity == 1 else 'No',
        'gen_hlth': gen_hlth,
        'diff_walk': 'Yes' if diff_walk == 1 else 'No',
        'prediction': 'Heart Disease Risk' if prediction == 1 else 'Healthy',
        'probability': prob_float,
        'risk_level': 'High' if prob_float >= 0.6 else ('Medium' if prob_float >= 0.3 else 'Low')
    }

    # Display results
    st.divider()
    st.subheader('📊 Prediction Results')
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        diagnosis = 'Heart Disease' if prediction == 1 else 'Healthy'
        if prediction == 1:
            st.error(f'⚠️ {diagnosis}')
        else:
            st.success(f'✅ {diagnosis}')
    
    with col_b:
        st.metric('Risk Probability', f'{prob_float*100:.1f}%')
    
    with col_c:
        if prob_float < 0.3:
            risk = '🟢 Low Risk'
            st.success(risk)
        elif prob_float < 0.6:
            risk = '🟡 Medium Risk'
            st.warning(risk)
        else:
            risk = '🔴 High Risk'
            st.error(risk)

    # Risk Assessment Meter
    st.divider()
    st.subheader('🎯 Risk Assessment Meter')
    
    col_meter1, col_meter2, col_meter3 = st.columns(3)
    
    with col_meter1:
        if prob_float < 0.3:
            st.success("🟢 **LOW RISK**")
        else:
            st.write("⚪ LOW")
    
    with col_meter2:
        if 0.3 <= prob_float < 0.6:
            st.warning("🟡 **MEDIUM RISK**")
        else:
            st.write("⚪ MEDIUM")
    
    with col_meter3:
        if prob_float >= 0.6:
            st.error("🔴 **HIGH RISK**")
        else:
            st.write("⚪ HIGH")
    
    # Progress bar
    st.progress(prob_float)
    st.caption(f"Risk Score: {prob_float*100:.2f}%")

    # Recommendation
    st.divider()
    st.subheader('💊 Medical Recommendation')
    
    if prob_float < 0.3:
        st.success('''
**Low Risk Assessment**
- Maintain healthy lifestyle habits
- Continue regular physical activity
- Schedule routine checkups annually
- Monitor blood pressure and cholesterol levels
        ''')
    elif prob_float < 0.6:
        st.warning('''
**Medium Risk Assessment**
- Consult with healthcare provider soon
- Monitor cardiovascular health closely
- Consider lifestyle modifications
- Schedule follow-up tests (ECG, stress test)
- Discuss preventive medications with doctor
        ''')
    else:
        st.error('''
**⚠️ High Risk Assessment**
- **Immediate medical consultation strongly recommended!**
- Schedule comprehensive cardiac evaluation
- Discuss treatment options with cardiologist
- Implement lifestyle changes immediately
- May require medication or intervention
- Do not delay seeking medical attention
        ''')

    # Risk Factors Summary
    st.divider()
    st.subheader('⚠️ Identified Risk Factors')
    
    risk_factors = []
    if high_bp == 1:
        risk_factors.append("• High Blood Pressure")
    if high_chol == 1:
        risk_factors.append("• High Cholesterol")
    if smoker == 1:
        risk_factors.append("• Smoking")
    if diabetes > 0:
        risk_factors.append("• Diabetes/Pre-diabetes")
    if stroke == 1:
        risk_factors.append("• Previous Stroke")
    if phys_activity == 0:
        risk_factors.append("• Lack of Physical Activity")
    if gen_hlth >= 4:
        risk_factors.append("• Poor General Health")
    
    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.success("✅ No major risk factors identified")

# Chatbot Section (only show after prediction)
if st.session_state.prediction_made:
    st.divider()
    st.subheader('🤖 Ask Medical Questions')
    st.write('Have questions about your results? Ask our AI assistant!')

    # Get API key from environment
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found in environment variables!")
        st.info("Please create a .env file with: GROQ_API_KEY=your_api_key_here")
    else:
        try:
            client = Groq(api_key=api_key)
            
            # Clear chat button (moved up for better UX)
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn2:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    st.rerun()

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about your results..."):

                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Create context with patient info
                context = f"""
You are a helpful medical assistant analyzing heart disease prediction results.

Patient Information:
- Age: {st.session_state.patient_context['age']} years
- Sex: {st.session_state.patient_context['sex']}
- High Blood Pressure: {st.session_state.patient_context['high_bp']}
- High Cholesterol: {st.session_state.patient_context['high_chol']}
- Smoker: {st.session_state.patient_context['smoker']}
- Diabetes: {st.session_state.patient_context['diabetes']}
- Previous Stroke: {st.session_state.patient_context['stroke']}
- Physical Activity: {st.session_state.patient_context['phys_activity']}
- Difficulty Walking: {st.session_state.patient_context['diff_walk']}
- General Health: {st.session_state.patient_context['gen_hlth']}/5

Prediction Results:
- Diagnosis: {st.session_state.patient_context['prediction']}
- Risk Probability: {st.session_state.patient_context['probability']*100:.1f}%
- Risk Level: {st.session_state.patient_context['risk_level']}

Answer clearly and empathetically. Remind user this is informational only and not medical advice.

Question: {prompt}
"""

                # Generate response from Groq
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = client.chat.completions.create(
                                model="moonshotai/kimi-k2-instruct-0905",
                                messages=[{"role": "user", "content": context}]
                            )
                            reply = response.choices[0].message.content

                            st.markdown(reply)
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.info("Please check your GROQ_API_KEY in .env file.")

# Model performance metrics - moved to the very end
st.divider()
with st.expander("ℹ️ About This Model"):
    st.write("**Model Information:**")
    st.write("This prediction system uses an XGBoost machine learning model trained on the BRFSS Heart Disease Dataset.")
    
    st.write("")
    st.write("**Performance Metrics:**")
    
    col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
    
    with col_perf1:
        st.metric("Accuracy", "67.36%")
        st.caption("Overall correctness")
    
    with col_perf2:
        st.metric("ROC-AUC", "0.832")
        st.caption("Discrimination ability")
    
    with col_perf3:
        st.metric("Recall", "85%")
        st.caption("Disease detection rate")
    
    with col_perf4:
        st.metric("CV Recall", "96.68%")
        st.caption("Cross-validation score")
    
    st.info("""
    **What do these numbers mean?**
    - **Accuracy (67%)**: The model correctly identifies heart disease or healthy patients 67% of the time.
    - **ROC-AUC (0.832)**: Measures how well the model distinguishes between patients with and without heart disease (higher is better, max is 1.0).
    - **Recall (85%)**: Out of all patients with heart disease, the model correctly identifies 85% of them.
    - **CV Recall (96.68%)**: The model's consistency in detecting heart disease across different data samples.
    
    ⚠️ **Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice.
    """)
