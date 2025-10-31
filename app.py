import streamlit as st
import joblib
import pandas as pd

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
    chol_check = st.selectbox('Cholesterol Check (Last 5 years)', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    stroke = st.selectbox('History of Stroke', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    diabetes = st.selectbox('Diabetes', [0, 1, 2], format_func=lambda x: 'No' if x == 0 else 'Pre-diabetes' if x == 1 else 'Yes')
    
with col2:
    st.subheader('👤 Lifestyle & Demographics')
    smoker = st.selectbox('Smoker', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    phys_activity = st.selectbox('Physical Activity (Last 30 days)', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    age_input = st.number_input('Age (years)', min_value=18, max_value=120, value=50, step=1)
    age_category = age_to_category(age_input)
    st.caption(f'Age category: {age_category} (Used for prediction)')

st.divider()
st.subheader('💪 Health Status')
col3, col4, col5 = st.columns(3)

with col3:
    gen_hlth = st.slider('General Health', 1, 5, 3, help='1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor')
with col4:
    ment_hlth = st.slider('Mental Health (Bad days/month)', 0, 30, 0)
with col5:
    phys_hlth = st.slider('Physical Health (Bad days/month)', 0, 30, 0)

diff_walk = st.selectbox('Difficulty Walking or Climbing Stairs', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

st.divider()

# Predict button
if st.button('🔍 Predict Heart Disease Risk', type='primary', use_container_width=True):
    # Prepare data
    patient_data = pd.DataFrame([{
        'HighBP': high_bp,
        'HighChol': high_chol,
        'CholCheck': chol_check,
        'BMI': bmi,
        'Smoker': smoker,
        'Stroke': stroke,
        'Diabetes': diabetes,
        'PhysActivity': phys_activity,
        'GenHlth': gen_hlth,
        'MentHlth': ment_hlth,
        'PhysHlth': phys_hlth,
        'DiffWalk': diff_walk,
        'Sex': sex,
        'Age': age_category
    }])
    
    # Make prediction
    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0, 1]
    prob_float = float(probability)
    
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
    if bmi > 30:
        risk_factors.append("• Obesity (BMI > 30)")
    if phys_activity == 0:
        risk_factors.append("• Lack of Physical Activity")
    if gen_hlth >= 4:
        risk_factors.append("• Poor General Health")
    
    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.success("✅ No major risk factors identified")

# Footer
st.divider()
st.info('ℹ️ **Disclaimer:** This is a predictive model for educational and screening purposes only. It should NOT replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.')

# Model performance metrics
with st.expander("📈 Model Performance Metrics"):
    col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
    with col_perf1:
        st.metric("Accuracy", "67.36%")
    with col_perf2:
        st.metric("ROC-AUC", "0.832")
    with col_perf3:
        st.metric("Recall", "85%")
    with col_perf4:
        st.metric("CV Recall", "96.68%")
    
    st.caption("Model: XGBoost Classifier v3.0")
    st.caption("Training Data: BRFSS Heart Disease Dataset")