import streamlit as st
import joblib
from preprocess import prepare_input_features

# Page Setup
st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="wide")

st.title(" Customer Churn Prediction App")
st.write("Is app ke zariye aap customer ke details daal kar check kar sakte hain ki wo business chhod kar jayega (Churn) ya nahi.")

# Model & Scaler Load
@st.cache_resource
def load_assets():
    model = joblib.load('model/logistic_regression_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("Model & Scaler successfully loaded!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")

# Sidebar Inputs
st.sidebar.header("Customer Information")

user_inputs = {
    'tenure': st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12),
    'monthly_charges': st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0),
    'contract': st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"]),
    'internet_service': st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    'payment_method': st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]),
    'paperless_billing': st.sidebar.radio("Paperless Billing", ["Yes", "No"])
}

# Calculated Total Charges
user_inputs['total_charges'] = st.sidebar.number_input(
    "Total Charges ($)", 
    min_value=18.0, 
    max_value=8500.0, 
    value=user_inputs['tenure'] * user_inputs['monthly_charges']
)

# Prediction Logic
if st.button("Predict Churn Risk"):
    # Preprocessing file se cleaned Dataframe pass karein
    df_features = prepare_input_features(user_inputs, scaler.feature_names_in_)
    
    # Scale Features
    scaled_input = scaler.transform(df_features)
    
    # Model Predictions
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    
    # UI Output
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error(f" High Churn Risk! (Probability: {probability:.2%})")
            st.write("Customer ke platform chhodne ke chances zyada hain. Incentive offer karein!")
        else:
            st.success(f"Low Churn Risk! (Probability: {probability:.2%})")
            st.write("Customer satisfied lag raha hai.")

    with col2:
        st.metric("Churn Probability", f"{probability:.2%}")
        st.progress(float(probability))