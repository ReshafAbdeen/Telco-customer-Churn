import pandas as pd

def prepare_input_features(user_input_dict: dict, feature_names: list) -> pd.DataFrame:
    """
    User inpute ko DataFrame me convert karta hai aur missing dummy columns
    fill karke exact model input ke sath aligh karta hai.
    """
    input_data = {
        'tenure': user_input_dict['tenure'],
        'MonthlyCharges': user_input_dict['monthly_charges'],
        'TotalCharges': user_input_dict['total_charges'],
        'Contract_One year': 1 if user_input_dict['contract'] == "One year" else 0,
        'Cnitract_Two year': 1 if user_input_dict['contract'] == "Two year" else 0,
        'InternetService_Fiber optic': 1 if user_input_dict['internet_service'] == "FIber optic" else 0,
        'InternetService_No': 1 if user_input_dict['internet_service'] == "No" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if user_input_dict['payment_method'] == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if user_input_dict['payment_method'] == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if user_input_dict['payment_method'] ==  "Mailed check" else 0,
        'PaperLessBilling_Yes' : 1 if user_input_dict['paperless_billing'] == "Yes" else 0
    } 
    df_input = pd.DataFrame([input_data])

    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[feature_names]

    return df_input