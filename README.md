# Customer Churn Prediction

This project predicts whether a telecom customer will leave the company (churn) or stay.

Companies lose money when customers leave. If we know who is likely to leave, the company can offer them a discount or help before they go.

## Dataset

- Telco Customer Churn dataset (from Kaggle)
- About 7,000 customers
- Information used: tenure, monthly charges, contract type, internet service, payment method and more
- Target column: `Churn` (Yes or No)

## What I did

1. Cleaned the data (removed `customerID`, fixed blank values in `TotalCharges`)
2. Converted text columns into numbers using one-hot encoding
3. Split the data into 80% training and 20% testing (same churn ratio in both)
4. Scaled the features with `StandardScaler`
5. Trained and compared four models: Logistic Regression, Random Forest, XGBoost and Decision Tree (balanced)
6. Checked Accuracy, Precision, Recall, F1 and ROC-AUC, not only accuracy
7. Saved the best model and built a small Streamlit app

## Results

All scores are on the test set (1,409 customers).

| Model | Accuracy | Precision (churn) | Recall (churn) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.820 | 0.68 | 0.60 | 0.862 |
| Random Forest | 0.796 | 0.66 | 0.47 | 0.842 |
| XGBoost | 0.794 | 0.64 | 0.52 | 0.838 |
| Decision Tree (balanced) | 0.745 | 0.51 | 0.79 | 0.839 |

**Best model: Logistic Regression**

Why I chose it:
- It has the highest Accuracy (0.820) and the highest ROC-AUC (0.862)
- It also has the best Precision and a good Recall among the first three models
- It is simple, fast and easy to explain
- The dataset is small and most features have a simple relationship with churn, so complex models like Random Forest and XGBoost did not help

## Important point about Recall

Only about 27% of customers in the data are churners, so accuracy alone can be misleading.
Recall is important here because missing a customer who is about to leave costs more than giving a discount to a customer who would have stayed.

The Decision Tree with `class_weight="balanced"` gave the highest churn Recall (0.79), but its Precision (0.51), Accuracy and overall ranking quality were lower. So it catches more churners but also gives many false alarms.
Logistic Regression gave the best balance, so I used it as the final model.

## Limitations

- Only the Decision Tree used balanced class weights, so the comparison is not fully equal
- Models were not tuned much (default or basic settings)
- The default threshold of 0.5 was used

## Key findings

- Customers with a month-to-month contract leave more often
- Customers with low tenure (new customers) leave more often
- Customers with higher monthly charges are more likely to leave

## Project structure

```
Telco Customer Churn/
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
├── train.py
├── app.py
├── requirements.txt
└── README.md
```

## How to run

1. Install the libraries

```
pip install -r requirements.txt
```

2. Train the model

```
python train.py
```

3. Start the app

```
streamlit run app.py
```

The app asks for some customer details and shows the churn probability.
The demo app uses 6 main features only (tenure, monthly charges, contract, internet service, tech support, paperless billing). The other features are kept at default values.

## Tools used

Python, Pandas, Scikit-learn, XGBoost, Joblib, Streamlit

## Future work

- Try `class_weight="balanced"` on Logistic Regression to improve Recall
- Try a lower decision threshold (like 0.3 or 0.4) to catch more churners
- Use SHAP to explain each prediction
- Tune the models with RandomizedSearchCV
- Add all features to the app
