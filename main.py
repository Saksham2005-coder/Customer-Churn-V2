
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Create a dummy dataset
data = {
    "tenure": [1, 2, 5, 10, 15, 24, 30, 36, 48, 60,
               3, 6, 12, 18, 25, 40, 50, 55, 8, 20,
               4, 9, 14, 22, 32, 45, 52, 58, 7, 28],
    "monthly_charges": [90, 85, 95, 80, 75, 60, 55, 50, 45, 40,
                        100, 95, 85, 70, 65, 50, 45, 40, 90, 70,
                        95, 85, 80, 65, 55, 48, 42, 38, 92, 60],
    "contract": [
        "Month-to-month", "Month-to-month", "Month-to-month",
        "Month-to-month", "One year", "One year", "One year",
        "Two year", "Two year", "Two year",
        "Month-to-month", "Month-to-month", "Month-to-month",
        "One year", "One year", "Two year", "Two year",
        "Two year", "Month-to-month", "One year",
        "Month-to-month", "Month-to-month", "One year",
        "One year", "Two year", "Two year", "Two year",
        "Two year", "Month-to-month", "One year"
    ],
    "support_calls": [5, 6, 4, 4, 3, 2, 2, 1, 1, 0,
                      7, 5, 4, 3, 2, 1, 0, 1, 6, 3,
                      6, 5, 3, 2, 1, 1, 0, 0, 5, 2],
    "churn": [
        "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "No",
        "Yes", "Yes", "Yes", "No", "No", "No", "No", "No", "Yes", "No",
        "Yes", "Yes", "No", "No", "No", "No", "No", "No", "Yes", "No"
    ]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# Step 2: Convert categorical features into numbers
encoder = LabelEncoder()
df["contract"] = encoder.fit_transform(df["contract"])
df["churn"] = encoder.fit_transform(df["churn"])

# Step 3: Separate features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Step 5: Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Predict churn for a new customer
# Enter: tenure, monthly_charges, contract, support_calls

print("\nNew Customer Prediction")

tenure = float(input("Enter customer tenure in months: "))
monthly_charges = float(input("Enter monthly charges: "))

print("Contract options: Month-to-month, One year, Two year")
contract = input("Enter contract type: ").strip()

support_calls = int(input("Enter number of support calls: "))

# Encode contract using the encoder's learned classes
if contract not in encoder.classes_:
    print("Invalid contract type!")
else:
    contract_encoded = encoder.transform([contract])[0]

    customer = pd.DataFrame(
        [[tenure, monthly_charges, contract_encoded, support_calls]],
        columns=["tenure", "monthly_charges", "contract", "support_calls"]
    )

    customer_scaled = scaler.transform(customer)
    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0][1]

    if prediction == 1:
        print("\nPrediction: CUSTOMER LIKELY TO CHURN")
    else:
        print("\nPrediction: CUSTOMER LIKELY TO STAY")

    print("Estimated churn probability:", round(probability * 100, 2), "%")
