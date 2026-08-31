import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# =========================================================
# LOAD MODEL, SCALER AND THRESHOLD
# =========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("model/xgb_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    threshold = joblib.load("model/threshold.pkl")

    return model, scaler, threshold


model, scaler, threshold = load_artifacts()


# =========================================================
# TITLE
# =========================================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter the transaction details below to check whether "
    "the transaction is potentially fraudulent."
)


# =========================================================
# USER INPUTS
# =========================================================

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)

avg_transaction_amount = st.number_input(
    "Average Transaction Amount",
    min_value=0.0,
    value=100.0
)

hour = st.number_input(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)

is_international = st.selectbox(
    "International Transaction?",
    ["No", "Yes"]
)

account_age_days = st.number_input(
    "Account Age (Days)",
    min_value=0,
    value=365
)

transactions_last_24h = st.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    value=5
)

failed_transactions = st.number_input(
    "Failed Transactions",
    min_value=0,
    value=0
)

card_present = st.selectbox(
    "Card Present?",
    ["No", "Yes"]
)

transaction_type = st.selectbox(
    "Transaction Type",
    [
        "online",
        "POS",
        "ATM",
        "Bank Transfer"
    ]
)

merchant = st.selectbox(
    "Merchant Category",
    [
        "Healthcare",
        "grocery",
        "Travel",
        "Fuel",
        "Entertainment",
        "Electronics",
        "Clothing",
        "Restaurant"
    ]
)

location = st.selectbox(
    "Location",
    [
        "chennai",
        "Mumbai",
        "Bengaluru",
        "Kolkata",
        "Hyderabad",
        "Delhi",
        "Vijayawada",
        "Pune"
    ]
)

device_type = st.selectbox(
    "Device Type",
    [
        "mobile",
        "Desktop",
        "Tablet",
        "POS Terminal"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Check Transaction"):

    # -----------------------------------------------------
    # Convert Yes / No values to 0 / 1
    # -----------------------------------------------------

    international_value = (
        1 if is_international == "Yes" else 0
    )

    card_present_value = (
        1 if card_present == "Yes" else 0
    )


    # -----------------------------------------------------
    # Create all model features
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "amount": [amount],

        "hour": [hour],

        "is_international": [
            international_value
        ],

        "account_age_days": [
            account_age_days
        ],

        "transactions_last_24h": [
            transactions_last_24h
        ],

        "avg_transaction_amount": [
            avg_transaction_amount
        ],

        "failed_transactions": [
            failed_transactions
        ],

        "card_present": [
            card_present_value
        ],

        "transaction_type_online": [
            1 if transaction_type == "online" else 0
        ],

        "transaction_type_POS": [
            1 if transaction_type == "POS" else 0
        ],

        "transaction_type_ATM": [
            1 if transaction_type == "ATM" else 0
        ],

        "transasction_type_Bank_Transfer": [
            1 if transaction_type == "Bank Transfer" else 0
        ],

        "merchant_Healthcare": [
            1 if merchant == "Healthcare" else 0
        ],

        "merchant_grocery": [
            1 if merchant == "grocery" else 0
        ],

        "merchant_Travel": [
            1 if merchant == "Travel" else 0
        ],

        "merchant_Fuel": [
            1 if merchant == "Fuel" else 0
        ],

        "merchant_Entertainment": [
            1 if merchant == "Entertainment" else 0
        ],

        "merchant_Electronics": [
            1 if merchant == "Electronics" else 0
        ],

        "merchant_Clothing": [
            1 if merchant == "Clothing" else 0
        ],

        "merchant_Restaurant": [
            1 if merchant == "Restaurant" else 0
        ],

        "location_chennai": [
            1 if location == "chennai" else 0
        ],

        "location_Mumbai": [
            1 if location == "Mumbai" else 0
        ],

        "location_Bengaluru": [
            1 if location == "Bengaluru" else 0
        ],

        "location_Kolkata": [
            1 if location == "Kolkata" else 0
        ],

        "location_Hyderabad": [
            1 if location == "Hyderabad" else 0
        ],

        "location_Delhi": [
            1 if location == "Delhi" else 0
        ],

        "location_Vijayawada": [
            1 if location == "Vijayawada" else 0
        ],

        "location_Pune": [
            1 if location == "Pune" else 0
        ],

        "device_type_mobile": [
            1 if device_type == "mobile" else 0
        ],

        "device_type_Desktop": [
            1 if device_type == "Desktop" else 0
        ],

        "device_type_Tablet": [
            1 if device_type == "Tablet" else 0
        ],

        "device_type_POS_Terminal": [
            1 if device_type == "POS Terminal" else 0
        ]
    })


    # =====================================================
    # ENSURE EXACT FEATURE ORDER
    # =====================================================

    expected_features = [
        "amount",
        "hour",
        "is_international",
        "account_age_days",
        "transactions_last_24h",
        "avg_transaction_amount",
        "failed_transactions",
        "card_present",
        "transaction_type_online",
        "transaction_type_POS",
        "transaction_type_ATM",
        "transasction_type_Bank_Transfer",
        "merchant_Healthcare",
        "merchant_grocery",
        "merchant_Travel",
        "merchant_Fuel",
        "merchant_Entertainment",
        "merchant_Electronics",
        "merchant_Clothing",
        "merchant_Restaurant",
        "location_chennai",
        "location_Mumbai",
        "location_Bengaluru",
        "location_Kolkata",
        "location_Hyderabad",
        "location_Delhi",
        "location_Vijayawada",
        "location_Pune",
        "device_type_mobile",
        "device_type_Desktop",
        "device_type_Tablet",
        "device_type_POS_Terminal"
    ]

    input_data = input_data[expected_features]


    # =====================================================
    # SCALE NUMERICAL FEATURES
    # =====================================================

    num_cols = [
        "amount",
        "avg_transaction_amount",
        "hour"
    ]

    input_data[num_cols] = scaler.transform(
        input_data[num_cols]
    )


    # =====================================================
    # PREDICT FRAUD PROBABILITY
    # =====================================================

    fraud_probability = model.predict_proba(
        input_data
    )[0][1]


    # =====================================================
    # APPLY THRESHOLD = 0.4
    # =====================================================

    prediction = int(
        fraud_probability >= threshold
    )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.subheader("Prediction Result")

    st.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )


    if prediction == 1:

        st.error(
            "🚨 FRAUDULENT TRANSACTION DETECTED"
        )

        st.warning(
            "This transaction has been classified as potentially fraudulent."
        )

    else:

        st.success(
            "✅ LEGITIMATE TRANSACTION"
        )

        st.info(
            "This transaction has been classified as legitimate."
        )


    # =====================================================
    # SHOW INPUT DETAILS
    # =====================================================

    st.subheader("Transaction Details")

    st.write(
        f"**Amount:** ₹{amount:,.2f}"
    )

    st.write(
        f"**Average Transaction Amount:** "
        f"₹{avg_transaction_amount:,.2f}"
    )

    st.write(
        f"**Transaction Hour:** {hour}"
    )

    st.write(
        f"**International:** {is_international}"
    )

    st.write(
        f"**Transaction Type:** {transaction_type}"
    )

    st.write(
        f"**Merchant:** {merchant}"
    )

    st.write(
        f"**Location:** {location}"
    )

    st.write(
        f"**Device:** {device_type}"
    )