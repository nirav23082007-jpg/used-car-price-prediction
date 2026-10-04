"""Streamlit app: predict the resale price of a used car.

Run with:  streamlit run app.py
Needs car_price_model.pkl (created by car_price_prediction.ipynb) next to this file.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

CURRENT_YEAR = 2026  # must match the notebook (car_age = CURRENT_YEAR - year)
CAT_FEATURES = ["brand", "fuel", "seller_type", "transmission", "owner"]
MODEL_PATH = Path(__file__).parent / "car_price_model.pkl"

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def format_inr(value: float) -> str:
    """Format a number with Indian digit grouping, e.g. 1234567 -> 12,34,567."""
    s = str(int(round(value)))
    if len(s) <= 3:
        return "₹" + s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + ",".join(groups + [tail])


def predict_price(model, brand, year, km_driven, fuel, seller_type, transmission, owner) -> float:
    row = pd.DataFrame([{
        "brand": brand,
        "car_age": CURRENT_YEAR - year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }])
    return float(model.predict(row)[0])   # the saved model already returns INR


try:
    model = load_model()
except FileNotFoundError:
    st.error("car_price_model.pkl not found. Run the notebook first to create it.")
    st.stop()

# Dropdown options come straight from the categories the model saw in training
encoder = model.regressor_.named_steps["prep"].named_transformers_["cat"]
options = {name: list(cats) for name, cats in zip(CAT_FEATURES, encoder.categories_)}

st.title("🚗 Used Car Price Predictor")
st.caption("Estimate the resale price of a used car from CarDekho listings data.")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", sorted(options["brand"]), index=sorted(options["brand"]).index("Maruti")
                         if "Maruti" in options["brand"] else 0)
    year = st.number_input("Year of manufacture", min_value=1992, max_value=CURRENT_YEAR, value=2015, step=1)
    km_driven = st.number_input("Kilometres driven", min_value=0, max_value=500_000, value=60_000, step=1_000)
    fuel = st.selectbox("Fuel type", options["fuel"])
with col2:
    transmission = st.selectbox("Transmission", options["transmission"])
    seller_type = st.selectbox("Seller type", options["seller_type"])
    owner = st.selectbox("Owner", options["owner"])

if st.button("Predict price", type="primary"):
    price = predict_price(model, brand, int(year), km_driven, fuel, seller_type, transmission, owner)
    st.metric("Estimated selling price", format_inr(price))
    st.caption(f"Car age used by the model: {CURRENT_YEAR - int(year)} years")
    if fuel in ("CNG", "LPG", "Electric") or owner == "Test Drive Car":
        st.warning("The training data has very few cars like this, so treat the estimate with extra caution.")

st.info("This is a rough estimate. On held-out data the typical error was about ₹1.5 lakh (roughly 20–35% of the price). "
        "Car model/variant, condition and city are not considered.")
