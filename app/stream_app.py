import streamlit as st
import joblib, os, sys, pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
pipeline = joblib.load("models/model.pkl")


st.title("🏠 House Price Predictor")

col1, col2 = st.columns(2)

with col1:
    GrLivArea   = st.number_input("Above-ground living area (sq ft)", 500, 5000, 1500)
    OverallQual = st.slider("Overall quality (1–10)", 1, 10, 5)
    GarageCars  = st.number_input("Garage capacity (cars)", 0, 4, 2)
    TotalBsmtSF = st.number_input("Basement area (sq ft)", 0, 3000, 800)
    YearBuilt   = st.number_input("Year built", 1872, 2010, 1990)

with col2:
    QUAL = ["Po", "Fa", "TA", "Gd", "Ex"]
    Neighborhood = st.selectbox("Neighborhood", ["NAmes", "CollgCr", "OldTown",
                                                  "Edwards", "Somerst", "NridgHt"])
    ExterQual    = st.selectbox("Exterior quality",  QUAL, index=2)
    KitchenQual  = st.selectbox("Kitchen quality",   QUAL, index=2)
    BsmtQual     = st.selectbox("Basement quality",  QUAL, index=2)
    GarageQual   = st.selectbox("Garage quality",    QUAL, index=2)

if st.button("Predict price"):
    # Columns the pipeline was trained on
    import numpy as np
    
    # Load train to get all expected columns
    TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
    train_df = pd.read_csv("data/raw/train.csv")
    train_df = train_df.drop(columns=["SalePrice", "Id"], errors="ignore")
    
    # Start with one empty row matching all training columns
    input_df = pd.DataFrame([train_df.mode().iloc[0]])  # fill with mode defaults
    
    # Override with user inputs
    input_df["GrLivArea"]   = GrLivArea
    input_df["OverallQual"] = OverallQual
    input_df["GarageCars"]  = GarageCars
    input_df["TotalBsmtSF"] = TotalBsmtSF
    input_df["YearBuilt"]   = YearBuilt
    input_df["Neighborhood"]= Neighborhood
    input_df["ExterQual"]   = ExterQual
    input_df["KitchenQual"] = KitchenQual
    input_df["BsmtQual"]    = BsmtQual
    input_df["GarageQual"]  = GarageQual

    pred = pipeline.predict(input_df)[0]
    st.success(f"### Estimated Sale Price: ${np.expm1(pred):,.0f}")
    st.balloons()