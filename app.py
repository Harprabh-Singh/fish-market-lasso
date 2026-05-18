import os

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = "fish-market"
DATA_FILE = os.path.join(DATA_DIR, "Fish.csv")

st.title("Fish Market Regression with Lasso Regularization")
st.markdown(
    "This app loads the Fish Market dataset from a local file, trains a simple regression model, and shows Lasso regularization results from scratch."
)

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    return pd.read_csv(DATA_FILE)

df = load_data()
if df is None:
    st.error(
        "Dataset not found. Please place `Fish.csv` inside a local `fish-market` folder inside this project."
    )
    st.write("Expected path:", DATA_FILE)
    st.stop()

st.subheader("Dataset Preview")
st.write(df.head())
st.write("**Rows:**", df.shape[0], "**Columns:**", df.shape[1])

if st.checkbox("Show dataset statistics"):
    st.write(df.describe())

# Dataset preprocessing
x = df.drop(columns="Species", axis=1).drop(columns="Weight", axis=1).values
# The original code uses `df.drop(columns="Weight",axis=1)` and `df["Weight"]`
y = df["Weight"].values.reshape(-1, 1)

split = int(0.8 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

mean = x_train.mean(axis=0)
standard_dev = x_train.std(axis=0)
x_train = (x_train - mean) / standard_dev
x_test = (x_test - mean) / standard_dev

lr = 0.01
epochs = 100
penalty = 0.5

# jai mata di bolke model prediction start

def regression(x, y, regularization=None):
    w = np.zeros((x.shape[1], 1))
    b = 0
    n = x.shape[0]
    for i in range(epochs):
        y_cap = x.dot(w) + b
        error = y - y_cap
        mse = (1 / n) * np.sum(error**2)
        dw = -(2 / n) * x.T.dot(error)
        db = -(2 / n) * np.sum(error)

        if regularization == "Lasso":
            dw = dw + penalty * np.sign(w)
        w = w - lr * dw
        b = b - lr * db
    return w, b

w_lin, b_lin = regression(x_train, y_train)
w_lass, b_lass = regression(x_train, y_train, regularization="Lasso")

st.subheader("Model Results")
col1, col2 = st.columns(2)
with col1:
    st.write("**Linear regression (same code implementation)**")
    st.write("Weights:")
    st.write(w_lin.flatten())
    st.write("Bias:", float(b_lin))
with col2:
    st.write("**Lasso regression**")
    st.write("Weights:")
    st.write(w_lass.flatten())
    st.write("Bias:", float(b_lass))


def predict(x, w, b):
    return x.dot(w) + b


def mse_score(x, y, w, b):
    y_pred = predict(x, w, b)
    return float(np.mean((y - y_pred) ** 2))

st.subheader("Test Performance")
st.write("MSE on test set (Linear):", mse_score(x_test, y_test, w_lin, b_lin))
st.write("MSE on test set (Lasso):", mse_score(x_test, y_test, w_lass, b_lass))