# Fish Market Regression with Lasso Regularization

A Streamlit project that loads the Fish Market dataset and runs a simple regression model with Lasso regularization from scratch.

## Files

- `app.py` - Streamlit app with the provided regression/Lasso code.
- `requirements.txt` - Python dependencies.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place `Fish.csv` inside a local `fish-market` folder in this project.

3. Run the app:
   ```bash
   python -m streamlit run app.py
   ```

If `streamlit` is not recognized, use `python -m streamlit run app.py` instead of `streamlit run app.py`.

## Note

- The dataset is loaded from `fish-market/Fish.csv`.