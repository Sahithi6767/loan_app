# Loan Approval Predictor

A Streamlit app for loan default risk prediction using an XGBoost model.

## 📁 Project structure

- `app.py` : Streamlit application
- `loan_risk_model.joblib` : pre-trained model artifact
- `requirements.txt` : dependencies
- `loan.ipynb` : exploratory notebook (optional)
- `.devcontainer/` : VS Code devcontainer config (optional)

## 🚀 Quick start

1. Create and activate a Python venv

   Windows:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run Streamlit
   ```bash
   streamlit run app.py
   ```

4. Open browser at `http://localhost:8501`

## 🧪 Model checks

- The model expects feature order:
  1. `loan_to_income_ratio`
  2. `LTV_percent`
  3. `interest_burden`
  4. `upfront_charge_percent`
  5. `long_term_flag`
  6. `emi_to_income_ratio`
  7. `high_risk`
  8. `approval_readiness_score`
  9. `DTI_Category`

- `app.py` aligns columns to this order to avoid `feature_names mismatch`.

## 🧾 GitHub setup

```bash
git init
git add .
git commit -m "Initial loan approval app"
# create repo on GitHub via CLI or UI
# e.g. gh repo create yourname/loanapproval --public --source . --push
```

## 🧰 Optional: use devcontainer

1. Install Docker and VS Code Remote Containers
2. Open folder in VS Code
3. Click `Reopen in Container`

The container auto-installs requirements and extensions.

## 📌 Notes

- If `loan_risk_model.joblib` is big, consider Git LFS for repository size management.
