import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file
import joblib
import io

app = Flask(__name__)

if not os.environ.get("TABPFN_TOKEN"):
    raise RuntimeError("TABPFN_TOKEN is missing.")

try:
    model = joblib.load('enterprise_model.pkl')
    schema = joblib.load('model_schema.pkl')
except FileNotFoundError:
    raise RuntimeError("Model artifacts not found. Please run train_engine.py first.")

# Load training split for prototype attribution if available
try:
    X_train_raw = joblib.load('X_train_raw.pkl')
    y_train_raw = joblib.load('y_train_raw.pkl')
except FileNotFoundError:
    X_train_raw, y_train_raw = None, None

test_auc = "N/A"
if os.path.exists('metrics.txt'):
    with open('metrics.txt', 'r') as f:
        test_auc = f.read().strip()

@app.route('/', methods=['GET'])
def index():
    return render_template('dashboard.html', test_auc=test_auc, result=None, batch_results=None, portfolio_summary=None, error=None)

def find_influential_training_rows(input_df, X_train_raw, y_train_raw, top_k=3):
    """
    Computes nearest historical training rows using Euclidean distance on scaled features
    to simulate TabPFN's underlying instance-based prototype weighting.
    """
    numeric_cols = input_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0 or X_train_raw is None:
        return []
    
    train_subset = X_train_raw[numeric_cols].fillna(0)
    input_subset = input_df[numeric_cols].fillna(0)
    
    distances = np.linalg.norm(train_subset.values - input_subset.values, axis=1)
    nearest_indices = np.argsort(distances)[:top_k]
    
    influential_cases = []
    for idx in nearest_indices:
        actual_label = y_train_raw.iloc[idx]
        influential_cases.append({
            "index": int(idx),
            "outcome": "DEFAULT (1)" if actual_label == 1 else "SECURE (0)",
            "match_score": round(float(1 / (1 + distances[idx])) * 100, 1)
        })
    return influential_cases

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        form_data = request.form.to_dict()
        input_dict = {}
        for col in schema:
            if col in form_data and form_data[col] != '':
                val = form_data[col]
                try:
                    val = float(val) if '.' in val else int(val)
                except ValueError:
                    pass
                input_dict[col] = [val]
            else:
                input_dict[col] = [0]
                
        input_df = pd.DataFrame(input_dict)

        # Get probability distribution [P(Low Risk), P(High Risk)]
        probas = model.predict_proba(input_df)[0]
        p_low, p_high = float(probas[0]), float(probas[1])
        
        confidence_margin = abs(p_high - p_low)
        risk_level = "HIGH RISK / DEFAULT" if p_high > 0.5 else "LOW RISK / SECURE"
        
        influential_cases = find_influential_training_rows(input_df, X_train_raw, y_train_raw)

        result = {
            "risk_probability": round(p_high * 100, 2),
            "secure_probability": round(p_low * 100, 2),
            "confidence_margin": round(confidence_margin * 100, 2),
            "classification": risk_level,
            "influential_cases": influential_cases
        }
        
        return render_template('dashboard.html', test_auc=test_auc, result=result, batch_results=None, portfolio_summary=None, error=None)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "limit" in err_str.lower():
            error_message = "API Rate Limit Notice: Token threshold reached. Resets at midnight UTC[cite: 8]."
        else:
            error_message = f"Inference Error: {err_str}"
        return render_template('dashboard.html', test_auc=test_auc, result=None, batch_results=None, portfolio_summary=None, error=error_message)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        file = request.files.get('batch_file')
        if not file or file.filename == '':
            return render_template('dashboard.html', test_auc=test_auc, error="No CSV file uploaded.")
            
        df = pd.read_csv(file)
        
        # Vectorized portfolio inference dataframe alignment
        inference_df = pd.DataFrame(index=df.index)
        for col in schema:
            if col in df.columns:
                inference_df[col] = df[col]
            else:
                inference_df[col] = 0
                
        # Single vectorized forward pass for the entire portfolio batch
        probas = model.predict_proba(inference_df)
        df['Default_Probability_Pct'] = np.round(probas[:, 1] * 100, 2)
        
        batch_df = df.sort_values(by="Default_Probability_Pct", ascending=False)
        
        total_exposure = float(batch_df['credit_amount'].sum()) if 'credit_amount' in batch_df.columns else 0.0
        avg_default_prob = float(batch_df['Default_Probability_Pct'].mean())
        high_risk_count = int((batch_df['Default_Probability_Pct'] > 50).sum())
        secure_count = len(batch_df) - high_risk_count
        
        portfolio_summary = {
            "total_records": len(batch_df),
            "total_exposure": f"${total_exposure:,.2f}",
            "avg_default_prob": round(avg_default_prob, 2),
            "high_risk_count": high_risk_count,
            "secure_count": secure_count
        }

        return render_template('dashboard.html', test_auc=test_auc, batch_results=batch_df.to_dict(orient='records'), portfolio_summary=portfolio_summary, result=None, error=None)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "limit" in err_str.lower():
            error_message = "API Rate Limit Notice during batch processing. Resets at midnight UTC[cite: 8]."
        else:
            error_message = f"Batch Error: {err_str}"
        return render_template('dashboard.html', test_auc=test_auc, error=error_message)

if __name__ == '__main__':
    app.run(debug=True, port=5000)