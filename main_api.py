# main_api.py
# API-only entrypoint extracted for deployment.
# This file is provided in addition to the original full notebook file (original_full.py).
# It runs the Flask API and binds to the PORT environment variable (required by Render).

import os
import math
import re
import joblib
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS

# ML / Torch imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
CORS(app)

# Load the trained model (ensure these files are included in the deployment ZIP)
model_path = os.path.join(os.path.dirname(__file__), 'phishing_detector_complete.pkl')
slm_path = os.path.join(os.path.dirname(__file__), 'phishing_detector_complete_slm.pt')

print("\\n🔄 Loading trained model for API...")
model_data = joblib.load(model_path)
ml_model = model_data['model']
scaler = model_data['scaler']

# Load SLM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
slm_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
slm_model.eval()

slm_classifier = nn.Sequential(
    nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, 2)
).to(device)

if os.path.exists(slm_path):
    checkpoint = torch.load(slm_path, map_location=device)
    slm_classifier.load_state_dict(checkpoint['classifier'])
    print("✅ SLM loaded with trained weights")
else:
    print("⚠ SLM checkpoint not found; running with default SLM head (untrained)")

def entropy(text):
    if not text:
        return 0
    p = [text.count(c) / len(text) for c in set(text)]
    return -sum(x * math.log2(x) for x in p if x > 0)

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        'url_len': len(url),
        'domain_len': len(domain),
        'path_len': len(path),
        'num_dot': url.count('.'),
        'num_hyph': url.count('-'),
        'num_slash': url.count('/'),
        'has_https': 1 if parsed.scheme == 'https' else 0,
        'entropy_url': entropy(url),
        'entropy_domain': entropy(domain),
        'entropy_path': entropy(path),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'num_special': sum(not c.isalnum() for c in url),
        'num_params': url.count('='),
        'has_ip': 1 if re.search(r'\\d+\\.\\d+\\.\\d+\\.\\d+', domain) else 0
    }
    return features

def analyze_with_slm(url):
    try:
        parsed = urlparse(url)
        text = f"URL: {url} | Protocol: {parsed.scheme} | Domain: {parsed.netloc} | Path: {parsed.path}"

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = slm_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            logits = slm_classifier(emb)
            probs = torch.softmax(logits, dim=1)[0]

        phishing_prob = probs[1].item()
        legit_prob = probs[0].item()

        return {
            'slm_prediction': int(phishing_prob > 0.5),
            'slm_phishing_probability': round(phishing_prob, 4),
            'slm_legitimate_probability': round(legit_prob, 4),
            'slm_confidence': round(max(phishing_prob, legit_prob), 4)
        }
    except Exception as e:
        print("SLM analysis failed:", e)
        return None

@app.route('/')
def home():
    return jsonify({
        'message': 'Phishing Detection API - Trained and Ready!',
        'version': '1.0',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict',
            'batch': 'POST /batch-predict'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'slm_loaded': os.path.exists(slm_path)
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'URL required'}), 400

    url = data['url'].strip()
    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'URL must start with http:// or https://'}), 400

    try:
        features = extract_features(url)
        X = scaler.transform([list(features.values())])
        ml_pred = ml_model.predict(X)[0]
        ml_prob = ml_model.predict_proba(X)[0]

        result = {
            'url': url,
            'ml_prediction': int(ml_pred),
            'ml_phishing_probability': round(float(ml_prob[1]), 4),
            'ml_legitimate_probability': round(float(ml_prob[0]), 4),
            'prediction_label': 'Phishing' if ml_pred == 1 else 'Legitimate'
        }

        slm_result = analyze_with_slm(url)
        if slm_result:
            result.update(slm_result)
            ensemble_prob = ml_prob[1] * 0.6 + slm_result['slm_phishing_probability'] * 0.4
            result['ensemble_prediction'] = int(ensemble_prob > 0.5)
            result['ensemble_probability'] = round(float(ensemble_prob), 4)
            result['final_prediction'] = 'Phishing' if ensemble_prob > 0.5 else 'Legitimate'
            result['confidence'] = round(float(max(ensemble_prob, 1 - ensemble_prob)), 4)
        else:
            result['final_prediction'] = result['prediction_label']
            result['confidence'] = round(float(max(ml_prob)), 4)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    if not data or 'urls' not in data:
        return jsonify({'error': 'urls array required'}), 400

    urls = data['urls']
    if len(urls) > 100:
        return jsonify({'error': 'Max 100 URLs per batch'}), 400

    results = []
    for url in urls:
        try:
            features = extract_features(url)
            X = scaler.transform([list(features.values())])
            ml_pred = ml_model.predict(X)[0]
            ml_prob = ml_model.predict_proba(X)[0][1]

            results.append({
                'url': url,
                'prediction': int(ml_pred),
                'probability': round(float(ml_prob), 4),
                'label': 'Phishing' if ml_pred == 1 else 'Legitimate'
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e)})

    return jsonify({'results': results})

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT)