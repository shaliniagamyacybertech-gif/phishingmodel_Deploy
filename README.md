Phishing API - Render deployment bundle
=======================================

Files included:
- original_full.py
  This is the unmodified file you provided (kept exactly as-is).
- main_api.py
  A deployment entrypoint that runs only the Flask API and respects the PORT env var (required by Render).
  NOTE: main_api.py is provided as an extra file so the original file remains unchanged.
- phishing_detector_complete.pkl
  (not included) -- add this file to the ZIP before deploying.
- phishing_detector_complete_slm.pt
  (optional) SLM checkpoint; add if available.
- requirements.txt
  Python dependencies.
- render.yaml
  Render service config.

Deployment steps (Render):
1. Add the missing model files into this folder:
   - phishing_detector_complete.pkl
   - phishing_detector_complete_slm.pt  (optional but recommended)

2. Push this repository to GitHub.

3. On Render dashboard:
   - Create a new Web Service from the GitHub repo.
   - Render will run the startCommand in render.yaml:
     gunicorn main_api:app --bind 0.0.0.0:$PORT --workers 1 --threads 4

4. After deployment, test:
   GET  https://<your-service>.onrender.com/health
   POST https://<your-service>.onrender.com/predict  (JSON: {"url":"http://example.com"})

Important:
- You asked to not modify your original code. I kept original_full.py unchanged and included main_api.py as a separate deployment entrypoint.
- You MUST include the model pickle and SLM checkpoint files in the ZIP before deploying.
