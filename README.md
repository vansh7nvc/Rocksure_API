# Rockfall Prediction API

Simple FastAPI app that loads a scikit-learn pipeline from `full_rockfall_pipeline.pkl` and exposes a `/predict` endpoint.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Set an API key (optional) and run on port 8002 (default):

```powershell
$env:API_KEY = "changeme"; python .\main.py
```

3. Example request (replace with your feature JSON):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8002/predict -Headers @{"access_token"="changeme"} -Body (@{data=@{feature1=1; feature2=2}} | ConvertTo-Json)
```

Notes
- Place `full_rockfall_pipeline.pkl` next to `main.py`.
- Set `API_PORT` to change the local port. Defaults to 8002 to avoid conflicts with other APIs.
