import sys
from pathlib import Path

# ensure project root in path (assumes this tests/ is two levels below project root)
ROOT = Path(__file__).resolve().parents[2]
import sys
from pathlib import Path

# ensure project root in path (tests/ is directly under project root)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import main


class DummyModel:
    def predict(self, df):
        # always return zeros with same length as df
        return [0] * len(df)


def test_health_and_predict(monkeypatch):
    # inject dummy model
    monkeypatch.setattr(main, "model", DummyModel())

    client = TestClient(main.app)

    r = client.get("/health")
    assert r.status_code == 200
    # model is set by monkeypatch
    assert r.json().get("model_loaded") is True

    payload = {"data": {"feature1": 1, "feature2": 2}}
    headers = {"access_token": "changeme"}
    r = client.post("/predict", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
