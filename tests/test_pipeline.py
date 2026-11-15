import pytest
from fastapi.testclient import TestClient
from src.data_cleaner import load_and_clean_data
from src.main import app


def test_data_cleaner_repairs_anomalies(tmp_path):
    dirty_content = (
        '"data","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"\n'
        '"1",2015-02-11\n'
        '14:48:00,21.76,31.13,437.33,1029.66,0.005021,1\n'
        '"3",2015-02-11 14:50:00.21.7675,31.12,434.0,1003.75,0.005021,0\n'
    )
    sample_file = tmp_path / "dirty_sample.txt"
    sample_file.write_text(dirty_content, encoding="utf-8")

    df = load_and_clean_data(str(sample_file))
    assert len(df) == 2
    assert df.loc[0, "Temperature"] == pytest.approx(21.76)
    assert df.loc[1, "Temperature"] == pytest.approx(21.7675)
    assert df["Occupancy"].tolist() == [1, 0]


def test_predict_endpoint_valid_event():
    payload = {
        "timestamp": "2015-02-11 14:48:00",
        "temperature": 21.76,
        "humidity": 31.13,
        "light": 437.33,
        "co2": 1029.66,
        "humidity_ratio": 0.005021,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["occupancy_prediction"] in [0, 1]
        assert 0.0 <= body["confidence_score"] <= 1.0
        assert body["actuator_directive"] in ["ENABLE_HVAC_AND_LIGHTS", "STANDBY_MODE"]


def test_predict_endpoint_rejects_bad_schema():
    bad_payload = {
        "timestamp": "2015-02-11 14:48:00",
        "temperature": "CORRUPT_SENSOR_STRING",
        "humidity": 31.13,
        "light": 437.33,
        "co2": 1029.66,
        "humidity_ratio": 0.005021,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422
