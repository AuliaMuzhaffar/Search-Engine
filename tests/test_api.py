import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_stats_endpoint():
    response = client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_indexed_terms" in data

def test_search_empty_query():
    # Empty query string (query parameter validation or app check)
    response = client.get("/api/v1/search?q=")
    # FastAPI returns 422 Unprocessable Entity for query constraints (min_length=1)
    assert response.status_code == 422

def test_search_invalid_method():
    response = client.get("/api/v1/search?q=covid&method=invalid")
    assert response.status_code == 400
    assert "Invalid ranking method" in response.json()["detail"]

def test_search_valid():
    response = client.get("/api/v1/search?q=covid&method=hybrid")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "covid"
    assert data["method"] == "hybrid"
    assert "results" in data
    assert "latency_ms" in data
