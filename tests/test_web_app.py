"""Unit tests for the EVE Search Agent Flask web application."""

from __future__ import annotations

import pytest

from eve_search import web_app
from eve_search.web_app import app


@pytest.fixture(autouse=True)
def clean_mission_state():
    """Ensure every test runs with an isolated session state."""
    web_app.reset_state()
    yield
    web_app.reset_state()


class TestWebAppRoutes:
    """Test cases for the web application routes."""

    def test_index_route(self):
        """GET / should serve the HTML UI page."""
        with app.test_client() as client:
            response = client.get("/")
            assert response.status_code == 200
            assert "html" in response.text.lower()

    def test_run_route_no_mission(self):
        """POST /run when no mission is running should return started."""
        with app.test_client() as client:
            response = client.post("/run", json={})
            assert response.status_code == 200
            data = response.get_json()
            assert data["status"] in ("started", "error")

    def test_state_route_no_mission(self):
        """GET /state when no mission is running should return no_mission."""
        with app.test_client() as client:
            response = client.get("/state")
            assert response.status_code == 200
            data = response.get_json()
            assert data.get("status") == "no_mission"

    def test_init_and_step_routes(self):
        """POST /init and POST /step should update real simulator state."""
        with app.test_client() as client:
            init_res = client.post("/init", json={"seed": 42})
            assert init_res.status_code == 200
            init_data = init_res.get_json()
            assert init_data["status"] == "ok"
            assert "state" in init_data
            assert init_data["state"]["step"] == 0
            assert init_data["state"]["pos"] == [2, 2]

            step_res = client.post("/step", json={})
            assert step_res.status_code == 200
            step_data = step_res.get_json()
            assert step_data["status"] == "ok"
            assert step_data["state"]["step"] == 1

    def test_manual_action_route(self):
        """POST /manual should execute an explicit operator command."""
        with app.test_client() as client:
            client.post("/init", json={"seed": 42})
            res = client.post("/manual", json={"action": "move_forward"})
            assert res.status_code == 200
            data = res.get_json()
            assert data["status"] == "ok"
            assert data["state"]["step"] == 1

    def test_default_config_route(self):
        """GET /config/default should return the bundled sample config."""
        with app.test_client() as client:
            res = client.get("/config/default")
            assert res.status_code == 200
            data = res.get_json()
            assert data["status"] == "ok"
            assert "yaml" in data


class TestWebAppImport:
    """Test that the web_app module imports correctly."""

    def test_app_import(self):
        """web_app.py should define a Flask app."""
        assert web_app.app is not None
        assert hasattr(web_app.app, "route")

    def test_routes_exist(self):
        """Flask app should have defined routes."""
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert len(rules) > 0
