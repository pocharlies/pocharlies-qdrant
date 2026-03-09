"""Comprehensive unit tests for PicqerClient."""

import dataclasses

import httpx
import pytest
import respx

from picqer_client import PicqerClient, PicqerConfig

BASE_URL = "https://testshop.picqer.com/api/v1"


# ---------------------------------------------------------------------------
# TestPicqerConfig
# ---------------------------------------------------------------------------
class TestPicqerConfig:
    def test_base_url(self, config):
        assert config.base_url == "https://testshop.picqer.com/api/v1"

    def test_config_is_immutable(self, config):
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            config.subdomain = "other"


# ---------------------------------------------------------------------------
# TestGet
# ---------------------------------------------------------------------------
class TestGet:
    @respx.mock
    async def test_get_single_resource(self, client):
        payload = {"idproduct": 1, "name": "Widget"}
        respx.get(f"{BASE_URL}/products/1").mock(
            return_value=httpx.Response(200, json=payload)
        )
        result = await client.get("/products/1")
        assert result == payload

    @respx.mock
    async def test_get_list_resource(self, client):
        payload = [{"idproduct": 1}, {"idproduct": 2}]
        respx.get(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(200, json=payload)
        )
        result = await client.get("/products")
        assert isinstance(result, list)
        assert len(result) == 2

    @respx.mock
    async def test_get_with_params(self, client):
        respx.get(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(200, json=[])
        )
        await client.get("/products", params={"search": "widget"})
        req = respx.calls.last.request
        assert b"search=widget" in req.url.raw_path

    @respx.mock
    async def test_get_404_returns_error_dict(self, client):
        respx.get(f"{BASE_URL}/products/999").mock(
            return_value=httpx.Response(404, json={"error": "Not found"})
        )
        result = await client.get("/products/999")
        assert result["error"] is True
        assert result["status_code"] == 404

    @respx.mock
    async def test_get_500_returns_error_dict(self, client):
        respx.get(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        result = await client.get("/products")
        assert result["error"] is True
        assert result["status_code"] == 500

    @respx.mock
    async def test_get_204_returns_ok(self, client):
        respx.get(f"{BASE_URL}/products/1").mock(
            return_value=httpx.Response(204)
        )
        result = await client.get("/products/1")
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# TestGetList
# ---------------------------------------------------------------------------
class TestGetList:
    @respx.mock
    async def test_single_page(self, client):
        items = [{"id": i} for i in range(50)]
        respx.get(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(200, json=items)
        )
        result = await client.get_list("/products")
        assert len(result) == 50

    @respx.mock
    async def test_pagination_two_pages(self, client):
        page1 = [{"id": i} for i in range(100)]
        page2 = [{"id": i} for i in range(100, 130)]
        route = respx.get(f"{BASE_URL}/products").mock(
            side_effect=[
                httpx.Response(200, json=page1),
                httpx.Response(200, json=page2),
            ]
        )
        result = await client.get_list("/products")
        assert len(result) == 130
        assert route.call_count == 2

    @respx.mock
    async def test_pagination_respects_max_results(self, client):
        page1 = [{"id": i} for i in range(100)]
        respx.get(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(200, json=page1)
        )
        result = await client.get_list("/products", max_results=50)
        assert len(result) == 50

    @respx.mock
    async def test_pagination_error_on_second_page(self, client):
        page1 = [{"id": i} for i in range(100)]
        respx.get(f"{BASE_URL}/products").mock(
            side_effect=[
                httpx.Response(200, json=page1),
                httpx.Response(500, text="Server Error"),
            ]
        )
        result = await client.get_list("/products")
        assert isinstance(result, dict)
        assert result["error"] is True

    @respx.mock
    async def test_empty_list(self, client):
        respx.get(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(200, json=[])
        )
        result = await client.get_list("/products")
        assert result == []


# ---------------------------------------------------------------------------
# TestPost
# ---------------------------------------------------------------------------
class TestPost:
    @respx.mock
    async def test_post_creates_resource(self, client):
        payload = {"idproduct": 42, "name": "New Widget"}
        respx.post(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(201, json=payload)
        )
        result = await client.post("/products", data={"name": "New Widget"})
        assert result == payload

    @respx.mock
    async def test_post_sends_json_body(self, client):
        body = {"name": "New Widget", "price": 9.99}
        respx.post(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(201, json={"idproduct": 1})
        )
        await client.post("/products", data=body)
        req = respx.calls.last.request
        import json
        sent = json.loads(req.content)
        assert sent == body

    @respx.mock
    async def test_post_action_no_body(self, client):
        respx.post(f"{BASE_URL}/orders/1/process").mock(
            return_value=httpx.Response(200, json={"status": "processed"})
        )
        result = await client.post("/orders/1/process")
        assert result == {"status": "processed"}

    @respx.mock
    async def test_post_422_validation_error(self, client):
        respx.post(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(
                422, json={"error_message": "Name is required"}
            )
        )
        result = await client.post("/products", data={})
        assert result["error"] is True
        assert result["status_code"] == 422


# ---------------------------------------------------------------------------
# TestPut
# ---------------------------------------------------------------------------
class TestPut:
    @respx.mock
    async def test_put_updates_resource(self, client):
        payload = {"idproduct": 1, "name": "Updated Widget"}
        respx.put(f"{BASE_URL}/products/1").mock(
            return_value=httpx.Response(200, json=payload)
        )
        result = await client.put("/products/1", data={"name": "Updated Widget"})
        assert result == payload

    @respx.mock
    async def test_put_empty_response(self, client):
        respx.put(f"{BASE_URL}/products/1").mock(
            return_value=httpx.Response(204)
        )
        result = await client.put("/products/1", data={"name": "Updated"})
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# TestDelete
# ---------------------------------------------------------------------------
class TestDelete:
    @respx.mock
    async def test_delete_resource(self, client):
        respx.delete(f"{BASE_URL}/products/1").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        result = await client.delete("/products/1")
        assert result == {"success": True}

    @respx.mock
    async def test_delete_204(self, client):
        respx.delete(f"{BASE_URL}/products/1").mock(
            return_value=httpx.Response(204)
        )
        result = await client.delete("/products/1")
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# TestAuth
# ---------------------------------------------------------------------------
class TestAuth:
    @respx.mock
    async def test_basic_auth_header_sent(self, client):
        respx.get(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(200, json=[])
        )
        await client.get("/products")
        req = respx.calls.last.request
        auth_header = req.headers.get("authorization", "")
        assert auth_header.startswith("Basic ")

    @respx.mock
    async def test_user_agent_header(self, client):
        respx.get(f"{BASE_URL}/products").mock(
            return_value=httpx.Response(200, json=[])
        )
        await client.get("/products")
        req = respx.calls.last.request
        ua = req.headers.get("user-agent", "")
        assert "SkirmshopMCP" in ua


# ---------------------------------------------------------------------------
# TestConnectionErrors
# ---------------------------------------------------------------------------
class TestConnectionErrors:
    @respx.mock
    async def test_network_error_returns_error_dict(self, client):
        respx.get(f"{BASE_URL}/products").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        result = await client.get("/products")
        assert result["error"] is True
        assert "Connection refused" in result["detail"]

    @respx.mock
    async def test_timeout_returns_error_dict(self, client):
        respx.get(f"{BASE_URL}/products").mock(
            side_effect=httpx.ReadTimeout("Read timed out")
        )
        result = await client.get("/products")
        assert result["error"] is True
        assert "timed out" in result["detail"].lower()
