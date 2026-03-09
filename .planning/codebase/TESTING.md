# Testing Patterns

**Analysis Date:** 2026-03-09

## Test Framework

**Runner:** None configured

**There are zero test files in this codebase.** No test framework, test runner, test configuration, or test dependencies exist. A thorough search for `*test*`, `*spec*`, `pytest`, `unittest`, `jest`, `vitest`, `mocha` found nothing outside of third-party `.venv` directories.

**No testing dependencies** appear in any `requirements.txt`:
- `rag-service/requirements.txt` - 21 dependencies, none test-related
- `agent-service/requirements.txt` - 15 dependencies, none test-related
- `mcp-server/requirements.txt` - 7 dependencies, none test-related

**No CI/CD pipeline** exists (no `.github/workflows/`, no `Jenkinsfile`, no `.gitlab-ci.yml`).

## Current Validation Approach

The codebase relies entirely on:
1. **Manual testing via curl/httpx** against running Docker containers
2. **Docker Compose smoke testing** (`docker compose up -d` then hit endpoints)
3. **Python syntax checking** via `py_compile` (seen in `.claude/settings.local.json`)

## Recommended Test Setup

Given the Python 3.11 stack with FastAPI, the recommended test framework is:

**pytest + pytest-asyncio + httpx** for unit and integration tests.

### Recommended Dependencies

Add to each service's `requirements.txt` (or a separate `requirements-dev.txt`):
```
pytest>=7.0
pytest-asyncio>=0.23.0
pytest-cov>=4.0
httpx>=0.26.0  # already present; used as test client
```

### Recommended Configuration

Create `pyproject.toml` or `pytest.ini` in each service directory:
```ini
# pytest.ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_functions = test_*
```

### Run Commands
```bash
pytest                    # Run all tests
pytest --tb=short -q      # Quick summary
pytest --cov=. --cov-report=html  # Coverage
pytest -x                 # Stop on first failure
pytest -k "test_search"   # Run specific tests
```

## Recommended Test File Organization

**Location:** Co-located `tests/` directory within each service

**Naming:** `test_{module}.py` matching the source module name

**Recommended structure:**
```
rag-service/
  tests/
    __init__.py
    conftest.py              # Shared fixtures
    test_shopify_client.py
    test_product_indexer.py
    test_web_indexer.py
    test_translator.py
    test_reranker.py
    test_app.py              # FastAPI endpoint integration tests
    test_webhook_handler.py
agent-service/
  tests/
    __init__.py
    conftest.py
    test_chat.py
    test_tasks.py
    test_health.py
    test_mcp_manager.py
    test_supervisor.py
mcp-server/
  tests/
    __init__.py
    test_server.py
```

## Recommended Test Structure

### Unit Test Pattern

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shopify_client import ShopifyClient


class TestShopifyClient:
    """Tests for ShopifyClient product text/metadata extraction."""

    def setup_method(self):
        self.client = ShopifyClient(
            shop_domain="test.myshopify.com",
            access_token="test-token",
        )

    def test_extract_product_text_basic(self):
        product = {
            "title": "Tokyo Marui Hi-Capa 5.1",
            "vendor": "Tokyo Marui",
            "product_type": "GBB Pistol",
            "body_html": "<p>Gas blowback pistol</p>",
            "tags": "gbb, pistol",
            "variants": [{"price": "159.99"}],
        }
        text = self.client.extract_product_text(product)
        assert "Tokyo Marui Hi-Capa 5.1" in text
        assert "Brand: Tokyo Marui" in text
        assert "Gas blowback pistol" in text

    def test_extract_metadata_includes_required_fields(self):
        product = {
            "id": 123,
            "title": "Test Gun",
            "handle": "test-gun",
            "vendor": "TestBrand",
            "product_type": "AEG",
            "tags": "aeg, rifle",
            "body_html": "",
            "variants": [{"price": "99.99", "sku": "TB-001", "inventory_quantity": 5}],
            "status": "active",
        }
        metadata = self.client.extract_metadata(product)
        assert metadata["shopify_id"] == 123
        assert metadata["price"] == 99.99
        assert metadata["sku"] == "TB-001"
        assert metadata["status"] == "active"

    def test_parse_airsoft_specs_fps(self):
        product = {
            "title": "Gun 350 FPS",
            "body_html": "",
            "tags": "",
            "product_type": "",
        }
        specs = self.client.parse_airsoft_specs(product)
        assert specs["fps"] == 350
```

### Async Test Pattern

```python
import pytest
from unittest.mock import AsyncMock, patch

from translator import TranslationPipeline, GlossaryStore, pack_batches


class TestTranslationPipeline:
    """Tests for the translation pipeline."""

    def test_pack_batches_single(self):
        texts = ["Hello world"]
        batches = pack_batches(texts, max_input_tokens=100)
        assert len(batches) == 1
        assert batches[0] == [0]

    def test_pack_batches_splits_large(self):
        texts = ["word " * 100 for _ in range(5)]  # ~500 words each
        batches = pack_batches(texts, max_input_tokens=200)
        assert len(batches) > 1

    @pytest.mark.asyncio
    async def test_translate_batch_calls_llm(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='["Hola mundo"]'))]
        mock_client.chat.completions.create.return_value = mock_response

        pipeline = TranslationPipeline(llm_client=mock_client)
        job = await pipeline.translate_batch(
            texts=["Hello world"],
            source_lang="en",
            target_lang="es",
        )
        assert job.status == "completed"
        assert len(job.results) == 1
```

### FastAPI Integration Test Pattern

```python
import pytest
from httpx import AsyncClient, ASGITransport

from main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
```

## Recommended Mocking Patterns

**What to Mock:**
- External API calls: Shopify REST/GraphQL, LLM completions, Qdrant client operations
- Redis client: use `fakeredis[aioredis]` or `AsyncMock`
- ML model inference: `SentenceTransformer.encode()`, `CrossEncoder.predict()`
- HTTP requests: `httpx.AsyncClient` responses

**What NOT to Mock:**
- Pydantic model validation
- Pure data transformation functions (e.g., `extract_product_text`, `parse_airsoft_specs`)
- Utility functions (e.g., `pack_batches`, `estimate_tokens`, `compute_hash`)

**Mocking pattern:**
```python
# Mock Qdrant client
@pytest.fixture
def mock_qdrant():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.create_collection.return_value = None
    client.upsert.return_value = None
    client.query_points.return_value = MagicMock(points=[])
    return client

# Mock SentenceTransformer
@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 768
    model.encode.return_value = [[0.1] * 768]
    return model

# Mock Redis
@pytest.fixture
async def mock_redis():
    redis = AsyncMock()
    redis.ping.return_value = True
    redis.hgetall.return_value = {}
    redis.hset.return_value = None
    redis.get.return_value = None
    redis.set.return_value = None
    return redis
```

## Fixtures and Factories

**Recommended test data location:** `tests/conftest.py` per service

**Shared fixtures:**
```python
# rag-service/tests/conftest.py
import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def sample_shopify_product():
    return {
        "id": 12345,
        "title": "Tokyo Marui Hi-Capa 5.1 Gold Match",
        "handle": "tokyo-marui-hi-capa-5-1-gold-match",
        "vendor": "Tokyo Marui",
        "product_type": "GBB Pistol",
        "body_html": "<p>Premium gas blowback pistol. 300 FPS with 0.20g BBs.</p>",
        "tags": "gbb, pistol, tokyo marui, hi-capa",
        "variants": [{
            "id": 1,
            "price": "189.99",
            "compare_at_price": "199.99",
            "sku": "TM-HC51-GM",
            "inventory_quantity": 8,
        }],
        "image": {"src": "https://cdn.shopify.com/s/test.jpg"},
        "status": "active",
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-06-20T15:30:00Z",
    }


@pytest.fixture
def sample_crawl_html():
    return """
    <html>
    <head><title>Test Airsoft Store</title></head>
    <body>
        <h1>Airsoft Rifles</h1>
        <p>Browse our selection of AEG and GBB rifles from top brands.</p>
        <div class="product">
            <h2>G&G CM16 Raider 2.0</h2>
            <p>Electric airsoft rifle. 350 FPS. Full metal gearbox.</p>
            <span class="price">EUR 179.99</span>
        </div>
    </body>
    </html>
    """
```

## Coverage

**Requirements:** None enforced (no tests exist)

**Recommended coverage targets:**
- Critical business logic (product extraction, translation, search): 80%+
- API endpoints: 70%+
- Utility modules: 90%+
- Infrastructure/glue code (lifespan, config): 50%+

**View coverage:**
```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
open htmlcov/index.html
```

## Test Types

**Unit Tests:**
- Pure function testing: `parse_airsoft_specs()`, `extract_product_text()`, `pack_batches()`, `estimate_tokens()`, `compute_hash()`
- Class method testing with mocked dependencies: `ProductIndexer.search()`, `GlossaryStore.get_relevant()`
- No real network calls, no real databases

**Integration Tests:**
- FastAPI `TestClient`/`AsyncClient` against endpoints with mocked service layer
- Redis integration with `fakeredis`
- Database integration with test PostgreSQL (agent-service)

**E2E Tests:**
- Not recommended for initial implementation
- Would require Docker Compose test environment with Qdrant + Redis + Postgres

## Priority Test Targets

Based on codebase analysis, these modules have the most testable logic and highest value:

1. **`rag-service/shopify_client.py`** - Pure data extraction, no external deps needed for testing `extract_*` and `parse_airsoft_specs`
2. **`rag-service/translator.py`** - `pack_batches()`, `_parse_translations()`, `normalize_specs()`, `normalize_brand()` are all pure functions
3. **`rag-service/product_indexer.py`** - `_generate_id()`, search logic with mocked Qdrant
4. **`rag-service/sync_state.py`** - `ContentHashStore.compute_hash()` is pure, rest needs Redis mock
5. **`agent-service/api/tasks.py`** - CRUD endpoints with mocked database session
6. **`agent-service/graphs/supervisor.py`** - Graph construction logic
7. **`mcp-server/server.py`** - Tool functions with mocked RAG service

---

*Testing analysis: 2026-03-09*
