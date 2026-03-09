import pytest
from picqer_client import PicqerClient, PicqerConfig

TEST_CONFIG = PicqerConfig(
    subdomain="testshop",
    api_key="test-api-key-123",
    timeout=5.0,
)


@pytest.fixture
def config():
    return TEST_CONFIG


@pytest.fixture
async def client(config):
    c = PicqerClient(config)
    yield c
    await c.close()
