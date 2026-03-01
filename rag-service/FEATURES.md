# RAG Service — Features Added (2026-03-01)

## Shopify GraphQL + Bulk Operations

- **GraphQL Admin API** (v2025-01) — replaces REST-only pagination with efficient GraphQL queries
- **Bulk Operations** — async JSONL export for products, collections, and pages (handles 3,799+ products without pagination)
- **Single-item GraphQL fetch** — `fetch_product(gid)`, `fetch_collection(gid)` for real-time webhook updates
- **Incremental fetch** — `fetch_products_updated_since(since)`, `fetch_collections_updated_since(since)` for delta syncs
- **Flatten utilities** — convert nested GraphQL responses (edges/nodes) to flat dicts for indexing

**File:** `shopify_graphql.py` (new)
**Modified:** `shopify_client.py` (API version bump, graphql property, collection/page extraction methods)

---

## Sync State Tracking (Redis-backed)

- **SyncStateStore** — tracks sync operations with status, cursor, items processed, errors, and timestamps
  - `create_sync(sync_type)` → unique sync ID
  - `update_sync()`, `complete_sync()` — lifecycle management
  - `get_last_cursor()` — ISO timestamp for incremental syncs
  - `get_sync_history(limit)` — audit trail
  - Redis keys: `catalog:sync:{id}`, `catalog:sync:history`

**File:** `sync_state.py` (new)

---

## Content Hash Dedup (Redis-backed)

- **ContentHashStore** — SHA-256 content hashing to skip re-embedding unchanged items
  - `has_changed(item_key, content_hash)` → bool
  - `set_hash(item_key, content_hash)` — update after successful upsert
  - Redis keys: `catalog:hash:{item_key}`
- Integrated into `product_indexer.py` — `_process_product_batch()` checks hash before embedding

**File:** `sync_state.py` (ContentHashStore class)
**Modified:** `product_indexer.py` (content hash integration)

---

## Collections & Pages Indexing

- **New Qdrant collections:** `product_collections`, `product_pages` (same hybrid vector config: dense BGE + sparse BM25)
- **CatalogIndexer** class:
  - `index_collections(collections_data)` — embed and upsert Shopify collections
  - `index_pages(pages_data)` — embed and upsert Shopify pages
  - `search_collections(query, top_k)` — semantic collection search with hybrid scoring
  - `search_pages(query, top_k)` — semantic page search
  - `get_product_by_id_or_handle()` — exact product lookup by Shopify ID or handle
  - `get_collection_products(id_or_handle)` — list products in a collection
  - `get_inventory(id_or_handle)` — stock levels for a product
  - `get_stats()` — point counts for all catalog collections

**File:** `catalog_indexer.py` (new)

---

## Shopify Webhook Handler

- **HMAC-SHA256 verification** — validates webhook authenticity using `SHOPIFY_WEBHOOK_SECRET`
- **Topic routing:**
  - `products/create`, `products/update` → fetch full product via GraphQL, re-embed, upsert to Qdrant
  - `products/delete` → remove from Qdrant
  - `collections/create`, `collections/update` → fetch full collection, re-embed, upsert
  - `collections/delete` → remove from Qdrant
  - `inventory_levels/update` → update inventory payload in Qdrant
- **Async processing** — responds 200 immediately, processes in background

**File:** `webhook_handler.py` (new)

---

## New API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/collections/search` | Semantic search across collections |
| GET | `/collections/{id_or_handle}/products` | List products in a collection |
| GET | `/products/{id_or_handle}` | Full product detail by ID or handle |
| GET | `/products/{id_or_handle}/inventory` | Stock levels for a product |
| POST | `/catalog/search` | Unified search across products, collections, and pages |
| GET | `/products/sync/history` | View sync state history |
| POST | `/catalog/full-sync` | Trigger bulk GraphQL sync (products → collections → pages) |
| POST | `/webhooks/shopify` | HMAC-verified Shopify webhook receiver |

**Modified:** `app.py`

---

## New MCP Tools

| Tool | Proxies To | Description |
|------|-----------|-------------|
| `search_collections` | POST `/collections/search` | Semantic search for Shopify collections |
| `get_collection_products` | GET `/collections/{id}/products` | List products in a collection |
| `get_product` | GET `/products/{id_or_handle}` | Get full product detail |
| `get_inventory` | GET `/products/{id}/inventory` | Get stock levels |
| `search_catalog` | POST `/catalog/search` | Unified search (products + collections + pages) |

**Modified:** `mcp-server/server.py`

---

## DGX Proxy Integration

The DGX MCP server (`mcpservers-pocharlies`) now proxies all catalog queries to `https://rag.e-dani.com` instead of running its own Shopify sync. The DGX's local Shopify infrastructure (GraphQL client, bulk operations, webhook server, sync jobs) has been removed and replaced with an HTTP proxy (`CatalogProxyHandler`).

**Env var:** `SAUVAGE_RAG_URL=https://rag.e-dani.com`
