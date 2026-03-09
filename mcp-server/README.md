# Picqer WMS MCP Server

Complete [Picqer](https://picqer.com) warehouse management API exposed as an MCP server. 43 tools covering products, stock, orders, customers, picklists, purchase orders, receipts, returns, backorders, warehouses, locations, suppliers, webhooks, tasks, stats, and more.

## Prerequisites

You need a Picqer account with an API key. Find your subdomain and API key in **Picqer > Settings > API Keys**.

| Variable | Description | Example |
|----------|-------------|---------|
| `PICQER_SUBDOMAIN` | Your Picqer subdomain | `mycompany` |
| `PICQER_API_KEY` | Admin API key | `Abc123...` |

## Installation

### Option 1: pip install (recommended)

```bash
pip install git+https://github.com/pocharlies/pocharlies-qdrant.git#subdirectory=mcp-server
```

Then run:
```bash
PICQER_SUBDOMAIN=mycompany PICQER_API_KEY=your-key picqer-mcp
```

### Option 2: Clone and run

```bash
git clone https://github.com/pocharlies/pocharlies-qdrant.git
cd pocharlies-qdrant/mcp-server
pip install -r requirements.txt
PICQER_SUBDOMAIN=mycompany PICQER_API_KEY=your-key python picqer_server.py
```

### Option 3: Docker

```bash
docker build -f mcp-server/Dockerfile.picqer -t picqer-mcp ./mcp-server
docker run -e PICQER_SUBDOMAIN=mycompany -e PICQER_API_KEY=your-key -p 8001:8001 picqer-mcp
```

### Option 4: SSE mode (for remote/multi-agent setups)

```bash
PICQER_SUBDOMAIN=mycompany PICQER_API_KEY=your-key picqer-mcp sse
# or: python picqer_server.py sse
```

Listens on `http://0.0.0.0:8001/sse` by default. Override with `MCP_HOST` and `MCP_PORT`.

---

## Client Configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "picqer-wms": {
      "command": "picqer-mcp",
      "env": {
        "PICQER_SUBDOMAIN": "mycompany",
        "PICQER_API_KEY": "your-api-key"
      }
    }
  }
}
```

Or with Python directly:

```json
{
  "mcpServers": {
    "picqer-wms": {
      "command": "python",
      "args": ["/path/to/picqer_server.py"],
      "env": {
        "PICQER_SUBDOMAIN": "mycompany",
        "PICQER_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Claude Code (CLI)

Add to `.claude/settings.json` or project `.mcp.json`:

```json
{
  "mcpServers": {
    "picqer-wms": {
      "command": "picqer-mcp",
      "env": {
        "PICQER_SUBDOMAIN": "mycompany",
        "PICQER_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Cursor

Add to Cursor settings (Settings > MCP Servers):

```json
{
  "picqer-wms": {
    "command": "picqer-mcp",
    "env": {
      "PICQER_SUBDOMAIN": "mycompany",
      "PICQER_API_KEY": "your-api-key"
    }
  }
}
```

### Windsurf / Codeium

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "picqer-wms": {
      "command": "picqer-mcp",
      "env": {
        "PICQER_SUBDOMAIN": "mycompany",
        "PICQER_API_KEY": "your-api-key"
      }
    }
  }
}
```

### OpenAI Agents SDK / LangGraph / Any MCP Client (SSE)

Point your MCP client to the SSE endpoint:

```
http://localhost:8001/sse
```

Example with `mcp_servers.json`:

```json
{
  "servers": {
    "picqer-wms": {
      "type": "sse",
      "url": "http://localhost:8001/sse",
      "description": "Picqer WMS - warehouse management, stock, orders, picklists"
    }
  }
}
```

---

## Tools (43 total)

### Products
| Tool | Actions |
|------|---------|
| `picqer_products` | list, get, create, update, activate, deactivate |
| `picqer_product_stock` | get, get_warehouse, change, move, history |
| `picqer_product_images` | list, add, delete |
| `picqer_product_parts` | list, get, add, update, remove, produce |
| `picqer_product_locations` | list, assign, update, remove, warehouse_settings, update_warehouse |
| `picqer_product_comments` | list_comments, add_comment, price_history, expected |

### Orders
| Tool | Actions |
|------|---------|
| `picqer_orders` | list, get, create, update, delete |
| `picqer_order_actions` | process, pause, resume, reopen, change_to_concept, undo_cancellation, allocate, deallocate, prioritise, process_backorders, anonymize, product_status |
| `picqer_order_products` | add, update, remove |
| `picqer_order_fields` | set_field, remove_field, list_tags, add_tag, remove_tag, list_comments, add_comment |
| `picqer_webshop_orders` | list, get |

### Customers
| Tool | Actions |
|------|---------|
| `picqer_customers` | list, get, create, update, delete |
| `picqer_customer_addresses` | list_addresses, add_address, update_address, delete_address, list_tags, add_tag, remove_tag, list_comments, add_comment |

### Picklists
| Tool | Actions |
|------|---------|
| `picqer_picklists` | list, get, update, close, pick, pickall, assign, unassign, snooze, pause, resume, cancel |
| `picqer_picklist_products` | update_pick_locations, packaging_advice, set_packaging, pdf, packing_pdf |
| `picqer_picklist_shipments` | list, create, methods |
| `picqer_picklist_comments` | list, add |
| `picqer_picklist_batches` | list, get, create, add_picklist, remove_picklist, assign, pdf, list_comments, add_comment |

### Purchasing & Receiving
| Tool | Actions |
|------|---------|
| `picqer_purchase_orders` | list, get, create, update, mark_purchased, close, cancel, list_comments, add_comment |
| `picqer_purchase_order_products` | list, add, update, remove |
| `picqer_receipts` | list, get, create, complete, delete, expected_products, expected_product, receive, revert, list_comments, add_comment |

### Returns
| Tool | Actions |
|------|---------|
| `picqer_returns` | list, get, create, update, delete, receive, add_log, logs, statuses, reasons |
| `picqer_return_products` | list, add, update, remove (both returned and replacement) |
| `picqer_return_comments` | list, add |

### Warehouse & Locations
| Tool | Actions |
|------|---------|
| `picqer_backorders` | list, get, delete, process |
| `picqer_warehouses` | list, get, stock |
| `picqer_locations` | list, get, create, update, delete, products, set_intent |
| `picqer_location_types` | list, get, create, update, delete, set_default |

### Admin & Reference
| Tool | Actions |
|------|---------|
| `picqer_suppliers` | list, get, create, update, list_comments, add_comment |
| `picqer_users` | list, get, me |
| `picqer_tags` | list, get, create, update, delete |
| `picqer_resource_tags` | list, add, remove (on products/orders/customers) |
| `picqer_comments` | list, get, delete (global) |
| `picqer_webhooks` | list, get, create, delete, reactivate |
| `picqer_shipments` | list, get |
| `picqer_tasks` | list, get, create, update, delete, complete, uncomplete |
| `picqer_stats` | list, get |
| `picqer_reference_data` | product_fields, order_fields, customer_fields, pricelists, vat_groups, shipping_providers, templates, packing_stations |
| `picqer_packagings` | list, get, create, update |
| `picqer_fulfilment` | list, get, create, update, report |
| `picqer_picking_containers` | list, get, create, update, delete, link |

### Utilities
| Tool | Actions |
|------|---------|
| `picqer_test_connection` | Test API connection |
| `picqer_api` | Generic: any GET/POST/PUT/DELETE to any endpoint |

---

## Architecture

```
picqer_client.py   - PicqerConfig (frozen dataclass) + PicqerClient (async httpx)
picqer_server.py   - FastMCP server with 43 @mcp.tool() functions
```

The client uses dependency injection (accepts `PicqerConfig` via constructor), making it fully testable without real API calls. 56 unit tests included (100% client coverage).

## License

MIT
