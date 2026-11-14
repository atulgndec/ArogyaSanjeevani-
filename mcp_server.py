import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from arogya.lib import vector_db

app = FastAPI()
MODE = os.getenv("AROGYA_MCP_MODE", "retrieval")


@app.get("/healthz")
def health():
    return {"ok": True, "mode": MODE}


@app.post("/mcp")
async def mcp(request: Request):
    body = await request.json()
    method = body.get("method")
    rpc_id = body.get("id")

    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "arogya-" + MODE, "version": "1.0"}, "capabilities": {"tools": {}}}
    elif method == "tools/list":
        tools = [{"name": "search_policy", "description": "Search Arogya policy clauses", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}] if MODE == "retrieval" else [{"name": "list_project_files", "description": "List project files", "inputSchema": {"type": "object", "properties": {}}}]
        result = {"tools": tools}
    elif method == "tools/call":
        params = body.get("params", {})
        if MODE == "retrieval" and params.get("name") == "search_policy":
            docs = vector_db.hybrid_clauses(params.get("arguments", {}).get("query", ""), k=5)
            text = json.dumps([{d.metadata["clause"]: d.page_content} for d in docs])
        elif MODE == "filesystem":
            root = Path("/workspace")
            text = json.dumps([str(p.relative_to(root)) for p in sorted(root.rglob("*")) if p.is_file()][:200])
        else:
            text = json.dumps({"error": "Unknown tool"})
        result = {"content": [{"type": "text", "text": text}]}
    else:
        return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32601, "message": "Method not found"}}
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


@app.get("/sse")
async def sse():
    async def events():
        while True:
            yield "event: ready\ndata: {\"status\":\"ready\"}\n\n"
            await asyncio.sleep(15)

    return StreamingResponse(events(), media_type="text/event-stream")
