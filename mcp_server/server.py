"""
Meeting Intelligence MCP Server

Exposes the FastAPI backend as MCP tools for Claude Desktop.
All 5 tools map to REST endpoints on the backend.

Setup in Claude Desktop config:
{
  "mcpServers": {
    "meeting-intelligence": {
      "command": "python",
      "args": ["C:/Meeting/mcp_server/server.py"],
      "env": { "BACKEND_URL": "http://localhost:8000" }
    }
  }
}
"""

import asyncio
import json
import os
import sys

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

server = Server("meeting-intelligence")


# ─── Tool Definitions ─────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="upload_meeting",
            description=(
                "Upload a meeting transcript for analysis. "
                "The agent will extract decisions, action items, conflicts, and unresolved questions. "
                "Returns a full analysis report."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Meeting title"},
                    "project": {"type": "string", "description": "Project name"},
                    "transcript": {"type": "string", "description": "Raw meeting transcript text"},
                },
                "required": ["title", "project", "transcript"],
            },
        ),
        Tool(
            name="search_past_decisions",
            description=(
                "Semantic search over all past meeting decisions. "
                "Returns relevant decisions matching the query, with cache hit/miss info."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you're looking for (natural language)"},
                    "project": {"type": "string", "description": "Optional: limit search to a specific project"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_action_items",
            description=(
                "Retrieve action items from all past meetings. "
                "Filter by owner (person's name), status (pending/in_progress/done), or project."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string", "description": "Filter by owner name (optional)"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done"],
                        "description": "Filter by status (optional)",
                    },
                    "project": {"type": "string", "description": "Filter by project name (optional)"},
                },
            },
        ),
        Tool(
            name="find_contradictions",
            description=(
                "Find contradictions between decisions made in different meetings for the same project. "
                "Uses semantic similarity + LLM verification to surface real conflicts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name to scan for contradictions"},
                },
                "required": ["project"],
            },
        ),
        Tool(
            name="summarize_project_history",
            description=(
                "Get a full summary of all decisions, open action items, and unresolved conflicts "
                "for a project. Includes an LLM-generated executive narrative."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project name"},
                },
                "required": ["project"],
            },
        ),
    ]


# ─── Tool Handlers ────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=120.0) as client:
        try:
            if name == "upload_meeting":
                resp = await client.post(
                    "/meetings/upload",
                    data={
                        "title": arguments["title"],
                        "project": arguments["project"],
                        "transcript": arguments["transcript"],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                analysis = data["analysis"]
                output = (
                    f"✅ **Meeting processed** (ID: {data['meeting_id']}) "
                    f"via {analysis.get('llm_provider_used', 'unknown')} "
                    f"in {analysis.get('processing_time_ms', '?')}ms\n\n"
                    f"**Decisions ({len(analysis['decisions'])}):**\n"
                    + "\n".join(f"- {d['text']}" for d in analysis["decisions"])
                    + f"\n\n**Action Items ({len(analysis['action_items'])}):**\n"
                    + "\n".join(f"- [{a.get('owner','?')}] {a['text']} (due: {a.get('deadline','TBD')})" for a in analysis["action_items"])
                    + f"\n\n**Conflicts ({len(analysis['conflicts'])}):**\n"
                    + "\n".join(f"- {c['issue']}" for c in analysis["conflicts"])
                    + f"\n\n**Unresolved Questions ({len(analysis['unresolved_questions'])}):**\n"
                    + "\n".join(f"- {q['question']}" for q in analysis["unresolved_questions"])
                )

            elif name == "search_past_decisions":
                params = {"q": arguments["query"]}
                if arguments.get("project"):
                    params["project"] = arguments["project"]
                resp = await client.get("/decisions/search", params=params)
                resp.raise_for_status()
                data = resp.json()
                cache_status = "🟢 CACHE HIT" if data["cache_hit"] else "🔴 CACHE MISS"
                output = (
                    f"{cache_status} ({data['latency_ms']}ms) — Query: \"{data['query']}\"\n\n"
                    + "\n\n".join(
                        f"**[Meeting {r['meeting_id']}] {r['meeting_title']}** ({r['project']})\n"
                        f"Score: {r['score']}\n{r['text']}"
                        for r in data["results"]
                    ) or "No results found."
                )

            elif name == "get_action_items":
                params = {k: v for k, v in arguments.items() if v is not None}
                resp = await client.get("/action-items", params=params)
                resp.raise_for_status()
                items = resp.json()
                if not items:
                    output = "No action items found matching the criteria."
                else:
                    output = f"**{len(items)} action item(s):**\n" + "\n".join(
                        f"- [{a.get('status')}] {a['text']} | Owner: {a.get('owner','?')} | Due: {a.get('deadline','TBD')}"
                        for a in items
                    )

            elif name == "find_contradictions":
                resp = await client.get("/contradictions", params={"project": arguments["project"]})
                resp.raise_for_status()
                items = resp.json()
                if not items:
                    output = f"No contradictions found in project '{arguments['project']}'. ✅"
                else:
                    output = f"**⚠️ {len(items)} contradiction(s) found:**\n\n" + "\n\n".join(
                        f"**Meeting {c['meeting_id_a']}**: {c['decision_a']}\n"
                        f"**Meeting {c['meeting_id_b']}**: {c['decision_b']}\n"
                        f"💬 {c['contradiction_explanation']}"
                        for c in items
                    )

            elif name == "summarize_project_history":
                resp = await client.get(f"/project/{arguments['project']}/history")
                resp.raise_for_status()
                data = resp.json()
                output = (
                    f"# Project: {data['project']} ({data['total_meetings']} meetings)\n\n"
                    f"## Summary\n{data['summary']}\n\n"
                    f"## All Decisions ({len(data['all_decisions'])})\n"
                    + "\n".join(f"- {d['text']}" for d in data["all_decisions"])
                    + f"\n\n## Open Action Items ({len(data['open_action_items'])})\n"
                    + "\n".join(f"- [{a.get('owner','?')}] {a['text']} (due: {a.get('deadline','TBD')})" for a in data["open_action_items"])
                    + f"\n\n## Unresolved Conflicts ({len(data['unresolved_conflicts'])})\n"
                    + "\n".join(f"- {c['issue']}" for c in data["unresolved_conflicts"])
                )
            else:
                output = f"Unknown tool: {name}"

        except httpx.HTTPStatusError as e:
            output = f"❌ Backend error {e.response.status_code}: {e.response.text}"
        except Exception as e:
            output = f"❌ Error: {str(e)}"

    return [TextContent(type="text", text=output)]


# ─── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
