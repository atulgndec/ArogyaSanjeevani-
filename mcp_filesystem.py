"""FastMCP tools for bounded, read-only project file access."""
import os
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

mcp = FastMCP("Arogya Project Files")


def _resolve(relative_path: str) -> Path:
    root = Path(os.getenv("AROGYA_MCP_WORKSPACE", "/workspace")).resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise ToolError("Requested path is outside the configured project root.")
    return target


def _is_hidden(path: Path) -> bool:
    root = Path(os.getenv("AROGYA_MCP_WORKSPACE", "/workspace")).resolve()
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def list_project_files(
    relative_path: Annotated[str, Field(min_length=1, max_length=500)] = ".",
) -> list[str]:
    """List at most 500 files below the configured read-only project root."""
    directory = _resolve(relative_path)
    if not directory.is_dir() or _is_hidden(directory):
        raise ToolError("Requested path is not a directory.")
    return [
        path.relative_to(Path(os.getenv("AROGYA_MCP_WORKSPACE", "/workspace")).resolve()).as_posix()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not _is_hidden(path)
    ][:500]


def read_project_file(
    relative_path: Annotated[str, Field(min_length=1, max_length=500)],
) -> str:
    """Read a UTF-8 project file no larger than one megabyte."""
    file_path = _resolve(relative_path)
    if not file_path.is_file() or _is_hidden(file_path):
        raise ToolError("Requested path is not a file.")
    if file_path.stat().st_size > 1_000_000:
        raise ToolError("Requested file exceeds the one-megabyte read limit.")
    try:
        return file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ToolError(f"Unable to read requested text file: {type(error).__name__}.") from error


mcp.tool(list_project_files)
mcp.tool(read_project_file)
app = mcp.http_app(path="/mcp")
