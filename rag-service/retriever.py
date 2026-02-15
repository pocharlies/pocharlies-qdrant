"""
Retriever and Tool Implementations for RAG
"""

import os
import subprocess
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from urllib.parse import urlparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion,
)
from sentence_transformers import SentenceTransformer
from sparse_encoder import encode_sparse_query


def _make_qdrant_client(url: str, api_key: Optional[str] = None) -> QdrantClient:
    """Create QdrantClient, handling HTTPS URLs (qdrant_client 1.7 needs explicit port/https)."""
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return QdrantClient(
            host=parsed.hostname,
            port=parsed.port or 443,
            https=True,
            api_key=api_key,
        )
    return QdrantClient(url=url, api_key=api_key)


@dataclass
class RetrievalResult:
    path: str
    repo: str
    chunk_idx: int
    start_line: int
    end_line: int
    text: str
    score: float
    symbols: List[str]


class CodeRetriever:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        collection_name: str = "code_index"
    ):
        self.client = _make_qdrant_client(qdrant_url, qdrant_api_key)
        self.model = SentenceTransformer(embedding_model)
        self.collection_name = collection_name

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        repo_filter: Optional[str] = None,
        language_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant code chunks using hybrid search (dense + sparse BM25 with RRF fusion).
        """
        # BGE models benefit from query prefix for retrieval
        query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"

        dense_vector = self.model.encode(
            query_with_prefix,
            normalize_embeddings=True
        ).tolist()

        sparse_vector = encode_sparse_query(query)

        # Build filter
        filter_conditions = []
        if repo_filter:
            filter_conditions.append(
                FieldCondition(key="repo", match=MatchValue(value=repo_filter))
            )
        if language_filter:
            filter_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=language_filter))
            )

        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", filter=search_filter, limit=top_k * 3),
                Prefetch(query=sparse_vector, using="sparse", filter=search_filter, limit=top_k * 3),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            RetrievalResult(
                path=hit.payload['path'],
                repo=hit.payload['repo'],
                chunk_idx=hit.payload['chunk_idx'],
                start_line=hit.payload['start_line'],
                end_line=hit.payload['end_line'],
                text=hit.payload['text'],
                score=hit.score,
                symbols=hit.payload.get('symbols', [])
            )
            for hit in results.points
        ]


class CodeTools:
    """
    Tool implementations for LLM function calling
    """

    def __init__(self, repos_base_path: str = "/repos"):
        self.repos_base = Path(repos_base_path)

    def search_repo(
        self,
        query: str,
        repo: Optional[str] = None,
        limit: int = 50
    ) -> str:
        """
        Search repository using ripgrep.
        Returns matching lines with file paths.
        """
        search_path = self.repos_base / repo if repo else self.repos_base

        if not search_path.exists():
            return f"Error: Path {search_path} does not exist"

        try:
            cmd = [
                "rg",
                "-n",  # Line numbers
                "--max-count", "5",  # Max matches per file
                "-i",  # Case insensitive
                query,
                str(search_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            lines = result.stdout.strip().split('\n')[:limit]
            return '\n'.join(lines) if lines[0] else "No matches found"

        except subprocess.TimeoutExpired:
            return "Error: Search timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 200,
        repo: Optional[str] = None
    ) -> str:
        """
        Read specific lines from a file.
        """
        if repo:
            full_path = self.repos_base / repo / path
        else:
            full_path = self.repos_base / path

        if not full_path.exists():
            return f"Error: File {path} not found"

        if not full_path.is_file():
            return f"Error: {path} is not a file"

        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)

            result_lines = []
            for i in range(start_idx, end_idx):
                result_lines.append(f"{i + 1}: {lines[i].rstrip()}")

            return '\n'.join(result_lines)

        except Exception as e:
            return f"Error reading file: {str(e)}"

    def list_files(
        self,
        path: str = ".",
        repo: Optional[str] = None,
        pattern: str = "*"
    ) -> str:
        """
        List files in a directory.
        """
        if repo:
            full_path = self.repos_base / repo / path
        else:
            full_path = self.repos_base / path

        if not full_path.exists():
            return f"Error: Path {path} not found"

        try:
            files = list(full_path.glob(pattern))
            result = []
            for f in sorted(files)[:100]:
                rel = f.relative_to(full_path)
                marker = "/" if f.is_dir() else ""
                result.append(f"{rel}{marker}")

            return '\n'.join(result) if result else "No files found"

        except Exception as e:
            return f"Error: {str(e)}"

    def run_command(
        self,
        command: str,
        repo: Optional[str] = None,
        timeout: int = 60
    ) -> str:
        """
        Run a shell command in a repository directory.
        SECURITY: Limited to safe commands only.
        """
        # Whitelist of allowed commands
        allowed_prefixes = [
            'git status', 'git log', 'git diff', 'git show', 'git branch',
            'ls', 'cat', 'head', 'tail', 'wc',
            'grep', 'find', 'tree',
            'python -m py_compile',  # Syntax check
            'npm run lint', 'npm test', 'npm run build',
            'pytest', 'cargo check', 'cargo build', 'go build', 'go test'
        ]

        is_allowed = any(
            command.strip().startswith(prefix)
            for prefix in allowed_prefixes
        )

        if not is_allowed:
            return f"Error: Command not allowed. Allowed prefixes: {', '.join(allowed_prefixes)}"

        cwd = self.repos_base / repo if repo else self.repos_base

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = f"exit_code={result.returncode}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout[:5000]}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr[:2000]}"

            return output

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {str(e)}"


# Tool definitions for OpenAI function calling format
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_repo",
            "description": "Search for text/code patterns in the repository using ripgrep. Use this to find where something is defined or used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search pattern (regex supported)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name to search in (optional)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a specific file, optionally specifying line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file relative to repo root"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (default: 1)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (default: 200)"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name (optional)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to repo root"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name (optional)"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (default: *)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the repository. Limited to safe commands like git, ls, grep, lint, test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository to run command in (optional)"
                    }
                },
                "required": ["command"]
            }
        }
    }
]
