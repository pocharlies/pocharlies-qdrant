"""
Code Indexer for RAG Pipeline
Indexes repository code into Qdrant vector database
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File extensions to index
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java',
    '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php', '.swift',
    '.kt', '.scala', '.sh', '.bash', '.yaml', '.yml', '.json',
    '.toml', '.md', '.rst', '.txt', '.sql', '.html', '.css', '.scss',
    '.vue', '.svelte', '.astro'
}

# Files/directories to skip
SKIP_PATTERNS = {
    'node_modules', '.git', '__pycache__', '.venv', 'venv',
    'dist', 'build', '.cache', '.pytest_cache', 'target',
    '.idea', '.vscode', '.min.js', '.min.css', 'package-lock.json',
    'yarn.lock', 'Cargo.lock', 'poetry.lock', '.next', '.nuxt',
    'coverage', '.coverage', 'htmlcov', '.tox', '.eggs'
}


@dataclass
class CodeChunk:
    path: str
    repo: str
    language: str
    chunk_idx: int
    start_line: int
    end_line: int
    text: str
    symbols: List[str]


class CodeIndexer:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        collection_name: str = "code_index"
    ):
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.model = SentenceTransformer(embedding_model)
        self.collection_name = collection_name
        self.dim = self.model.get_sentence_embedding_dimension()

        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")

    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped"""
        path_str = str(path)
        for pattern in SKIP_PATTERNS:
            if pattern in path_str:
                return True
        return False

    def _get_language(self, path: Path) -> str:
        """Detect language from file extension"""
        ext_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.tsx': 'typescript', '.jsx': 'javascript', '.go': 'go',
            '.rs': 'rust', '.java': 'java', '.cpp': 'cpp', '.c': 'c',
            '.h': 'c', '.hpp': 'cpp', '.cs': 'csharp', '.rb': 'ruby',
            '.php': 'php', '.swift': 'swift', '.kt': 'kotlin',
            '.scala': 'scala', '.sh': 'bash', '.yaml': 'yaml',
            '.yml': 'yaml', '.json': 'json', '.md': 'markdown',
            '.sql': 'sql', '.html': 'html', '.css': 'css',
            '.vue': 'vue', '.svelte': 'svelte'
        }
        return ext_map.get(path.suffix.lower(), 'text')

    def _chunk_code(
        self,
        content: str,
        max_chars: int = 2000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Chunk code with line awareness.
        Tries to break at natural boundaries when possible.
        """
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_chars = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_len = len(line) + 1  # +1 for newline

            if current_chars + line_len > max_chars and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_line': start_line,
                    'end_line': i - 1
                })

                # Start new chunk with overlap
                overlap_lines = []
                overlap_chars = 0
                for prev_line in reversed(current_chunk):
                    if overlap_chars + len(prev_line) > overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_chars += len(prev_line) + 1

                current_chunk = overlap_lines
                current_chars = overlap_chars
                start_line = i - len(overlap_lines)

            current_chunk.append(line)
            current_chars += line_len

        # Don't forget last chunk
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'start_line': start_line,
                'end_line': len(lines)
            })

        return chunks

    def _extract_symbols(self, text: str, language: str) -> List[str]:
        """Extract function/class names from code (simple regex-based)"""
        import re
        symbols = []

        patterns = {
            'python': [
                r'def\s+(\w+)\s*\(',
                r'class\s+(\w+)\s*[:\(]',
                r'async\s+def\s+(\w+)\s*\('
            ],
            'javascript': [
                r'function\s+(\w+)\s*\(',
                r'class\s+(\w+)\s*[{\s]',
                r'const\s+(\w+)\s*=\s*(?:async\s*)?\(',
                r'(\w+)\s*:\s*(?:async\s*)?\('
            ],
            'typescript': [
                r'function\s+(\w+)\s*[<\(]',
                r'class\s+(\w+)\s*[<{\s]',
                r'interface\s+(\w+)\s*[<{]',
                r'type\s+(\w+)\s*[<=]'
            ],
            'go': [
                r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(',
                r'type\s+(\w+)\s+struct'
            ],
            'rust': [
                r'fn\s+(\w+)\s*[<\(]',
                r'struct\s+(\w+)\s*[<{]',
                r'impl\s+(?:<[^>]+>\s+)?(\w+)'
            ]
        }

        for pattern in patterns.get(language, []):
            matches = re.findall(pattern, text)
            symbols.extend(matches)

        return list(set(symbols))

    def _generate_id(self, repo: str, path: str, chunk_idx: int) -> int:
        """Generate deterministic ID for a chunk"""
        key = f"{repo}:{path}:{chunk_idx}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:16], 16)

    def index_repository(
        self,
        repo_path: str,
        repo_name: Optional[str] = None,
        batch_size: int = 100
    ) -> int:
        """
        Index an entire repository.
        Returns number of chunks indexed.
        """
        repo_path = Path(repo_path)
        repo_name = repo_name or repo_path.name

        logger.info(f"Indexing repository: {repo_name} at {repo_path}")

        points = []
        total_files = 0

        for file_path in repo_path.rglob('*'):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in CODE_EXTENSIONS:
                continue
            if self._should_skip(file_path):
                continue

            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if not content.strip():
                    continue

                rel_path = str(file_path.relative_to(repo_path))
                language = self._get_language(file_path)
                chunks = self._chunk_code(content)

                for idx, chunk_data in enumerate(chunks):
                    text = chunk_data['text']
                    symbols = self._extract_symbols(text, language)

                    # Create embedding
                    embedding = self.model.encode(
                        text,
                        normalize_embeddings=True
                    ).tolist()

                    point = PointStruct(
                        id=self._generate_id(repo_name, rel_path, idx),
                        vector=embedding,
                        payload={
                            'repo': repo_name,
                            'path': rel_path,
                            'language': language,
                            'chunk_idx': idx,
                            'start_line': chunk_data['start_line'],
                            'end_line': chunk_data['end_line'],
                            'symbols': symbols,
                            'text': text
                        }
                    )
                    points.append(point)

                total_files += 1

                # Batch upsert
                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    logger.info(f"Indexed {len(points)} chunks...")
                    points = []

            except Exception as e:
                logger.warning(f"Error indexing {file_path}: {e}")
                continue

        # Final batch
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        total_chunks = self.client.count(
            collection_name=self.collection_name
        ).count

        logger.info(
            f"Indexed {total_files} files, "
            f"{total_chunks} total chunks in collection"
        )

        return total_chunks

    def delete_repository(self, repo_name: str) -> bool:
        """Delete all chunks from a repository"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="repo",
                            match=MatchValue(value=repo_name)
                        )
                    ]
                )
            )
            logger.info(f"Deleted chunks for repository: {repo_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting repository {repo_name}: {e}")
            return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python indexer.py <repo_path> [repo_name]")
        sys.exit(1)

    repo_path = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else None

    indexer = CodeIndexer(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )

    indexer.index_repository(repo_path, repo_name)
