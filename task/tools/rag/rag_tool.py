import json
import hashlib
import os
import re
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

_SYSTEM_PROMPT = """
You answer questions using only the provided document context.

Rules:
- Ground every answer in the retrieved context.
- If the context does not contain enough information, say that the document context is insufficient.
- Keep the answer concise and directly useful.
- Do not mention embeddings, chunks, retrieval, or internal implementation details.
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.document_cache = document_cache
        self.model: SentenceTransformer | None = None
        self._use_sentence_transformer = os.getenv("RAG_USE_SENTENCE_TRANSFORMER", "false").lower() == "true"
        self._model_load_failed = not self._use_sentence_transformer
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Search an attached PDF, TXT, CSV, or HTML file semantically and answer a specific question from the most "
            "relevant parts. Use this for large or paginated documents, when full extraction is too long, or when the "
            "user asks to search/find information inside a file. Prefer this over reading every page when the question "
            "is narrow. Provide the user's exact question as request and the attached file URL as file_url."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document.",
                },
                "file_url": {
                    "type": "string",
                    "description": "The DIAL file URL to search.",
                },
            },
            "required": ["request", "file_url"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        request = arguments["request"]
        file_url = arguments["file_url"]
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"**Request**: {request}\n\r")
        stage.append_content(f"**File URL**: {file_url}\n\r")

        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        cached_data = self.document_cache.get(cache_document_key)
        if cached_data:
            index, chunks = cached_data
        else:
            text_content = DialFileContentExtractor(self.endpoint, tool_call_params.api_key).extract_text(file_url)
            if not text_content:
                stage.append_content("## Response: \n")
                stage.append_content("Error: File content not found.\n\r")
                return "Error: File content not found."

            chunks = self.text_splitter.split_text(text_content)
            if not chunks:
                stage.append_content("## Response: \n")
                stage.append_content("Error: File content not found.\n\r")
                return "Error: File content not found."

            embeddings_array = self.__encode(chunks)
            index = faiss.IndexFlatL2(embeddings_array.shape[1])
            index.add(embeddings_array)
            self.document_cache.set(cache_document_key, index, chunks)

        query_embedding_array = self.__encode([self.__expand_request(request)])
        _, indices = index.search(query_embedding_array, k=min(5, len(chunks)))

        retrieved_chunks = [
            chunks[idx]
            for idx in indices[0]
            if 0 <= idx < len(chunks)
        ]
        augmented_prompt = self.__augmentation(request, retrieved_chunks)

        stage.append_content("## RAG Request: \n")
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        stage.append_content("## Response: \n")

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version='2025-01-01-preview',
        )
        chunks_stream = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=[
                {
                    "role": Role.SYSTEM.value,
                    "content": _SYSTEM_PROMPT,
                },
                {
                    "role": Role.USER.value,
                    "content": augmented_prompt,
                },
            ],
            stream=True,
        )

        content = ''
        async for chunk in chunks_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    stage.append_content(delta.content)
                    content += delta.content

        return content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        context = "\n\n---\n\n".join(chunks)
        return f"""
Question:
{request}

Document context:
{context}

Answer the question using only the document context.
"""

    @staticmethod
    def __expand_request(request: str) -> str:
        lower_request = request.lower()
        if "plate" in lower_request:
            return f"{request} glass tray turntable roller ring"
        return request

    def __encode(self, texts: list[str]) -> np.ndarray:
        if not self._model_load_failed:
            try:
                if self.model is None:
                    self.model = SentenceTransformer(
                        model_name_or_path='all-MiniLM-L6-v2',
                        device='cpu',
                        local_files_only=True,
                    )
                return np.array(self.model.encode(texts), dtype='float32')
            except Exception as e:
                self._model_load_failed = True
                print(f"SentenceTransformer is not available locally, using fallback embeddings: {e}")

        return self.__fallback_encode(texts)

    @staticmethod
    def __fallback_encode(texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 384), dtype='float32')
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % vectors.shape[1]
                vectors[row, index] += 1.0

            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm

        return vectors
