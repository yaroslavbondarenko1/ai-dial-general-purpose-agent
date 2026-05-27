from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'MCPClient':
        """Async factory method to create and connect MCPClient"""
        client = cls(mcp_server_url)
        await client.connect()
        return client

    async def connect(self):
        """Connect to MCP server"""
        if self.session:
            return

        self._streams_context = streamablehttp_client(self.server_url)
        read_stream, write_stream, _ = await self._streams_context.__aenter__()

        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()

        initialize_result = await self.session.initialize()
        print(f"[MCPClient] Connected to {self.server_url}: {initialize_result}")


    async def get_tools(self) -> list[MCPToolModel]:
        """Get available tools from MCP server"""
        await self.connect()
        if not self.session:
            raise RuntimeError("MCP session is not initialized")

        result = await self.session.list_tools()
        return [
            MCPToolModel(
                name=tool.name,
                description=tool.description or tool.title or tool.name,
                parameters=tool.inputSchema,
            )
            for tool in result.tools
        ]

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        await self.connect()
        if not self.session:
            raise RuntimeError("MCP session is not initialized")

        result: CallToolResult = await self.session.call_tool(tool_name, tool_args)
        content_parts: list[str] = []
        for content in result.content:
            if isinstance(content, TextContent):
                content_parts.append(content.text)
            elif isinstance(content, ImageContent):
                content_parts.append(
                    f"[Image content: {content.mimeType}, base64 length {len(content.data)}]"
                )
            elif isinstance(content, EmbeddedResource):
                resource = content.resource
                if isinstance(resource, TextResourceContents):
                    content_parts.append(resource.text)
                elif isinstance(resource, BlobResourceContents):
                    content_parts.append(resource.blob)
            else:
                content_parts.append(str(content))

        return "\n\n".join(content_parts)

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        """Get specific resource content"""
        await self.connect()
        if not self.session:
            raise RuntimeError("MCP session is not initialized")

        result: ReadResourceResult = await self.session.read_resource(uri)
        if not result.contents:
            return ""

        resource = result.contents[0]
        if isinstance(resource, TextResourceContents):
            return resource.text
        if isinstance(resource, BlobResourceContents):
            return resource.blob

        raise ValueError(f"Unsupported resource content type: {type(resource)}")

    async def close(self):
        """Close connection to MCP server"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

        self.session = None
        self._session_context = None
        self._streams_context = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False
