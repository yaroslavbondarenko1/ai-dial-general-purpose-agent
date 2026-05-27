import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import StrictStr, AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        self.dial_endpoint = dial_endpoint
        self.mcp_client = mcp_client
        self._code_execute_tool: Optional[MCPToolModel] = None
        for mcp_tool_model in mcp_tool_models:
            if mcp_tool_model.name == tool_name:
                self._code_execute_tool = mcp_tool_model
                break

        if not self._code_execute_tool:
            raise ValueError(f"Cannot set up PythonCodeInterpreterTool: MCP tool '{tool_name}' was not found")

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        mcp_client = await MCPClient.create(mcp_url)
        mcp_tool_models = await mcp_client.get_tools()
        return cls(
            mcp_client=mcp_client,
            mcp_tool_models=mcp_tool_models,
            tool_name=tool_name,
            dial_endpoint=dial_endpoint,
        )

    @property
    def show_in_stage(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        code = arguments["code"]
        session_id = arguments.get("session_id")
        stage = tool_call_params.stage

        stage.append_content("## Request arguments: \n")
        stage.append_content(f"```python\n\r{code}\n\r```\n\r")
        if session_id and session_id != "0":
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")

        content = await self.mcp_client.call_tool(self.name, arguments)
        execution_result = _ExecutionResult.model_validate_json(content)

        if execution_result.files:
            dial_client = Dial(base_url=self.dial_endpoint, api_key=tool_call_params.api_key)
            files_home = dial_client.my_appdata_home()
            file_url_prefix = "files/"
            if files_home is None:
                files_home = dial_client.my_files_home()
                file_url_prefix = ""

            for file_reference in execution_result.files:
                file_name = file_reference.name
                mime_type = file_reference.mime_type
                resource = await self.mcp_client.get_resource(AnyUrl(file_reference.uri))

                if isinstance(resource, bytes):
                    file_content = resource
                elif mime_type.startswith("text/") or mime_type in ["application/json", "application/xml"]:
                    file_content = resource.encode("utf-8")
                else:
                    file_content = base64.b64decode(resource)

                file_url = f"{file_url_prefix}{(files_home / file_name).as_posix()}"
                dial_client.files.upload(file_url, file_content)

                attachment = Attachment(
                    type=StrictStr(mime_type),
                    title=StrictStr(file_name),
                    url=StrictStr(file_url),
                )
                stage.add_attachment(
                    type=attachment.type,
                    title=attachment.title,
                    url=attachment.url,
                )
                tool_call_params.choice.add_attachment(
                    type=attachment.type,
                    title=attachment.title,
                    url=attachment.url,
                )
                execution_result.output.append(f"Generated file uploaded to {file_url}")

        if execution_result.output:
            execution_result.output = [
                output if len(output) <= 1000 else f"{output[:1000]}..."
                for output in execution_result.output
            ]

        execution_result_json = execution_result.model_dump_json(indent=2)
        stage.append_content("## Response: \n")
        stage.append_content(f"```json\n\r{execution_result_json}\n\r```\n\r")
        return execution_result_json
