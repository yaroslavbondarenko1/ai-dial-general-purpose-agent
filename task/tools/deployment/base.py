import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.pop("prompt")

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version="2025-01-01-preview",
        )
        chunks = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=[
                {
                    "role": Role.USER.value,
                    "content": prompt,
                }
            ],
            stream=True,
            extra_body={
                "custom_fields": {
                    "configuration": arguments,
                }
            },
            **self.tool_parameters,
        )

        content = ""
        attachments = []
        async for chunk in chunks:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta and delta.content:
                tool_call_params.stage.append_content(delta.content)
                content += delta.content

            custom_content = delta.custom_content if delta else None
            if custom_content and custom_content.attachments:
                for attachment in custom_content.attachments:
                    attachments.append(attachment)
                    tool_call_params.stage.add_attachment(
                        type=attachment.type,
                        title=attachment.title,
                        data=attachment.data,
                        url=attachment.url,
                        reference_type=attachment.reference_type,
                        reference_url=attachment.reference_url,
                    )

        return Message(
            role=Role.TOOL,
            content=StrictStr(content) if content else None,
            custom_content=CustomContent(attachments=attachments),
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
            name=StrictStr(tool_call_params.tool_call.function.name),
        )
