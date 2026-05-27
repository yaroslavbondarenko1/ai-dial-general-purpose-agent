from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        message = await super()._execute(tool_call_params)
        if not isinstance(message, Message):
            return message

        attachments = message.custom_content.attachments if message.custom_content else []
        for attachment in attachments or []:
            if attachment.type in ["image/png", "image/jpeg"] and attachment.url:
                tool_call_params.choice.append_content(f"\n\r![image]({attachment.url})\n\r")
                tool_call_params.choice.add_attachment(
                    type=attachment.type,
                    title=attachment.title,
                    data=attachment.data,
                    url=attachment.url,
                    reference_type=attachment.reference_type,
                    reference_url=attachment.reference_url,
                )

        if not message.content:
            message.content = StrictStr("The image has been successfully generated according to request and shown to user!")

        return message

    @property
    def deployment_name(self) -> str:
        return "gpt-image-1.5-2025-12-16"

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return (
            "Generate an image from a text prompt using GPT Image. Use this when the user asks to create, generate, draw, or visualize "
            "a picture. Provide a detailed prompt that describes the subject, style, composition, lighting, and any "
            "important constraints. Do not use this tool for editing existing images."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated.",
                },
                "size": {
                    "type": "string",
                    "enum": ["auto", "1024x1024", "1536x1024", "1024x1536"],
                    "description": "Image size. Use auto unless the user asks for square, landscape, or portrait output.",
                },
                "quality": {
                    "type": "string",
                    "enum": ["auto", "low", "medium", "high"],
                    "description": "Rendering quality. Use auto unless the user explicitly asks for a faster, cheaper, or higher-quality image.",
                },
                "background": {
                    "type": "string",
                    "enum": ["auto", "transparent", "opaque"],
                    "description": "Background style. Use transparent only when the user asks for transparency, stickers, icons, or cutouts.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["png", "jpeg", "webp"],
                    "description": "Image output format. Use png by default, jpeg for smaller photographic output, and webp when requested.",
                },
                "output_compression": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Compression level for jpeg or webp output, from 0 to 100.",
                },
            },
            "required": ["prompt"],
        }
