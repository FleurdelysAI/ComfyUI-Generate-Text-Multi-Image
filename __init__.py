"""ComfyUI custom-node entry point."""

from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .generate_text_multiimage import TextGenerateMultiImage


class GenerateTextMultiImageExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [TextGenerateMultiImage]


async def comfy_entrypoint() -> GenerateTextMultiImageExtension:
    return GenerateTextMultiImageExtension()
