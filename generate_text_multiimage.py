"""Generate Text node with dynamically growing image inputs.

This is a ComfyUI V3 custom node. It mirrors the built-in TextGenerate node's
sampling and output behavior while accepting multiple reference-image sockets.
"""

from __future__ import annotations

import inspect
import re

import torch
from comfy_api.latest import io
from comfy_extras.nodes_textgen import TextGenerate as _BuiltinTextGenerate


class TextGenerateMultiImage(_BuiltinTextGenerate):
    """The built-in Generate Text node with autogrowing image inputs."""

    @classmethod
    def define_schema(cls):
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01),
                ],
            ),
            io.DynamicCombo.Option(key="off", inputs=[]),
        ]

        image_template = io.Autogrow.TemplatePrefix(
            input=io.Image.Input("image"),
            prefix="image_",
            min=0,
            max=16,
        )

        return io.Schema(
            node_id="TextGenerateMultiImage",
            display_name="Generate Text (Multi-Image)",
            category="text",
            search_aliases=["LLM", "gemma", "vision", "multi image", "reference images"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Autogrow.Input(
                    "images",
                    template=image_template,
                    tooltip="Connect multiple reference images. A new image socket appears as you connect the last one.",
                ),
                io.Image.Input(
                    "video",
                    optional=True,
                    tooltip="Video frames as image batch. Assumed to be 24 FPS; subsampled to 1 FPS internally.",
                ),
                io.Audio.Input("audio", optional=True),
                io.Int.Input("max_length", default=512, min=1, max=32768),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Boolean.Input(
                    "thinking", optional=True, default=False,
                    tooltip="Operate in thinking mode if the model supports it.",
                ),
                io.Boolean.Input(
                    "use_default_template", optional=True, default=True,
                    tooltip="Use the built in system prompt/template if the model has one.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "mtp",
                    options=["auto", "off", "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"],
                    default="auto",
                    optional=True,
                    tooltip=(
                        "Speculative decoding with the checkpoint's multi-token-prediction head. "
                        "No effect without MTP weights."
                    ),
                ),
                io.String.Input(
                    "system_prompt", force_input=True, optional=True,
                    tooltip="Replaces the system prompt in the model's chat template. Ignored when the default template is not used.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.String.Output(display_name="thinking"),
            ],
        )

    @staticmethod
    def _image_slot_sort_key(item):
        name = str(item[0])
        match = re.search(r"(\d+)$", name)
        if match:
            return (0, int(match.group(1)))
        return (1, name)

    @classmethod
    def _collect_images(cls, images: io.Autogrow.Type) -> list[torch.Tensor]:
        """Return individual one-image tensors in socket order, including batches."""
        if not images:
            return []

        image_list: list[torch.Tensor] = []
        for _name, value in sorted(images.items(), key=cls._image_slot_sort_key):
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected an IMAGE tensor for {_name}, got {type(value).__name__}.")
            if value.ndim != 4:
                raise ValueError(
                    f"Input {_name} has an unexpected shape {tuple(value.shape)}; "
                    "ComfyUI IMAGE inputs should have shape [batch, height, width, channels]."
                )
            for index in range(value.shape[0]):
                image_list.append(value[index:index + 1])
        return image_list

    @staticmethod
    def _supports_image_list(clip) -> bool:
        """Check whether this loaded tokenizer explicitly supports `images=[...]`."""
        tokenizer = getattr(clip, "tokenizer", None)
        tokenize_with_weights = getattr(tokenizer, "tokenize_with_weights", None)
        if tokenize_with_weights is None:
            return False
        try:
            return "images" in inspect.signature(tokenize_with_weights).parameters
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _make_image_batch(image_list: list[torch.Tensor]) -> torch.Tensor | None:
        """Combine images for tokenizers that only support ComfyUI's `image` batch."""
        if not image_list:
            return None

        reference_shape = tuple(image_list[0].shape[1:])
        incompatible = [
            tuple(image.shape[1:])
            for image in image_list
            if tuple(image.shape[1:]) != reference_shape
        ]
        if incompatible:
            raise ValueError(
                "This model's tokenizer does not expose a separate-image list interface, and the connected "
                "images have different dimensions. Resize them to the same width and height before connecting "
                "them (for example, with ImageScale), or use a vision model whose tokenizer supports multiple "
                "separate images."
            )
        return torch.cat(image_list, dim=0)

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        max_length,
        sampling_mode,
        images,
        thinking=False,
        use_default_template=True,
        video=None,
        audio=None,
        mtp="auto",
        system_prompt="",
    ) -> io.NodeOutput:
        mtp = {"off": False, "auto": True}.get(mtp, mtp)
        image_list = cls._collect_images(images)

        tokenize_kwargs = {
            "skip_template": not use_default_template,
            "min_length": 1,
            "thinking": thinking,
            "video": video,
            "audio": audio,
            "system_prompt": system_prompt if use_default_template else "",
        }

        # Qwen3-VL-derived tokenizers (and other compatible tokenizers) accept a
        # list of individually sized images. This avoids forcing different-sized
        # references into one IMAGE batch and preserves their individual dimensions.
        if image_list and cls._supports_image_list(clip):
            tokenize_kwargs["images"] = image_list
        else:
            tokenize_kwargs["image"] = cls._make_image_batch(image_list)

        tokens = clip.tokenize(prompt, **tokenize_kwargs)

        # Keep sampling behavior in sync with the built-in Generate Text node.
        sampling_mode = sampling_mode or {}
        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
        presence_penalty = sampling_mode.get("presence_penalty", 0.0)

        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            mtp=mtp,
        )

        generated_text = clip.decode(generated_ids)
        reasoning, separator, text = generated_text.partition("</think>")

        # Match the built-in node's handling of models that return an unfinished
        # thinking block because generation stopped at max_length.
        if not separator and not reasoning.lstrip().startswith("<think>"):
            if isinstance(tokens, dict):
                tokens = next(iter(tokens.values()))
            prompt_ids = [
                token[0]
                for token in tokens[-1]
                if isinstance(token, tuple) and isinstance(token[0], int)
            ]
            if (
                not prompt.rstrip().endswith("<think>")
                and not clip.decode(prompt_ids[-4:]).rstrip().endswith("<think>")
            ):
                reasoning, text = "", generated_text

        return io.NodeOutput(text.strip(), reasoning.replace("<think>", "", 1).strip())
