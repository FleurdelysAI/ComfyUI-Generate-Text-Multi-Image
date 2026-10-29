# Generate Text (Multi-Image) for ComfyUI

A standalone ComfyUI V3 custom node based on the built-in **Generate Text** node. It adds dynamic reference-image sockets using `io.Autogrow`, while retaining the built-in text-generation sampling controls and outputs.

## Install

1. Copy the entire `ComfyUI-GenerateText-MultiImage` folder into your ComfyUI `custom_nodes` directory.
2. Restart ComfyUI completely.
3. Search for **Generate Text (Multi-Image)** in the node menu under **text**.

No JavaScript files or extra Python packages are required beyond a ComfyUI version that supports the V3 API, `io.Autogrow`, and the built-in `comfy_extras.nodes_textgen` module.

## How to use

Connect your `CLIP` from the relevant text/vision model loader, enter a prompt, then connect one or more `IMAGE` outputs to the dynamically growing image sockets. When you connect the last visible image socket, ComfyUI adds another socket automatically. Up to 16 sockets are allowed by default; edit `max=16` in `generate_text_multiimage.py` to change that limit.

Each socket can carry a normal ComfyUI `IMAGE` batch. The node processes sockets in numerical order and preserves the order of images inside each batch.

## Image-size compatibility

Some vision tokenizers explicitly support a list of separate images. For those tokenizers, this node passes each image separately, so the images can have different dimensions.

For tokenizers that only support the standard `image` batch input, all connected images must have the same height, width, and channel count. If they do not, the node raises a readable error instead of silently distorting the references. Use an `ImageScale`/resize node before those inputs to make their dimensions match.

The model/tokenizer itself must support multiple images in one prompt. This node adds image sockets and forwards the images; it cannot make a single-image-only model understand multiple visual references.

## Files

- `__init__.py` registers this V3 extension with ComfyUI.
- `generate_text_multiimage.py` contains the node implementation.
