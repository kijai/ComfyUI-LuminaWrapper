# WORK IN PROGRESS

## Currently requires `flash_attn` !

For Linux users this doesn't mean anything but `pip install flash_attn`. 

However doing same on Windows currently will most likely fail if you do not have a build environment setup, and even if you do it can take an hour to build.
Alternative for Windows can be pre-built wheels from here, has to match your python environment:
https://github.com/bdashore3/flash-attention/releases

## Text encoder setup

Lumina-next uses Google's Gemma-2b -LLM: https://huggingface.co/google/gemma-2b
To download it you need to consent to their terms. This means having Hugginface account and requesting access (it's instant once you do it).

Either download it yourself to `ComfyUI/models/LLM/gemma-2b` (don't need the gguf -file) or let the node autodownload it.

![image](https://github.com/kijai/ComfyUI-LuminaWrapper/assets/40791699/d1efae46-590a-441e-ad42-9590062b3837)

Original repo:
https://github.com/Alpha-VLLM/Lumina-T2X
