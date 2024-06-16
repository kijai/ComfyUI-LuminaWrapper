# WORK IN PROGRESS

## Currently requires `flash_attn` !

For Linux users this doesn't mean anything but `pip install flash_attn`. 

However doing same on Windows currently will most likely fail if you do not have a build environment setup, and even if you do it can take an hour to build.
Alternative for Windows can be pre-built wheels from here, has to match your python environment:
https://github.com/bdashore3/flash-attention/releases

## Text encoder setup

Lumina-next uses Google's Gemma-2b -LLM: https://huggingface.co/google/gemma-2b
To download it you need to consent to their terms. This means having Hugginface account and requesting access (it's instant once you do it).

Then either download it yourself to `ComfyUI/models/LLM/gemma-2b` (don't need the gguf -file) or provide your access token to the node (or the hf_token.json file) and it will autodownload it, this would only be required on the first run. 
**Don't leave your token to the node or it might be saved in the result metadata!**


![image](https://github.com/kijai/ComfyUI-LuminaWrapper/assets/40791699/d20cb547-cb8f-43d1-96a5-570601d152c4)

Original repo:
https://github.com/Alpha-VLLM/Lumina-T2X
