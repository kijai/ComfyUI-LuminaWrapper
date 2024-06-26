# WORK IN PROGRESS

# Installation
- Clone this repo into `custom_nodes` folder.
- Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in ComfyUI_windows_portable -folder:

  `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-LuminaWrapper\requirements.txt`
  
## Note: Sampling is slow without `flash_attn` !

For Linux users this doesn't mean anything but `pip install flash_attn`. 

However doing same on Windows currently will most likely fail if you do not have a build environment setup, and even if you do it can take an hour to build.
Alternative for Windows can be pre-built wheels from here, has to match your python environment:
https://github.com/bdashore3/flash-attention/releases

If flash_attn is not installed, attention code will fallback to torch SDP attention, which is at least twice as slow and memory hungry.

## Text encoder setup

Lumina-next uses Google's Gemma-2b -LLM: https://huggingface.co/google/gemma-2b
To download it you need to consent to their terms. This means having Hugginface account and requesting access (it's instant once you do it).

Either download it yourself to `ComfyUI/models/LLM/gemma-2b` (don't need the gguf -file) or let the node autodownload it.

## Lumina models

The nodes support the Lumina-next text to image models:

https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT

https://huggingface.co/Alpha-VLLM/Lumina-Next-T2I

They are automatically downloaded to `ComfyUI/models/lumina`

# Examples
The workflows are including in the examples -folder
![image](https://github.com/kijai/ComfyUI-LuminaWrapper/assets/40791699/d1efae46-590a-441e-ad42-9590062b3837)

![lumina_composition_example](https://github.com/kijai/ComfyUI-LuminaWrapper/assets/40791699/99603330-903a-444f-a23f-3ac0f332e73e)

![lumina_i2i_example](https://github.com/kijai/ComfyUI-LuminaWrapper/assets/40791699/680c032e-b700-4ec4-9484-977710228043)


Original repo:

https://github.com/Alpha-VLLM/Lumina-T2X
