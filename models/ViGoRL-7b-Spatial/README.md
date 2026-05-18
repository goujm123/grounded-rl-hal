---
library_name: transformers
pipeline_tag: image-text-to-text
base_model: 
- Qwen/Qwen2.5-VL-7B-Instruct
---

# ViGoRL: Visually Grounded Reinforcement Learning for Visual Reasoning

This model card describes the ViGoRL (**Vi**sually **G**r**o**unded **R**einforcement **L**earning) model, introduced in our paper ["Grounded Reinforcement Learning for Visual Reasoning"](https://arxiv.org/abs/2505.23678).

**Authors:** Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J. Tarr, Aviral Kumar, Katerina Fragkiadaki

---

## Model Overview

ViGoRL is a vision-language model fine-tuned using reinforcement learning (RL) to explicitly anchor textual reasoning steps to visual coordinates. Inspired by human visual cognition, ViGoRL employs multi-turn visual grounding, dynamically zooming into image regions to perform fine-grained visual reasoning and grounding.

This model was trained using supervised fine-tuning (SFT) on visually-grounded reasoning traces generated via Monte Carlo Tree Search (MCTS), followed by reinforcement learning with Group Relative Policy Optimization (GRPO).

---

## Model Details

* **Base Architecture:** Qwen2.5-Vision-Language (3B or 7B parameters)
* **Training Paradigm:**

  * Supervised Fine-Tuning on MCTS-generated reasoning traces
  * Group Relative Policy Optimization (GRPO)
  * Multi-turn visual grounding with dynamic zoom-in feedback (if "Multiturn" appears in name)

---

## Use Cases

This model excels in visual reasoning tasks that require precise visual grounding and region-level reasoning. Please see model name for specific domain.

* **Spatial Reasoning:** SAT-2, BLINK, RoboSpatial
* **Visual Search:** V\*Bench
* **Web Interaction and Grounding:** ScreenSpot (Pro and V2), VisualWebArena

---

## Usage

You can load this model easily using Hugging Face's Transformers library:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# # default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "", torch_dtype="auto", device_map="auto"
# ) # replace with any of the ViGoRL models

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "path/to/image.png",
            },
            {"type": "text", "text": "QUERY HERE"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text) # this will output a single tool call turn of the model if version is multiturn.
```

**Important**: This model requires a system prompt for proper usage. Please see the model's chat template for details.

---

## Datasets and Training Data

Training datasets and generated reasoning chains are publicly available:

* [Code](https://github.com/Gabesarch/grounded-rl)
* [ViGoRL Datasets on Hugging Face](https://huggingface.co/datasets/gsarch/vigorl_datasets)

---

## Citation

If you use ViGoRL in your research or applications, please cite our paper:

```bibtex
@article{sarch2025vigorl,
    title={Grounded Reinforcement Learning for Visual Reasoning},
    author={Sarch, Gabriel and Saha, Snigdha and Khandelwal, Naitik and Jain, Ayush and Tarr, Michael J and Kumar, Aviral and Fragkiadaki, Katerina},
    year={2025}
}
```

---

## Contact

For questions, feedback, or collaborations, please reach out to Gabriel Sarch or open an issue in our [GitHub repository](https://github.com/Gabesarch/grounded-rl).

---