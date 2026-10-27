from openai import OpenAI, OpenAIError
import os
import sys
from PIL import Image
from typing import List, Tuple, Optional
import json
import logging
import re
import time
import hashlib
from vlmsearch.image_geometry import ImagePayloadCache, NORMALIZED_COORDINATE_INSTRUCTION

# Configure logging to suppress debug messages for this module
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger(__name__).setLevel(logging.WARNING)
TIMEOUT = float(os.getenv("QWEN_VLLM_TIMEOUT", "360"))
BASE_URL = os.getenv("QWEN_VLLM_BASE_URL", f"http://localhost:{os.getenv('PORT', '9011')}/v1")
SERVED_MODEL_NAME = os.getenv("QWEN_VLLM_MODEL", "qwen_vllm")
EVIDENCE_BLOCK_RE = re.compile(r"<evidence>\s*(.*?)\s*</evidence>", re.IGNORECASE | re.DOTALL)
# Constrain only the evidence protocol. Native normalized coordinates must be
# complete numeric pairs in [0,1000]; range/frame validation still runs later.
NORMALIZED_NUMBER_PATTERN = r"(?:1000(?:\.0{1,3})?|[0-9]{1,3}(?:\.[0-9]{1,3})?)"
EVIDENCE_OUTPUT_REGEX = (
    r"<evidence>(?:[^()<>\r\n（）]|\n|\(" + NORMALIZED_NUMBER_PATTERN
    + r",[ ]{0,2}" + NORMALIZED_NUMBER_PATTERN + r"\))+</evidence>"
)

client = OpenAI(
    api_key=os.getenv("QWEN_VLLM_API_KEY", "qwen"),
    base_url=BASE_URL,
    timeout=TIMEOUT,
    max_retries=0,
)

class Qwen_VLLM():
    """
    Wraps a Qwen2_VL model for single-step generation 
    of 'thoughts' or final answers. Loaded one instance per Accelerate process.
    """

    def __init__(
        self,
        max_new_tokens: int = 64,
        temperature: float = 0.,
        top_p: float = None,
        examples: str = None,
        thought_token_begin: str = "<thought>",
        thought_token_end: str = "</thought>",
        final_token_begin: str = "<final>",
        final_token_end: str = "</final>",
        multicrop: bool = False,
        repetition_penalty: float = 0.0,
        **kwargs,
    ):
        """
        :param pretrained: Name (or path) of the Hugging Face model to load.
        :param device_map: Passed to from_pretrained for device placement. Usually "auto" with Accelerate.
        :param device: The device ("cuda" or "cpu").
        :param max_new_tokens: Default maximum tokens for a single generation step.
        :param temperature, top_p: Sampling parameters.
        :param use_cache: Whether to use the model's cache for faster inference.
        """
        super().__init__()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = kwargs.get("top_k", -1)
        self.thought_token_begin = thought_token_begin
        self.thought_token_end = thought_token_end
        self.final_token_begin = final_token_begin
        self.final_token_end = final_token_end
        self.multicrop = multicrop
        self.repetition_penalty = repetition_penalty
        self.max_evidence_steps = int(os.getenv("QWEN38_MAX_EVIDENCE_STEPS", "1"))
        self.evidence_max_new_tokens = int(os.getenv("QWEN38_EVIDENCE_MAX_NEW_TOKENS", "128"))
        if self.max_evidence_steps <= 0 or self.evidence_max_new_tokens <= 0:
            raise ValueError("Qwen3.8 evidence step and token budgets must be positive integers")
        self.generation_policy = {
            "version": 4, "protocol": "normalized_1000_evidence",
            "coordinate_syntax": "grammar_numeric_pairs_0_1000_v1",
            "evidence_regex_sha256": hashlib.sha256(EVIDENCE_OUTPUT_REGEX.encode()).hexdigest(),
            "max_evidence_steps": self.max_evidence_steps,
            "evidence_max_new_tokens": self.evidence_max_new_tokens,
            "budget_action": "force_final_answer",
            "point_binding": "adjacent_named_object_or_claim_specific_feature",
            "claim_scope": "minimal_relevant_visible_fact",
        }
        self.image_payload_cache = ImagePayloadCache(
            min_pixels=int(os.getenv("QWEN38_MIN_PIXELS", "65536")),
            max_pixels=int(os.getenv("QWEN38_MAX_PIXELS", "1048576")),
            max_entries=int(os.getenv("QWEN38_IMAGE_CACHE_ENTRIES", "16")),
            max_bytes=int(os.getenv("QWEN38_IMAGE_CACHE_BYTES", str(64 * 1024 * 1024))),
        )
        self.last_image_geometry = None
        self.metrics = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                        "image_encode_seconds": 0.0, "image_cache_hits": 0, "image_cache_misses": 0,
                        "uploaded_image_bytes": 0, "api_seconds": 0.0,
                        "evidence_budget_forced_finals": 0}
        if examples is not None:
            with open(examples, "r") as f:
                self.examples = json.load(f)
            for example in self.examples:
                example["image"] = Image.open(example["image"])
        else:
            self.examples = None

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _encode_image(self, img: Image.Image) -> str:
        started = time.perf_counter()
        hits, misses = self.image_payload_cache.hits, self.image_payload_cache.misses
        uri, geometry = self.image_payload_cache.encode(img)
        self.metrics["image_encode_seconds"] += time.perf_counter() - started
        self.metrics["image_cache_hits"] += self.image_payload_cache.hits - hits
        self.metrics["image_cache_misses"] += self.image_payload_cache.misses - misses
        self.metrics["uploaded_image_bytes"] += len(uri)
        self.last_transport_geometry = geometry
        return uri

    @staticmethod
    def _previous_evidence_steps(previous_thoughts):
        """Count actual generated blocks, never the question or few-shot prompt."""
        count = 0
        for text, maybe_image in previous_thoughts[1:]:
            if maybe_image is None:
                count += sum(bool(match.group(1).strip()) for match in EVIDENCE_BLOCK_RE.finditer(text))
        return count

    def generate_single_thought(
        self,
        system_prompt: str,
        previous_thoughts: List[Tuple[str, Optional[Image.Image]]],
        force_final: bool = False,
        no_sample: bool = False
    ) -> str:
        """
        Generate a *single* piece of text from the model. 
        Incorporates:
          - A system prompt
          - A chain of previous thoughts (with optional images)
          - Possibly forces a <final> token if `force_final` is True

        Returns:
          The newly generated 'thought' or final answer.
        """

        # Build messages for Qwen2-VL
        # Only a prompt with the explicit contract may declare its coordinates
        # normalized. Legacy clients must not acquire a frame by inference.
        normalized_protocol = NORMALIZED_COORDINATE_INSTRUCTION in system_prompt
        bounded_evidence_protocol = (normalized_protocol and self.thought_token_begin == "<evidence>"
                                     and self.thought_token_end == "</evidence>")
        previous_evidence_steps = self._previous_evidence_steps(previous_thoughts) if bounded_evidence_protocol else 0
        if (bounded_evidence_protocol and not force_final
                and previous_evidence_steps >= self.max_evidence_steps):
            force_final = True
            self.metrics["evidence_budget_forced_finals"] += 1
        self.last_image_geometry = None
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        ]

        if self.examples is not None:
            for example in self.examples:
                base64_string = self._encode_image(example["image"])
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"{base64_string}"}
                            },
                            {"role": "user", "content": [{"type": "text", "text": example["question"]}]}
                        ]
                    }
                    )
                if example["true_answer"] is not None:
                    thoughts = example["thoughts"]
                else:
                    thoughts = example["thoughts"][:-1]
                for thought in thoughts:
                    messages.append({"role": "user", "content": [{"type": "text", "text": f"{self.thought_token_begin} {thought} {self.thought_token_end}"}]})
                if example["true_answer"] is not None:
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": f"{self.final_token_begin} {example['true_answer']} {self.final_token_end}"}]})
                else:
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": f"{self.thought_token_begin} {example['thoughts'][-1]} {self.thought_token_end}"}]})
        
        # We'll add a single 'user' turn containing all previous_thoughts
        if previous_thoughts:
            # Accumulate the user content
            for thought_idx, (text_str, maybe_image) in enumerate(previous_thoughts):
                if maybe_image is not None:
                    # Strip the <image> placeholder from text if needed
                    text_str = text_str.replace("<image>", "")
                    text_str = text_str.strip()
                    # Convert the image to base64
                    base64_string = self._encode_image(maybe_image)
                    if thought_idx == 0:
                        if normalized_protocol:
                            self.last_image_geometry = self.last_transport_geometry
                            width, height = self.last_image_geometry["original_size"]
                            tw, th = self.last_image_geometry["transport_size"]
                            text_str += (f"\n\nImage geometry: original {width} x {height} pixels; "
                                         f"displayed {tw} x {th} pixels. Point coordinates must use "
                                         "the normalized 0-1000 grid specified above.")
                            if bounded_evidence_protocol:
                                if force_final:
                                    text_str += ("\nThe observation budget is closed. Return only the final "
                                                 "yes/no answer; do not generate another observation.")
                                elif previous_evidence_steps == 0:
                                    text_str += ("\nNext response: ONE atomic factual observation, at most 40 words. "
                                                 "Use named object or part (x, y), then its visible predicate. "
                                                 "Each pair must contain exactly two numbers and one comma; "
                                                 "put approximation words outside and use a separate pair for each point. "
                                                 "Never use prose or literal x/y placeholders inside a pair. "
                                                 "For a feature-dependent claim, point to that exact visible feature. "
                                                 "Omit irrelevant or uncertain fine qualifiers; a broader supported "
                                                 "posture or appearance is preferable to an invented subtype. "
                                                 "After this observation, answer immediately unless an essential "
                                                 "new fact is permitted by a remaining evidence budget.")
                                else:
                                    text_str += ("\nPrior evidence is available. Answer now unless an essential "
                                                 "new visible fact is needed; do not repeat an earlier observation.")
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {
                                        "url": f"{base64_string}"}
                                    },
                                    {"type": "text", "text": text_str}
                                ]
                            }
                        )
                    else:
                        if self.multicrop:
                            # assume that >1 thoughts are the cropped images
                            messages.append({"role": "assistant", "content": [{"type": "text", "text": text_str}]})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "<observation>\nHere is the crop of the image centered on the coordinate:"},
                                        {"type": "image_url", "image_url": {
                                            "url": f"{base64_string}"}
                                        },
                                        {"type": "text", "text": "\n</observation>"},
                                    ]
                                }
                            )
                        else:
                            raise ValueError("Multicrop is not enabled")
                else:
                    text_str = text_str.replace(f"{self.thought_token_begin}", "")
                    text_str = text_str.replace(f"{self.thought_token_end}", "")
                    text_str = text_str.strip()
                    label = "Previous evidence" if self.thought_token_begin == "<evidence>" else "Previous thought"
                    messages[-1]['content'][-1]['text'] += f"\n\n{label} {thought_idx}: {text_str}"
        
        # If user wants to force a <final> token, we append it
        add_answer_begin = False
        if force_final:
            if self.multicrop or self.thought_token_begin == "<evidence>":
                prefix = (self.final_token_begin if self.thought_token_begin == "<evidence>"
                          else f"<think> Based on all the information I've gathered, I'll now provide my final answer. </think>\n{self.final_token_begin}")
                messages.append({"role": "assistant", "content": [{"type": "text", "text": prefix}]})
                continue_final_message = True
                add_generation_prompt = False
                add_answer_begin = True
                no_sample = True
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": f"\nYour turn! Max thoughts reached. You should now provide the final answer in the format {self.final_token_begin} ... {self.final_token_end}"}]})
                continue_final_message = False
                add_generation_prompt = True
        else:
            continue_final_message = False
            add_generation_prompt = True
        
        if no_sample:
            temperature = 0.
        else:
            temperature = self.temperature

        try:
            request_started = time.perf_counter()
            self.metrics["api_calls"] += 1
            completion = client.chat.completions.create(
                model=SERVED_MODEL_NAME,
                messages=messages,
                max_tokens=(min(self.max_new_tokens, self.evidence_max_new_tokens)
                            if bounded_evidence_protocol and not force_final else self.max_new_tokens),
                temperature=temperature,
                top_p=self.top_p,
                # Keep one visible evidence step per MCTS expansion even when
                # the teacher would continue directly into its final answer.
                stop=([self.final_token_end] if force_final and self.thought_token_begin == "<evidence>"
                      else [self.thought_token_end, self.final_token_end]
                      if self.thought_token_begin == "<evidence>" else None),
                extra_body={
                    **({"structured_outputs": {"regex": EVIDENCE_OUTPUT_REGEX}}
                       if bounded_evidence_protocol and not force_final else {}),
                    "include_stop_str_in_output": self.thought_token_begin == "<evidence>",
                    "continue_final_message": continue_final_message,
                    "add_generation_prompt": add_generation_prompt,
                    "repetition_penalty": self.repetition_penalty,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": self.top_k,
                    }
            )
            self.metrics["api_seconds"] += time.perf_counter() - request_started
            usage = getattr(completion, "usage", None)
            if usage is not None:
                for metric in ("prompt_tokens", "completion_tokens"):
                    count = getattr(usage, metric, None)
                    if isinstance(count, int):
                        self.metrics[metric] += count
        except OpenAIError as exc:
            raise RuntimeError(
                f"vLLM request to {BASE_URL} failed or timed out after {TIMEOUT:g}s. "
                "Check the latest vllm_logs/vllm_logfile_*.txt for server-side errors."
            ) from exc

        answer = completion.choices[0].message.content or ""

        if add_answer_begin and self.final_token_begin not in answer:
            answer = f"{self.final_token_begin}{answer}"

        return answer

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def generate_until(self, requests):
        raise NotImplementedError("TODO: Implement generate_until")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("TODO: Implement multi-round generation")

        
if __name__ == "__main__":
    image_path = "data/mllm_hal/images/AMBER_152.jpg"
    question = "Is the floor yellow in this image?"

    system_prompt = """You are a careful visual verifier answering a yes/no question about an image.
    When ready to answer, output only:
    <answer> yes </answer>
    or
    <answer> no </answer>
    """
    image = Image.open(image_path)

    model = Qwen_VLLM(
        max_new_tokens=512,
        temperature=0.0,
        top_p=1.0,
        thought_token_begin="<think>",
        thought_token_end="</think>",
        final_token_begin="<answer>",
        final_token_end="</answer>",
        repetition_penalty=1.05,
    )

    print(f"Sending request to {BASE_URL} ...", file=sys.stderr, flush=True)

    response = model.generate_single_thought(
        system_prompt=system_prompt,
        previous_thoughts=[(question, image)],
        force_final=False,
        no_sample=True,
    )

    if response.strip():
        print(response, flush=True)
    else:
        print("[empty response from model]", file=sys.stderr, flush=True)
