#!/usr/bin/env python3
"""Check teacher vision inference and, optionally, the existing MCTS client.

Run after starting scripts/vllm/serve_qwen38.sh. The HTTP check requires Pillow;
--legacy-client also requires the project's OpenAI client dependencies.
"""

import argparse
import base64
import io
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def request_json(base_url, api_key, path, timeout, payload=None):
    headers = {"Authorization": f"Bearer {api_key}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path, data=data, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {request.full_url}: {detail}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9011/v1")
    parser.add_argument("--api-key", default="qwen")
    parser.add_argument("--model", default="qwen_vllm")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--legacy-client", action="store_true")
    parser.add_argument(
        "--legacy-image", type=Path, default=REPO_ROOT / "demo/examples/man.jpg"
    )
    args = parser.parse_args()

    models = request_json(args.base_url, args.api_key, "/models", args.timeout)
    available = [item["id"] for item in models["data"]]
    if args.model not in available:
        raise RuntimeError(f"Expected model {args.model!r}; server advertises {available}")

    # No downloaded test data or external media is needed for vision validation.
    with io.BytesIO() as image_bytes:
        Image.new("RGB", (256, 256), (255, 0, 0)).save(image_bytes, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(
            image_bytes.getvalue()
        ).decode("ascii")
    started = time.monotonic()
    response = request_json(
        args.base_url,
        args.api_key,
        "/chat/completions",
        args.timeout,
        {
            "model": args.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "What color fills this image? Reply with one color word only."},
                ],
            }],
            "temperature": 0,
            "max_tokens": 64,
        },
    )
    answer = response["choices"][0]["message"].get("content") or ""
    if not re.search(r"\bred\b", answer, re.IGNORECASE):
        raise RuntimeError(f"Vision smoke check expected red, received {answer!r}")
    result = {
        "vision": "passed",
        "model": args.model,
        "answer": answer,
        "seconds": round(time.monotonic() - started, 2),
        "usage": response.get("usage"),
    }

    if args.legacy_client:
        if args.model != "qwen_vllm":
            raise RuntimeError("The existing MCTS client requires --model qwen_vllm")
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from openai import OpenAI
        from vlmsearch.models import qwen_vllm

        # Keep the wrapper itself unchanged while honoring the smoke-test URL.
        qwen_vllm.client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=0,
        )
        model = qwen_vllm.Qwen_VLLM(
            max_new_tokens=256,
            temperature=0,
            top_p=1,
            thought_token_begin="<think>",
            thought_token_end="</think>",
            final_token_begin="<answer>",
            final_token_end="</answer>",
            repetition_penalty=1.05,
        )
        with Image.open(args.legacy_image) as source_image:
            image = source_image.convert("RGB")
        legacy_answer = model.generate_single_thought(
            system_prompt=(
                "Answer the image question using exactly one <answer> ... </answer> "
                "block. Put a brief description inside the tags."
            ),
            previous_thoughts=[("What is visible in this image?", image)],
            force_final=True,
            no_sample=True,
        )
        if not re.search(r"<answer>\s*\S.*?</answer>", legacy_answer, re.DOTALL):
            raise RuntimeError(f"MCTS client did not return an answer block: {legacy_answer!r}")
        result.update(legacy_client="passed", legacy_answer=legacy_answer)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
