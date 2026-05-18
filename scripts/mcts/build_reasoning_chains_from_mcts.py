import argparse
import json
from typing import List, Dict, Any
import os
import random
from PIL import Image
from io import BytesIO
import base64
import wandb
import ast
from PIL import ImageDraw
import re
from pathlib import Path


random.seed(42)


ANSWER_TAG_PATTERN = re.compile(r"(<answer>\s*)(.*?)(\s*</answer>)", re.IGNORECASE | re.DOTALL)


def normalize_amber_question_prompt(question: str) -> str:
    return question.replace("Answer with the text of the option.", "Answer with the option letter only.")


def answer_to_amber_option(answer: str) -> str | None:
    normalized = answer.strip().lower()
    normalized = re.sub(r"^[\s\"'{}\[\]()]+|[\s\"'{}\[\]().!,;:]+$", "", normalized)

    choice_match = re.match(r"^([ab])\s*[\.)]?\s*(.*)$", normalized)
    if choice_match:
        choice, rest = choice_match.groups()
        rest = rest.strip()
        if not rest or rest in {"yes", "no"}:
            return choice.upper()

    if normalized == "yes":
        return "A"
    if normalized == "no":
        return "B"

    yes_no_match = re.search(r"\b(yes|no)\b", normalized)
    if yes_no_match:
        return "A" if yes_no_match.group(1) == "yes" else "B"

    return None


def convert_amber_answer_tags(chain_text: str) -> str:
    def replace_answer(match: re.Match) -> str:
        option = answer_to_amber_option(match.group(2))
        if option is None:
            return match.group(0)
        return f"{match.group(1)}{option}{match.group(3)}"

    return ANSWER_TAG_PATTERN.sub(replace_answer, chain_text)

# ---------------------------------------------------------------------------
# 1) Helpers to build reasoning chains from your MCTS data
# ---------------------------------------------------------------------------
def process_text_chain(chain: List[str]) -> tuple[str, str]:
    """
    1. Removes first line of chain if it contains the word "<image>"
    2. Removes <think>, </think>, <answer>, </answer>
    3. Joins the chain together
    4. Returns (joined_chain, final_answer)
    """

    if chain and (chain[0].startswith("<image>") or chain[0].endswith("<image>")):
        chain = chain[1:]

    final_answer = chain[-1]
    final_answer = final_answer.replace("<answer>", "").replace("</answer>", "").strip()
    chain = chain[:-1]

    # Remove any <think>, </think>, etc. from the lines
    cleaned = []
    for line in chain:
        line = line.replace("<think>", "").replace("</think>", "")
        line = line.replace("<answer>", "").replace("</answer>", "")
        cleaned.append(line.strip())

    joined_chain = " ".join(cleaned)
    return joined_chain, final_answer


def build_reasoning_chains_from_rollouts(
    node_data: Dict[str, Any],
    backtrack_message: str = "Wait, this seems off. Let's try something else.",
    thought_start_tag: str = "<think>",
    thought_end_tag: str = "</think>",
    answer_start_tag: str = "<answer>",
    answer_end_tag: str = "</answer>",
) -> List[str]:
    """
    Return all possible reasoning chains from this node (recursively) as strings.
    Each chain includes wrong attempts (if any) plus a backtrack message, then a correct attempt.
    """
    rollouts = node_data.get("rollouts", [])
    correct_rollouts = []
    wrong_rollouts = []
    for r in rollouts:
        if r["reward"] >= 1.0:
            correct_rollouts.append(r)
        else:
            wrong_rollouts.append(r)

    child_nodes = node_data.get("children", [])
    is_terminal = node_data.get("is_terminal", False)

    all_chains = []

    # 1) Build chains from ephemeral rollouts at this node
    #    (Wrong -> backtrack -> Correct) and also purely Correct
    for wrong_r in wrong_rollouts:
        wrong_chain, _ = process_text_chain(wrong_r["ephemeral_texts"])
        # Insert a backtrack line after the wrong chain
        wrong_chain += f"\n{backtrack_message}"
        if correct_rollouts:
            for correct_r in correct_rollouts:
                correct_chain, correct_ans = process_text_chain(correct_r["ephemeral_texts"])
                combined_chain = wrong_chain + "\n" + correct_chain
                # Format it with <think> ... </think> plus <answer> ... </answer>:
                combined_chain = (
                    f"{thought_start_tag}\n{combined_chain}\n{thought_end_tag}\n"
                    f"{answer_start_tag} {correct_ans} {answer_end_tag}"
                )
                all_chains.append(combined_chain)

    for correct_r in correct_rollouts:
        chain_text, final_ans = process_text_chain(correct_r["ephemeral_texts"])
        chain_text = (
            f"{thought_start_tag}\n{chain_text}\n{thought_end_tag}\n"
            f"{answer_start_tag} {final_ans} {answer_end_tag}"
        )
        all_chains.append(chain_text)

    # 2) Recurse into children if not terminal
    if not is_terminal:
        for child in child_nodes:
            child_chains = build_reasoning_chains_from_rollouts(child, backtrack_message)
            all_chains.extend(child_chains)

    return all_chains


def build_all_reasoning_for_sample(sample_json: Dict[str, Any]) -> tuple[List[str], float]:
    """
    Build all possible reasoning chains from the top-level 'tree' in sample_json.
    Returns (list_of_chains, root_node_value).
    """
    root_node = sample_json["tree"]
    root_node_value = root_node.get("value", 0.0)
    chains = build_reasoning_chains_from_rollouts(root_node)
    # Deduplicate if needed:
    unique_chains = list(set(chains))
    return unique_chains, root_node_value

# ---------------------------------------------------------------------------
# 2) Helper to extract intermediate points from <think> text
# ---------------------------------------------------------------------------
def extract_think_points(chain_text: str):
    """
    Finds all <think>...</think> sections in chain_text, then extracts
    every coordinate (x, y) from those sections in order.
    Returns a list of (x, y, index).
    """
    # Find all <think> ... </think> blocks (could be multiple if there's a backtrack)
    think_blocks = re.findall(r"<think>(.*?)</think>", chain_text, flags=re.DOTALL)
    all_points = []
    point_idx = 1

    for block in think_blocks:
        # Find coords of form (123, 456) or ( 123 , 456 )
        coords = re.findall(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", block)
        for (x_str, y_str) in coords:
            try:
                x = float(x_str)
                y = float(y_str)
                all_points.append((x, y, point_idx))
                point_idx += 1
            except:
                pass  # if there's a parse error, skip

    return all_points


def parse_args():
    parser = argparse.ArgumentParser(description="Linearize MCTS rollout trees into SFT reasoning-chain data.")
    parser.add_argument(
        "--input",
        default="data/mcts/MCTS_AMBER_72b_20260424_103335",
        help="Rollout JSONL file or directory containing rollouts_*.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files. Defaults to <input-dir>/reasoning_chains.",
    )
    parser.add_argument(
        "--prompt-type",
        choices=["amber", "web_grounding", "spatial", "web_action", "vstar"],
        default="amber",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.05,
        help="Validation split. Use <1 for a fraction, or >=1 for an absolute sample count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-chains-per-sample", type=int, default=10000)
    parser.add_argument("--train-output-name", default="reasoning_chains_train.json")
    parser.add_argument("--val-output-name", default="reasoning_chains_val.json")
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--no-draw-points", action="store_true")
    return parser.parse_args()


def collect_data_files(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_dir():
        rollout_files = sorted(path.glob("rollouts_*.jsonl"))
        if rollout_files:
            return rollout_files
        return sorted(p for p in path.glob("*.jsonl") if not p.name.startswith("errors_"))
    return [path]


def resolve_output_dir(input_path: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    path = Path(input_path)
    base_dir = path if path.is_dir() else path.parent
    return base_dir / "reasoning_chains"

# ---------------------------------------------------------------------------
# 2) Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    rng = random.Random(args.seed)
    data_str = args.input
    log_to_wandb = args.log_to_wandb
    prompt_type = args.prompt_type

    system_prompt_web_grounding = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step, verifying each step and grounding every step to a specific point in the image.\n\nAll reasoning processes must be enclosed within a single set of '<think>' tags, with each reasoning step explicitly referencing a coordinate:\n\n<think>\n[Reasoning text with grounded points inline] (x1, y1). [Further reasoning] (x2, y2), [Final refinement] (x3, y3).\n</think>\n\nThe final answer should be enclosed in '<answer>' tags in the format:\n<answer> (xf, yf) </answer>\n\nYour task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.\n- Aim to point to the center or a representative point within the described area/element/object as accurately as possible.\n- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.\n- The final output should be the single most precise coordinate for the requested element.\n- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."

    system_prompt_spatial="A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step by checking and verifying possible solutions and image regions, while grounding reasoning steps to specific objects and their relationships in the image using (x,y) coordinates. There may be one image or two images concatenated together, in which case the Assistant must compare the spatial relationships between the two images.\n\nAll reasoning processes must be enclosed within a single set of '<think>' tags, and reasoning steps must include specific reference coordinates:\n\nFor example, <think>\n{Reasoning text}. {Further reasoning text} {more reasoning} \n</think>\n\nThe final answer should be enclosed in '<answer>' tags in the format:\n<answer> {text of selected answer choice} </answer>\n\nThe Assistant must help the user identify the correct answer choice from the options provided.\n-Your answer should be the **exact text** of the selected answer option, without additional explanations or reasoning or the option text. For example, if the answer is A. right , your response should just be <answer>right</answer> (not <answer>A. right</answer>).\n-If the correct answer is unclear, select the most relevant option based on the spatial relationships and dynamics within the image.\n- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."

    system_prompt_web_action="""You are a helpful Assistant tasked with navigating a web browser. These tasks will be accomplished through the use of specific actions you can issue. Your task is to choose the action that makes the most progress towards an objective. You should systematically reason through the problem step by step by checking and verifying possible actions and webpage regions, while grounding reasoning steps to specific (x, y) points in the image:\nEach reasoning step must be enclosed within '<think>' tags and reference exactly one specific coordinate (x, y):\n<think>\n[Reasoning text with grounded points inline] (x_1, y_1). [Further reasoning] (x_2, y_2), ..., [Final reasoning] (x_n, y_n).\n</think>\nWhen ready to provide the final answer, enclose it within '<answer>' tags:\n<answer> {action} </answer>\n- Each reasoning step must explicitly describe and evaluate the region’s relevance to the objective and proposing an action.\n- Never repeat coordinates from previous steps.\n- Look at diverse webpage regions to figure out which action should be taken.\n- Verify your selection by examining multiple possible solutions.\n\n**Inputs**\nHere's the information you'll have:\n1. OBJECTIVE: This is the task you are trying to complete.\n2. The web page screenshot: This is a screenshot of the current webpage you are on, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.\n3. PREVIOUS ACTIONS: This is the actions that you have performed prior to getting to the current page, but instead of the button id, the button text of the actions taken on the previously navigated pages are provided.\n\n**Action Space**\nYou can take the following actions:\n1. ```click [id]```: This action clicks on an element with a specific id on the webpage.\n2. ```type [id] [content]```: Use this to type the content into the field with id. By default, typing the content simulates pressing the "Enter" key afterward to submit the text.\n3. ```scroll [down]```: Scroll the page down.\n4. ```go_back```: Navigate to the previously viewed page.\n5. ```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If no answer is required, output empty brackets.\n\n**Guidelines**\nTo be successful, it is very important to follow the following rules:\n2. Generate the final action in the correct format. For example, '<answer> click [1234] </answer>'.\n3. Issue the stop action (i.e. stop [answer]) when you think you have achieved the objective. Don't generate anything after stop.\n4. In your final answer, you should only output a single action and should never output a prediction involving taking multiple actions."""

    system_prompt_qa="""You are an assistant answering a visual question by reasoning through image regions. You must systematically examine and verify relevant regions of the image, grounding each reasoning step to a specific (x, y) coordinate.

All reasoning steps must be enclosed within '<think>' tags and each step must start with an absolute (x, y) coordinate, followed by a description and evaluation of the corresponding image region. 
When confident in the answer, provide it inside '<answer>' tags:

<think>\n[Reasoning text with grounded points inline] (x_1, y_1). [Further reasoning] (x_2, y_2), ..., [Final reasoning] (x_n, y_n).\n</think>\nWhen ready to provide the final answer, enclose it within '<answer>' tags:\n<answer> {final answer} </answer>

Instructions:
- Always begin a reasoning step with an (x, y) coordinate.
- Coordinates must be absolute image points formatted as integers: (x, y).
- Regions refer to spatially distinct parts of the image: quadrants (e.g., top-left), discrete objects (e.g., bottle), or structural zones (e.g., background).
- Explore diverse, even less likely, regions early on to ensure broad coverage.
- Reason about a region's relevance to the question and—if visible—its relation to prior steps.
- Aim to choose accurate, representative coordinates within each region."""

    system_prompt_amber="""A conversation between User and Assistant. The User asks a yes/no multiple-choice question about an image, and the Assistant answers by carefully verifying whether the claim is supported by visible evidence.

The Assistant must first reason inside a single <think> section, then provide the final answer inside an <answer> section. The reasoning should use coordinates as evidence anchors for the visible regions being checked.

The final answer must be exactly one of:
<answer> A </answer>
<answer> B </answer>

Rules:
- Base the answer only on visible evidence in the image.
- Ground each relevant evidence check with a representative absolute image coordinate formatted as (x, y).
- Coordinates are evidence anchors for reasoning, not the final answer.
- For existence questions, verify whether the queried object is actually visible and distinguish it from similar objects.
- For attribute questions, verify the specific property directly from the image.
- For relation questions, verify both entities first, then verify the claimed relation.
- Use the checked visual evidence to choose the better-supported option letter.
- Answer A when the claim is supported by visible evidence.
- Answer B when the claim is contradicted or not supported by visible evidence.
- If the evidence is ambiguous, still choose A or B based on the strongest visible evidence; do not answer "I don't know."
- If the user prompt asks for the text of the option, still follow this system instruction and answer with the option letter only.
- Do not use world knowledge to infer details that are not visible.
- The <answer> section must contain only A or B."""

    if prompt_type == "amber":
        system_prompt = system_prompt_amber
    elif prompt_type == "web_grounding":
        system_prompt = system_prompt_web_grounding
    elif prompt_type == "spatial":
        system_prompt = system_prompt_spatial
    elif prompt_type == "web_action":
        system_prompt = system_prompt_web_action
    elif prompt_type == "vstar":
        # NOTE: V* single turn not tested yet
        system_prompt = system_prompt_qa
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    val_size = args.val_size
    draw_points = not args.no_draw_points
    max_chains_per_sample = args.max_chains_per_sample

    data_files = collect_data_files(data_str)

    # We'll store:
    # - all textual chains to write to a .txt file
    # - for each chain, we create a separate SFT entry
    all_chains_text = []
    sample_entry_groups = []
    all_images_processed = set()
    chain_global_id = 0
    correct_count = 0
    total_samples = 0

    for i, data_file in enumerate(data_files):
        with open(data_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if args.max_samples is not None and total_samples >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                data_json = json.loads(line)

                error = data_json.get("error", "")
                if error:
                    print(f"error processing {data_file}:{line_number}: {error}")
                    continue

                question = data_json.get("question", "")
                if prompt_type == "amber":
                    question = normalize_amber_question_prompt(question)
                image = data_json.get("image", "")
                true_answer = data_json.get("true_answer", "")
                gt_answer = answer_to_amber_option(true_answer) if prompt_type == "amber" else true_answer
                if gt_answer is None:
                    gt_answer = true_answer
                chains, root_node_value = build_all_reasoning_for_sample(data_json)
                total_samples += 1
                if root_node_value > 0:
                    correct_count += 1

                if len(chains) > max_chains_per_sample:
                    chains = rng.sample(chains, max_chains_per_sample)

                sample_entries = []
                for c in chains:
                    if c.count("<image>") > 0:
                        c = c.replace("<image>", "").replace("</image>", "")
                    if prompt_type == "amber":
                        c = convert_amber_answer_tags(c)

                    all_chains_text.append(c)

                    chain_entry = {
                        "id": f"{i}_{line_number}_{chain_global_id}",
                        "metadata": {},
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": question
                            },
                            {
                                "role": "assistant",
                                "content": c
                            }
                        ],
                        "images": [image],
                        "gt_answer": gt_answer,
                    }
                    sample_entries.append(chain_entry)
                    all_images_processed.add(image)
                    chain_global_id += 1

                if sample_entries:
                    sample_entry_groups.append(sample_entries)
        if args.max_samples is not None and total_samples >= args.max_samples:
            break
        
    if total_samples == 0:
        print("No samples found. Exiting.")
        exit()

    sample_indices = list(range(len(sample_entry_groups)))
    rng.shuffle(sample_indices)
    if val_size < 1:
        val_count = int(len(sample_indices) * val_size)
    else:
        val_count = int(val_size)
    if len(sample_indices) > 1:
        val_count = min(max(val_count, 1), len(sample_indices) - 1)
    else:
        val_count = 0
    val_indices = set(sample_indices[:val_count])
    sft_entries_train = []
    sft_entries_val = []
    for sample_idx, entries in enumerate(sample_entry_groups):
        if sample_idx in val_indices:
            sft_entries_val.extend(entries)
        else:
            sft_entries_train.extend(entries)

    accuracy = correct_count / total_samples
    print(f"Processed {total_samples} rollout samples.")
    print(f"Root node correctness ratio: {accuracy:.3f} ({correct_count}/{total_samples})")

    # 2) Write out all chains to a text file
    out_dir = resolve_output_dir(data_str, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_path = out_dir / "reasoning_chains.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        sep = "\n\n----------------------\n\n----------------------\n\n"
        for ch in all_chains_text:
            f.write(ch + sep)
    print(f"Wrote {len(all_chains_text)} total chains to {txt_path}")

    # 3) Train/Val split at the chain level has been done during processing

    train_json_path = out_dir / args.train_output_name
    val_json_path = out_dir / args.val_output_name

    with train_json_path.open("w", encoding="utf-8") as f:
        json.dump(sft_entries_train, f, indent=2)
    print(f"Saved train SFT data to: {train_json_path}  (count={len(sft_entries_train)})")

    with val_json_path.open("w", encoding="utf-8") as f:
        json.dump(sft_entries_val, f, indent=2)
    print(f"Saved val SFT data to: {val_json_path}  (count={len(sft_entries_val)})")

    # 4) (Optional) Log to W&B
    if log_to_wandb:
        wandb.init(project="vlm-search", name="mcts_reasoning_chains_sft")
        max_samples = 50
        # Shuffle the data
        sft_entries = sft_entries_train + sft_entries_val
        random_indices = rng.sample(range(len(sft_entries)), min(max_samples, len(sft_entries)))
        html_content = "<html><body>"
        for idx in random_indices:
            entry = sft_entries[idx]
            question = entry["messages"][1]["content"]
            chain_text = entry["messages"][2]["content"]
            image_path = entry["images"][0]
            true_answer = entry["gt_answer"]
            # Extract final answer from <answer> tags, if present
            final_answer = ""
            if "<answer>" in chain_text and "</answer>" in chain_text:
                try:
                    final_answer = chain_text.split("<answer>")[1].split("</answer>")[0].strip()
                except:
                    pass

            img_html = ""
            if image_path and os.path.exists(image_path):
                with Image.open(image_path) as pil_img:
                    if draw_points:
                        # Prepare to draw on the image
                        draw = ImageDraw.Draw(pil_img)
                        circle_radius = 7

                        # 1) Draw intermediate points from <think> text
                        intermediate_points = extract_think_points(chain_text)
                        for (x, y, point_num) in intermediate_points:
                            draw.ellipse(
                                (x - circle_radius, y - circle_radius,
                                 x + circle_radius, y + circle_radius),
                                fill="green",
                            )
                            draw.text((x + 15, y), str(point_num), fill="green")

                        # 2) If final answer parse is valid, draw it in red
                        try:
                            pred_ans = ast.literal_eval(final_answer)
                            if (
                                isinstance(pred_ans, (list, tuple)) and
                                len(pred_ans) == 2
                            ):
                                px, py = float(pred_ans[0]), float(pred_ans[1])
                                draw.ellipse(
                                    (px - circle_radius, py - circle_radius,
                                     px + circle_radius, py + circle_radius),
                                    fill="red",
                                )
                                draw.text((px + 15, py), "PRED", fill="red")
                        except:
                            pass

                        # 3) If ground truth is provided, draw it in blue
                        try:
                            gt_ans = ast.literal_eval(true_answer)
                            if (
                                isinstance(gt_ans, (list, tuple)) and
                                len(gt_ans) == 2
                            ):
                                gx, gy = float(gt_ans[0]), float(gt_ans[1])
                                draw.ellipse(
                                    (gx - circle_radius, gy - circle_radius,
                                     gx + circle_radius, gy + circle_radius),
                                    fill="blue",
                                )
                                draw.text((gx + 15, gy), "GT", fill="blue")
                        except:
                            pass

                    # Convert to RGB for displaying
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")

                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    enc = base64.b64encode(buf.getvalue()).decode("utf-8")
                    img_html = f'<img src="data:image/png;base64,{enc}" style="max-width:600px;" />'
                # except Exception as e:
                #     print(f"Could not open {image_path}: {e}")

            row_html = f"""
            <div style="border:1px solid #ddd; padding:10px; margin:10px 0;">
                <p><b>Question:</b> {question}</p>
                {img_html}
                <p><b>Chain:</b> {chain_text.replace('<','&lt;').replace('>','&gt;')}</p>
            </div>
            """
            html_content += row_html

        html_content += "</body></html>"
        wandb.log({"mcts_reasoning_chains": wandb.Html(html_content)})
        wandb.finish()
