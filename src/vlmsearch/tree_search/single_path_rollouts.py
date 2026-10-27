# tree_search.py

from __future__ import annotations
from typing import Optional, List
import math
from PIL import Image
import logging
import wandb  # Added import for Weights & Biases
import numpy as np
import re
from tqdm import tqdm
import ast
logger = logging.getLogger(__name__)


from qwen_vl_utils import fetch_image
from PIL import ImageDraw
import json
TOOL_RE   = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
def _parse_coordinate(text: str):
    """
    Extract a coordinate from text of the form '(x, y)'.
    Returns (x, y) as a tuple of ints if found, otherwise None.
    """
    try:
        text = text.strip()
        json_text = json.loads(text)
        return json_text["arguments"]["coordinate"]
    except Exception as e:
        return None

def _get_point_crop(image: Image.Image, point: List[int], offset: int = 75, crop_size: int = 512, draw_dot: bool = True) -> Image.Image:
    """
    Get a crop of the image centered on the point with a side length of 2*offset.
    Also draws a dot at the point location within the crop.
    """
    x, y = point
    width, height = image.size

    # Ensure crop boundaries are within image dimensions
    left = max(0, x - offset)
    top = max(0, y - offset)
    right = min(width, x + offset)
    bottom = min(height, y + offset)
    
    # Ensure that right > left and bottom > top
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    
    # Create the crop
    crop = image.crop((left, top, right, bottom))

    if draw_dot:
        # Draw a dot at the point location (relative to the crop)
        draw = ImageDraw.Draw(crop)
        dot_x = x - left
        dot_y = y - top
        radius = 7
        draw.ellipse(
            [(dot_x - radius, dot_y - radius), (dot_x + radius, dot_y + radius)],
            fill="red",
            outline="white",
            width=2
        )

    # Super-resolution to crop_size x crop_size
    # crop = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    crop = fetch_image({"image": crop, "resized_width": crop_size, "resized_height": crop_size})

    return crop

def check_for_crop(text: str, image: Image.Image, offset: int = 50, crop_size: int = 512, draw_dot: bool = True) -> Optional[Image.Image]:
    m = TOOL_RE.search(text)
    if m:
        coord = _parse_coordinate(m.group(1))
        try:
            crop = _get_point_crop(image, coord, offset=offset, crop_size=crop_size, draw_dot=draw_dot)
            return crop
        except Exception as e:
            return None
    return None


class TreeNode:
    """
    Represents a single node in the tree.
    Each node has:
      - a reference to the parent
      - a list of children
      - a 'thought' text
      - a heuristic or computed value
      - a flag if it's terminal (<final>) or not
    """

    def __init__(
        self,
        thought_text: str,
        image: Image = None,
        parent: Optional[TreeNode] = None
    ):
        self.thought_text = thought_text
        self.image = image
        self.parent = parent
        self.children: List[TreeNode] = []
        self.value: float = 0.0   # heuristic or MCTS Q-value
        self.visit_count: int = 0 # for MCTS
        self.is_terminal: bool = False

    def add_child(self, child_node: TreeNode) -> None:
        self.children.append(child_node)

    def update_value(self, new_value: float) -> None:
        """
        Updates the node's value (placeholder for MCTS backprop).
        """
        self.value = new_value

    def increment_visits(self) -> None:
        self.visit_count += 1

class SinglePathRollouts:
    """
    Manages expansion of the tree. 
    Later can be adapted to MCTS: sample multiple rollouts from a node,
    update Q-values, etc.
    """

    def __init__(
        self,
        llm_wrapper,
        judge,
        system_prompt: str,
        max_depth: int = 5,
        n_rollouts: int = 3,
        add_thought_number_system_prompt: bool = False, 
        generate_cold_start: bool = False,
        generate_upfront: bool = False,
        rollout_no_thinking: bool = False,
        first_rollout_no_sample: bool = False,
        thought_token_begin: str = "<think>",
        thought_token_end: str = "</think>",
        final_token_begin: str = "<answer>",
        final_token_end: str = "</answer>",
        max_image_side: int = None,
        max_pixels: int = None,
        check_for_crop: bool = False,
        crop_offset: int = 50,
        crop_size: int = 512,
        draw_dot: bool = True,
    ):
        """
        llm_wrapper: An instance of LLMWrapper for generation.
        system_prompt: The system prompt appended to every generation step.
        max_depth: Maximum depth before forcing <final>.
        """
        self.llm = llm_wrapper
        self.judge = judge
        self.system_prompt = system_prompt
        self.max_depth = max_depth
        self.n_rollouts = n_rollouts
        self.add_thought_number_system_prompt = add_thought_number_system_prompt
        self.generate_cold_start = generate_cold_start
        self.generate_upfront = generate_upfront
        self.rollout_no_thinking = rollout_no_thinking
        self.first_rollout_no_sample = first_rollout_no_sample
        self.thought_token_begin = thought_token_begin
        self.thought_token_end = thought_token_end
        self.final_token_begin = final_token_begin
        self.final_token_end = final_token_end
        self.max_image_side = max_image_side
        self.max_pixels = max_pixels
        self.check_for_crop = check_for_crop
        self.crop_offset = crop_offset
        self.crop_size = crop_size
        self.draw_dot = draw_dot

        print(f"Max pixels: {self.max_pixels}; Max image side: {self.max_image_side}")

    def search(
        self, 
        input_query: str, 
        input_image_path: str,
        true_answer: str,
        worker_id: int,
        ) -> str:
        """
        Main entry point for performing the tree search on a single prompt/data sample.
        
        We will perform N full rollouts from the root. Each rollout:
        1) Expands a single path from the root until terminal or max depth.
        2) Obtains a heuristic value at the leaf node.
        3) Propagates that value back up to all ancestors.
        4) Logs the full trajectory in wandb.

        Returns the final answer from the last terminal node found, or a fallback if none found.
        """

        if self.add_thought_number_system_prompt:
            to_add = f" There should be atleast {np.random.randint(3, 8)} thoughts before providing the final answer."
            system_prompt = self.system_prompt + to_add
        else:
            system_prompt = self.system_prompt

        input_image = Image.open(input_image_path)

        width, height = input_image.size
        max_side = max(width, height)
        scale_factor = 1.0
        if self.max_image_side is not None and max_side > self.max_image_side:
            scale_factor = self.max_image_side / max_side
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            input_image = input_image.resize((new_width, new_height))
            if self.judge.judge_type == "point_matching":
                gt_point = tuple(ast.literal_eval(true_answer))
                gt_point = tuple(int(x * scale_factor) for x in gt_point)
                true_answer = str(gt_point)
            elif self.judge.judge_type == "point_in_bbox":
                gt_bbox = tuple(ast.literal_eval(true_answer))
                gt_bbox = tuple(int(x * scale_factor) for x in gt_bbox)
                true_answer = str(gt_bbox)

        width, height = input_image.size
        scale_factor = 1.0
        if self.max_pixels is not None and width * height > self.max_pixels:

            input_image = fetch_image({"image": input_image, "max_pixels": self.max_pixels})

            # get scale factor
            scale_factor_width = input_image.size[0] / width
            scale_factor_height = input_image.size[1] / height
            scale_factor = scale_factor_width
            if self.judge.judge_type == "point_matching":
                gt_point = tuple(ast.literal_eval(true_answer))
                gt_point = (int(gt_point[0] * scale_factor_width), int(gt_point[1] * scale_factor_height))
                true_answer = str(gt_point)
            elif self.judge.judge_type == "point_in_bbox":
                gt_bbox = tuple(ast.literal_eval(true_answer))
                gt_bbox = (int(gt_bbox[0] * scale_factor_width), int(gt_bbox[1] * scale_factor_height), int(gt_bbox[2] * scale_factor_width), int(gt_bbox[3] * scale_factor_height))
                true_answer = str(gt_bbox)

        # Create root node
        root = TreeNode(
            thought_text=input_query, 
            image=input_image, 
            parent=None
        )

        final_answer = None
        judge_score = None
        search_outputs = []
        for rollout_index in tqdm(range(self.n_rollouts), desc=f"Rollouts worker {worker_id}", leave=False):
            logger.debug(f"Starting rollout {rollout_index+1}/{self.n_rollouts}...")
            # Perform a single rollout (root -> leaf)
            
            # if rollout_index == 0 and self.first_rollout_no_sample:
            #     no_sample = True
            # else:
            #    no_sample = False
            no_sample = False
            
            path_nodes, depth = self._do_single_rollout(
                root, 
                system_prompt,
                no_sample
                )

            if self.generate_cold_start and depth < 3 :
                logger.debug(f"Rollout depth is less than 3! Doing another rollout")
                judge_score = 0.0
                search_output = self.format_search_output(
                        input_query,
                        input_image_path,
                        true_answer,
                        self.system_prompt,
                        path_nodes, 
                        judge_score,
                        scale_factor,
                    )
                search_outputs.append(search_output)
                continue
            
            #check the format of each node in the path. If the format is incorrect, we do another rollout.
            node_index = 0
            total_nodes = len(path_nodes)
            invalid_format = False
            
            for node in path_nodes:
                #skip the first and the last node
                if node_index == 0 or node_index == total_nodes - 1:
                    node_index += 1
                    continue

                if self.judge.format_check(node.thought_text):
                    node_index += 1
                    continue
                else:
                    logger.debug(f"Invalid format found in the path! Doing another rollout")
                    invalid_format = True
                    break
            
            
            if invalid_format:
                judge_score = 0.0
                search_output = self.format_search_output(
                        input_query,
                        input_image_path,
                        true_answer,
                        self.system_prompt,
                        path_nodes, 
                        judge_score,
                        scale_factor,
                    )
                search_outputs.append(search_output)
                continue
            else:
                logger.debug(f"Valid format found in the path! Continuing...")

            # path_nodes now contains all nodes from root to leaf for this rollout
            leaf_node = path_nodes[-1]
            # If leaf is terminal, we parse out <final> if it exists, or treat its text as final.
            if leaf_node.is_terminal:
                final_answer = leaf_node.thought_text  # already set to the final answer after <final>
                
                logger.debug(f"================= Judging =================")
        
                judge_score = self.judge.judge(
                    input_query=input_query,
                    image=input_image,
                    gt_ans=true_answer,
                    pred_ans=(getattr(leaf_node, "raw_final_answer", final_answer) if self.judge.judge_type == "string_match" else final_answer),
                    scale_factor=scale_factor,
                )
                logger.debug(f"Judge score: {judge_score}; Pred answer: {final_answer}; True answer: {true_answer}")

                logger.debug(f"================= Judging Done =================")

            else:
                # If not terminal, we can continue, or handle differently if we want.
                final_answer = "No final node found in this rollout."

            #judge_score should replace the heuristic value and if no final answer is found, then the judge score is 0.0.
            if judge_score is not None:
                leaf_value = judge_score
            else:
                leaf_value = 0.0

            leaf_node.value = leaf_value

            # Backprop the leaf_value up the path
            self._backprop_value(path_nodes, leaf_value)

            # # TURNED OFF FOR NOW: Log to wandb for easy visualization
            # rollout_html = self._log_rollout_to_wandb(
            #     rollout_index,
            #     path_nodes,
            #     leaf_value,
            #     input_query,
            #     input_image,
            #     true_answer,
            #     judge_score,
            # )

            #If the response is correct, we go to the next training example. If the response is incorrect, we do another rollout.
            search_output = self.format_search_output(
                input_query,
                input_image_path,
                true_answer,
                self.system_prompt,
                path_nodes, 
                judge_score,
                scale_factor,
            )
            search_outputs.append(search_output)
            
            if self.generate_cold_start:
                if judge_score == 1.0:
                    logger.debug(f"Correct answer found in rollout! Going on next Sample")
                    return search_outputs
                
                if judge_score == 0.0:
                    logger.debug(f"Incorrect answer found in rollout! Doing another rollout")
                    continue

        return search_outputs
    
    def _do_single_rollout(
        self, 
        root: TreeNode,
        system_prompt: str,
        no_sample: bool = False
    ) -> List[TreeNode]:
        """
        Execute a single rollout (root -> leaf) by sequentially generating
        next thoughts until a terminal node is encountered or max_depth is reached.
        Each new node becomes the sole child in this path.
        """
        path = []
        current_node = root
        depth = 0

        # if rollout_no_thinking, must have generate_upfront
        if self.rollout_no_thinking and not self.generate_upfront:
            raise ValueError("rollout_no_thinking must have generate_upfront")

        if self.generate_upfront:
            previous_thoughts = self._collect_path_thoughts(current_node)
            all_text = self.llm.generate_single_thought(
                system_prompt=system_prompt,
                previous_thoughts=previous_thoughts,
                force_final=False,
                no_sample=no_sample
            )
            if self.rollout_no_thinking:
                leaf_node = TreeNode(thought_text=all_text, parent=current_node)
                leaf_node.is_terminal = True
                current_node.add_child(leaf_node)
                path= [current_node, leaf_node]
                return path, 1
            pattern = re.compile(fr"({self.thought_token_begin}.*?{self.thought_token_end}|{self.final_token_begin}.*?{self.final_token_end})", re.DOTALL)
            matches = pattern.findall(all_text)

        # Move down the tree until we can't
        while True:
            path.append(current_node)
            current_node.increment_visits()

            # Stop if already terminal (possibly from a previous rollout).
            if current_node.is_terminal:
                break

            # If at or beyond max depth, force final
            force_final = (depth >= self.max_depth)

            # Prepare the prompt for the next generation
            previous_thoughts = self._collect_path_thoughts(current_node)

            # Generate the next thought
            if self.generate_upfront:
                if not matches:
                    next_text = all_text
                else:
                    next_text = matches.pop(0)
            else:
                next_text = self.llm.generate_single_thought(
                    system_prompt=system_prompt,
                    previous_thoughts=previous_thoughts,
                    force_final=force_final,
                    no_sample=no_sample
                )
            logger.debug(f"Generated text at depth {depth}: {next_text}")

            # if self.generate_upfront:
            #     final_answer_text = next_text
            #     leaf_node = TreeNode(thought_text=final_answer_text, parent=current_node)
            #     leaf_node.is_terminal = True
            #     current_node.add_child(leaf_node)
            #     path.append(leaf_node)
            #     break

            # Check if <final> was produced
            if self.final_token_begin in next_text:

                if self.thought_token_begin in next_text:
                    # account for the case where the model outputs <think> and <answer> tags
                    thought_text = next_text.split(self.final_token_begin)[0].strip()
                    # Create a new child node for the next thought
                    child_node = TreeNode(thought_text=thought_text, parent=current_node)
                    current_node.add_child(child_node)
                    current_node = child_node

                # Mark terminal
                # Extract the part after <final> as the final answer
                final_answer_text = next_text.split(self.final_token_begin)[1].strip()
                final_answer_text = final_answer_text.split(self.final_token_end)[0].strip()
                leaf_node = TreeNode(thought_text=final_answer_text, parent=current_node)
                leaf_node.is_terminal = True
                leaf_node.raw_final_answer = all_text if self.generate_upfront else next_text
                current_node.add_child(leaf_node)
                path.append(leaf_node)
                break
            else:
                
                # Create a new child node for the next thought
                child_node = TreeNode(thought_text=next_text, parent=current_node)
                if self.check_for_crop:
                    crop = check_for_crop(next_text, root.image, self.crop_offset, self.crop_size, draw_dot=self.draw_dot)
                    if crop is not None:
                        child_node.image = crop
                current_node.add_child(child_node)
                current_node = child_node

            depth += 1
            if depth > self.max_depth:
                # We've gone one step beyond max_depth if forced_final didn't yield <final>.
                # We'll consider this a leaf (non-terminal but can't go deeper).
                break

        return path, depth

    def _heuristic(self, text: str) -> float:
        """
        A placeholder heuristic function. 
        For advanced usage, you might incorporate a learned reward model or an MLE-based scoring.
        Here, we just return a constant for demonstration.
        """
        # Example: reward could be 1.0, or based on length, etc.
        return 1.0

    def _backprop_value(self, path_nodes: List[TreeNode], leaf_value: float) -> None:
        """
        Propagate the leaf value back up all nodes in the path.
        This is a simple approach that sets each ancestor's value to `leaf_value`.
        In a real MCTS, you'd average or do something more sophisticated.
        """
        for node in path_nodes:
            node.value = leaf_value

    def _collect_path_thoughts(self, node: TreeNode) -> List[str]:
        """
        Collect all thoughts from root to 'node' to feed as context.
        Each item in the returned list is (thought_text, image), but you can structure differently if needed.
        """
        path_texts = []
        current = node
        while current is not None:
            path_texts.append((current.thought_text, current.image))
            current = current.parent
        path_texts.reverse()
        return path_texts

    def _log_rollout_to_wandb(
        self,
        rollout_index: int,
        path_nodes: List[TreeNode],
        final_value: float,
        input_query: str,
        input_image: Image,
        true_answer: str,
        judge_score: float = None,
    ) -> None:
        """
        Logs the entire path of a single rollout to WandB in an HTML format:
        - The input query (root)
        - The input image
        - The final value (leaf_value)
        - The full sequence of thoughts from root to leaf as an HTML list
        """
        import base64

        def node_to_html(node: TreeNode) -> str:
            """
            Converts a node into an HTML representation.
            Includes its thought_text and its value (if visited).
            """
            text_snippet = node.thought_text.replace("<", "&lt;").replace(">", "&gt;")
            if node.is_terminal:
                # Append final indicator to the text
                text_snippet = self.final_token_begin + " " + text_snippet
            text_snippet = text_snippet.replace(self.final_token_begin, "Final:")
            value = f"(Value: {node.value:.2f})"
            return f"<li>{text_snippet} {value}</li>"

        if input_image:
            # `input_image` is a PIL image, convert it to base64
            from io import BytesIO
            buffer = BytesIO()
            input_image.save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Add the <img> tag to the HTML
            image_html = f'<img src="data:image/png;base64,{encoded_image}" alt="Input Image" style="max-width: 100%; height: auto;">'
        else:
            image_html = ""

        # Create an HTML list representing the path
        true_answer_html = f"<li>True Answer: {true_answer}</li>"
        judge_score_html = f"<li>Judge Score: {judge_score:.2f}</li>" if judge_score is not None else ""

        rollout_html = "<ul>" + image_html + "".join(node_to_html(node) for node in path_nodes) + true_answer_html +  judge_score_html + "</ul>" 
        # # Log to WandB
        # wandb_run.log({
        #     "rollout_index": rollout_index,
        #     # "query": input_query,
        #     "image": wandb.Image(input_image) if input_image else None,
        #     "rollout_path_html": wandb.Html(rollout_html),
        #     "final_value": final_value,
        #     "judge_score": judge_score,
        # })
        return rollout_html

    def format_search_output(
        self, 
        input_query: str, 
        image_path: str, 
        true_answer: str, 
        system_prompt: str,
        path_nodes: List[TreeNode],
        judge_score: float,
        scale_factor: float
    ) -> dict:
        """
        Formats the final tree rollout (thoughts and answer) into a dictionary
        suitable for serialization (e.g., JSON).

        Args:
            input_query: The original user query.
            image_path: A string path to the input image (if available).
            true_answer: The known correct answer for evaluation.
            path_nodes: The list of tree nodes from the final rollout 
                        (root to leaf), where the last node may be terminal.
            judge_score: The final judge score for the rollout.

        Returns:
            A dict containing question, image, true_answer, thoughts, and final_answer.
        """
        # Extract thoughts and final_answer
        thoughts = []
        final_answer = None
        
        for idx, node in enumerate(path_nodes):
            # skip first node
            if idx == 0:
                continue
            if not node.is_terminal:
                # Nodes that aren't terminal are "thought" nodes
                thought_text = node.thought_text
                thoughts.append(thought_text)
            else:
                # Terminal node text is taken as the final answer
                final_answer = node.thought_text

        # Construct the output dictionary
        return {
            "question": input_query,
            "image": image_path,
            "true_answer": true_answer,
            "thoughts": thoughts,
            "final_answer": final_answer,
            "raw_final_answer": getattr(path_nodes[-1], "raw_final_answer", final_answer),
            "system_prompt": system_prompt,
            "judge_score": judge_score,
            "scale_factor": scale_factor,
        }


    def _get_depth(self, node: TreeNode) -> int:
        """
        Get the depth of 'node' by walking upward.
        """
        depth = 0
        current = node
        while current.parent is not None:
            current = current.parent
            depth += 1
        return depth
