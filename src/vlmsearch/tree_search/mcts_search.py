from __future__ import annotations
from typing import Optional, List
import math
from PIL import Image
import logging
import numpy as np
import re
import json
from tqdm import tqdm
import datetime

logger = logging.getLogger(__name__)

def serialize_tree(root: TreeNode) -> None:
    """
    Recursively turn a TreeNode (and its descendants) into a nested dictionary,
    then write to a JSON file.
    """
    def _node_to_dict(node: TreeNode) -> dict:
        return {
            "thought_text": node.thought_text,
            "raw_final_answer": node.raw_final_answer,
            "is_terminal": node.is_terminal,
            "value": node.value,
            "visit_count": node.visit_count,
            # You could store rollouts if desired, or any other data
            "rollouts": node.rollouts,
            "children": [_node_to_dict(child) for child in node.children],
            "used_coords": list(node.used_coords),
        }

    tree_dict = _node_to_dict(root)
    return tree_dict


def deserialize_tree(filename: str) -> TreeNode:
    """
    Read a JSON file and recursively construct a TreeNode structure.
    """
    def _dict_to_node(d: dict, parent: Optional[TreeNode] = None) -> TreeNode:
        node = TreeNode(
            thought_text=d["thought_text"],
            parent=parent
        )
        node.is_terminal = d["is_terminal"]
        node.raw_final_answer = d.get("raw_final_answer")
        node.value = d["value"]
        node.visit_count = d["visit_count"]
        node.rollouts = d["rollouts"]
        # Recreate children
        for child_dict in d["children"]:
            child_node = _dict_to_node(child_dict, parent=node)
            node.children.append(child_node)
        return node

    with open(filename, "r", encoding="utf-8") as f:
        tree_dict = json.load(f)
    return _dict_to_node(tree_dict["tree"])

class TreeNode:
    """
    Represents a single node in the tree.
    Each node has:
      - a reference to the parent
      - a list of children
      - a 'thought' text
      - a value (Q-value for MCTS)
      - a visit count (N-value for MCTS)
      - a flag if it's terminal (<final>) or not
    """

    def __init__(
        self,
        thought_text: str,
        image: Image = None,
        parent: Optional[TreeNode] = None,
        used_coords: Optional[set] = None,
    ):
        self.thought_text = thought_text
        self.raw_final_answer = None
        self.image = image
        self.parent = parent
        self.children: List[TreeNode] = []

        # MCTS-related variables
        self.value: float = 0.0      # running average of returns
        self.visit_count: int = 0    # how many times this node was visited
        self.is_terminal: bool = False
        self.rollouts = []

        # ### FLAG (1): Keep track of all used coords so far
        # Inherit parent's set of used coords, add any found in self.thought_text
        if used_coords is None:
            used_coords = set()
        else:
            used_coords = set(used_coords)  # copy
        # parse out new coords from the text and add them
        coords_in_text = self._parse_coords(thought_text)
        used_coords.update(coords_in_text)

        self.used_coords = used_coords

    def add_child(self, child_node: TreeNode) -> None:
        self.children.append(child_node)

    def increment_visits(self) -> None:
        self.visit_count += 1

    def update_value_mcts(self, reward: float) -> None:
        """
        Update the node's average value with a new sample (reward).
        A simple way is to do incremental mean:
          new_average = old_average + (reward - old_average) / N
        """
        old_value = self.value
        self.value += (reward - old_value) / (self.visit_count)

    def _parse_coords(self, text: str) -> List[str]:
        """
        Parse (x, y) patterns from the text, returning them as strings.
        E.g., "(123, 456)" -> "123,456" or keep as "(123,456)".
        You can store them in your own standard form.
        """
        matches = re.findall(r"\(\s*\d+\s*,\s*\d+\s*\)", text)
        return [m for m in matches]


class MonteCarloTreeSearch:
    """
    A class that implements basic MCTS with:
     - Selection (UCB-based)
     - Expansion (creating a single new child from the LLM if not terminal)
     - Rollout (simulate until final with ephemeral steps)
     - Backprop (propagate final reward up)
    
    Only the final reward from the LLM judge is used for backprop.
    """

    def __init__(
        self,
        llm_wrapper,
        judge,
        system_prompt: str,
        # Common hyperparams
        n_simulations: int = 10,      # total MCTS simulations
        c_puct: float = 1.4,          # exploration constant in UCB
        max_depth: int = 5,           # expansion limit depth
        rollout_max_steps: int = 5,   # max steps in rollout
        n_rollouts_per_node: int = 2, # number of rollouts per node
        num_children_per_expand: int = 3, # number of children to expand
        # Additional toggles from original code
        add_thought_number_system_prompt: bool = False,
        generate_cold_start: bool = False,
        generate_upfront: bool = False,
        rollout_no_thinking: bool = False,
        first_rollout_no_sample: bool = False,
        thought_token_begin: str = "<think>",
        thought_token_end: str = "</think>",
        final_token_begin: str = "<answer>",
        final_token_end: str = "</answer>",
        avoid_repeat_coords: bool = True,
    ):
        """
        Args:
            llm_wrapper: An LLM interface for generating text.
            judge: A judging interface to produce final reward from answer correctness.
            system_prompt: A system prompt to prefix every generation.
            n_simulations: How many MCTS simulations to run from the root.
            c_puct: Exploration constant in the UCB formula.
            max_depth: The maximum expansion depth; if reached, we force <final>.
            rollout_max_steps: Max number of generation steps in a single rollout.
        """
        self.llm = llm_wrapper
        self.judge = judge
        self.system_prompt = system_prompt
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.rollout_max_steps = rollout_max_steps

        # Original toggles
        self.add_thought_number_system_prompt = add_thought_number_system_prompt
        self.generate_cold_start = generate_cold_start
        self.generate_upfront = generate_upfront
        self.rollout_no_thinking = rollout_no_thinking
        self.first_rollout_no_sample = first_rollout_no_sample
        self.n_rollouts_per_node = n_rollouts_per_node
        self.num_children_per_expand = num_children_per_expand
        self.thought_token_begin = thought_token_begin
        self.thought_token_end = thought_token_end
        self.final_token_begin = final_token_begin
        self.final_token_end = final_token_end

        self.avoid_repeat_coords = avoid_repeat_coords

    def search(
        self,
        input_query: str,
        input_image_path: str,
        true_answer: str,
        worker_id: int = 0
    ) -> List[dict]:
        """
        Main entry point for performing the MCTS search on a single prompt/data sample.
        We create a root node, run `n_simulations` of MCTS, then pick the best child or 
        best final node. This returns a list of rollouts (similar to the original code).

        Returns:
            A list of search outputs (dicts) for each simulation, 
            containing the final answer, judge score, etc.
        """

        global_start_time = datetime.datetime.now()

        # Optionally augment the system prompt with some random instruction
        if self.add_thought_number_system_prompt:
            extra_str = f"\n- There should be at least {np.random.randint(3, 8)} thoughts before providing the final answer."
            system_prompt = self.system_prompt + extra_str
        else:
            system_prompt = self.system_prompt

        # Open the image
        input_image = Image.open(input_image_path)

        # Create the root node
        root = TreeNode(
            thought_text=input_query,
            image=input_image,
            parent=None,
            used_coords=set()
        )

        # search_outputs = []
        for sim_idx in tqdm(range(self.n_simulations), desc=f"MCTS Simulations worker {worker_id}", leave=False):
            start_time = datetime.datetime.now()
            logger.debug(f"=== MCTS Simulation {sim_idx+1}/{self.n_simulations} ===")

            # 1) SELECTION: get a path from root to a leaf or expandable node
            path = self._select(root)
            leaf_node = path[-1]
            
            # If leaf is already terminal, no expansion needed; just backprop its known value
            if leaf_node.is_terminal:
                # If it hasn't been judged yet, do so
                if leaf_node.visit_count == 0:
                    # Possibly do a final judge if the node has a <final> text
                    judge_score = self._judge_node(leaf_node, input_query, input_image, true_answer)
                    leaf_node.visit_count = 1
                    leaf_node.value = judge_score
                # Backprop the existing value
                self._backprop(path, leaf_node.value)
                
                # # Log an output record
                # search_output = self._format_search_output(
                #     input_query, input_image_path, true_answer, system_prompt,
                #     path_nodes=path,
                #     judge_score=leaf_node.value
                # )
                # search_outputs.append(search_output)
                continue

            # 2) EXPANSION: expand the leaf node if possible
            new_children = self._expand(leaf_node, system_prompt, num_children=self.num_children_per_expand)

            for child in new_children:
                for rollout_i in range(self.n_rollouts_per_node):
                    reward, final_answer = self._rollout(child, system_prompt, input_query, input_image, true_answer)
                    self._backprop(path + [child], reward)

            # for rollout_i in range(self.n_rollouts_per_node):

            #     # 3) ROLLOUT: from the newly created child, do an ephemeral rollout to get final reward
            #     reward, final_answer = self._rollout(new_child, system_prompt, input_query, input_image, true_answer)

            #     # 4) BACKPROP: pass reward up the path
            #     self._backprop(path + [new_child], reward)

            # search_outputs.append(search_output)

            end_time = datetime.datetime.now()
            search_time = end_time - start_time
            logger.debug(f"MCTS Simulation {sim_idx+1}/{self.n_simulations} time: {search_time}")

            # # If we want to cut off early when we find a perfect answer:
            # if self.generate_cold_start and reward == 1.0:
            #     logger.debug("Perfect answer found; stopping search early.")
            #     break

        global_end_time = datetime.datetime.now()
        global_search_time = global_end_time - global_start_time
        global_search_time = global_search_time.total_seconds()
        logger.debug(f"Total MCTS search time: {global_search_time}")
        
        # Record the search output for this simulation
        search_output = self._format_search_output(
            input_query, 
            input_image_path, 
            true_answer, 
            system_prompt,
            root=root,
            judge_score=reward,
            global_search_time=global_search_time
        )

        return [search_output]

    def _select(self, root: TreeNode) -> List[TreeNode]:
        """
        SELECTION PHASE:
        Traverse the tree from `root` down to a leaf or node with unvisited children
        using the UCB policy. Return the path of nodes from root to that node.

        If any node has an unvisited child, we stop once we reach that node
        (so expansion can happen there).
        """
        path = []
        current = root

        while True:
            path.append(current)

            # If terminal, we're done
            if current.is_terminal:
                return path

            # If no children, we can't go deeper
            if not current.children:
                return path

            # Check if any child is unvisited (visit_count == 0).
            unvisited = [child for child in current.children if child.visit_count == 0]
            if unvisited:
                # We'll stop here and expand one of the unvisited children
                return path

            # Otherwise, pick the best child by UCB
            current = max(current.children, key=lambda c: self._ucb_score(current, c))

    def _expand(self, node: TreeNode, system_prompt: str, num_children: int = 3) -> List[TreeNode]:
        """
        EXPANSION PHASE:
        Generate multiple new children (num_children) from the LLM for `node` if not terminal.
        Each LLM call can produce a different text if you set some randomness (temperature).
        If the child text contains <final>, mark it terminal.
        Add all children to `node.children` and return them.
        """
        depth = self._get_depth(node)
        force_final = (depth >= self.max_depth)

        # Prepare the LLM call with all prior thoughts along the path
        previous_thoughts = self._collect_path_thoughts(node)

        new_children = []
        for i in range(num_children):
            # Generate the next text (possibly with some sampling randomness)
            next_text = self.llm.generate_single_thought(
                system_prompt=system_prompt,
                previous_thoughts=previous_thoughts,
                force_final=force_final
            )
            logger.debug(f"EXPANSION child {i+1}/{num_children}: generated text => {next_text}")

            # Check for <final>
            if self.final_token_begin in next_text:
                final_answer_text = next_text.split(self.final_token_begin)[1].split(self.final_token_end)[0].strip()
                final_answer_text = re.sub(r'^[A-Z]\. ', '', final_answer_text)
                child_node = TreeNode(thought_text=final_answer_text, parent=node)
                child_node.is_terminal = True
                child_node.raw_final_answer = next_text
            else:
                # Normal child node
                child_node = TreeNode(
                    thought_text=next_text, 
                    parent=node,
                    used_coords=node.used_coords
                )

            # ### FLAG (1): If avoid_repeat_coords is True, skip child if it references an old coord
            if self.avoid_repeat_coords and not child_node.is_terminal and self._is_coord_repeat(child_node, node):
                logger.debug(f"Skipping child because of repeated coords: {child_node.thought_text}")
                continue

            node.add_child(child_node)
            new_children.append(child_node)

        return new_children


    def _rollout(
        self,
        start_node: TreeNode,
        system_prompt: str,
        input_query: str,
        input_image: Image,
        true_answer: str
    ) -> (float, str):
        """
        ROLLOUT PHASE:
        From the newly expanded child node, we do ephemeral steps (not stored in the real tree)
        until we reach <final> or exceed rollout_max_steps.

        Then we pass the final answer to the judge to get a single reward.
        Return (reward, final_answer).
        """
        depth = 0
        current_text = start_node.thought_text
        previous_thoughts = self._collect_path_thoughts(start_node)
        is_terminal = start_node.is_terminal

        # Keep track of how many coords are repeated vs. new in this rollout
        ephemeral_used_coords = set(start_node.used_coords)
        num_same_points = 0
        num_different_points = 0

        # If the expanded child is already terminal, no further steps needed
        if is_terminal:
            final_answer = current_text
            final_answer = re.sub(r'^[A-Z]\. ', '', final_answer)
            reward = self._judge_rollout(
                input_query, input_image, true_answer, final_answer, start_node.raw_final_answer
            )
            logger.debug(f"ROLLOUT: final answer => {final_answer}")
            logger.debug(f"ROLLOUT: reward => {reward}")
            rollout = {
                "final_answer": final_answer,
                "raw_final_answer": start_node.raw_final_answer,
                "reward": reward,
                "ephemeral_texts": [t for (t, _) in previous_thoughts],
                "ephemeral_depth": self._get_depth(start_node),
                "depth": depth,
                "num_same_points": num_same_points,
                "num_different_points": num_different_points
            }
            start_node.rollouts.append(rollout)
            return reward, final_answer

        # Otherwise, generate ephemeral steps
        ephemeral_texts = [thought for thought, _ in previous_thoughts]
        ephemeral_depth = self._get_depth(start_node)

        logger.debug(f"ROLLOUT: starting with text => {ephemeral_texts}")

        while depth < self.rollout_max_steps and ephemeral_depth < self.max_depth:

            # If we're at our last allowed step or near max_depth, force a final generation
            if depth == (self.rollout_max_steps - 1) or ephemeral_depth == (self.max_depth - 1):
                force_final = True
            else:
                force_final = False

            # Generate the next text chunk
            next_text = self.llm.generate_single_thought(
                system_prompt=system_prompt,
                previous_thoughts=previous_thoughts,
                force_final=force_final
            )

            logger.debug(f"ROLLOUT: generated text => {next_text}")

            # Count coordinate repeats vs. new
            newly_parsed_coords = start_node._parse_coords(next_text)
            for coord in newly_parsed_coords:
                if coord in ephemeral_used_coords:
                    num_same_points += 1
                else:
                    ephemeral_used_coords.add(coord)
                    num_different_points += 1

            # Append to ephemeral text, increment counters
            ephemeral_texts.append(next_text)
            previous_thoughts.append((next_text, None))
            ephemeral_depth += 1
            depth += 1

            # If this generation included a final answer, we stop
            if self.final_token_begin in next_text:
                final_answer = next_text.split(self.final_token_begin)[1].split(self.final_token_end)[0].strip()
                final_answer = re.sub(r'^[A-Z]\. ', '', final_answer)
                
                reward = self._judge_rollout(
                    input_query, input_image, true_answer, final_answer, next_text
                )
                
                logger.debug(f"ROLLOUT: final answer => {final_answer}")
                logger.debug(f"ROLLOUT: reward => {reward}")
                rollout = {
                    "final_answer": final_answer,
                    "raw_final_answer": next_text,
                    "reward": reward,
                    "ephemeral_texts": ephemeral_texts,
                    "ephemeral_depth": ephemeral_depth,
                    "depth": depth,
                    "num_same_points": num_same_points,
                    "num_different_points": num_different_points
                }
                start_node.rollouts.append(rollout)
                return reward, final_answer

        # If we ran out of steps or hit max depth without <final>,
        # treat the last text as final (though possibly incomplete).
        final_answer = ephemeral_texts[-1]
        final_answer = re.sub(r'^[A-Z]\. ', '', final_answer)
        reward = self._judge_rollout(input_query, input_image, true_answer, final_answer)
        logger.debug(f"ROLLOUT: final answer => {final_answer}")
        logger.debug(f"ROLLOUT: reward => {reward}")
        rollout = {
            "final_answer": final_answer,
            "raw_final_answer": final_answer,
            "reward": reward,
            "ephemeral_texts": ephemeral_texts,
            "ephemeral_depth": ephemeral_depth,
            "depth": depth,
            "num_same_points": num_same_points,
            "num_different_points": num_different_points
        }
        start_node.rollouts.append(rollout)
        return reward, final_answer


    def _backprop(self, path: List[TreeNode], reward: float) -> None:
        """
        BACKPROP PHASE:
        Update all nodes in `path` with the final rollout reward.
        We do an incremental mean update of .value, and increment .visit_count.
        """
        for node in path:
            node.increment_visits()
            node.update_value_mcts(reward)

    def _judge_rollout(self, input_query, input_image, true_answer, final_answer, raw_final_answer=None):
        """
        For the rollout, only the final answer is judged. We get a single numerical reward.
        """
        score = self.judge.judge(
            input_query=input_query,
            image=input_image,
            gt_ans=true_answer,
            pred_ans=(raw_final_answer if raw_final_answer is not None and self.judge.judge_type == "string_match" else final_answer)
        )
        return score

    def _judge_node(self, node: TreeNode, input_query, input_image, true_answer):
        """
        If we ever land on a node that is terminal but unvisited, we do a final judge.
        """
        final_ans = node.thought_text
        if node.raw_final_answer is not None and self.judge.judge_type == "string_match":
            final_ans = node.raw_final_answer
        score = self.judge.judge(
            input_query=input_query,
            image=input_image,
            gt_ans=true_answer,
            pred_ans=final_ans
        )
        return score

    def _ucb_score(self, parent: TreeNode, child: TreeNode) -> float:
        """
        Standard UCB formula: Q + c * sqrt( ln(N_parent) / (N_child) )
        where Q = child.value / child.visit_count (if you prefer average),
        but here we directly store child.value as an incremental mean,
        so child.value itself can be used as Q.
        """
        c = self.c_puct

        # Exploitation term: child.value is already the average
        exploitation = child.value

        # Exploration term
        # (We add 1 to ensure no division by zero.)
        exploration = c * math.sqrt(
            math.log(parent.visit_count + 1) / (child.visit_count + 1)
        )

        return exploitation + exploration

    def _get_depth(self, node: TreeNode) -> int:
        """
        Get the depth of `node` by walking upward.
        """
        depth = 0
        current = node
        while current.parent is not None:
            current = current.parent
            depth += 1
        return depth

    def _collect_path_thoughts(self, node: TreeNode) -> List[str]:
        """
        Collect all node texts from root to this node (exclusive),
        so we can feed them as context to the LLM for next step generation.
        """
        path_texts = []
        current = node
        while current is not None:
            path_texts.append((current.thought_text, current.image))
            current = current.parent
        path_texts.reverse()
        return path_texts

    def _format_search_output(
        self,
        input_query: str,
        image_path: str,
        true_answer: str,
        system_prompt: str,
        root: TreeNode,
        # path_nodes: List[TreeNode],
        judge_score: float,
        global_search_time: float
    ) -> dict:
        """
        Formats the final rollout (or path) into a dictionary, similar to original code.
        The last node in `path_nodes` may be the newly expanded or final node.
        """

        tree = serialize_tree(root)

        return {
            "question": input_query,
            "image": image_path,
            "true_answer": true_answer,
            "tree": tree,
            "system_prompt": system_prompt,
            "global_search_time": global_search_time
        }

    # ### FLAG (1): Check if child references repeated coordinates
    def _is_coord_repeat(self, child_node: TreeNode, parent_node: TreeNode) -> bool:
        """
        If the newly parsed coords in the child’s text were already in parent's used_coords,
        return True (indicating we should skip or penalize).
        """
        newly_parsed = child_node._parse_coords(child_node.thought_text)
        # child_node.used_coords already includes parent's coords + newly parsed,
        # so to detect brand-new coords we can see if there's overlap:
        # But simpler is: If *any* of the newly parsed coords are in parent's used_coords,
        # we consider it a 'repeat.'
        for coord in newly_parsed:
            if coord in parent_node.used_coords:
                return True
        return False
