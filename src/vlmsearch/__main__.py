import multiprocessing
from functools import partial
import logging
import os
import json
import datetime
import traceback
import random
import shutil
import copy
import time
from PIL import Image
from tqdm import tqdm
import openai

from vlmsearch.arguments import get_args
from vlmsearch.models import get_model
from vlmsearch.tree_search.single_path_rollouts import SinglePathRollouts
from vlmsearch.tree_search.mcts_search import MonteCarloTreeSearch
from vlmsearch.reward_funcs.judge import Judge
from datasets import load_dataset
from vlmsearch.data_manifest import FrozenDataManifest, attach_manifest, tag_rollout
from vlmsearch.data_review import ReviewedTrainingData, attach_review
from vlmsearch.reward_funcs.answers import parse_final_answer

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
print(f"Logging level: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")


def model_metrics(model):
    return {key: value for key, value in getattr(model, "metrics", {}).items()
            if isinstance(value, (int, float))}


def tag_model_output(output, sample, model, started, before):
    tag_rollout(output, sample)
    output["search_elapsed_seconds"] = time.perf_counter() - started
    output["model_diagnostics"] = {key: value - before.get(key, 0)
                                   for key, value in model_metrics(model).items()}
    policy = getattr(model, "generation_policy", None)
    if policy:
        output["generation_policy"] = copy.deepcopy(policy)
    geometry = getattr(model, "last_image_geometry", None)
    if geometry:
        with Image.open(sample["image"]) as image:
            if list(image.size) != geometry.get("original_size"):
                raise ValueError("Teacher image geometry does not match the source image")
        output["image_geometry"] = copy.deepcopy(geometry)
    return output


def rollout_diagnostics(output):
    """Distinguish successful API calls from usable final-answer trajectories."""
    expected = str(output.get("true_answer", "")).strip().casefold()
    allowed = {"yes", "no"} if expected in {"yes", "no"} else {"a", "b", "c", "d"} if expected in {"a", "b", "c", "d"} else None
    trajectories = []
    def visit(node):
        trajectories.extend(node.get("rollouts", []))
        for child in node.get("children", []):
            visit(child)
    if output.get("tree"):
        visit(output["tree"])
    else:
        trajectories.append(output)
    return {
        "trajectories": len(trajectories),
        "valid_final_answers": sum(parse_final_answer(row.get("raw_final_answer") or row.get("final_answer"), allowed) is not None for row in trajectories),
        "positive_trajectories": sum(row.get("reward", row.get("judge_score", 0)) > 0 for row in trajectories),
    }

def init_model_and_judge(args):
    """
    Initialize the model, judge, and tree_searcher objects for each worker.
    """

    # actor model
    ModelClass = get_model(args.model)
    model_wrapper = ModelClass(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        examples=args.examples,
        thought_token_begin=args.thought_token_begin,
        thought_token_end=args.thought_token_end,
        final_token_begin=args.final_token_begin,
        final_token_end=args.final_token_end,
        pretrained=args.pretrained,
        first_rollout_no_sample=args.first_rollout_no_sample,
        multicrop=args.multicrop,
        repetition_penalty=args.repetition_penalty,
    )

    # judge
    judge = Judge(
        judge_llm_wrapper=model_wrapper,
        wandb_project="vlm-search",
        use_wandb=False,
        judge=args.judge,
        point_matching_threshold=args.point_matching_threshold,
        thought_token_begin=args.thought_token_begin,
        thought_token_end=args.thought_token_end,
        final_token_begin=args.final_token_begin,
        final_token_end=args.final_token_end,
    )

    # System prompt for actor model (either from args or default file)
    if args.system_prompt:
        system_prompt = args.system_prompt
    else:
        with open("src/vlmsearch/models/system_prompt.txt", "r") as f:
            system_prompt = f.read()
    logging.info(f"System prompt: {system_prompt}")

    # Choose search method
    if args.search_method == "single_path_rollouts":
        tree_searcher = SinglePathRollouts(
            llm_wrapper=model_wrapper,
            judge=judge,
            system_prompt=system_prompt,
            max_depth=args.max_depth,
            n_rollouts=args.n_rollouts,
            add_thought_number_system_prompt=args.add_thought_number_system_prompt,
            generate_cold_start=args.generate_cold_start,
            generate_upfront=args.generate_upfront,
            rollout_no_thinking=args.rollout_no_thinking,
            thought_token_begin=args.thought_token_begin,
            thought_token_end=args.thought_token_end,
            final_token_begin=args.final_token_begin,
            final_token_end=args.final_token_end,
            max_image_side=args.max_image_side,
            max_pixels=args.max_pixels,
            check_for_crop=args.check_for_crop,
            crop_offset=args.crop_offset,
            crop_size=args.crop_size,
            draw_dot=args.draw_dot,
            first_rollout_no_sample=args.first_rollout_no_sample,
        )
    elif args.search_method == "mcts":
        tree_searcher = MonteCarloTreeSearch(
            llm_wrapper=model_wrapper,
            judge=judge,
            system_prompt=system_prompt,
            max_depth=args.max_depth,
            c_puct=args.c_puct,
            n_simulations=args.n_simulations,
            rollout_max_steps=args.rollout_max_steps,
            add_thought_number_system_prompt=args.add_thought_number_system_prompt,
            generate_cold_start=args.generate_cold_start,
            generate_upfront=args.generate_upfront,
            rollout_no_thinking=args.rollout_no_thinking,
            first_rollout_no_sample=args.first_rollout_no_sample,
            n_rollouts_per_node=args.n_rollouts_per_node,
            num_children_per_expand=args.num_children_per_expand,
            thought_token_begin=args.thought_token_begin,
            thought_token_end=args.thought_token_end,
            final_token_begin=args.final_token_begin,
            final_token_end=args.final_token_end,
        )
    else:
        raise ValueError(f"Invalid search method: {args.search_method}")

    return model_wrapper, judge, tree_searcher

def process_samples(
    samples, 
    args, 
    save_tag,
    out_dir,
    worker_id=0,
    loaded_ids=None
    ):
    """
    Worker function to process a shard of samples.

    This function:
      - Initializes local model/judge/searcher
      - Iterates over its assigned samples, collecting rollouts
      - Every `checkpoint_interval` samples, it saves results to a JSONL
        (named with the worker_id) and clears them
      - Saves any leftover results at the end
      - Returns None or an empty list to avoid duplicates
    """
    model_wrapper, judge, tree_searcher = init_model_and_judge(args)

    results = []
    skipped_samples = []
    diagnostics = {"trajectories": 0, "valid_final_answers": 0, "positive_trajectories": 0}
    checkpoint_interval = getattr(args, "checkpoint_interval", 100)
    timeout_seconds = getattr(args, "timeout_seconds", 300)  # Default 5 minutes if not specified

    # TQDM so we see local progress for this worker
    for idx, sample in enumerate(
        tqdm(samples, desc=f"Worker {worker_id} PID={os.getpid()}", position=worker_id, leave=True)
    ):
        try:
            if loaded_ids and sample["id"] in loaded_ids:
                continue

            sample["input_query"] = sample["conversations"]["value"][0]
            sample["true_answer"] = sample["conversations"]["value"][1]
            started = time.perf_counter()
            before = model_metrics(model_wrapper)

            # Execute search with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                search_future = executor.submit(
                    tree_searcher.search,
                    input_query=sample["input_query"],
                    input_image_path=sample["image"],
                    true_answer=sample["true_answer"],
                    worker_id=worker_id,
                )
                
                try:
                    search_outputs = search_future.result(timeout=timeout_seconds)
                    if not search_outputs:
                        raise RuntimeError("Search returned no rollout records")
                    # Tag each output
                    for out in search_outputs:
                        tag_model_output(out, sample, model_wrapper, started, before)
                        for key, value in rollout_diagnostics(out).items():
                            diagnostics[key] += value
                    results.extend(search_outputs)
                except TimeoutError:
                    error_info = {
                        "id": sample["id"], 
                        "error": "timeout", 
                        "timeout_seconds": timeout_seconds,
                        "idx": idx
                    }
                    skipped_samples.append(error_info)
                    logging.error(f"Worker {worker_id}: Timeout for sample id={sample['id']} after {timeout_seconds} seconds")
                    continue

            # Checkpoint if needed
            if (idx + 1) % checkpoint_interval == 0:
                checkpoint_name = f"rollouts_{save_tag}_worker{worker_id}_ckpt{idx+1}.jsonl"
                checkpoint_path = os.path.join(out_dir, checkpoint_name)
                logging.debug(f"Worker {worker_id}: saving partial results to {checkpoint_path}")

                with open(checkpoint_path, "w") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                
                # Also save error log periodically
                if skipped_samples:
                    error_log_path = os.path.join(out_dir, f"errors_{save_tag}_worker{worker_id}_ckpt{idx+1}.jsonl")
                    with open(error_log_path, "w") as f:
                        for err in skipped_samples:
                            f.write(json.dumps(err) + "\n")

                results.clear()

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            error_info = {
                "id": sample["id"], 
                "error": "exception", 
                "error_message": error_msg,
                "idx": idx
            }
            skipped_samples.append(error_info)
            logging.error(f"Error in worker {worker_id} on sample id={sample['id']} idx={idx}: {e}")

    # After the loop, save any remaining results
    if results:
        final_ckpt_name = f"rollouts_{save_tag}_worker{worker_id}_final.jsonl"
        final_ckpt_path = os.path.join(out_dir, final_ckpt_name)
        logging.info(f"Worker {worker_id}: saving final {len(results)} results to {final_ckpt_path}")
        with open(final_ckpt_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    
    # Save final error log
    if skipped_samples:
        error_log_path = os.path.join(out_dir, f"errors_{save_tag}_worker{worker_id}_final.jsonl")
        logging.info(f"Worker {worker_id}: saving {len(skipped_samples)} error logs to {error_log_path}")
        with open(error_log_path, "w") as f:
            for err in skipped_samples:
                f.write(json.dumps(err) + "\n")

    # Return None or an empty list so that the main process won't gather duplicates
    return {"samples": len(samples), "errors": len(skipped_samples), **diagnostics}


def main():
    args = get_args()

    # We'll form a directory for each entire run. We'll store partial files
    # for each worker in that same directory, but with worker_id in the filename.
    save_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.save_tag:
        save_tag = f"{args.save_tag}_{save_tag}"

    out_dir = os.path.join(args.save_rollouts_dir, save_tag)
    # We want each worker to create the directory if not present
    # (in race conditions, you might want a safer approach, but for demonstration:
    os.makedirs(out_dir, exist_ok=False)

    # save args to out_dir
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)

    data_files_list = args.data_files.split(',')
    logging.info(f"Loading data from files: {data_files_list}")

    manifest = FrozenDataManifest(args.split_manifest, args.image_root) if args.split_manifest else None
    if manifest:
        shutil.copy2(manifest.path, os.path.join(out_dir, "split_manifest.json"))
    if args.review_overlay and not manifest:
        raise ValueError("A training review requires a frozen split manifest")
    review = ReviewedTrainingData(args.review_overlay, manifest) if args.review_overlay else None
    if review:
        shutil.copy2(review.path, os.path.join(out_dir, "training_review.json"))

    if args.use_python_mp:
        #--- Multiprocessing path ---
        raw_dataset = load_dataset(
            "src/vlmsearch/datasets/data_loader.py",
            data_files={"train": data_files_list},
            image_root=args.image_root,
            split="train",
            trust_remote_code=True
        )

        data_list = list(raw_dataset)
        if manifest:
            attach_manifest(data_list, manifest, args.data_split)
        if review:
            original_count = len(data_list)
            data_list = attach_review(data_list, review)
            logging.info("Training review admitted %s/%s questions", len(data_list), original_count)
            if not data_list:
                raise ValueError("Training review excluded every input sample")
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise ValueError("max_samples must be positive")
            random.Random(args.seed).shuffle(data_list)
            data_list = data_list[:args.max_samples]
        total_data_len = len(data_list)

        num_procs = args.num_processes if args.num_processes > 1 else 1
        

        # Split into shards
        if args.load_folder:
            data_files = [os.path.join(args.load_folder, f) for f in os.listdir(args.load_folder) if f.endswith(".jsonl")]
            loaded_ids = set()
            for data_file in data_files:
                with open(data_file, "r") as f:
                    for line in f:
                        sample = json.loads(line)
                        if sample["id"] not in loaded_ids:
                            loaded_ids.add(sample["id"])
        else:
            loaded_ids = None

        if num_procs == 1:
            summaries = [process_samples(data_list, args, save_tag, out_dir, 0, loaded_ids)]
        else:
            chunk_size = (total_data_len + num_procs - 1) // num_procs
            shards = [
                data_list[i*chunk_size : (i+1)*chunk_size]
                for i in range(num_procs)
            ]
            pool = multiprocessing.Pool(processes=num_procs)
            try:
                # starmap so each shard gets (samples, args, worker_id)
                summaries = pool.starmap(
                    process_samples,
                    [(shards[i], args, save_tag, out_dir, i, loaded_ids) for i in range(num_procs)]
                )
            finally:
                pool.close()
                pool.join()

        logging.info("All worker processes completed. Checkpoint files were saved by each worker.")
        summary = {
            "samples": sum(item["samples"] for item in summaries),
            "errors": sum(item["errors"] for item in summaries),
            "split_manifest_sha256": manifest.sha256 if manifest else None,
            "data_split": args.data_split,
            "review_overlay_id": review.id if review else None,
            "review_overlay_sha256": review.sha256 if review else None,
            **{key: sum(item[key] for item in summaries) for key in ("trajectories", "valid_final_answers", "positive_trajectories")},
        }
        with open(os.path.join(out_dir, "run_report.json"), "w") as handle:
            json.dump(summary, handle, indent=2)
        if summary["errors"]:
            raise RuntimeError(f"Rollout run had {summary['errors']} errors; see {out_dir}")
        if manifest and summary["trajectories"] and not summary["valid_final_answers"]:
            raise RuntimeError(f"No valid final answers were produced; inspect teacher formatting in {out_dir}")

    else:
        #--- Accelerate-based approach (unchanged) ---
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs
        from vlmsearch.dist_utils import gather_accelerator_results

        kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
        accelerator = Accelerator(kwargs_handlers=[kwargs_handler])

        raw_dataset = load_dataset(
            "src/vlmsearch/datasets/data_loader.py",
            data_files={"train": args.data_files},
            image_root=args.image_root,
            split="train"
        )

        data_list = list(raw_dataset)
        if manifest:
            attach_manifest(data_list, manifest, args.data_split)
        if review:
            data_list = attach_review(data_list, review)
            if not data_list:
                raise ValueError("Training review excluded every input sample")
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise ValueError("max_samples must be positive")
            random.Random(args.seed).shuffle(data_list)
            data_list = data_list[:args.max_samples]
        dataset_shard = data_list[accelerator.process_index::accelerator.num_processes]

        model_wrapper, judge, tree_searcher = init_model_and_judge(args)

        results = []
        errors = []
        checkpoint_interval = getattr(args, "checkpoint_interval", 100)

        for idx, sample in enumerate(tqdm(dataset_shard, desc="Processing dataset_shard", disable=not accelerator.is_main_process)):
            sample["input_query"] = sample["conversations"]["value"][0]
            sample["true_answer"] = sample["conversations"]["value"][1]
            search_outputs = []
            started = time.perf_counter()
            before = model_metrics(model_wrapper)

            try:
                search_outputs = tree_searcher.search(
                    input_query=sample["input_query"],
                    input_image_path=sample["image"],
                    true_answer=sample["true_answer"],
                )
                if not search_outputs:
                    raise RuntimeError("Search returned no rollout records")
            except openai.OpenAIError as e:
                logging.error(f"OpenAIError: {e}")
                errors.append({"id": sample["id"], "error": str(e)})
            except Exception as e:
                traceback.print_exc()
                logging.error(f"Error in search: {e}")
                search_outputs = []
                errors.append({"id": sample["id"], "error": str(e)})

            for out in search_outputs:
                tag_model_output(out, sample, model_wrapper, started, before)
            results.extend(search_outputs)

            if (idx + 1) % checkpoint_interval == 0:
                gathered = gather_accelerator_results(results, accelerator)
                results.clear()
                if accelerator.is_main_process:
                    partial_file = os.path.join(out_dir, f"rollouts_{save_tag}_{idx+1}.jsonl")
                    with open(partial_file, "w") as f:
                        for r in gathered:
                            f.write(json.dumps(r) + "\n")
                    logging.info(f"Partial checkpoint saved to {partial_file}")

        # Final gather & save
        gathered = gather_accelerator_results(results, accelerator)
        results.clear()

        if accelerator.is_main_process:
            final_file = os.path.join(out_dir, f"rollouts_{save_tag}_final.jsonl")
            with open(final_file, "w") as f:
                for r in gathered:
                    f.write(json.dumps(r) + "\n")
            logging.info(f"Final checkpoint saved to {final_file}")
        report = {"samples": len(dataset_shard), "errors": errors,
                  "split_manifest_sha256": manifest.sha256 if manifest else None}
        with open(os.path.join(out_dir, f"run_report_rank{accelerator.process_index}.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        if errors:
            raise RuntimeError(f"Rollout rank {accelerator.process_index} had {len(errors)} errors; see {out_dir}")


if __name__ == "__main__":
    main()
