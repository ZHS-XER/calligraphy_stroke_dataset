from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
ROBUSTNESS_DATA_ROOT = ROOT_DIR / "data" / "robustness"
ROBUSTNESS_OUTPUT_ROOT = ROOT_DIR / "experiment-data" / "experiments" / "robustness_dataset" / "data_analysis"

def load_all_samples(dataset_dir: Path) -> Dict[str, List[Dict]]:

    samples_by_model: Dict[str, List[Dict]] = defaultdict(list)
    recognition_files = list(dataset_dir.glob("*/recognition_*/*.json"))
    if not recognition_files:
        return {}

    for json_file in recognition_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"⚠️  Failed to read: {json_file} -> {exc}")
            continue

        model_name = json_file.parent.name
        character = data.get("character")
        remaining_strokes = data.get("remaining_strokes")
        runs = data.get("runs", [])
        if remaining_strokes is None:
            continue

        for run in runs:
            llm_output = run.get("llm_output")
            if not llm_output or not isinstance(llm_output, dict):
                continue
            sample = {
                "character": character,
                "remaining_strokes": remaining_strokes,
                "answer": llm_output.get("answer"),
                "probabilities": llm_output.get("probabilities", []),
                "run_id": run.get("run_id"),
            }
            samples_by_model[model_name].append(sample)

    return dict(samples_by_model)

def calculate_accuracy(samples: List[Dict]) -> float:
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s["answer"] == s["character"])
    return correct / len(samples)

def calculate_top3_accuracy(samples: List[Dict]) -> float:
    if not samples:
        return 0.0
    correct = 0
    for sample in samples:
        probs = [p for p in (sample.get("probabilities") or []) if isinstance(p, dict)]
        top3 = [p.get("char") for p in probs[:3] if p.get("char") is not None]
        if sample["character"] in top3:
            correct += 1
    return correct / len(samples)

def calculate_top1_diff(samples: List[Dict]) -> float:
    if not samples:
        return 0.0
    diffs: List[float] = []
    for sample in samples:
        probs = [p for p in (sample.get("probabilities") or []) if isinstance(p, dict)]
        if not probs:
            continue
        is_correct = sample["answer"] == sample["character"]
        if is_correct:
            diffs.append(0.0)
            continue
        top1_prob = float(probs[0].get("p") or 0.0)
        truth_prob = 0.0
        for prob_item in probs:
            if prob_item.get("char") == sample["character"]:
                truth_prob = float(prob_item.get("p") or 0.0)
                break
        diffs.append(top1_prob - truth_prob)
    return float(np.mean(diffs)) if diffs else 0.0

def calculate_ece(samples: List[Dict], num_bins: int = 10) -> float:
    if not samples:
        return 0.0

    confidences: List[float] = []
    correctness: List[float] = []
    for sample in samples:
        probs = [p for p in (sample.get("probabilities") or []) if isinstance(p, dict)]
        if not probs:
            continue
        top1_prob = float(probs[0].get("p") or 0.0)
        confidences.append(top1_prob)
        correctness.append(1.0 if sample["answer"] == sample["character"] else 0.0)

    if not confidences:
        return 0.0

    confidences = np.array(confidences)
    correctness = np.array(correctness)
    bin_bounds = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(num_bins):
        lower = bin_bounds[i]
        upper = bin_bounds[i + 1]
        if i == num_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        bin_size = np.sum(in_bin)
        if bin_size == 0:
            continue
        acc_bin = np.mean(correctness[in_bin])
        conf_bin = np.mean(confidences[in_bin])
        ece += abs(acc_bin - conf_bin) * (bin_size / total)

    return float(ece)

def compute_metrics(samples: List[Dict]) -> Dict[str, float]:
    return {
        "Acc": round(calculate_accuracy(samples), 3),
        "Top3": round(calculate_top3_accuracy(samples), 3),
        "ECE": round(calculate_ece(samples), 3),
        "Top1Diff": round(calculate_top1_diff(samples), 3),
        "Count": len(samples),
    }

def group_by_remaining(samples: List[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for sample in samples:
        remaining = sample.get("remaining_strokes")
        if remaining in (0, None):
            continue
        grouped[int(remaining)].append(sample)
    return grouped

def analyze_dataset(dataset_path: Path, output_dir: Path) -> None:
    print(f"\n=== Dataset: {dataset_path} ===")
    samples_by_model = load_all_samples(dataset_path)
    if not samples_by_model:
        print("⚠️  No recognition_* JSON found, skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples: List[Dict] = []

    for model_name in sorted(samples_by_model.keys()):
        model_samples = samples_by_model[model_name]
        grouped = group_by_remaining(model_samples)
        model_result = {"by_remaining_strokes": {}, "overall": {}}

        print(f"\n🧠 Model: {model_name}")
        print(f"{'Remain':>6} | {'Acc':>6} | {'Top3':>6} | {'ECE':>6} | {'T1Diff':>7} | {'Count':>6}")
        print("-" * 60)

        for remaining in sorted(grouped.keys()):
            metrics = compute_metrics(grouped[remaining])
            model_result["by_remaining_strokes"][str(remaining)] = metrics
            print(
                f"{remaining:>6} | "
                f"{metrics['Acc']:>6.3f} | {metrics['Top3']:>6.3f} | "
                f"{metrics['ECE']:>6.3f} | {metrics['Top1Diff']:>7.3f} | {metrics['Count']:>6}"
            )

        overall_samples = [
            s for s in model_samples if s.get("remaining_strokes") not in (None, 0)
        ]
        overall_metrics = compute_metrics(overall_samples)
        model_result["overall"] = overall_metrics
        print("-" * 60)
        print(
            f"{'All':>6} | "
            f"{overall_metrics['Acc']:>6.3f} | {overall_metrics['Top3']:>6.3f} | "
            f"{overall_metrics['ECE']:>6.3f} | {overall_metrics['Top1Diff']:>7.3f} | {overall_metrics['Count']:>6}"
        )

        model_file = output_dir / f"{model_name}.json"
        with open(model_file, "w", encoding="utf-8") as f:
            json.dump(model_result, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved: {model_file}")

        all_samples.extend(overall_samples)

    if all_samples:
        overall_metrics = compute_metrics(all_samples)
        summary = {
            "total_samples": len(all_samples),
            "overall_metrics": overall_metrics,
        }
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n📦 Summary saved: {summary_file}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze robustness datasets' recognition outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROBUSTNESS_DATA_ROOT,
        help="Robustness dataset root (default: data/robustness).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["1_dataset", "red_dataset", "green_dataset"],
        help="List of subdirectory names to analyze.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROBUSTNESS_OUTPUT_ROOT,
        help="Output root directory (default: experiment-data/experiments/robustness_dataset/data_analysis).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root_dir = args.root if args.root.is_absolute() else (ROOT_DIR / args.root).resolve()
    output_root = args.output if args.output.is_absolute() else (ROOT_DIR / args.output).resolve()
    for dataset_name in args.datasets:
        dataset_path = (root_dir / dataset_name).resolve()
        if not dataset_path.exists():
            print(f"\n⚠️  Dataset does not exist: {dataset_path}. Skipping.")
            continue
        dataset_output = (output_root / dataset_name).resolve()
        dataset_output.mkdir(parents=True, exist_ok=True)
        analyze_dataset(dataset_path, dataset_output)

if __name__ == "__main__":
    main()
