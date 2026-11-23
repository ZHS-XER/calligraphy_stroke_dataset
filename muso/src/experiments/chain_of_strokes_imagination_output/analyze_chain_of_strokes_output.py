from __future__ import annotations

import argparse
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

TARGET_REMAINING_STROKES = {3}
DIR_PREFIX = "chain_of_strokes_imagination_output"

def validate_llm_output(llm_output: Optional[dict]) -> bool:
    if not llm_output or not isinstance(llm_output, dict):
        return False
    answer = llm_output.get("answer")
    probabilities = llm_output.get("probabilities")
    if not answer or not isinstance(answer, str):
        return False
    if not probabilities or not isinstance(probabilities, list):
        return False
    return True

def normalize_probabilities(raw_probabilities: Iterable[dict]) -> List[dict]:
    cleaned: List[dict] = []
    for item in raw_probabilities or []:
        if not isinstance(item, dict):
            continue
        cleaned.append({"char": item.get("char"), "p": item.get("p")})
    return cleaned

def extract_samples_from_file(json_path: Path) -> List[dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"⚠️  Failed to parse: {json_path} -> {exc}")
        return []

    remaining = data.get("remaining_strokes")
    if remaining is None:
        return []
    remaining = int(remaining)

    if remaining not in TARGET_REMAINING_STROKES:
        return []

    base = {
        "character": data.get("character"),
        "remaining_strokes": remaining,
        "stroke_name": data.get("stroke_name"),
        "total_strokes": data.get("total_strokes"),
        "file": str(json_path),
    }

    samples: List[dict] = []

    runs = data.get("runs")
    if isinstance(runs, list) and runs:
        for run in runs:
            llm_output = run.get("llm_output")
            if not validate_llm_output(llm_output):
                continue
            sample = base.copy()
            sample.update(
                {
                    "answer": llm_output.get("answer"),
                    "probabilities": normalize_probabilities(llm_output.get("probabilities")),
                    "run_id": run.get("run_id"),
                }
            )
            samples.append(sample)
    else:
        llm_output = data.get("llm_output")
        if validate_llm_output(llm_output):
            sample = base.copy()
            sample.update(
                {
                    "answer": llm_output.get("answer"),
                    "probabilities": normalize_probabilities(llm_output.get("probabilities")),
                    "run_id": data.get("run_id"),
                }
            )
            samples.append(sample)

    return samples

def load_all_samples(dataset_dir: Path) -> Dict[str, List[dict]]:
    samples_by_model: Dict[str, List[dict]] = defaultdict(list)

    characters = [p for p in dataset_dir.iterdir() if p.is_dir()]
    print(f"📁 Found {len(characters)} character directories")

    for char_dir in characters:
        for model_dir in char_dir.glob(f"{DIR_PREFIX}*"):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for json_path in model_dir.glob("*.json"):
                samples = extract_samples_from_file(json_path)
                if not samples:
                    continue
                samples_by_model[model_name].extend(samples)

    total_samples = sum(len(v) for v in samples_by_model.values())
    print(f"✅ Collected {total_samples} samples across {len(samples_by_model)} model directories")
    return dict(samples_by_model)

def calculate_accuracy(samples: List[dict]) -> float:
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s.get("answer") == s.get("character"))
    return correct / len(samples)

def calculate_top3_accuracy(samples: List[dict]) -> float:
    if not samples:
        return 0.0
    correct = 0
    for sample in samples:
        probabilities = [p for p in sample.get("probabilities") if isinstance(p, dict)]
        top3 = [p.get("char") for p in probabilities[:3] if p.get("char") is not None]
        if sample.get("character") in top3:
            correct += 1
    return correct / len(samples)

def calculate_top1_diff(samples: List[dict]) -> float:
    if not samples:
        return 0.0
    diffs: List[float] = []
    for sample in samples:
        probabilities = [p for p in sample.get("probabilities") if isinstance(p, dict)]
        if not probabilities:
            continue

        if sample.get("answer") == sample.get("character"):
            diffs.append(0.0)
            continue

        top1_prob = probabilities[0].get("p")
        top1_prob = float(top1_prob) if top1_prob is not None else 0.0

        true_prob = 0.0
        for prob_item in probabilities:
            if prob_item.get("char") == sample.get("character"):
                true_prob = float(prob_item.get("p") or 0.0)
                break
        diffs.append(top1_prob - true_prob)

    if not diffs:
        return 0.0
    return float(np.mean(diffs))

def calculate_ece(samples: List[dict], num_bins: int = 10) -> float:
    if not samples:
        return 0.0

    confidences: List[float] = []
    correctness: List[float] = []

    for sample in samples:
        probabilities = [p for p in sample.get("probabilities") if isinstance(p, dict)]
        if not probabilities:
            continue
        top1_prob = probabilities[0].get("p")
        top1_prob = float(top1_prob) if top1_prob is not None else 0.0
        confidences.append(top1_prob)
        correctness.append(1.0 if sample.get("answer") == sample.get("character") else 0.0)

    if not confidences:
        return 0.0

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (confidences >= lower) & (confidences < upper if i < num_bins - 1 else confidences <= upper)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        avg_conf = float(np.mean(confidences[indices]))
        avg_acc = float(np.mean(correctness[indices]))
        ece += (len(indices) / total) * abs(avg_conf - avg_acc)
    return float(ece)

def summarize_model(samples: List[dict]) -> Dict[str, float]:
    return OrderedDict(
        {
            "count": len(samples),
            "acc": calculate_accuracy(samples),
            "top3": calculate_top3_accuracy(samples),
            "ece": calculate_ece(samples),
            "top1_diff": calculate_top1_diff(samples),
        }
    )

def main():
    parser = argparse.ArgumentParser(description="Analyze imagination Chain-of-Strokes outputs (remaining=3).")
    parser.add_argument(
        "-d", "--dataset",
        default="dataset",
        help="Dataset root that contains per-character folders (default: dataset)",
    )
    parser.add_argument(
        "-o", "--output",
        default="chain_of_strokes_imagination_output/data_analysis/summary.json",
        help="Path to save the aggregated summary JSON.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"❌ Dataset directory does not exist: {dataset_dir}")

    samples_by_model = load_all_samples(dataset_dir)
    if not samples_by_model:
        raise SystemExit("⚠️ No matching samples found")

    summary = OrderedDict()
    for model_name, samples in sorted(samples_by_model.items()):
        summary[model_name] = summarize_model(samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"✅ Analysis complete, results saved to {output_path}")

if __name__ == "__main__":
    main()
