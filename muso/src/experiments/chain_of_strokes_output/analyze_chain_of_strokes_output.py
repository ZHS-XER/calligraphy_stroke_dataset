from __future__ import annotations

import argparse
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

TARGET_REMAINING_STROKES = {1, 2, 3}
DIR_PREFIX = "chain_of_strokes_output"

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

def compute_metrics(samples: List[dict]) -> Dict[str, float]:
    return {
        "Acc": round(calculate_accuracy(samples), 3),
        "Top3Acc": round(calculate_top3_accuracy(samples), 3),
        "ECE": round(calculate_ece(samples), 3),
        "Top1Diff": round(calculate_top1_diff(samples), 3),
        "Count": len(samples),
    }

def analyze_model(model_name: str, samples: List[dict]) -> Tuple[dict, Dict[int, List[dict]]]:
    print(f"\n🧠 Model directory: {model_name}")
    filtered = [s for s in samples if s.get("remaining_strokes") in TARGET_REMAINING_STROKES]

    grouped: Dict[int, List[dict]] = defaultdict(list)
    for sample in filtered:
        grouped[sample["remaining_strokes"]].append(sample)

    print(f"{'Remain':>6} | {'Acc':>6} | {'Top3':>6} | {'ECE':>6} | {'T1Diff':>7} | {'Count':>6}")
    print("-" * 60)

    model_result: Dict[int, dict] = {}
    for remaining in sorted(TARGET_REMAINING_STROKES):
        metrics = compute_metrics(grouped.get(remaining, []))
        model_result[remaining] = metrics
        print(
            f"{remaining:>6} | "
            f"{metrics['Acc']:>6.3f} | {metrics['Top3Acc']:>6.3f} | "
            f"{metrics['ECE']:>6.3f} | {metrics['Top1Diff']:>7.3f} | {metrics['Count']:>6}"
        )

    overall = compute_metrics(filtered)
    print("-" * 60)
    print(
        f"{'All':>6} | "
        f"{overall['Acc']:>6.3f} | {overall['Top3Acc']:>6.3f} | "
        f"{overall['ECE']:>6.3f} | {overall['Top1Diff']:>7.3f} | {overall['Count']:>6}"
    )

    return {"by_remaining_strokes": model_result, "overall": overall}, grouped

def save_model_results(output_dir: Path, model_name: str, model_result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_result, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved: {output_path}")

def save_summary(output_dir: Path, aggregated: Dict[int, List[dict]]) -> None:
    ordered_summary: "OrderedDict[str, dict]" = OrderedDict()
    for remaining in sorted(TARGET_REMAINING_STROKES):
        ordered_summary[str(remaining)] = compute_metrics(aggregated.get(remaining, []))

    output_path = output_dir / "summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ordered_summary, f, ensure_ascii=False, indent=2)
    print(f"\n📦 Summary saved: {output_path}")

def analyze_chain_of_strokes(dataset_dir: Path, output_dir: Path) -> None:
    samples_by_model = load_all_samples(dataset_dir)
    if not samples_by_model:
        print("❌ No matching recognition results found")
        return

    aggregated: Dict[int, List[dict]] = defaultdict(list)

    for model_name in sorted(samples_by_model.keys()):
        model_samples = samples_by_model[model_name]
        model_result, grouped = analyze_model(model_name, model_samples)
        save_model_results(output_dir, model_name, model_result)

        for remaining, sample_list in grouped.items():
            aggregated[remaining].extend(sample_list)

    print("\n" + "=" * 60)
    print("📈 Cross-model overall stats")
    print("=" * 60)
    print(f"{'Remain':>6} | {'Acc':>6} | {'Top3':>6} | {'ECE':>6} | {'T1Diff':>7} | {'Count':>6}")
    print("-" * 60)

    all_samples: List[dict] = []
    for remaining in sorted(TARGET_REMAINING_STROKES):
        metrics = compute_metrics(aggregated.get(remaining, []))
        all_samples.extend(aggregated.get(remaining, []))
        print(
            f"{remaining:>6} | "
            f"{metrics['Acc']:>6.3f} | {metrics['Top3Acc']:>6.3f} | "
            f"{metrics['ECE']:>6.3f} | {metrics['Top1Diff']:>7.3f} | {metrics['Count']:>6}"
        )

    overall = compute_metrics(all_samples)
    print("-" * 60)
    print(
        f"{'All':>6} | "
        f"{overall['Acc']:>6.3f} | {overall['Top3Acc']:>6.3f} | "
        f"{overall['ECE']:>6.3f} | {overall['Top1Diff']:>7.3f} | {overall['Count']:>6}"
    )

    save_summary(output_dir, aggregated)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Chain-of-Strokes output recognition results.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "../dataset",
        help="Dataset directory (default: ../dataset relative to this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data_analysis",
        help="Output directory (default: ./data_analysis)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    analyze_chain_of_strokes(args.dataset.resolve(), args.output.resolve())
