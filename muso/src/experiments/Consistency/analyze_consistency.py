from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TARGET_REMAINING = {1, 2, 3}
METHODS = ("SC_mean", "SC_max")

def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def normalize_run_distribution(run: Dict) -> List[Tuple[str, float]]:
    llm_output = (run or {}).get("llm_output") or {}
    answer = (llm_output.get("answer") or "").strip()
    entries: List[Tuple[str, float]] = []
    for item in llm_output.get("probabilities") or []:
        if not isinstance(item, dict):
            continue
        char = (item.get("char") or "").strip()
        if not char:
            continue
        prob = to_float(item.get("p"))
        if prob <= 0:
            continue
        entries.append((char, prob))

    total = sum(prob for _, prob in entries)
    if total > 0:
        return [(char, prob / total) for char, prob in entries]
    if answer:
        return [(answer, 1.0)]
    return []

def finalize_probabilities(items: List[Tuple[str, float]]) -> List[Dict[str, float]]:
    filtered: List[Tuple[str, float]] = []
    for char, prob in items:
        char = (char or "").strip()
        prob = to_float(prob)
        if not char or prob <= 0:
            continue
        filtered.append((char, prob))

    if not filtered:
        return []

    total = sum(prob for _, prob in filtered)
    if total <= 0:
        return []

    normalized = [(char, prob / total) for char, prob in filtered]
    normalized.sort(key=lambda x: (-x[1], x[0]))
    return [{"char": char, "p": float(prob)} for char, prob in normalized]

def build_sample_payload(data: Dict, probabilities: List[Dict[str, float]], method: str) -> Optional[Dict]:
    if not probabilities:
        return None

    remaining = data.get("remaining_strokes")
    character = (data.get("character") or "").strip()
    if character == "":
        return None

    try:
        remaining_int = int(remaining)
    except (TypeError, ValueError):
        return None

    answer = (probabilities[0].get("char") or "").strip()
    if not answer:
        return None

    return {
        "character": character,
        "remaining_strokes": remaining_int,
        "answer": answer,
        "probabilities": probabilities,
        "method": method,
    }

def build_sc_mean_sample(data: Dict) -> Optional[Dict]:
    runs = data.get("runs") or []
    distributions = []
    for run in runs:
        dist = normalize_run_distribution(run)
        if dist:
            distributions.append(dist)

    if not distributions:
        return None

    prob_map = defaultdict(float)
    for dist in distributions:
        for char, prob in dist:
            prob_map[char] += prob

    run_count = len(distributions)
    averaged = [(char, total / run_count) for char, total in prob_map.items()]
    probabilities = finalize_probabilities(averaged)
    return build_sample_payload(data, probabilities, "SC_mean")

def build_sc_max_sample(data: Dict) -> Optional[Dict]:
    runs = data.get("runs") or []
    answers: List[str] = []
    for run in runs:
        llm_output = (run or {}).get("llm_output") or {}
        answer = (llm_output.get("answer") or "").strip()
        if answer:
            answers.append(answer)

    if not answers:
        return None

    counts = Counter(answers)
    total = len(answers)
    prob_items = [(char, count / total) for char, count in counts.items()]
    probabilities = finalize_probabilities(prob_items)
    return build_sample_payload(data, probabilities, "SC_max")

def collect_samples(dataset_dir: Path) -> Dict[str, Dict[str, List[Dict]]]:
    samples: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    pattern = dataset_dir.glob("*/recognition_consistency_*/*.json")
    files = list(pattern)
    print(f"📂 Found {len(files)} consistency result files")

    builders = {
        "SC_mean": build_sc_mean_sample,
        "SC_max": build_sc_max_sample,
    }

    for json_file in files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"⚠️  Failed to read {json_file}: {exc}")
            continue

        model_name = json_file.parent.name
        for method, builder in builders.items():
            sample = builder(data)
            if sample and sample["remaining_strokes"] in TARGET_REMAINING:
                samples[model_name][method].append(sample)

    return samples

def collect_from_recognition_1(dataset_dir: Path) -> Dict[str, List[Dict]]:

    result: Dict[str, List[Dict]] = {"SC_mean": [], "SC_max": []}
    files = list(dataset_dir.glob("*/recognition_1/*.json"))
    print(f"📂 gemini-2.5-pro recognition_1 files: {len(files)}")

    for json_file in files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"⚠️  Failed to read recognition_1 {json_file}: {exc}")
            continue

        sm = build_sc_mean_sample(data)
        sx = build_sc_max_sample(data)
        if sm and sm["remaining_strokes"] in TARGET_REMAINING:
            result["SC_mean"].append(sm)
        if sx and sx["remaining_strokes"] in TARGET_REMAINING:
            result["SC_max"].append(sx)

    return result

def calculate_accuracy(samples: List[Dict]) -> float:
    if not samples:
        return 0.0
    hits = sum(1 for sample in samples if sample.get("answer") == sample.get("character"))
    return hits / len(samples)

def calculate_top3_accuracy(samples: List[Dict]) -> float:
    if not samples:
        return 0.0
    hits = 0
    for sample in samples:
        probabilities = sample.get("probabilities") or []
        top3 = [
            (item.get("char") or "").strip()
            for item in probabilities[:3]
            if isinstance(item, dict)
        ]
        if sample.get("character") in top3:
            hits += 1
    return hits / len(samples)

def calculate_top1_diff(samples: List[Dict]) -> float:
    diffs: List[float] = []
    for sample in samples:
        probabilities = [p for p in sample.get("probabilities") if isinstance(p, dict)]
        if not probabilities:
            continue

        answer = (sample.get("answer") or "").strip()
        character = sample.get("character")
        if answer == character:
            diffs.append(0.0)
            continue

        top1_prob = to_float(probabilities[0].get("p"))
        true_prob = 0.0
        for prob_item in probabilities:
            if prob_item.get("char") == character:
                true_prob = to_float(prob_item.get("p"))
                break
        diffs.append(top1_prob - true_prob)

    if not diffs:
        return 0.0
    return float(np.mean(diffs))

def calculate_ece(samples: List[Dict], num_bins: int = 10) -> float:
    if not samples:
        return 0.0

    confidences: List[float] = []
    correctness: List[float] = []

    for sample in samples:
        probabilities = [p for p in sample.get("probabilities") if isinstance(p, dict)]
        if not probabilities:
            continue
        top1_prob = to_float(probabilities[0].get("p"))
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

def compute_metrics(samples: List[Dict]) -> Dict[str, float]:
    return {
        "Acc": round(calculate_accuracy(samples), 3),
        "Top3Acc": round(calculate_top3_accuracy(samples), 3),
        "ECE": round(calculate_ece(samples), 3),
        "Top1Diff": round(calculate_top1_diff(samples), 3),
        "Count": len(samples),
    }

def summarize_method(samples: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for sample in samples:
        remaining = sample.get("remaining_strokes")
        if remaining in TARGET_REMAINING:
            grouped[int(remaining)].append(sample)

    summary = {str(r): compute_metrics(grouped.get(r, [])) for r in sorted(TARGET_REMAINING)}
    return summary

def analyze_consistency(dataset_dir: str = "../dataset",
                        output_dir: str = "data_analysis"):
    base_dir = Path(__file__).parent
    dataset_path = (base_dir / dataset_dir).resolve()
    output_path = base_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("📊 Self-Consistency metrics (SC mean / SC max)")
    print(f"{'='*60}")
    print(f"Dataset dir: {dataset_path}")
    print(f"Output dir: {output_path}")
    print(f"{'='*60}\n")

    samples_by_model = collect_samples(dataset_path)

    pro_from_rec1 = collect_from_recognition_1(dataset_path)
    if any(pro_from_rec1.values()):
        samples_by_model["recognition_consistency_gemini-2.5-pro"] = pro_from_rec1
    if not samples_by_model:
        print("❌ No consistency samples found")
        return

    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for model_name in sorted(samples_by_model.keys()):
        print(f"\n🧠 Model: {model_name}")
        model_results: Dict[str, Dict[str, Dict[str, float]]] = {}

        for method in METHODS:
            method_samples = samples_by_model[model_name].get(method, [])
            print(f"\n[{method}]")
            print(f"{'Remain':>6} | {'Acc':>6} | {'Top3':>6} | {'ECE':>6} | {'T1Diff':>7} | {'Count':>6}")
            print("-" * 60)

            summary = summarize_method(method_samples)
            for remaining in sorted(TARGET_REMAINING):
                metrics = summary[str(remaining)]
                print(
                    f"{remaining:>6} | "
                    f"{metrics['Acc']:>6.3f} | {metrics['Top3Acc']:>6.3f} | "
                    f"{metrics['ECE']:>6.3f} | {metrics['Top1Diff']:>7.3f} | {metrics['Count']:>6}"
                )

            overall = compute_metrics(method_samples)
            print("-" * 60)
            print(
                f"{'All':>6} | "
                f"{overall['Acc']:>6.3f} | {overall['Top3Acc']:>6.3f} | "
                f"{overall['ECE']:>6.3f} | {overall['Top1Diff']:>7.3f} | {overall['Count']:>6}"
            )

            model_results[method] = {
                "by_remaining": summary,
                "overall": overall,
            }

        results[model_name] = model_results

    output_file = output_path / "consistency_metrics.json"
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Results saved to: {output_file}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute self-consistency SC mean / SC max metrics"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="../dataset",
        help="Dataset directory (default: ../dataset)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="data_analysis",
        help="Output directory (default: data_analysis)"
    )

    args = parser.parse_args()

    analyze_consistency(
        dataset_dir=args.dataset,
        output_dir=args.output_dir
    )
