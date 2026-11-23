import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict

MODEL_ALIASES = {
    "recognition_gpt-5-2025-08-07": "recognition_gpt-5",
}

def load_all_samples(dataset_dir: Path) -> Dict[str, List[Dict]]:

    samples_by_model: Dict[str, List[Dict]] = defaultdict(list)

    recognition_files = list(dataset_dir.glob("*/recognition_*/*.json"))
    model_names = sorted({path.parent.name for path in recognition_files})

    print(f"📂 Found {len(recognition_files)} recognition files across {len(model_names)} model directories")

    for json_file in recognition_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = json_file.parent.name
            model_name = MODEL_ALIASES.get(model_name, model_name)

            character = data.get('character')
            remaining_strokes = data.get('remaining_strokes')
            runs = data.get('runs', [])

            for run in runs:
                llm_output = run.get('llm_output')
                if not llm_output:
                    continue

                sample = {
                    'character': character,
                    'remaining_strokes': remaining_strokes,
                    'answer': llm_output.get('answer'),
                    'probabilities': llm_output.get('probabilities', []),
                    'run_id': run.get('run_id')
                }

                samples_by_model[model_name].append(sample)

        except Exception as e:
            print(f"⚠️  Failed to read file: {json_file.name} - {str(e)}")
            continue

    total_samples = sum(len(v) for v in samples_by_model.values())
    print(f"✅ Loaded {total_samples} samples")
    return dict(samples_by_model)

def calculate_accuracy(samples: List[Dict]) -> float:

    if not samples:
        return 0.0

    correct_count = sum(1 for s in samples if s['answer'] == s['character'])
    return correct_count / len(samples)

def calculate_top3_accuracy(samples: List[Dict]) -> float:

    if not samples:
        return 0.0

    correct_count = 0
    for sample in samples:
        raw_probabilities = sample.get('probabilities') or []
        probabilities = [p for p in raw_probabilities if isinstance(p, dict)]

        top3_chars = [
            p.get('char') for p in probabilities[:3]
            if p.get('char') is not None
        ]

        if sample['character'] in top3_chars:
            correct_count += 1

    return correct_count / len(samples)

def calculate_top1_diff(samples: List[Dict]) -> float:

    if not samples:
        return 0.0

    diffs = []

    for sample in samples:
        raw_probabilities = sample.get('probabilities') or []
        probabilities = [p for p in raw_probabilities if isinstance(p, dict)]
        if not probabilities:
            continue

        is_correct = (sample['answer'] == sample['character'])

        if is_correct:

            diffs.append(0.0)
        else:

            top1_prob = probabilities[0].get('p', 0.0)

            top1_prob = float(top1_prob) if top1_prob is not None else 0.0

            true_prob = 0.0
            for prob_item in probabilities:
                if prob_item.get('char') == sample['character']:
                    true_prob = prob_item.get('p', 0.0)

                    true_prob = float(true_prob) if true_prob is not None else 0.0
                    break

            diff = top1_prob - true_prob
            diffs.append(diff)

    if not diffs:
        return 0.0

    return np.mean(diffs)

def calculate_ece(samples: List[Dict], num_bins: int = 10) -> float:

    if not samples:
        return 0.0

    confidences = []
    correctness = []

    for sample in samples:
        raw_probabilities = sample.get('probabilities') or []
        probabilities = [p for p in raw_probabilities if isinstance(p, dict)]
        if not probabilities:
            continue

        top1_prob = probabilities[0].get('p', 0.0)
        top1_prob = float(top1_prob) if top1_prob is not None else 0.0
        confidences.append(top1_prob)

        is_correct = (sample['answer'] == sample['character'])
        correctness.append(1.0 if is_correct else 0.0)

    if not confidences:
        return 0.0

    confidences = np.array(confidences)
    correctness = np.array(correctness)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)

    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):

        lower_bound = bin_boundaries[i]
        upper_bound = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= lower_bound) & (confidences <= upper_bound)
        else:
            in_bin = (confidences >= lower_bound) & (confidences < upper_bound)

        bin_size = np.sum(in_bin)

        if bin_size > 0:

            acc_bin = np.mean(correctness[in_bin])

            conf_bin = np.mean(confidences[in_bin])

            ece += np.abs(acc_bin - conf_bin) * (bin_size / total_samples)

    return ece

def group_by_remaining_strokes(samples: List[Dict]) -> Dict[int, List[Dict]]:

    grouped = defaultdict(list)
    for sample in samples:
        remaining = sample['remaining_strokes']
        grouped[remaining].append(sample)
    return dict(grouped)

def analyze_baseline_1(dataset_dir: str = "../dataset",
                       output_dir: str = "data_analysis"):

    script_dir = Path(__file__).parent
    dataset_path = script_dir / dataset_dir
    output_path = script_dir / output_dir

    print(f"\n{'='*60}")
    print("📊 Analyzing recognition_* outputs")
    print(f"{'='*60}")
    print(f"Dataset directory: {dataset_path}")
    print(f"Output directory: {output_path}")
    print(f"{'='*60}\n")

    output_path.mkdir(parents=True, exist_ok=True)

    samples_by_model = load_all_samples(dataset_path)

    if not samples_by_model:
        print("❌ No sample data found")
        return

    results: Dict[str, Dict[str, Dict]] = {}
    all_samples: List[Dict] = []

    print(f"\n{'='*60}")
    print("📊 Per-model metrics")
    print(f"{'='*60}")

    for model_name in sorted(samples_by_model.keys()):
        model_samples = samples_by_model[model_name]
        if not model_samples:
            continue

        grouped_samples = group_by_remaining_strokes(model_samples)
        remaining_values = sorted(grouped_samples.keys())

        model_result = {
            "by_remaining_strokes": {},
            "overall": {}
        }

        print(f"\n🧠 Model: {model_name}")
        print(f"{'Remaining':>10} | {'Acc':>6} | {'Top3':>6} | {'ECE':>6} | {'T1Diff':>6} | {'Count':>6}")
        print(f"{'-'*70}")

        for remaining in remaining_values:
            group_samples = grouped_samples[remaining]

            acc = calculate_accuracy(group_samples)
            top3 = calculate_top3_accuracy(group_samples)
            ece = calculate_ece(group_samples)
            top1_diff = calculate_top1_diff(group_samples)
            count = len(group_samples)

            metrics = {
                "Acc": round(acc, 3),
                "Top3": round(top3, 3),
                "ECE": round(ece, 3),
                "Top1Diff": round(top1_diff, 3),
                "Count": count
            }
            model_result["by_remaining_strokes"][str(remaining)] = metrics

            print(f"{remaining:>10} | {metrics['Acc']:>6.3f} | {metrics['Top3']:>6.3f} | "
                  f"{metrics['ECE']:>6.3f} | {metrics['Top1Diff']:>6.3f} | {metrics['Count']:>6}")

        overall_samples = [
            s for s in model_samples
            if s.get('remaining_strokes') not in (None, 0)
        ]
        if overall_samples:
            overall_acc = calculate_accuracy(overall_samples)
            overall_top3 = calculate_top3_accuracy(overall_samples)
            overall_ece = calculate_ece(overall_samples)
            overall_top1_diff = calculate_top1_diff(overall_samples)
            overall_count = len(overall_samples)
        else:
            overall_acc = overall_top3 = overall_ece = overall_top1_diff = 0.0
            overall_count = 0
        overall_metrics = {
            "Acc": round(overall_acc, 3),
            "Top3": round(overall_top3, 3),
            "ECE": round(overall_ece, 3),
            "Top1Diff": round(overall_top1_diff, 3),
            "Count": overall_count
        }
        model_result["overall"] = overall_metrics

        print(f"{'-'*70}")
        print(f"{'Overall(1-3)':>10} | {overall_acc:>6.3f} | {overall_top3:>6.3f} | "
              f"{overall_ece:>6.3f} | {overall_top1_diff:>6.3f} | {overall_count:>6}")

        results[model_name] = model_result
        all_samples.extend(model_samples)

    print(f"\n{'='*60}\n")

    nonzero_samples = [
        s for s in all_samples
        if s.get('remaining_strokes') not in (None, 0)
    ]

    if nonzero_samples:
        overall_acc = calculate_accuracy(nonzero_samples)
        overall_top3 = calculate_top3_accuracy(nonzero_samples)
        overall_ece = calculate_ece(nonzero_samples)
        overall_top1_diff = calculate_top1_diff(nonzero_samples)
        print("📈 Overall stats across models (remaining 1-3):")
        print(f"  - Total samples: {len(nonzero_samples)}")
        print(f"  - Overall accuracy: {overall_acc:.3f}")
        print(f"  - Overall Top-3 accuracy: {overall_top3:.3f}")
        print(f"  - Overall ECE: {overall_ece:.3f}")
        print(f"  - Overall Top-1 Diff: {overall_top1_diff:.3f}")
        print(f"\n{'='*60}\n")
    else:
        overall_acc = overall_top3 = overall_ece = overall_top1_diff = 0.0
        print("⚠️ No remaining-1-to-3 samples available for overall stats")
        print(f"\n{'='*60}\n")

    for model_name, model_result in results.items():
        model_file = output_path / f"{model_name}.json"
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model_result, f, ensure_ascii=False, indent=2)
        print(f"✅ Model metrics saved to: {model_file}")

    summary = {
        "total_samples": len(nonzero_samples),
        "overall_metrics": {
            "Acc": round(overall_acc, 3),
            "Top3": round(overall_top3, 3),
            "ECE": round(overall_ece, 3),
            "Top1Diff": round(overall_top1_diff, 3)
        }
    }
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Summary saved to: {summary_file}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze baseline_1 recognition results"
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

    analyze_baseline_1(
        dataset_dir=args.dataset,
        output_dir=args.output_dir
    )
