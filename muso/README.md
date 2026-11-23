# MUSO | Chinese Multimodal Stroke Order Dataset

MUSO extracts ordered stroke images from Chinese calligraphy workbooks and evaluates visual-language models on partial-character recognition and reasoning.

## What is included

| Path | Contents |
| --- | --- |
| `src/calligraphy_stroke_dataset/` | Extract workbook grids, build stroke metadata and assemble character folders. |
| `src/experiments/` | Baseline, Chain-of-Thought, few-shot, self-consistency, Chain-of-Frames and Chain-of-Strokes scripts, with Chinese and English prompts where available. |
| `data/dataset.zip` | Dataset snapshot with 2,318 character folders, 24,775 stroke PNGs and one `stroke_info.json` per character. |
| `data/inputs/` | Source workbook, character list and few-shot character list. |
| `data/metadata/` | Stroke dictionary and character mapping. |

The older implementation and its data are in the repository root. Run the commands below from `muso/`.

## Set up the dataset

Use Python 3.13 and `uv`.

```bash
cd muso
uv sync --locked
unzip -q -n data/dataset.zip -d data
```

The archive extracts to `data/dataset/`. The `src/experiments/dataset` link then resolves to that folder. To rebuild the dataset from the workbook, run the following in a fresh copy before extracting the archive.

```bash
uv run python -m calligraphy_stroke_dataset.prepare_character_info
uv run python -m calligraphy_stroke_dataset.main data/inputs/1000.pdf --output data/intermediate/grid_rows
uv run python -m calligraphy_stroke_dataset.allocate_characters
```

## Run an experiment

Set `MUSO_API_URL` to your model provider's full chat-completions endpoint and set `MUSO_API_KEY` in the shell. Use `--help` on each script to see its supported arguments.

```bash
uv run python src/experiments/chain_of_frames_input/recognize.py \
  --dataset data/dataset --model gemini-2.5-pro --test 1
```

CoT and Few-shot use additional demonstration images from `data/prompt_examples/prompt_dataset/`. The expected character list is in `data/inputs/hanzi_prompt.txt`. Supply those images before reproducing the few-shot runs. Scripts using a test or robustness subset also need the corresponding local inputs under `data/test_dataset/` or `data/robustness/`.
