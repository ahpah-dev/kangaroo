# 🦘 Kangaroo

**Lightweight LoRA / QLoRA fine-tuning for your own GPU, as a desktop app.**

Kangaroo is a small, readable alternative to heavier fine-tuning stacks. It is a ~1k-line Python engine on top of plain
PyTorch + Transformers + PEFT, with an Electron app on top. It focuses on the things that cost the most memory and time
in a LoRA run, and leaves everything else to the libraries you already trust.

![Kangaroo desktop app](docs/screenshot.png)

<sup>A 40-step run on the tiny offline test model that ships in `tests/`, on CPU. Not a benchmark.</sup>

## Status — please read

* **Verified:** the engine (35 pytest tests) and the desktop app (6 Node tests plus a scripted end-to-end run of the real
  Electron UI: train → test → export) on **CPU**, using a tiny random model. The loss implementation is checked against
  the dense loss and against Hugging Face's own loss.
* **Not yet verified:** the **CUDA path** (bf16 autocast, fused AdamW, the auto memory probe, 4-bit QLoRA) has been written
  but **not run on a GPU yet**. Expect to find bugs there, and please open issues.
* **No benchmarks yet.** Kangaroo does **not** claim to be faster than Unsloth. Unsloth ships hand-written fused Triton
  kernels that target raw compute speed; Kangaroo does not have those. Its savings come from memory, padding and
  overhead (below), and whether that adds up to a speed-up on your hardware is something to measure — see
  [Benchmark it yourself](#benchmark-it-yourself).

## What it does differently

| Technique | Effect |
|---|---|
| **Chunked, checkpointed cross-entropy** (`loss.py`) | Never materialises the `[tokens × vocab]` fp32 logits (often the largest tensor in a LoRA step with 128k–256k vocabularies). Identical result to the dense loss (tested, including gradients and logit soft-capping). |
| **Only supervised positions hit the LM head** | With completion-only loss, prompt tokens skip the vocab projection entirely. |
| **Token-budget, length-bucketed batching** (`data.py`) | Sorts within a window and fills each micro-batch to a padded-token budget, so there is little padding waste and memory per step is predictable. No cross-example attention leakage (unlike naive packing). The app reports padding efficiency. |
| **Auto activation checkpointing** | On CUDA, probes the biggest batch with checkpointing *off*; if it fits, you skip the recompute cost, otherwise it turns checkpointing on. |
| **Lean training loop** (`trainer.py`) | No `Trainer`/Accelerate layers. SDPA attention, bf16 autocast over fp32 LoRA weights, fused AdamW, token-weighted loss normalisation across gradient accumulation. |
| **Optional QLoRA** | NF4 + double quantisation via bitsandbytes (CUDA only). |
| **Small footprint** | Hard dependencies: `torch`, `transformers`, `peft`, `tokenizers`. Everything else is optional. |

## Install

### Windows installer (easiest)

Download **`Kangaroo-Setup-0.1.0.exe`** from the [latest release](https://github.com/ahpah-dev/kangaroo/releases/latest)
and run it. On first launch the app opens **Settings** and offers an **Install engine** button, which creates a private
Python environment and installs PyTorch (CUDA 12.8 if an NVIDIA GPU is detected, otherwise CPU) plus Transformers and PEFT.
You need **Python 3.10+** installed beforehand (python.org; tick "Add python.exe to PATH"). The GPU download is about 3 GB.

> The installer is **not code-signed**, so Windows SmartScreen will warn you ("Windows protected your PC" → *More info* →
> *Run anyway*). It is built from this repository by `npm run dist` in `app/`.

### From source

Requirements: Python ≥ 3.10, Node ≥ 20, and (for GPU training) an NVIDIA GPU with a recent driver.

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # CUDA 12.8 build (RTX 50-series OK)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Torch cpu
```

```bash
# Linux / macOS
bash scripts/setup.sh            # or: bash scripts/setup.sh cu126 | cpu
```

The script creates `.venv`, installs PyTorch + Kangaroo (`pip install -e ".[quant,hub]"`) and the Electron app.
Then start the app:

```bash
cd app
npm start
```

The **Settings** tab shows what the engine found (GPU, VRAM, library versions) and lets you point at a different Python.

> **Node 26 note:** if `npm install` leaves Electron without its binary, run `node node_modules/electron/install.js`;
> if that exits silently without creating `node_modules/electron/dist/electron.exe`, unzip the cached
> `electron-v*-win32-x64.zip` from `%LOCALAPPDATA%\electron\Cache` into `node_modules/electron/dist` and write
> `electron.exe` to `node_modules/electron/path.txt`.

## Using the app

1. **Train** — pick a base model (Hugging Face id or local folder) and a dataset, adjust LoRA / optimisation settings,
   press *Start training*. Live loss curve, tokens/s, peak VRAM and ETA. *Stop & save* ends after the current step and
   still writes the adapter.
2. **Test** — generate with the adapter you just trained (streams tokens).
3. **Export** — merge the adapter into the base weights and save standard safetensors. For GGUF, run llama.cpp's
   `convert_hf_to_gguf.py` on the output folder.

### Dataset formats

Local `.jsonl`, `.json`, `.csv`, `.tsv`, or any Hub dataset id (needs `datasets`). The format is auto-detected:

| Format | Record |
|---|---|
| `alpaca` | `{"instruction": "...", "input": "...", "output": "..."}` |
| `chat` | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]}` or ShareGPT `{"conversations": [{"from": "human", "value": ...}]}` |
| `prompt_completion` | `{"prompt": "...", "completion": "..."}` |
| `text` | `{"text": "..."}` (loss on everything) |

The model's own chat template is used when it has one. For multi-turn chats, loss is applied to the **final assistant
turn**. Prompts are masked unless *Also train on prompts* is on.

## Command line

The app is a thin shell around this CLI; everything it does you can script:

```bash
python -m kangaroo train --model Qwen/Qwen2.5-0.5B-Instruct --dataset data.jsonl --output-dir out --epochs 1
python -m kangaroo train --config run.json            # JSON with the same keys; flags override it
python -m kangaroo generate --adapter out/adapter --prompt "Hello"
python -m kangaroo export --adapter out/adapter --output-dir merged
python -m kangaroo info
```

`train` accepts every field of [`TrainConfig`](src/kangaroo/config.py) as a flag (`--lora-r 32`, `--no-rslora`,
`--gradient-checkpointing off`, …). Stdout is a JSON-lines event stream (`progress`, `done`, `error`, …); see
[docs/DESIGN.md](docs/DESIGN.md).

## Benchmark it yourself

`train` ends with a `done` event reporting average real tokens/s and peak VRAM. To compare fairly with another stack, use
the same model, dataset, `max_seq_len`, LoRA rank/targets, precision and effective batch, and compare **tokens/s,
peak VRAM and final eval loss**. Kangaroo's `batch_tokens` × `grad_accum` sets its effective batch in tokens, so match it to
the other tool's `batch_size × seq_len × grad_accum`. If you do, please share the numbers in an issue.

## Limitations

* Text-only causal LMs that Transformers supports and whose backbone is exposed as `model.base_model` (Llama, Qwen,
  Mistral, Gemma, Phi, SmolLM, …). Vision / multimodal models are not supported.
* LoRA adapters only (no full fine-tuning, no embedding training, no DoRA yet). Single GPU.
* No custom Triton kernels and no GGUF export of its own.
* The engine runs on CPU for testing, but training real models there is slow.
* The installer is Windows-only, unsigned, and does not bundle Python (it sets up a venv from your system Python).

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest                 # engine tests (CPU, offline, ~30 s)
cd app && npm install && node --test
cd app && npm run dist           # builds dist/Kangaroo-Setup-<version>.exe
bash scripts/smoke-e2e.sh        # drives the real Electron UI end to end (CPU, offline)
```

Layout: `src/kangaroo/` engine · `app/` Electron shell (`engine.js` supervises the Python process, `renderer/` is plain
HTML/JS with no framework) · `tests/` · `scripts/`.

## License

[The Unlicense](LICENSE) — public domain. Do whatever you like.
