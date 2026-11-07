# Design notes

## Process model

```
Electron main (app/main.js)
  └─ Engine (app/engine.js) ──spawn──▶  python -m kangaroo <train|generate|export|info> --config <tmp.json>
        ▲   JSON lines on stdout                │
        └───────────────────────────────────────┘      stderr = library noise (shown dimmed in the log)
```

* The renderer is sandboxed (`contextIsolation`, no Node integration, strict CSP). It talks to the main process only
  through the small API in `preload.js`.
* The engine is spawned with `shell: false`; configuration travels via a temp JSON file, never the command line, so UI
  values cannot be interpreted by a shell. Only four commands are whitelisted.
* **Stopping:** the app creates a `STOP` file path and passes it to the trainer as `stop_file`. The trainer checks it
  every step, stops, and still saves the adapter. If the process has not exited after 20 s the tree is killed.
* Python's stdout is reserved for protocol events (`events.capture_stdout()` redirects stray prints to stderr).

## Events

Every line on stdout is `{"event": <name>, ...}`:

| event | when | fields |
|---|---|---|
| `phase` | stage change | `name`: `data` / `model` / `train` |
| `data` | after tokenisation | `format, records, examples, val_examples, tokens, target_tokens` |
| `model` | after LoRA attach | `trainable_params, total_params` |
| `plan` | before training | `total_steps, batches_per_epoch, groups_per_epoch, padding_efficiency` |
| `checkpointing` | policy decided | `mode`: `on` / `off` |
| `progress` | every `logging_steps` | `step, total_steps, epoch, loss, lr, grad_norm, tok_s, eta_s, mem_gb` |
| `eval` | validation | `step, val_loss` |
| `token` | `generate` | `text` |
| `done` | success (terminal) | train: `adapter_dir, final_loss, avg_tok_s, peak_mem_gb, stopped, …` |
| `error` | failure (terminal) | `message, traceback` |
| `log`, `info` | misc | |

## Training loop

1. Render each record to `(prompt, completion)` text (model chat template when available), tokenise separately,
   mask the prompt with `-100`, truncate the completion if needed.
2. Build micro-batches: shuffle, sort inside a 512-example window, fill to `batch_tokens` padded tokens
   (`max_len × n_examples`), shuffle the batches. A group of `grad_accum` micro-batches is one optimiser step.
3. Forward through the HF backbone only (`lm.base_model`) to get final hidden states, select supervised positions,
   and compute the summed cross-entropy in checkpointed chunks (`loss.py`).
4. Each micro-batch loss is divided by the **total supervised tokens of the whole accumulation group**, so every token
   counts equally regardless of how the batches were cut.
5. Clip, step AdamW (fused on CUDA), cosine/linear/constant schedule with linear warmup.

## Why chunked loss is exact

`sum_i CE(softmax(h_i W^T), y_i)` is a sum over independent rows, so chunking rows changes nothing mathematically.
`torch.utils.checkpoint` drops each chunk's logits after the forward and recomputes them in the backward, so only one
chunk of `[chunk × vocab]` fp32 logits exists at a time. `tests/test_loss.py` verifies values and gradients against the
dense computation (also with Gemma-style logit soft-capping).

## Known gaps / ideas

* Run and tune the CUDA path; publish real benchmarks against other stacks.
* Fused RMSNorm / RoPE / MLP kernels would close the compute gap with hand-written-kernel libraries.
* Optional sequence packing with varlen attention; DoRA; resume from checkpoint; multi-turn all-assistant-turn loss.
* Packaged installers (electron-builder) with a bundled, self-installing Python environment.
