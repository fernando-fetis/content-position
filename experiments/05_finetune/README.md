# 05_finetune

Partial RoPE followed by fine-tuning, the conversion step of `sec:planes` (see Sources): in Qwen2.5-0.5B and Qwen2.5-1.5B, each KV head keeps the rotation of a quarter of its planes, chosen by the rules of `04_planes` (our position-dependent variance $Var(p_j) + Var(epsilon_(i j))$ per plane, the cost of `prop:the-cost-of-removing-a-rotation`, and the published rules it is compared with), and the whole model is fine-tuned on FineWeb-Edu under that edit; at fixed token budgets along fine-tuning, each run is evaluated as `04_planes` evaluates its edits (next-token NLL on WikiText-2 and code, accuracy on retrieval, binding and the needle in a haystack) against the intact model fine-tuned on the same tokens. In Qwen2.5-1.5B the same score calibrated at longer lengths tests the calibration length that `cor:slow-planes-and-long-contexts` sets, which concerns the edited model before any training.

Importance: medium.

## Sources
- Definitions: `docs/notes.typ` (`prop:the-cost-of-removing-a-rotation`, `def:content-positional-and-interaction-parts`, `cor:slow-planes-and-long-contexts`).
- Methodology: `docs/notes.typ` (`sec:removing-rotation`: *Evaluación*, the evaluation it shares with the edits without training; `sec:planes`: `tab:plane-rules`; `sec:finetune`: its first paragraph and the list from *Edición diferenciable* to *Evaluación*).
- Results: `docs/notes.typ` (`sec:finetune`, from its *Predicciones* to *Alcance*: `tab:finetune-tasks`, `fig:finetune`).

## Files
- `run.py`: the stages `compute` (per model and run: fine-tuning with evaluations along it), `summarize`, `figures`, `check` (both partial RoPEs against the intact model with every plane kept, the trainable one against the exact edit of 04 with alternate planes and with 04's masks, and the clipping of the evaluation steps).
- `train.py`: the training tokens (FineWeb-Edu, or a text of `data/` for the smoke run), the learning-rate schedule, the evaluation steps and the fine-tuning loop.
- `figures.py`: the figure listed below.
- `config.yaml`: every parameter; `smoke:` for a quick CPU run.
- The edits are shared code: `src/edits/partial_rope.py` (`partial_rope`, the exact edit of 04, and `partial_rope_train`, the differentiable one used here); the evaluation is `src/eval/text_tasks.py`.

## Running
On a SLURM cluster, one job per model and run (`psym-05-<model>-<run>`, `src/remote/launch.py`, with `--rule <run>`), after 04's job of the same model (`psym-04-<model>`), whose selection holds the kept planes; the jobs download the FineWeb-Edu shard (`HF_ONLINE=1`) and cache its tokens per tokenizer in `results/tokens__<tokenizer>.npy`. `make summarize|figures|check E=05_finetune`.

## Main config keys
- `runs`: Qwen2.5-0.5B, `intact` (full RoPE, fine-tuned on the same tokens), `variance` (ours), `norm` (MHA2MLA), `mkl` (MHA2MLA-VLM) and `high` (the fastest planes); Qwen2.5-1.5B, `intact`, `variance`, `variance_4096`, `variance_16000` (our score calibrated at 4096 and 16000 tokens) and `norm`. `rule` picks one run per job.
- `fraction` (0.25) of the planes of each KV head that keep their rotation, read from `selection_experiment` (`04_planes`, masks `kept.<rule>_<fraction>`).
- Training: `data` (the first documents of the first shard of FineWeb-Edu sample-10BT at a pinned revision), `tokens` (50M per run), `seq` (2048), `batch` (32 sequences per step), `micro` (8; 4 for Qwen2.5-1.5B), `optimizer` (AdamW, learning rate 5e-5, warm-up over 2% of the steps, cosine decay to 0.1 of it, weight decay 0.1, betas (0.9, 0.95), clipping at 1), `train_dtype` (bfloat16 autocast).
- Evaluation at `eval_tokens` (0, 5M, 12.5M, 25M, 50M): `text` (second half of the WikiText-2 test set, Python standard library), `text_n` (16 windows) at `text_tokens` (1024, 4096); `tasks` (retrieval, binding, needle in a haystack), `task_n` (32 short, 50 long), `lengths` (4000, 16000).
- `check`: windows of 256 tokens, tolerances 1e-5.

## Summary (`results/summary.json`, key `models.<model>`)
`steps` (the evaluation steps) and `tokens` (training tokens seen at each); per corpus and length (`wiki_1024`, ...), rule and step: `gap`, the gap (the mean over windows of the rule's NLL per token minus that of the intact model fine-tuned on the same tokens, nats); `versus_variance`, each rule minus `variance` (mean and `ci`, paired 95% BCa over windows); per corpus and length and rule: `ratio_to_variance`, the final gap over that of `variance`; `tasks.<task>.<run>`, the accuracy at each step. `meta`: the run metadata.

## Figures (labels in `docs/notes.typ`)
- `finetune.pdf`: `fig:finetune`.

## Notes
Config keys and keys of the raw results whose names differ from the text (existing results store them):
- `rule` (config): the run of a job, a rule of `04_planes` or `intact`.
- `loss` (raw, `<model>__<run>`): the training loss per update (mean next-token NLL over the batch, nats per token), not the loss of the evaluation texts; `step_<s>.text` holds those, per window.
