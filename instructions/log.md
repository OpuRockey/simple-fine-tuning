There's no dedicated log file — here's what exists and where:

**Training logs (the loss/accuracy numbers)** — printed to the console only, but the full history is saved in the checkpoint:

- `models\checkpoints\checkpoint-120\trainer_state.json` — contains `log_history` with every logged step (train loss, `mean_token_accuracy`, `grad_norm`) and every epoch's `eval_loss` / `eval_mean_token_accuracy`. This is the machine-readable log. Quickest way to read it:

```bash
uv run python -c "import json; h=json.load(open('models/checkpoints/checkpoint-120/trainer_state.json'))['log_history']; [print(e.get('epoch'), e.get('eval_loss', e.get('loss',''))) for e in h]"
```

- Earlier checkpoints got deleted by `save_total_limit=1` — only the last one (`checkpoint-120`) keeps a `trainer_state.json`, so terminal output from a run is currently the only complete record.
- Why no TensorBoard/W&B file: `train.py` sets `report_to=[]` deliberately (no external tracking), and console prints go to your terminal, not to a file.

**Everything else** — `prepare_dataset.py`, `test_model.py`, `export_gguf.py` are all print-only to stdout. No file logging anywhere in the project (standard for small scripts; fine at this scale).

If you want durable logs, the minimal habit is to tee the console:

```bash
uv run python scripts/train.py --epochs 20 2>&1 | Tee-Object -FilePath runs\last_train.log
```

(`mkdir runs` first; it's gitignored territory, or add `runs/` to .gitignore if you commit.) And if you'd rather have structured, queryable logging, the `trainer_state.json` `log_history` is already the right artifact — it just gets overwritten each run unless you copy it per run like above.