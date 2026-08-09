# AirComp

A research prototype comparing conventional text-based AI-agent communication against a
**semantic / Joint Source-Channel Coding (JSCC)** pipeline that transmits compressed LLM latent
representations over a simulated noisy wireless channel, using a bilateral negotiation task as
the testbed.

See [CLAUDE.md](CLAUDE.md) for the full architecture, design rationale, and repository layout.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126   # GPU build; omit for CPU-only
python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct

# Conventional (text + digital channel) baseline
python evaluate.py run-baseline --episodes 50 --snr-db 10 --channel-mode raw

# Collect a JSCC training set, then train the semantic encoder/decoder
python train.py collect-dataset --episodes 500 --out data/jscc_dataset.pt
python train.py train-jscc --dataset data/jscc_dataset.pt --out checkpoints/jscc_v1.pt

# Compare both pipelines across an SNR sweep
python evaluate.py snr-sweep --checkpoint checkpoints/jscc_v1.pt --episodes 100 --out results/sweep.json

pytest -m "not slow" -q
```

## Limitations

- **No real wireless or network transmission.** Both agents run in the same Python process on
  one machine; the "channel" is a mathematical noise model applied to tensors/bit arrays, not
  real RF or network transport. This is a deliberate scoping choice for validating the
  algorithmic hypothesis first (see CLAUDE.md).
- **Bits vs. symbols are not directly comparable.** The sweep reports both a raw payload-size
  comparison (bits for the digital pipeline, `k` real-valued symbols for the semantic pipeline)
  and a Shannon-capacity-equivalent bit estimate for the semantic channel
  (`k * 0.5*log2(1+SNR_linear)`) so this isn't misread as an apples-to-apples bandwidth claim.
- **The LLM is frozen** during Phase 1 JSCC training (used only as a feature extractor); joint
  fine-tuning of the LLM itself is future work, as is the differentiable task-outcome Phase 2
  fine-tuning pass described in CLAUDE.md.
- Evaluated with a single small model (Qwen2.5-1.5B-Instruct) and a synthetic 3-item negotiation
  task; results may not generalize to larger models or richer negotiation domains.
