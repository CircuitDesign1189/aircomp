# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project vision

AirComp is a research prototype exploring **AI-native wireless communication**: instead of AI
agents exchanging information via human-readable digital protocols (text tokenized into bits,
sent over conventional APIs with round-trip latency), agents compress their internal LLM
representations directly and exchange them over a channel modeled on real wireless impairments
(noise, limited SNR).

The concrete hypothesis under test: a **semantic / Joint Source-Channel Coding (JSCC)** pipeline
that transmits a compressed continuous latent vector (derived from an LLM's hidden state) over a
simulated analog AWGN channel degrades *gracefully* under low SNR, whereas a **conventional
pipeline** that serializes the same communicative content to text/JSON and transmits it as bits
over a simulated digital channel degrades *catastrophically* (a single flipped bit inside `{`,
`"`, `:` breaks JSON parsing entirely).

This connects two established research threads:
- **Over-the-Air Computation (AirComp)**: exploiting wireless superposition to compute a function
  of transmitted signals directly, rather than decode-then-compute.
- **Semantic communication / JSCC**: neural encoder/decoder pairs trained end-to-end to preserve
  task-relevant meaning under channel noise, rather than optimizing for exact bit recovery.

**Scope of the current prototype**: pure software simulation, single machine, single Python
process. Both negotiating agents run sequentially in one process; the "wireless channel" is a
tensor operation (`z + noise`), not real RF or even real network transport. This is a deliberate
choice (see "Explicitly out of scope" below) to validate the algorithmic hypothesis before adding
hardware/network complexity. It is also the standard evaluation methodology in the JSCC/semantic
communication literature.

## Architecture

Two pipelines are compared on the same task under matched channel conditions:

**Baseline (conventional) pipeline** — `airComp/baseline/`, `airComp/agents/baseline_agent.py`,
`airComp/channel/digital.py`:
1. An LLM agent generates a structured JSON proposal as text.
2. Text -> UTF-8 bits -> BPSK -> AWGN(`SNR_dB`) -> hard-decision demod -> bits -> UTF-8 (possibly
   corrupted) -> regex/JSON extraction -> pydantic validation.
3. Two channel modes: `raw` (no FEC — demonstrates the catastrophic failure cliff) and `arq`
   (CRC-8 detect-and-drop — the more realistic "modern digital comms" baseline).

**Semantic/JSCC pipeline** — `airComp/agents/semantic_agent.py`, `airComp/channel/analog.py`,
`airComp/jscc/`:
1. The LLM's hidden state is mean-pooled over only the tokens corresponding to the offer JSON
   (never private reasoning tokens — this keeps the comparison fair: both pipelines transmit the
   same communicative act, only the encoding/channel differ).
2. `SemanticEncoder` (MLP) compresses this to a `k`-dim, power-normalized real vector.
3. `AnalogAWGNChannel` adds Gaussian noise (differentiable, enabling end-to-end training).
4. `SemanticDecoder` (MLP) reconstructs a structured offer (item counts, action, an auxiliary
   continuous "intent" signal) directly — it does not attempt to reconstruct injectable LLM
   context across independently-instantiated model calls.
5. Training is supervised (Phase 1: frozen LLM as feature extractor, SNR-randomized) with an
   optional differentiable task-outcome fine-tuning phase (Phase 2).

**Task/environment** — `airComp/env/negotiation.py`: a simplified "Deal or No Deal"-style
bilateral bargaining game (Lewis et al. 2017). Two agents split a pool of 3 item types
(`book`/`hat`/`ball`) with private, independently-randomized per-item values (normalized so a full
pool is worth 100 points), over up to 10 alternating messages (`PROPOSE`/`ACCEPT`/`REJECT`).
No-deal or running out of rounds yields 0 utility for both agents.

**Evaluation** — `airComp/eval/snr_sweep.py`: runs both pipelines (baseline in both channel modes,
plus semantic) across an SNR grid with paired random seeds, tracking agreement rate, utility,
social welfare, Pareto efficiency, and effective bits/symbols transmitted. The core result is a
graceful-vs-catastrophic degradation comparison across SNR.

### Explicitly out of scope (for now)

No real RF transmission and no real network transport occur in this prototype — see the "Scope"
note above. Two possible future extensions, not yet implemented:
- Real socket-based transport (TCP/UDP) between two processes/machines, with noise injected
  synthetically at the sender (real LAN/Wi-Fi already error-corrects at the physical layer, so bit
  errors can't be observed without going below that stack).
- Real SDR hardware (USRP/HackRF/LimeSDR) for actual RF transmission. `airComp/channel/base.py`
  defines an abstract `Channel` interface specifically so a future SDR-backed implementation can
  be swapped in without touching agent/task code.

## Repository layout

```
airComp/
  config.py                 # dataclasses: ModelConfig, ChannelConfig, JSCCConfig, NegotiationConfig, TrainConfig
  env/
    negotiation.py          # Pool, Values, Offer, EpisodeState, generate_pool(), generate_values(), utility(), run_episode()
    scoring.py               # social_welfare(), pareto_frontier(), pareto_efficiency()
  agents/
    llm_backend.py           # LocalLLM: chat(), chat_with_hidden()
    prompts.py                # system prompt templates, JSON schema instructions
    parser.py                  # regex+JSON extraction, pydantic Offer schema, bounded-retry logic
    baseline_agent.py         # TextAgent: LLM text turn -> DigitalChannel -> parsed Offer
    semantic_agent.py         # SemanticAgent: hidden state -> SemanticEncoder -> AnalogAWGNChannel -> SemanticDecoder -> Offer
  channel/
    base.py                    # abstract Channel interface (future SDR backend implements this)
    digital.py                  # DigitalChannel: bits/BPSK/AWGN/demod, "raw"/"arq" modes
    analog.py                    # AnalogAWGNChannel(nn.Module): differentiable AWGN on real vectors
    fading.py                     # optional Rayleigh block-fading variant (stretch)
  jscc/
    modules.py                    # SemanticEncoder, SemanticDecoder (nn.Module)
    dataset.py                     # collect_dataset(): self-play via frozen LLM -> (hidden_state, offer) tuples
    losses.py                       # offer/action CE losses, aux MSE, expected-utility surrogate (Phase 2)
    train_jscc.py                    # training loop, SNR randomization, checkpointing
  baseline/
    run_baseline.py                   # orchestrates N episodes of TextAgent<->DigitalChannel<->TextAgent
  eval/
    metrics.py                         # agreement_rate, avg_utility, rounds_to_agreement, pareto_efficiency, effective_bits
    snr_sweep.py                        # both pipelines x both channel modes across an SNR grid, paired seeds
    plots.py                             # matplotlib comparison plots
  utils/
    seeding.py, logging.py, io.py         # RNG seeding, JSONL episode logging, path helpers
scripts/
  download_model.py                       # pre-downloads the local LLM into ./.hf_cache
configs/
  base.yaml, snr_sweep.yaml
tests/
  test_negotiation_env.py, test_digital_channel.py, test_analog_channel.py,
  test_jscc_modules.py, test_parser.py, test_end_to_end_baseline.py (marked `slow`)
train.py                                   # CLI: collect-dataset / train-jscc
evaluate.py                                 # CLI: run-baseline / snr-sweep
```

## Commands

```powershell
# Environment setup
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126   # GPU build; omit this line for CPU-only
pip install -r requirements.txt

# Model download (Qwen2.5-1.5B-Instruct by default — Apache-2.0, ungated)
python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct

# Baseline (conventional) pipeline
python evaluate.py run-baseline --config configs/base.yaml --episodes 50 --snr-db 10 --channel-mode raw

# JSCC dataset collection + training
python train.py collect-dataset --episodes 500 --out data/jscc_dataset.pt
python train.py train-jscc --dataset data/jscc_dataset.pt --epochs 30 --snr-range -5 20 --k 16 --out checkpoints/jscc_v1.pt

# Full SNR sweep (both pipelines, both channel modes)
python evaluate.py snr-sweep --checkpoint checkpoints/jscc_v1.pt --episodes 100 --snr-grid -10 -5 0 5 10 15 20 --out results/sweep.json

# Tests
pytest -m "not slow" -q     # fast: channel math, env logic, parser edge cases (no model download needed)
pytest -m slow -q           # integration: requires the downloaded model
```

## Key design decisions worth knowing before changing this code

- **Fairness between pipelines is load-bearing.** The semantic pipeline's sender-side hidden state
  is pooled *only* over the offer-JSON token span (never chain-of-thought/private reasoning), and
  the SNR sweep runs both pipelines against identical seeded pools/values. Any change to one
  pipeline's information budget should be mirrored or explicitly justified as a documented
  asymmetry.
- **Bits vs. symbols is not an apples-to-apples comparison.** Report both a raw payload-size
  comparison and a Shannon-capacity-equivalent bit estimate (`k * 0.5*log2(1+SNR_linear)`) for the
  semantic channel — don't present bit counts alone as a bandwidth-fairness claim.
- **Model choice**: Qwen2.5-1.5B-Instruct was picked over Llama-3.2-Instruct because it's ungated
  on Hugging Face (no license click-through) and fits fp16 comfortably without quantization.
  `bitsandbytes` is intentionally not a dependency (unreliable on Windows, unnecessary at this
  model size).
- **Invalid LLM output is a real, measured failure mode**, not something silently retried away —
  bounded retries (max 2) exist, but exhausting them counts as an implicit `REJECT` in the metrics.
- Environment: Python 3.13, RTX 3060 12GB. The default `pip install torch` resolves to a CPU-only
  build on this machine -- the CUDA build must be installed explicitly. As of the current driver
  (CUDA UMD Version 13.3), the matching wheel is `cu126` (`torch==2.13.0+cu126`); `cu121`/`cu124`
  do not publish a 2.13.0 build and `cu128`/`cu129` jump straight to torch >=2.7.0. Verified
  working: `torch.cuda.is_available()` is `True` and LLM generation runs ~3-4x faster than on CPU.
  `LocalLLM` auto-detects CUDA and falls back to CPU if unavailable, so no application code needs
  to change when the wheel is swapped.
