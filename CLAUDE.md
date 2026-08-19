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

`docs/results.md` is the current answer to "does this work?": **yes, 11–14 dB of effective SNR
gain against `compact_fec`** — down from the 20–25 dB Phase 3 claimed against verbose JSON, because
over half of that was source coding and missing FEC. Read it before quoting any number.

`docs/related_work.md` places this project against those threads plus two more (the 3GPP/AI-RAN
"AI-native air interface" standardization track, and the 2024-2026 LLM latent-communication
literature). Read it before writing anything that claims novelty. Two things here are *not*
novel, and claiming them will get the work dismissed: the two-sided encoder/decoder mechanism
(3GPP Release 20 carries it as a work item, targeting CSI), and validating JSCC on a real SDR
(arXiv:2410.17536 is a prototype-validation paper). What is unoccupied is the combination — LLM
latents as the payload, evaluated by downstream task utility, with hardware and simulation
paired episode-for-episode. The doc also lists the standard objections and what this repo
already contains to answer them.

**Scope**: there are now TWO channel implementations, and they share the DSP-facing seam.

- **Simulated** (`airComp/`): both agents run sequentially in one process and the "wireless
  channel" is a tensor operation (`z + noise`). This is the standard evaluation methodology in the
  JSCC/semantic communication literature, and it is what the SNR sweep in `airComp/eval/` uses.
- **Real RF** (`hwlab/`): the same latent vector is pulse-shaped, transmitted from one HackRF One
  and received by another over a conducted coax path with a calibrated attenuator. See
  `hwlab/README.md`. Both sweeps derive episode seeds from the same formula, so their curves are
  paired episode-for-episode and can be overlaid; a systematic gap between them is a bug, not a
  physical effect.

## Architecture

Two pipelines are compared on the same task under matched channel conditions:

**Baseline (conventional) pipeline** — `airComp/baseline/`, `airComp/agents/baseline_agent.py`,
`airComp/channel/digital.py`:
1. An LLM agent generates a structured JSON proposal as text.
2. Text -> UTF-8 bits -> BPSK -> AWGN(`SNR_dB`) -> hard-decision demod -> bits -> UTF-8 (possibly
   corrupted) -> regex/JSON extraction -> pydantic validation.
3. Two channel modes: `raw` (no FEC) and `arq` (CRC-8 detect-and-drop). **Neither corrects
   errors, and both put the LLM's entire completion on the wire (~1000 bits to convey 6.1 bits
   of offer).** They are the naive-digital reference point, not a fair baseline — see below.

**Compact (fair digital) baseline** — `airComp/agents/compact_agent.py`,
`airComp/baseline/offer_codec.py`: the same LLM turn, but the parsed offer is source-coded to a
fixed 8-bit frame (index into the pool's feasible count-vectors + action) and sent over the same
BPSK/AWGN channel. `compact_fec` adds Hamming(7,4), giving **16 channel uses — exactly the
semantic pipeline's k=16 real channel uses at the same SNR per real dimension.** This is the
only apples-to-apples comparison in the repo and the one any claim must be stated against.

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

- **Real network transport.** No socket-based transport (TCP/UDP) between processes or machines.
  Real LAN/Wi-Fi already error-corrects at the physical layer, so bit errors cannot be observed
  without going below that stack.
- **Over-the-air radiation.** The HackRF path is *conducted* only: coax plus a calibrated
  attenuator, nothing radiated. HackRF One carries no Japanese 技適 mark, so radiating outside a
  shielded box would violate 電波法. It is also the cleanest channel scientifically — no multipath,
  no fading, no external interference — i.e. as close to the AWGN model as a real radio gets.
- **Spatial separation between the two agents.** Both radios sit on one bench driven by one
  process.

Note that "no real RF" is no longer on this list: `hwlab/` transmits for real. The abstract
`Channel` seam in `airComp/channel/base.py` is what let that be added without touching agent or
task code, and `hwlab/radio/backend.py` carries the same idea one level lower.

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
    compact_agent.py           # CompactAgent: same turn, offer source-coded to 8 bits (the fair baseline)
    semantic_agent.py         # SemanticAgent: hidden state -> SemanticEncoder -> AnalogAWGNChannel -> SemanticDecoder -> Offer
  channel/
    base.py                    # abstract Channel interface (future SDR backend implements this)
    digital.py                  # DigitalChannel: bits/BPSK/AWGN/demod, "raw"/"arq"/"fec" modes
    analog.py                    # AnalogAWGNChannel(nn.Module): differentiable AWGN on real vectors
    fading.py                     # optional Rayleigh block-fading variant (stretch)
  jscc/
    modules.py                    # SemanticEncoder, SemanticDecoder (nn.Module)
    dataset.py                     # collect_dataset(): self-play via frozen LLM -> (hidden_state, offer) tuples
    losses.py                       # offer/action CE losses, aux MSE, expected-utility surrogate (Phase 2)
    train_jscc.py                    # training loop, SNR randomization, checkpointing
  baseline/
    run_baseline.py                   # orchestrates N episodes of TextAgent<->DigitalChannel<->TextAgent
    offer_codec.py                     # Offer <-> fixed 8-bit frame, conditioned on the shared pool
  eval/
    metrics.py                         # agreement_rate, avg_utility, rounds_to_agreement, pareto_efficiency, effective_bits
    normalize.py                        # floor/ceiling normalisation + effective SNR gain -- the headline number
    reconstruction.py                    # is the decoder using the channel, or emitting a prior?
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
pip install torch                     # CPU build. This machine has an AMD GPU; see the environment note below -- do NOT install a CUDA wheel.
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

- **Fairness between pipelines is load-bearing, and it has already been got wrong once.** The
  hidden state is pooled *only* over the offer-JSON token span, and the sweep uses identical
  seeded pools/values — but Phase 3 still compared a 16-symbol latent against ~1000 bits of prose
  and reported the resulting 20–25 dB as a JSCC result. It was mostly source coding. Before
  claiming any gain, check all four budgets: **payload information content, channel uses, error
  correction, and number of turns.** `compact_fec` equalises all four; `raw`/`arq` equalise none.
- **Never compare raw agreement rates across pipelines.** They share neither a zero nor a one.
  Floor (measured at −60 dB): semantic **0.48** — its decoder always emits a valid offer, so two
  priors agree by coincidence — versus 0.02–0.10 for the compact pipelines, whose undecodable
  frames are implicit REJECTs. Ceiling (measured at +40 dB): 0.90–0.98, differing because
  `CompactAgent` must parse its own JSON and `SemanticAgent` does not. Use
  `airComp/eval/normalize.py`, which divides each curve by its own floor-to-ceiling range.
- **`lost_message_ends_episode` is a fairness knob, not a detail.** Default `True` reproduces
  Phase 3, but it ends the episode on the first undecodable message — which only digital
  pipelines can produce. Worth ~2 dB; set `False` (`--survive-lost-messages`) for fair numbers.
- **Bits vs. symbols is not an apples-to-apples comparison.** Report both a raw payload-size
  comparison and a Shannon-capacity-equivalent bit estimate (`k * 0.5*log2(1+SNR_linear)`) for the
  semantic channel — don't present bit counts alone as a bandwidth-fairness claim. Note the
  estimate has a floor problem of its own: at −15 dB it says 0.83 bits/episode reach the receiver
  while agreement reads 0.74, which is only possible because 0.48 of that is prior.
- **Model choice**: Qwen2.5-1.5B-Instruct was picked over Llama-3.2-Instruct because it's ungated
  on Hugging Face (no license click-through) and fits fp16 comfortably without quantization.
  `bitsandbytes` is intentionally not a dependency (unreliable on Windows, unnecessary at this
  model size).
- **Invalid LLM output is a real, measured failure mode**, not something silently retried away —
  bounded retries (max 2) exist, but exhausting them counts as an implicit `REJECT` in the metrics.
- **Environment: Python 3.14, and the GPU is an AMD Radeon RX 9060 XT -- there is no NVIDIA card
  in this machine** (verified 2026-08-18: no `VEN_10DE` device present). This supersedes an earlier
  note in this file describing an RTX 3060 and a working `cu126` build; the hardware was changed.
  **Do not install a CUDA wheel** -- `torch.cuda.is_available()` cannot become `True` here, so the
  2.5 GB download buys nothing. `LocalLLM` already falls back to CPU, so `configs/*.yaml` saying
  `device: "cuda"` is harmless.
- **PyTorch has no GPU path here, but ONNX Runtime does -- and this is now wired up.**
  `torch-directml` publishes no wheel for Python 3.14 and ROCm is Linux-only, so `torch` stays
  CPU-only. `onnxruntime-genai-directml` drives the Radeon instead; set `model.backend: "onnx-dml"`
  (the shipped configs already do) and build the model once with
  `python scripts/build_genai_model.py`. See `airComp/agents/llm_onnx.py`.
- **Use genai, not a generic ONNX graph.** Measured at a 250-token prompt:
  CPU torch **345 ms/token**, generic ONNX on DirectML **189 ms/token**, genai on DirectML
  **11 ms/token**. Batch-1 decode is hundreds of tiny operators, so it is bound by per-operator
  dispatch, not bandwidth -- confirmed twice over, since raising KV-cache traffic 5x cost only 8%
  and the *int4* generic graph was **slower** than fp16 (dequantization adds ops). genai's fused
  decode kernels are what removes that overhead. A GEMV microbenchmark predicted 5.7x and was
  badly misleading about which bottleneck mattered.
- **Never install `optimum-onnx`**: it downgrades transformers 5.15 to 4.57 and replaces
  onnxruntime-directml with the CPU build. The KV-cache handling is genai's, so it is not needed.
- **End to end**: dataset collection went **38.9 -> 9.1 s per episode (4.3x)**, and yield rose from
  1.29 to 2.00 examples/episode because the int4 model emits better-formed JSON. A 500-episode
  collection is ~1.3 hours instead of ~5.4.
- **Hidden states stay on CPU torch by design.** `chat_with_hidden` generates on the GPU but pools
  from the torch model, so the pooled vector is numerically identical to everything collected
  before the port -- no train/inference shift in the JSCC decoder. That prefill (~2.8 s) is now the
  dominant per-turn cost and is the obvious next thing to move.
- **Do not "optimize" the CPU path to bfloat16.** It looks right -- decode is memory-bandwidth
  bound and bf16 halves the weight traffic -- but this CPU is AVX2-only with no AVX512-BF16, so
  bf16 GEMM is emulated. Decode gains nothing and the compute-bound prefill in `chat_with_hidden`
  goes from 2.8 s to 17 s; a full episode measured **32 s -> 60 s**. Raising `torch.set_num_threads`
  from 8 to 16 also changes nothing (312 vs 315 ms/token), which is what confirms the bandwidth
  limit. Measure the prefill, not just tokens/s, before believing any dtype change.
