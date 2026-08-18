"""CLI: run-baseline / snr-sweep / plot / decoder-check."""
from __future__ import annotations

import argparse

from airComp.baseline.run_baseline import run as run_baseline_fn
from airComp.eval.plots import plot_sweep
from airComp.eval.snr_sweep import ALL_PIPELINES, run_sweep


def cmd_run_baseline(args):
    run_baseline_fn(args.config, args.episodes, args.snr_db, args.channel_mode, args.out)


def cmd_snr_sweep(args):
    run_sweep(args.config, args.checkpoint, args.episodes, args.snr_grid, args.out, tuple(args.pipelines))


def cmd_plot(args):
    """Overlay every pipeline from every results file on one axis.

    Takes several files because the simulated sweep and the hardware sweep are
    separate runs; the point of the figure is to see them together.
    """
    series = plot_sweep(args.results, args.out, args.metric, args.floor)
    for label, points in sorted(series.items()):
        span = f"{min(points):+.0f}..{max(points):+.0f} dB"
        print(f"  {label:16s} {len(points):2d} points over {span}")
    print(f"wrote {args.out}")


def cmd_decoder_check(args):
    """Is the semantic pipeline communicating at all, or emitting a prior?

    A flat task-metric curve does not distinguish those, so this scores decoded
    offers against the offers that were sent. See airComp/eval/reconstruction.py.
    """
    import torch

    from airComp.config import ITEM_TYPES, JSCCConfig
    from airComp.eval.reconstruction import format_table, reconstruction_table, verdict
    from airComp.jscc.modules import SemanticDecoder, SemanticEncoder

    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg: JSCCConfig = ckpt.get("jscc_cfg", JSCCConfig())
    encoder = SemanticEncoder(ckpt["input_dim"], cfg.encoder_hidden_dims, cfg.k)
    decoder = SemanticDecoder(cfg.k, cfg.decoder_hidden_dims, len(ITEM_TYPES), cfg.max_count, cfg.aux_dim)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    examples = torch.load(args.dataset, weights_only=False)
    table = reconstruction_table(encoder, decoder, examples, args.snr_grid, cfg.max_count)
    print(format_table(table))
    return 0 if verdict(table)[0] else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_baseline = sub.add_parser("run-baseline")
    p_baseline.add_argument("--config", default="configs/base.yaml")
    p_baseline.add_argument("--episodes", type=int, default=50)
    p_baseline.add_argument("--snr-db", type=float, default=10.0)
    p_baseline.add_argument("--channel-mode", default="raw", choices=["raw", "arq"])
    p_baseline.add_argument("--out", default="results/baseline.jsonl")
    p_baseline.set_defaults(func=cmd_run_baseline)

    p_sweep = sub.add_parser("snr-sweep")
    p_sweep.add_argument("--config", default="configs/snr_sweep.yaml")
    p_sweep.add_argument("--checkpoint", required=True)
    p_sweep.add_argument("--episodes", type=int, default=100)
    p_sweep.add_argument("--snr-grid", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 15, 20])
    p_sweep.add_argument("--out", default="results/sweep.json")
    p_sweep.add_argument("--pipelines", nargs="+", default=list(ALL_PIPELINES), choices=list(ALL_PIPELINES),
                         help="subset to run; the baselines dominate the cost at low SNR")
    p_sweep.set_defaults(func=cmd_snr_sweep)

    p_plot = sub.add_parser("plot", help="overlay sweep results (simulated and/or hardware)")
    p_plot.add_argument("--results", nargs="+", required=True,
                        help="one or more sweep JSON files; non-series keys are ignored")
    p_plot.add_argument("--metric", default="agreement_rate")
    p_plot.add_argument("--out", default="results/sweep.png")
    p_plot.add_argument("--floor", type=float, default=None,
                        help="draw a no-information reference line; the semantic decoder scores "
                             "0.48 agreement on a channel carrying nothing, so omitting it overstates the tail")
    p_plot.set_defaults(func=cmd_plot)

    p_check = sub.add_parser(
        "decoder-check",
        help="does the decoder use the channel, or emit a prior? Run before claiming graceful degradation",
    )
    p_check.add_argument("--checkpoint", required=True)
    p_check.add_argument("--dataset", required=True)
    p_check.add_argument("--snr-grid", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 20])
    p_check.set_defaults(func=cmd_decoder_check)

    args = parser.parse_args()
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
