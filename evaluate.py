"""CLI: run-baseline / snr-sweep / plot / decoder-check."""
from __future__ import annotations

import argparse

from airComp.baseline.run_baseline import run as run_baseline_fn
from airComp.eval.plots import plot_latent_examples, plot_sweep
from airComp.eval.snr_sweep import ALL_PIPELINES, run_sweep


def cmd_run_baseline(args):
    run_baseline_fn(args.config, args.episodes, args.snr_db, args.channel_mode, args.out)


def cmd_snr_sweep(args):
    run_sweep(args.config, args.checkpoint, args.episodes, args.snr_grid, args.out,
              tuple(args.pipelines), args.survive_lost_messages)


def cmd_plot(args):
    """Overlay every pipeline from every results file on one axis.

    Takes several files because the simulated sweep and the hardware sweep are
    separate runs; the point of the figure is to see them together.
    """
    series = plot_sweep(args.results, args.out, args.metric, args.floor, args.floor_label)
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


def cmd_embed_check(args):
    """Is the injectable embed head communicating, or emitting a prior?

    Embed-head analog of decoder-check: cosine similarity to embed_target
    instead of exact offer match. Needs a checkpoint trained with
    --embed-loss-weight and a dataset with embed_target set (see
    train.py backfill-embed-targets). See airComp/eval/reconstruction.py.
    """
    import torch

    from airComp.config import ITEM_TYPES, JSCCConfig
    from airComp.eval.reconstruction import embed_reconstruction_table, embed_verdict, format_embed_table
    from airComp.jscc.modules import SemanticDecoder, SemanticEncoder

    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg: JSCCConfig = ckpt.get("jscc_cfg", JSCCConfig())
    if cfg.embed_dim is None:
        raise SystemExit(f"{args.checkpoint} has no embed_dim (jscc_cfg.embed_dim is None) -- "
                         f"train with --embed-loss-weight > 0 first")
    encoder = SemanticEncoder(ckpt["input_dim"], cfg.encoder_hidden_dims, cfg.k)
    decoder = SemanticDecoder(cfg.k, cfg.decoder_hidden_dims, len(ITEM_TYPES), cfg.max_count, cfg.aux_dim,
                              embed_dim=cfg.embed_dim)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    examples = torch.load(args.dataset, weights_only=False)
    table = embed_reconstruction_table(encoder, decoder, examples, args.snr_grid, cfg.max_count)
    print(format_embed_table(table))
    return 0 if embed_verdict(table)[0] else 1


def cmd_visualize_latent(args):
    """Render a few example latent vectors as PNGs, clean vs. channel-noisy.

    For seeing, not measuring: makes concrete what "analog" means for the
    semantic pipeline. See airComp/eval/plots.py:plot_latent_examples.
    """
    import torch

    from airComp.channel.analog import AnalogAWGNChannel
    from airComp.config import JSCCConfig
    from airComp.jscc.modules import SemanticEncoder

    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg: JSCCConfig = ckpt.get("jscc_cfg", JSCCConfig())
    encoder = SemanticEncoder(ckpt["input_dim"], cfg.encoder_hidden_dims, cfg.k)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    examples = torch.load(args.dataset, weights_only=False)
    written = plot_latent_examples(encoder, AnalogAWGNChannel(), examples, args.out_dir,
                                    snr_db=args.snr_db, n=args.n)
    for path in written:
        print(f"wrote {path}")


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
    p_sweep.add_argument("--survive-lost-messages", action="store_true",
                         help="an undecodable message costs a turn instead of ending the episode; "
                              "equalises the number of attempts each pipeline gets")
    p_sweep.set_defaults(func=cmd_snr_sweep)

    p_plot = sub.add_parser("plot", help="overlay sweep results (simulated and/or hardware)")
    p_plot.add_argument("--results", nargs="+", required=True,
                        help="one or more sweep JSON files; non-series keys are ignored")
    p_plot.add_argument("--metric", default="agreement_rate")
    p_plot.add_argument("--out", default="results/sweep.png")
    p_plot.add_argument("--floor", type=float, default=None,
                        help="draw a no-information reference line; the semantic decoder scores "
                             "0.48 agreement on a channel carrying nothing, so omitting it overstates the tail")
    p_plot.add_argument("--floor-label", default="no-information floor",
                        help="name the pipeline the floor belongs to; floors differ per pipeline")
    p_plot.set_defaults(func=cmd_plot)

    p_check = sub.add_parser(
        "decoder-check",
        help="does the decoder use the channel, or emit a prior? Run before claiming graceful degradation",
    )
    p_check.add_argument("--checkpoint", required=True)
    p_check.add_argument("--dataset", required=True)
    p_check.add_argument("--snr-grid", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 20])
    p_check.set_defaults(func=cmd_decoder_check)

    p_embed_check = sub.add_parser(
        "embed-check",
        help="does the injectable embed head use the channel, or emit a prior? "
             "Embed-head analog of decoder-check",
    )
    p_embed_check.add_argument("--checkpoint", required=True)
    p_embed_check.add_argument("--dataset", required=True)
    p_embed_check.add_argument("--snr-grid", type=float, nargs="+", default=[-10, -5, 0, 5, 10, 20])
    p_embed_check.set_defaults(func=cmd_embed_check)

    p_viz = sub.add_parser(
        "visualize-latent",
        help="render a few example latent vectors as PNGs, clean vs. channel-noisy",
    )
    p_viz.add_argument("--checkpoint", required=True)
    p_viz.add_argument("--dataset", required=True)
    p_viz.add_argument("--out-dir", default="results/latent_examples")
    p_viz.add_argument("--n", type=int, default=3)
    p_viz.add_argument("--snr-db", type=float, nargs="+", default=[20.0, -5.0])
    p_viz.set_defaults(func=cmd_visualize_latent)

    args = parser.parse_args()
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
