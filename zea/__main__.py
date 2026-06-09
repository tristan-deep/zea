"""Entry point for the zea toolbox.

Usage::

    zea process <dataset> <save_dir> [options]   # batch beamform a dataset
    zea app [--share] [--server_port PORT]       # launch the Gradio visualizer
    zea --device DEVICE process ...              # specify device for processing, e.g.
                                                 # 'cuda:0', 'cpu', or 'auto:1' (default)

"""

import argparse


def get_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser with ``process`` and ``app`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="zea",
        description="zea ultrasound toolbox.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    # ── device  ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--device",
        type=str,
        default="auto:1",
        help=(
            "Device to use for processing. Can be a specific device (e.g. 'cuda:0', 'cpu') "
            "or 'auto:1' to automatically select the best available device."
        ),
    )

    # ── process ──────────────────────────────────────────────────────────────
    from zea.data.process import get_parser as _process_parser

    subparsers.add_parser(
        "process",
        help="Beamform a zea dataset using a pipeline YAML config.",
        parents=[_process_parser(add_help=False)],
    )

    # ── app ──────────────────────────────────────────────────────────────────
    app_p = subparsers.add_parser(
        "app",
        help="Launch the interactive Gradio dataset visualizer.",
    )
    app_p.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    app_p.add_argument(
        "--server_port",
        type=int,
        default=None,
        help="Port for the Gradio server to listen on. Defaults to 7860.",
    )

    return parser


def main() -> None:
    """Dispatch to the requested subcommand."""
    args = get_parser().parse_args()

    from zea.internal.device import init_device

    init_device(args.device)

    if args.command == "process":
        from zea.data.process import run_processing

        config_path = args.config or f"{args.dataset}/config.yaml"
        run_processing(
            args.dataset,
            config_path,
            args.key,
            args.n_frames,
            args.save_dir,
            args.save_as,
            args.keep_keys,
            args.timings,
            args.num_threads,
            args.overwrite,
            args.keep_dynamic_range,
            args.revision,
            args.config_revision,
        )

    elif args.command == "app":
        from zea.data.app import CSS, build_interface

        import gradio as gr

        demo = build_interface()
        demo.launch(
            share=args.share,
            server_port=args.server_port,
            theme=gr.themes.Soft(primary_hue="violet", secondary_hue="yellow"),
            css=CSS,
        )


if __name__ == "__main__":
    main()
