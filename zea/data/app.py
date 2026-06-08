"""Gradio visualiser for zea datasets.

Usage:
    python -m zea.data.app
    python -m zea.data.app --share
    python -m zea.data.app --server_port 7861
"""

import argparse
import base64
import contextlib
import io
import os
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("ZEA_LOG_LEVEL", "WARNING")

import numpy as np
from keras import ops

from zea import display, io_lib
from zea.config import Config
from zea.data.datasets import Dataset
from zea.data.file import File
from zea.data.process import _get_config_parameters
from zea.internal.device import init_device
from zea.ops.pipeline import Pipeline

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "gradio is required for the zea app. Install with: pip install 'zea[app]'"
    ) from exc


# ── Logo ───────────────────────────────────────────────────────────────────────

_LOGO_PATH = Path(__file__).parent.parent.parent / "docs/_static/zea-logo.png"


def _logo_html(height: int = 30) -> str:
    try:
        with open(_LOGO_PATH, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="height:{height}px;width:auto;max-height:{height}px;'
            'vertical-align:middle;margin-right:8px;display:inline-block" />'
        )
    except Exception:
        return ""


# ── Colours from the zea logo ─────────────────────────────────────────────────

_YELLOW = "#f5c518"
_PURPLE = "#9333ea"

# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "PICMUS — experiment contrast speckle RF": {
        "dataset": (
            "hf://zeahub/picmus/database/experiments/contrast_speckle/"
            "contrast_speckle_expe_dataset_rf"
        ),
        "config": "hf://zeahub/picmus/config_rf.yaml",
        "key": "data/raw_data",
    },
    "PICMUS — experiment resolution distortion IQ": {
        "dataset": (
            "hf://zeahub/picmus/database/experiments/resolution_distorsion/"
            "resolution_distorsion_expe_dataset_iq"
        ),
        "config": "hf://zeahub/picmus/config_iq.yaml",
        "key": "data/raw_data",
    },
    "zea cardiac 2026": {
        "dataset": "hf://zeahub/zea-cardiac-2026",
        "config": "hf://zeahub/zea-cardiac-2026/config.yaml",
        "key": "data/raw_data",
    },
    "zea carotid 2023": {
        "dataset": "hf://zeahub/zea-carotid-2023",
        "config": "hf://zeahub/zea-carotid-2023/config.yaml",
        "key": "data/raw_data",
    },
}

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
footer { display: none !important; }
.status-box { max-height: 380px; overflow-y: auto; scroll-behavior: smooth; }
"""

_SCROLL_JS = """
() => {
    requestAnimationFrame(() => {
        const el = document.querySelector('.status-box');
        if (el) el.scrollTop = el.scrollHeight;
    });
}
"""

# ── Stop signal ───────────────────────────────────────────────────────────────

_stop_event = threading.Event()

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)


def _is_hf(path: str) -> bool:
    return str(path).strip().startswith("hf://")


def _enrich_error(exc: Exception) -> str:
    """Return a user-friendly string for common HF and runtime errors."""
    try:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
        )
        from huggingface_hub.utils import HFValidationError

        if isinstance(exc, GatedRepoError):
            return (
                str(exc)
                + "\n\nThis repository is gated. Accept the terms on Hugging Face "
                "and set the HF_TOKEN environment variable."
            )
        if isinstance(exc, RepositoryNotFoundError):
            return (
                str(exc)
                + "\n\nRepository not found. Check that the path is correct. "
                "If the repo is private, set the HF_TOKEN environment variable."
            )
        if isinstance(exc, EntryNotFoundError):
            return str(exc) + "\n\nFile not found in the repository. Check the path."
        if isinstance(exc, HFValidationError):
            return str(exc) + "\n\nInvalid Hugging Face repository ID format."
    except ImportError:
        pass
    return str(exc)


def _html_pass(msg: str) -> str:
    return f'<p style="margin:2px 0;color:#22c55e">&#10004; {msg}</p>'


def _html_fail(msg: str, err: Exception | str | None = None) -> str:
    html = f'<p style="margin:2px 0;color:#ef4444">&#10008; {msg}</p>'
    if err is not None:
        detail = _enrich_error(err) if isinstance(err, Exception) else str(err)
        escaped = (
            detail.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        html += (
            f'<p style="margin:2px 0 2px 1.5em;font-size:0.85em;color:#ef4444">'
            f"{escaped}</p>"
        )
    return html


def _html_warn(msg: str) -> str:
    return f'<p style="margin:2px 0;color:{_YELLOW}">&#9888; {msg}</p>'


def _html_info(msg: str) -> str:
    return f'<p style="margin:2px 0;color:{_YELLOW}">&#8250; {msg}</p>'


def _html_progress(current: int, total: int) -> str:
    pct = int(current / total * 100)
    return (
        f'<div style="margin:4px 0">'
        f'<span style="color:{_YELLOW};font-size:0.9em">Processing frame {current}/{total}</span>'
        f'<div style="background:#374151;border-radius:3px;height:5px;margin-top:3px">'
        f'<div style="background:{_PURPLE};border-radius:3px;height:5px;width:{pct}%"></div>'
        f'</div></div>'
    )


def _fetch_hf_revisions(path: str) -> list[str]:
    """Return branches + tags for the given hf:// path's repo."""
    try:
        from huggingface_hub import list_repo_refs

        parts = path.removeprefix("hf://").strip("/").split("/")
        if len(parts) < 2 or not parts[1]:
            return ["main"]
        repo_id = "/".join(parts[:2])
        refs = list_repo_refs(repo_id, repo_type="dataset")
        branches = [b.name for b in refs.branches]
        tags = [t.name for t in refs.tags]
        all_revs = branches + tags
        return all_revs if all_revs else ["main"]
    except Exception:
        return ["main"]


def _load_config_text(path: str, revision: str | None = None) -> str:
    """Fetch YAML from path (hf:// or local) and return as a string."""
    path = (path or "").strip()
    revision = (revision or "").strip() or None
    if not path:
        return "# No config path specified."
    try:
        if path.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            parts = path.removeprefix("hf://").split("/")
            repo_id = "/".join(parts[:2])
            filepath = "/".join(parts[2:])
            local = hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type="dataset",
                revision=revision,
            )
            with open(local) as fh:
                return fh.read()
        else:
            with open(path) as fh:
                return fh.read()
    except Exception as exc:
        return f"# Failed to load config:\n# {exc}"


# ── Core check pipeline ────────────────────────────────────────────────────────


def run_checks(
    dataset_path: str,
    config_path: str,
    dataset_revision: str | None = None,
    config_revision: str | None = None,
    key: str = "data/raw_data",
    file_index: int = 0,
    start_frame: int = 0,
    n_frames: int = 1,
    keep_keys: tuple = ("maxval",),
    stop_check=None,
):
    """Validate and beamform frame(s) from a zea dataset; yields ``(html, image)`` pairs.

    Each yield updates the UI after one check step. Stops immediately after any
    failure. For ``n_frames == 1`` the image is a PIL Image; for ``n_frames > 1``
    it is a file-path string pointing to an animated GIF.
    """
    file_index = int(file_index)
    start_frame = int(start_frame)
    n_frames = max(1, int(n_frames))
    lines: list[str] = []

    def _stopped():
        return stop_check is not None and stop_check()

    def _emit(line, image=None):
        lines.append(line)
        return "".join(lines), image

    def _replace_last(line, image=None):
        if lines:
            lines[-1] = line
        else:
            lines.append(line)
        return "".join(lines), image

    dataset_hf_kwargs = {"revision": dataset_revision} if dataset_revision else {}
    eff_config_rev = config_revision if config_revision is not None else dataset_revision
    config_hf_kwargs = {"revision": eff_config_rev} if eff_config_rev else {}

    # HF token check ────────────────────────────────────────────────────
    if _is_hf(dataset_path) or _is_hf(config_path):
        has_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if not has_token:
            try:
                from huggingface_hub import get_token as _get_hf_token

                has_token = bool(_get_hf_token())
            except Exception:
                pass
        if not has_token:
            yield _emit(
                _html_warn(
                    "No HF token found — set the HF_TOKEN environment variable or run "
                    "<code>huggingface-cli login</code>. "
                    "Private repos will fail and downloads may be rate-limited."
                )
            )

    # 1. Open dataset ──────────────────────────────────────────────────
    _src = "from HF" if _is_hf(dataset_path) else "from disk"
    yield _emit(_html_info(f"Opening dataset {_src}…"))
    try:
        ds = Dataset(dataset_path, validate=False, **dataset_hf_kwargs)
    except Exception as exc:
        yield _replace_last(_html_fail("Open dataset", exc))
        return

    num_files = len(ds)
    if num_files == 0:
        yield _replace_last(_html_fail("Open dataset", "Dataset is empty (0 files)."))
        return
    if file_index >= num_files:
        yield _replace_last(
            _html_fail(
                "File index out of range",
                f"File index {file_index} >= {num_files} files in dataset.",
            )
        )
        return
    yield _replace_last(_html_pass(f"Dataset opened — {num_files} file(s)"))
    if _stopped():
        return

    file_path = ds.file_paths[file_index]
    ds.close()

    # 2. Load config ────────────────────────────────────────────────────
    _src = "from HF" if _is_hf(config_path) else "from disk"
    yield _emit(_html_info(f"Loading config {_src}…"))
    try:
        config = Config.from_path(config_path, **config_hf_kwargs)
        config_params = _get_config_parameters(config)
    except Exception as exc:
        yield _replace_last(_html_fail("Load config", exc))
        return
    yield _replace_last(_html_pass("Config loaded"))
    if _stopped():
        return

    # 3. Build pipeline ─────────────────────────────────────────────────
    _src = "from HF" if _is_hf(config_path) else "from disk"
    yield _emit(_html_info(f"Building pipeline {_src}…"))
    try:
        pipeline = Pipeline.from_path(config_path, with_batch_dim=False, **config_hf_kwargs)
    except Exception as exc:
        yield _replace_last(_html_fail("Build pipeline", exc))
        return
    yield _replace_last(_html_pass("Pipeline built"))
    if _stopped():
        return

    # 4. Load parameters + validate frame range ─────────────────────────
    try:
        with File(file_path) as f:
            parameters = _run_quiet(f.load_parameters, **config_params)
            total_frames = f[key].shape[0]
    except Exception as exc:
        yield _emit(_html_fail("Load parameters", exc))
        return

    if start_frame >= total_frames:
        yield _emit(
            _html_fail(
                "Frame index out of range",
                f"Start frame {start_frame} >= {total_frames} frames in file.",
            )
        )
        return

    end_frame = min(start_frame + n_frames, total_frames)
    actual_n = end_frame - start_frame
    if actual_n < n_frames:
        yield _emit(
            _html_warn(
                f"Requested {n_frames} frames but only {actual_n} available "
                f"(frames {start_frame}–{end_frame - 1})."
            )
        )
    yield _emit(_html_pass(f"Parameters loaded — {total_frames} frame(s) in file"))
    if _stopped():
        return

    # 5. Prepare pipeline parameters ────────────────────────────────────
    try:
        params = _run_quiet(pipeline.prepare_parameters, parameters, **config_params)
    except Exception as exc:
        yield _emit(_html_fail("Prepare parameters", exc))
        return
    yield _emit(_html_pass("Parameters prepared"))
    if _stopped():
        return

    # 6. Process frames ─────────────────────────────────────────────────
    selected_transmits = np.array([int(t) for t in parameters.selected_transmits])
    dr = getattr(parameters, "dynamic_range", None)
    dynamic_range = tuple(dr) if dr is not None else (-60, 0)

    processed_frames: list[np.ndarray] = []
    for i, frame_idx in enumerate(range(start_frame, end_frame)):
        try:
            with File(file_path) as f:
                frame = np.asarray(f[key][frame_idx])
            frame = frame[selected_transmits]
            output = _run_quiet(pipeline, data=frame, **params)
            processed = output["data"]
            for k in keep_keys:
                if k in output:
                    params[k] = output[k]
        except Exception as exc:
            yield _emit(_html_fail(f"Run pipeline (frame {frame_idx})", exc))
            return

        processed_frames.append(ops.convert_to_numpy(processed))

        pbar = _html_progress(i + 1, actual_n)
        if i == 0:
            yield _emit(pbar)
        else:
            yield _replace_last(pbar)

        if _stopped():
            return

    # 7. Convert to image / GIF ─────────────────────────────────────────
    try:
        if actual_n == 1:
            result_image = display.to_8bit(processed_frames[0], dynamic_range, pillow=True)
        else:
            frames_u8 = [
                display.to_8bit(f, dynamic_range, pillow=False) for f in processed_frames
            ]
            video = np.stack(frames_u8, axis=0)
            try:
                fps = int(round(parameters.frames_per_second))
            except (ValueError, AttributeError):
                fps = 20
            tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
            io_lib.save_video(video, Path(tmp.name), fps=fps)
            result_image = tmp.name
    except Exception as exc:
        yield _emit(_html_fail("Convert to image", exc))
        return

    frame_label = (
        f"frame {start_frame}"
        if actual_n == 1
        else f"frames {start_frame}–{end_frame - 1}"
    )
    done_html = (
        f'<hr style="margin:6px 0;border-color:#374151">'
        f'<p style="margin:4px 0;color:{_YELLOW}"><b>&#10004; Processing done</b>'
        f' <span style="color:#6b7280">— file {file_index + 1}/{num_files}'
        f" &middot; {frame_label}</span></p>"
    )
    # Replace progress bar line with done message
    yield _replace_last(done_html, result_image)

    # Warnings ──────────────────────────────────────────────────────────
    if not _is_hf(dataset_path):
        yield _emit(_html_warn("Local dataset path — not yet on Hugging Face."), result_image)
    if not _is_hf(config_path):
        yield _emit(_html_warn("Local config path — not yet on Hugging Face."), result_image)
    if _is_hf(dataset_path) and _is_hf(config_path):
        rp = str(dataset_path).removeprefix("hf://").rstrip("/").split("/")[:2]
        cp = str(config_path).removeprefix("hf://").rstrip("/").split("/")[:2]
        if rp != cp:
            yield _emit(
                _html_warn("Dataset and config are on different HF repositories."),
                result_image,
            )


# ── Gradio interface ───────────────────────────────────────────────────────────


def build_interface() -> "gr.Blocks":
    """Build and return the Gradio Blocks interface."""

    logo = _logo_html(height=60)

    with gr.Blocks(title="zea visualizer") as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.HTML(
            f'<div style="display:flex;align-items:center;padding:8px 0 4px;'
            f'border-bottom:2px solid {_YELLOW};margin-bottom:6px">'
            f"{logo}"
            f'<span style="font-size:1.35em;font-weight:700;color:{_PURPLE}">zea</span>'
            f'<span style="font-size:1.35em;font-weight:400;margin-left:5px">'
            f"dataset visualizer</span>"
            f"</div>"
        )

        # ── Preset selector ─────────────────────────────────────────────────
        preset_selector = gr.Dropdown(
            label="Preset",
            choices=list(PRESETS.keys()),
            value=None,
            interactive=True,
            info="Select a preset to fill in the fields, or type a custom path.",
        )

        # ── Main row ────────────────────────────────────────────────────────
        with gr.Row():

            # Left: tabbed controls ─────────────────────────────────────────
            with gr.Column(scale=1, min_width=360):
                with gr.Tabs():

                    with gr.Tab("Settings"):
                        with gr.Row():
                            dataset_input = gr.Textbox(
                                label="Dataset path",
                                placeholder="hf://zeahub/… or /local/path",
                                scale=4,
                            )
                            dataset_rev_input = gr.Dropdown(
                                label="Revision",
                                choices=["main"],
                                value=None,
                                allow_custom_value=True,
                                interactive=False,
                                scale=1,
                                min_width=110,
                                info=" ",
                            )
                        with gr.Row():
                            config_input = gr.Textbox(
                                label="Config path",
                                placeholder="hf://… or /local/config.yaml",
                                scale=4,
                            )
                            config_rev_input = gr.Dropdown(
                                label="Revision",
                                choices=["main"],
                                value=None,
                                allow_custom_value=True,
                                interactive=False,
                                scale=1,
                                min_width=110,
                                info=" ",
                            )
                        key_input = gr.Textbox(label="Data key", value="data/raw_data")
                        with gr.Row():
                            file_index_input = gr.Number(
                                label="File index", value=0, minimum=0, step=1, precision=0
                            )
                            start_frame_input = gr.Number(
                                label="Start frame", value=0, minimum=0, step=1, precision=0
                            )
                            n_frames_input = gr.Number(
                                label="N frames (>1 → GIF)",
                                value=1,
                                minimum=1,
                                step=1,
                                precision=0,
                            )
                        with gr.Row():
                            run_btn = gr.Button("Run", variant="primary", scale=3)
                            stop_btn = gr.Button(
                                "Stop", variant="stop", scale=1, interactive=False
                            )

                    with gr.Tab("Config editor"):
                        load_config_btn = gr.Button("Load config from path", size="sm")
                        config_editor = gr.Code(
                            label="Config YAML",
                            language="yaml",
                            lines=22,
                        )

            # Right: image + status ─────────────────────────────────────────
            with gr.Column(scale=2):
                image_output = gr.Image(
                    label="Output",
                    type="filepath",
                    height=440,
                )
                status_output = gr.HTML(
                    label="Status",
                    elem_classes=["status-box"],
                )

        # ── Event wiring ────────────────────────────────────────────────────

        # Toggle revision interactive state on each keystroke (fast, no network)
        def _rev_toggle(path):
            return gr.update(interactive=_is_hf(path))

        dataset_input.change(_rev_toggle, inputs=[dataset_input], outputs=[dataset_rev_input])
        config_input.change(_rev_toggle, inputs=[config_input], outputs=[config_rev_input])

        # Fetch revisions + auto-fill config when user leaves dataset field
        def _on_dataset_blur(path):
            path = path.strip()
            if not _is_hf(path):
                return gr.update(), gr.update(interactive=False, choices=["main"], value=None)
            revisions = _fetch_hf_revisions(path)
            default = "main" if "main" in revisions else (revisions[0] if revisions else "main")
            config_prefill = f"{path.rstrip('/')}/config.yaml"
            return (
                gr.update(value=config_prefill),
                gr.update(interactive=True, choices=revisions, value=default),
            )

        dataset_input.blur(
            _on_dataset_blur,
            inputs=[dataset_input],
            outputs=[config_input, dataset_rev_input],
        )

        # Fetch config revisions when user leaves config field
        def _on_config_blur(path):
            path = path.strip()
            if not _is_hf(path):
                return gr.update(interactive=False, choices=["main"], value=None)
            revisions = _fetch_hf_revisions(path)
            default = "main" if "main" in revisions else (revisions[0] if revisions else "main")
            return gr.update(interactive=True, choices=revisions, value=default)

        config_input.blur(
            _on_config_blur,
            inputs=[config_input],
            outputs=[config_rev_input],
        )

        # Sync dataset ↔ config revision together (one-way per user action, no cascade)
        dataset_rev_input.change(
            lambda v: gr.update(value=v),
            inputs=[dataset_rev_input],
            outputs=[config_rev_input],
        )
        config_rev_input.change(
            lambda v: gr.update(value=v),
            inputs=[config_rev_input],
            outputs=[dataset_rev_input],
        )

        # Preset fills all fields (fetches revisions once)
        def _apply_preset(name):
            if name not in PRESETS:
                return (gr.update(),) * 6
            p = PRESETS[name]
            ds = p.get("dataset", "")
            cfg = p.get("config", "")
            ds_revs = _fetch_hf_revisions(ds) if _is_hf(ds) else ["main"]
            cfg_revs = _fetch_hf_revisions(cfg) if _is_hf(cfg) else ["main"]
            ds_def = "main" if "main" in ds_revs else (ds_revs[0] if ds_revs else "main")
            cfg_def = "main" if "main" in cfg_revs else (cfg_revs[0] if cfg_revs else "main")
            return (
                gr.update(value=ds),
                gr.update(value=cfg),
                gr.update(interactive=_is_hf(ds), choices=ds_revs, value=ds_def),
                gr.update(interactive=_is_hf(cfg), choices=cfg_revs, value=cfg_def),
                gr.update(value=p.get("key", "data/raw_data")),
                gr.update(value=None),  # clear image
            )

        preset_selector.change(
            _apply_preset,
            inputs=[preset_selector],
            outputs=[
                dataset_input,
                config_input,
                dataset_rev_input,
                config_rev_input,
                key_input,
                image_output,
            ],
        )

        # Config editor — pass revision so the correct tag/branch is fetched
        load_config_btn.click(
            _load_config_text, inputs=[config_input, config_rev_input], outputs=[config_editor]
        )

        # Run generator
        def _on_run(dataset, config, ds_rev, cfg_rev, key, file_idx, start_f, n_f, editor_yaml):
            _stop_event.clear()
            dataset = (dataset or "").strip()
            config = (config or "").strip()
            if not dataset:
                yield _html_fail("Please enter a dataset path."), None
                return
            config_resolved = config or f"{dataset}/config.yaml"
            tmp_cfg = None

            if editor_yaml and editor_yaml.strip() and not editor_yaml.strip().startswith("#"):
                tmp_cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
                tmp_cfg.write(editor_yaml)
                tmp_cfg.close()
                config_resolved = tmp_cfg.name
                cfg_rev = None

            try:
                for html, img in run_checks(
                    dataset,
                    config_resolved,
                    ds_rev if ds_rev else None,
                    cfg_rev if cfg_rev else None,
                    (key or "data/raw_data").strip() or "data/raw_data",
                    int(file_idx or 0),
                    int(start_f or 0),
                    int(n_f or 1),
                    stop_check=_stop_event.is_set,
                ):
                    if img is None:
                        yield html, None
                    elif isinstance(img, str):
                        yield html, img
                    else:
                        tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        img.save(tmp_png.name)
                        yield html, tmp_png.name
            except Exception as exc:
                import traceback as _tb
                yield (
                    _html_fail("Unexpected error", exc)
                    + f'<pre style="font-size:0.75em;color:#6b7280;white-space:pre-wrap">'
                    f"{_tb.format_exc()}</pre>"
                ), None
            finally:
                if tmp_cfg is not None:
                    try:
                        os.unlink(tmp_cfg.name)
                    except OSError:
                        pass

        run_event = run_btn.click(
            _on_run,
            inputs=[
                dataset_input,
                config_input,
                dataset_rev_input,
                config_rev_input,
                key_input,
                file_index_input,
                start_frame_input,
                n_frames_input,
                config_editor,
            ],
            outputs=[status_output, image_output],
        )

        def _on_stop():
            _stop_event.set()

        stop_btn.click(_on_stop, cancels=[run_event])

        # Auto-scroll status box on each update
        status_output.change(fn=None, js=_SCROLL_JS)

        # Pre-load default config at startup (no-op if config_input is empty)
        demo.load(_load_config_text, inputs=[config_input], outputs=[config_editor])

    return demo


# ── CLI ────────────────────────────────────────────────────────────────────────


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the zea Gradio visualizer.")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link."
    )
    parser.add_argument("--server_port", type=int, default=None, help="Port to listen on.")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    init_device()
    demo = build_interface()
    demo.launch(
        share=args.share,
        server_port=args.server_port,
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="yellow"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
