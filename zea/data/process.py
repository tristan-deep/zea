"""CLI for beamforming a zea dataset with a pipeline defined in a YAML config file.

Usage:
    python -m zea.data.process <dataset> <save_dir> --config <config.yaml>
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields as dataclass_fields
from pathlib import Path

import keras
import numpy as np
from keras import ops

from zea import display, io_lib, log
from zea.config import Config
from zea.data.dataloader import Dataloader
from zea.data.datasets import Dataset
from zea.data.file import File
from zea.data.spec import ScanSpec
from zea.internal.device import init_device
from zea.ops.pipeline import Pipeline
from zea.utils import FunctionTimer

SUPPORTED_FORMATS = ["gif", "mp4", "hdf5"]

try:
    import SimpleITK as sitk

    SUPPORTED_FORMATS += ["nii.gz"]
except ImportError:
    sitk = None


def get_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Beamform a zea dataset using a pipeline defined in a config YAML file. "
            "Processes frames sequentially to support temporal algorithms."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path/URI to the zea dataset (folder of HDF5 files or a single HDF5 file).",
    )
    parser.add_argument(
        "save_dir",
        type=Path,
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml. Defaults to <dataset>/config.yaml.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="data/raw_data",
        help="Data type to load from each file (e.g. data/raw_data, data/aligned_data).",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process per file (all frames when omitted).",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="gif",
        help=f"Output format. One of: {', '.join(SUPPORTED_FORMATS)}.",
    )
    parser.add_argument(
        "--keep_keys",
        nargs="+",
        default=["maxval"],
        help="Pipeline output keys to forward to the next frame iteration.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Record dataloader and pipeline timings and save to YAML files in save_dir.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads used by the dataloader. Default is 16.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "HuggingFace revision for the dataset (branch, tag, or commit hash). "
            "Only used for hf:// paths."
        ),
    )
    parser.add_argument(
        "--config_revision",
        type=str,
        default=None,
        help=(
            "HuggingFace revision for the config (branch, tag, or commit hash). "
            "Defaults to --revision if omitted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. Default is False.",
    )
    parser.add_argument(
        "--keep_dynamic_range",
        action="store_true",
        help=(
            "Store pipeline output as-is (float32 dB) instead of converting to uint8. "
            "Only valid when --save_as hdf5."
        ),
    )
    return parser


def _get_config_parameters(config: Config) -> dict:
    """Return the config parameters dict, handling missing or empty sections."""
    params = getattr(config, "parameters", None)
    if params is None:
        return {}
    return params.as_dict() if hasattr(params, "as_dict") else dict(params)


# Keys that carry raw RF / pre-beamformed data and always require a pipeline.
_PIPELINE_REQUIRED_KEYS = frozenset({"data/raw_data", "data/aligned_data/values"})


def _key_requires_pipeline(key: str) -> bool:
    """Return True if ``key`` holds raw RF/pre-beamformed data that needs a pipeline."""
    return (key or "").strip() in _PIPELINE_REQUIRED_KEYS


def _build_probe_dict(probe) -> dict:
    """Build a minimal probe dict for File.create() from a Probe object."""
    probe_dict = {}
    if getattr(probe, "name", None):
        probe_dict["name"] = probe.name
    if getattr(probe, "probe_geometry", None) is not None:
        probe_dict["probe_geometry"] = probe.probe_geometry
    for field in (
        "type",
        "probe_center_frequency",
        "probe_bandwidth_percent",
        "element_width",
        "element_height",
        "lens_sound_speed",
        "lens_thickness",
    ):
        val = getattr(probe, field, None)
        if val is not None:
            probe_dict[field] = val
    return probe_dict


def _run_passthrough(
    dataset_path: str,
    key: str,
    n_frames: int | None,
    save_dir: Path,
    save_as: str,
    overwrite: bool,
    **hf_kwargs,
) -> None:
    """Save data frames directly without a beamforming pipeline."""
    if save_as not in ("gif", "mp4", "hdf5"):
        raise ValueError(f"Passthrough mode only supports gif/mp4/hdf5, got {save_as!r}")

    ds = Dataset(dataset_path, validate=False, **hf_kwargs)
    file_paths = list(ds.file_paths)
    ds.close()

    pbar = keras.utils.Progbar(len(file_paths))
    for file_path in file_paths:
        with File(file_path) as f:
            data_key = f.format_key(key)
            arr = np.asarray(f[data_key][:n_frames] if n_frames else f[data_key][:])
            filestem = f.stem

        # Ensure (N, H, W) — squeeze any leading single-element dims
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # add frame axis

        if arr.dtype != np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            arr = (
                ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
                if hi > lo
                else np.zeros_like(arr, dtype=np.uint8)
            )

        save_path = save_dir / f"{filestem}.{save_as}"
        if save_path.exists() and not overwrite:
            log.warning(f"File {save_path} already exists. Use --overwrite to replace it.")
        else:
            if save_as in ("gif", "mp4"):
                io_lib.save_video(arr, save_path, fps=20)
            elif save_as == "hdf5":
                File.create(save_path, data={"image": {"values": arr}}, overwrite=overwrite)
            log.info(f"Saved {log.yellow(save_path)}")

        pbar.add(1)


def run_processing(
    dataset_path: str,
    config_path: str,
    key: str,
    n_frames: int | None,
    save_dir: Path,
    save_as: str = "gif",
    keep_keys=("maxval",),
    timings=False,
    num_threads=16,
    overwrite=False,
    keep_dynamic_range=False,
    revision: str | None = None,
    config_revision: str | None = None,
) -> None:
    if keep_dynamic_range and save_as != "hdf5":
        raise ValueError("--keep_dynamic_range is only supported with --save_as hdf5.")
    if save_as == "nii.gz" and sitk is None:
        raise ValueError("SimpleITK is not installed; cannot save as nii.gz.")
    if save_as not in SUPPORTED_FORMATS:
        raise ValueError(f"save_as must be one of {SUPPORTED_FORMATS}, got {save_as!r}")

    dataset_hf_kwargs = {"revision": revision} if revision is not None else {}
    config_hf_kwargs = (
        {"revision": config_revision if config_revision is not None else revision}
        if (config_revision or revision)
        else {}
    )
    config = Config.from_path(config_path, **config_hf_kwargs)
    config_params = _get_config_parameters(config)
    try:
        pipeline = Pipeline.from_path(config_path, with_batch_dim=False, **config_hf_kwargs)
    except (ValueError, KeyError) as exc:
        if _key_requires_pipeline(key):
            raise
        log.warning(
            f"No pipeline found in config ({exc}). "
            f"Key '{key}' does not require beamforming — saving data as-is."
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        _run_passthrough(
            dataset_path, key, n_frames, save_dir, save_as, overwrite, **dataset_hf_kwargs
        )
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = Dataset(dataset_path, validate=False, **dataset_hf_kwargs)
    dataloader = Dataloader(
        dataset_path,
        key=key,
        batch_size=None,
        shuffle=False,
        return_filename=True,
        limit_n_frames=n_frames,
        n_frames=1,
        num_threads=num_threads,
        insert_frame_axis=False,
        sort_files=True,
        **dataset_hf_kwargs,
    )
    dataset_files.close()

    iterator = iter(dataloader)
    total_batches = len(dataloader)

    get_data = lambda: next(iterator)
    prepare_parameters = pipeline.prepare_parameters
    pipeline_call = pipeline.__call__

    if timings:
        timer = FunctionTimer()
        get_data = timer(get_data, name="dataloader")
        prepare_parameters = timer(prepare_parameters, name="prepare_parameters")
        pipeline_call = timer(pipeline_call, name="pipeline")

    _DEFAULT_FPS = 20
    _scan_spec_fields = {f.name for f in dataclass_fields(ScanSpec)}

    prev_file_path = None
    data_output = []
    filestem = None
    parameters = None
    selected_transmits = None
    params = None
    fps = _DEFAULT_FPS

    def save_video_worker(
        video: np.ndarray,
        save_path: Path,
        src_file_path: str,
        fps: int,
    ):
        if save_path.exists() and not overwrite:
            log.warning(f"File {save_path} already exists. Use --overwrite to replace it.")
            return
        if save_as in ["mp4", "gif"]:
            io_lib.save_video(video, save_path, fps=fps)
        elif save_as == "hdf5":
            with File(src_file_path) as src:
                scan_dict = {
                    k: v for k, v in src.get_scan_parameters().items() if k in _scan_spec_fields
                }
                probe_dict = _build_probe_dict(src.probe)
            File.create(
                save_path,
                data={"image": {"values": video}},
                scan=scan_dict if scan_dict else None,
                probe=probe_dict if probe_dict else None,
                overwrite=overwrite,
            )
        elif save_as == "nii.gz":
            sitk.WriteImage(sitk.GetImageFromArray(video), str(save_path))
            log.info(f"Saved NIfTI to {log.yellow(save_path)}")

    pbar = keras.utils.Progbar(total_batches)

    with ThreadPoolExecutor(max_workers=1) as executor:
        save_future = None
        for i in range(total_batches + 1):
            if i < total_batches:
                frame, metadata = get_data()
                file_path = metadata["fullpath"]
            else:
                file_path = None  # sentinel to flush the last file

            if file_path != prev_file_path:
                if prev_file_path is not None:
                    video = np.stack([ops.convert_to_numpy(f) for f in data_output], axis=0)
                    save_path = save_dir / f"{filestem}.{save_as}"
                    if save_future is not None:
                        save_future.result()
                    save_future = executor.submit(
                        save_video_worker, video, save_path, prev_file_path, fps
                    )
                    data_output = []
                    if file_path is None:
                        break

                prev_file_path = file_path
                with File(file_path) as f:
                    filestem = f.stem
                    parameters = f.load_parameters(**config_params)

                selected_transmits = np.array([int(t) for t in parameters.selected_transmits])
                try:
                    fps = int(round(parameters.frames_per_second))
                except (ValueError, AttributeError):
                    fps = _DEFAULT_FPS

                params = prepare_parameters(parameters, **config_params)

            # slice to selected transmits (transmit axis = 0 when insert_frame_axis=False)
            frame = frame[selected_transmits]

            output = pipeline_call(data=frame, **params)
            processed_frame = output["data"]

            if not keep_dynamic_range:
                dr = getattr(parameters, "dynamic_range", None)
                dynamic_range = tuple(dr) if dr is not None else (-60, 0)
                processed_frame = display.to_8bit(processed_frame, dynamic_range, pillow=False)

            data_output.append(processed_frame)
            pbar.add(1)

            for key in keep_keys:
                if key in output:
                    params[key] = output[key]

            if timings:
                for tname in timer.timings.keys():
                    timer.append_to_yaml(save_dir / f"timings_{tname}.yaml", tname)

    if timings:
        timer.print()


def main() -> None:
    args = get_parser().parse_args()
    init_device()
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


if __name__ == "__main__":
    main()
