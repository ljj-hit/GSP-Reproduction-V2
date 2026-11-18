"""Strict CUDA checks for GSP-Reproduction-V2.

Formal training/evaluation never silently falls back to CPU.  If CUDA is not
available, this module raises an actionable error instead.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Optional, Sequence

import torch


CUDA_INSTALL_HINT = (
    "CUDA is required for formal GSP training. Install an NVIDIA driver and a "
    "CUDA-enabled PyTorch build, for example: pip install torch --index-url "
    "https://download.pytorch.org/whl/cu128 . Then run 'python check_cuda.py'."
)


def print_cuda_info(device: torch.device) -> None:
    """Print the environment details required at training start."""
    if not torch.cuda.is_available():
        print("CUDA available: False")
        print(CUDA_INSTALL_HINT)
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"NVIDIA GPU name: {torch.cuda.get_device_name(index)}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device index: {index} ({device})")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"GPU memory total: {properties.total_memory / (1024 ** 3):.2f} GiB")
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        print(f"GPU memory available: {free_bytes / (1024 ** 3):.2f} GiB")
        print(f"GPU memory total (runtime): {total_bytes / (1024 ** 3):.2f} GiB")
    except Exception as exc:  # explicit diagnostic, not a silent fallback
        print(f"GPU memory availability query failed: {exc}")


def require_cuda(device: str | torch.device = "cuda:0", *, verbose: bool = True) -> torch.device:
    """Validate CUDA availability/device index and return a CUDA device.

    The function deliberately raises rather than returning a CPU device.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(CUDA_INSTALL_HINT)
    if torch.version.cuda is None:
        raise RuntimeError(
            "This PyTorch build has no CUDA support. " + CUDA_INSTALL_HINT
        )
    selected = torch.device(device)
    if selected.type != "cuda":
        raise RuntimeError(
            f"Requested device '{selected}' is not CUDA. Formal training/evaluation "
            "must run on an NVIDIA GPU."
        )
    index = selected.index
    if index is None:
        index = torch.cuda.current_device()
        selected = torch.device(f"cuda:{index}")
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is invalid; visible device count is "
            f"{torch.cuda.device_count()}."
        )
    if verbose:
        print_cuda_info(selected)
    return selected


def assert_model_on_cuda(model: torch.nn.Module, device: str | torch.device) -> None:
    target = torch.device(device)
    if target.type != "cuda":
        raise RuntimeError(f"Expected a CUDA target device, got {target}")
    parameters = list(model.parameters())
    if not parameters:
        raise RuntimeError("Model has no parameters; cannot verify GPU placement")
    for index, parameter in enumerate(parameters):
        if parameter.device.type != "cuda" or parameter.device != target:
            raise RuntimeError(
                f"Model parameter {index} is on {parameter.device}, expected {target}"
            )


def assert_batch_on_cuda(
    batch: Mapping[str, Any],
    device: str | torch.device,
) -> None:
    target = torch.device(device)
    if target.type != "cuda":
        raise RuntimeError(f"Expected a CUDA target device, got {target}")
    tensor_count = 0
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor_count += 1
        if value.device.type != "cuda" or value.device != target:
            raise RuntimeError(
                f"Batch tensor '{key}' is on {value.device}, expected {target}"
            )
    if tensor_count == 0:
        raise RuntimeError("Batch contains no tensors; cannot verify GPU placement")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check strict CUDA requirements")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cuda", action="store_true", help="alias for --device cuda:0")
    args = parser.parse_args(argv)
    if args.cuda:
        args.device = "cuda:0"
    try:
        device = require_cuda(args.device, verbose=True)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    # Small smoke allocation to prove kernels can actually execute (not just
    # that the driver reports a device).
    try:
        probe = torch.arange(8, device=device, dtype=torch.float32).square().sum()
        torch.cuda.synchronize(device)
        print(f"CUDA tensor smoke test: sum(x^2) = {probe.item():.1f}")
        print("CUDA check passed.")
    except Exception as exc:
        print(f"ERROR: CUDA device is visible but a tensor operation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())