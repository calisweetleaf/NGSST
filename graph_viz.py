"""
Export and visualize HVT v2.0 graphs with minimal dependency footprint.

Capabilities:
- FX graph to DOT (primary, no extra deps if graphviz is unavailable).
- Optional torchviz autograd graph when torchviz + graphviz are installed.
- Optional ONNX export for Netron inspection.

Usage (PowerShell):
    python graph_viz.py --out-dir artifacts/graphs --torchviz --onnx
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

from hvt_v2 import HarmonicVisionTransformer

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HVT v2.0 computation graphs (FX, optional torchviz/ONNX)."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/graphs"),
        help="Output directory for generated graph artifacts.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for dummy input.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Frame height for dummy input.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Frame width for dummy input.",
    )
    parser.add_argument(
        "--torchviz",
        action="store_true",
        help="Emit torchviz autograd graph if torchviz+graphviz are installed.",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Export ONNX graph for Netron if available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Computation device for dummy run and export.",
    )
    return parser.parse_args()

class ForwardWrapper(torch.nn.Module):
    """Thin wrapper to expose the model forward for tracing/export."""

    def __init__(self, model: HarmonicVisionTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x, return_reconstruction=False)
        return outputs["logits"]

def build_dummy_inputs(
    batch: int,
    height: int,
    width: int,
    device: str,
) -> torch.Tensor:
    image = torch.randn(batch, 3, height, width, device=device)
    return image

def export_fx_graph(wrapper: ForwardWrapper, out_dir: Path) -> Path:
    wrapper.eval()
    traced = torch.fx.symbolic_trace(wrapper)
    dot_path = out_dir / "hvt_v2_fx.dot"
    try:
        from torch.fx.passes.graph_drawer import FxGraphDrawer

        drawer = FxGraphDrawer(traced, "HVT-v2-FX")
        dot_source = drawer.get_dot_graph().source
        dot_path.write_text(dot_source, encoding="utf-8")
        logging.info("FX DOT graph written to %s", dot_path)
    except ImportError:
        logging.warning(
            "torch.fx graph drawer dependency missing (graphviz). "
            "Writing tabular graph instead."
        )
        buf = io.StringIO()
        traced.graph.print_tabular(file=buf)
        fallback_path = out_dir / "hvt_v2_fx.txt"
        fallback_path.write_text(buf.getvalue(), encoding="utf-8")
        logging.info("FX tabular graph written to %s", fallback_path)
    return dot_path

def export_torchviz_graph(
    wrapper: ForwardWrapper,
    image: torch.Tensor,
    out_dir: Path,
) -> Optional[Path]:
    try:
        from torchviz import make_dot
    except ImportError:
        logging.warning("torchviz is not installed; skipping torchviz export.")
        return None

    image.requires_grad_(True)
    logits = wrapper(image)
    graph = make_dot(logits, params=dict(wrapper.named_parameters()))
    render_path = out_dir / "hvt_v2_torchviz"
    graph.format = "pdf"
    graph.render(str(render_path), cleanup=True)
    logging.info("Torchviz autograd graph written to %s.pdf", render_path)
    return render_path.with_suffix(".pdf")

def export_onnx_graph(
    wrapper: ForwardWrapper,
    image: torch.Tensor,
    out_dir: Path,
) -> Optional[Path]:
    onnx_path = out_dir / "hvt_v2.onnx"
    try:
        torch.onnx.export(
            wrapper,
            (image,),
            onnx_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch"},
            },
        )
        logging.info("ONNX graph written to %s", onnx_path)
        return onnx_path
    except Exception as exc:
        logging.error("ONNX export failed: %s", exc)
        return None

def main() -> None:
    setup_logging()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        logging.warning("CUDA requested but not available; falling back to CPU.")

    model = HarmonicVisionTransformer(
        num_classes=10,
        num_freq_bands=4,
        hidden_dim=64,
        num_routing_heads=4,
        num_evolution_layers=3
    ).to(device)
    wrapper = ForwardWrapper(model)

    image = build_dummy_inputs(
        batch=args.batch,
        height=args.height,
        width=args.width,
        device=str(device),
    )

    with torch.no_grad():
        _ = wrapper(image)

    export_fx_graph(wrapper, args.out_dir)

    if args.torchviz:
        export_torchviz_graph(wrapper, image, args.out_dir)

    if args.onnx:
        export_onnx_graph(wrapper, image, args.out_dir)

if __name__ == "__main__":
    main()
