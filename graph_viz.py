"""
Export and visualize NGSST graphs with minimal dependency footprint.

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

from ngsst_implementation import NGSST, NGSSTConfig

def setup_logging() -> None:
 logging.basicConfig(
 level=logging.INFO,
 format="%(asctime)s [%(levelname)s] %(message)s",
 )

def parse_args() -> argparse.Namespace:
 parser = argparse.ArgumentParser(
 description="Export NGSST computation graphs (FX, optional torchviz/ONNX)."
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
 "--frames",
 type=int,
 default=2,
 help="Temporal length for dummy input.",
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
 "--include-poses",
 action="store_true",
 default=True,
 help="Include camera poses to exercise geometric path.",
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

 def __init__(self, model: NGSST) -> None:
 super().__init__()
 self.model = model

 def forward(
 self, x: torch.Tensor, camera_poses: Optional[torch.Tensor] = None
 ) -> torch.Tensor:
 outputs = self.model(
 x,
 camera_poses=camera_poses,
 return_predictions=False,
 return_uncertainty=False,
 )
 return outputs["logits"]

def build_dummy_inputs(
 batch: int,
 frames: int,
 height: int,
 width: int,
 device: str,
 include_poses: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
 video = torch.randn(batch, frames, height, width, 3, device=device)
 poses: Optional[torch.Tensor] = None
 if include_poses:
 identity = torch.eye(4, device=device)
 poses = identity.unsqueeze(0).unsqueeze(0).repeat(batch, frames, 1, 1)
 return video, poses

def export_fx_graph(wrapper: ForwardWrapper, out_dir: Path) -> Path:
 wrapper.eval()
 traced = torch.fx.symbolic_trace(wrapper)
 dot_path = out_dir / "ngsst_fx.dot"
 try:
 from torch.fx.passes.graph_drawer import FxGraphDrawer

 drawer = FxGraphDrawer(traced, "NGSST-FX")
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
 fallback_path = out_dir / "ngsst_fx.txt"
 fallback_path.write_text(buf.getvalue(), encoding="utf-8")
 logging.info("FX tabular graph written to %s", fallback_path)
 return dot_path

def export_torchviz_graph(
 wrapper: ForwardWrapper,
 video: torch.Tensor,
 poses: Optional[torch.Tensor],
 out_dir: Path,
) -> Optional[Path]:
 try:
 from torchviz import make_dot # type: ignore
 except ImportError:
 logging.warning("torchviz is not installed; skipping torchviz export.")
 return None

 video.requires_grad_(True)
 if poses is not None:
 poses.requires_grad_(False)
 logits = wrapper(video, poses)
 graph = make_dot(logits, params=dict(wrapper.named_parameters()))
 render_path = out_dir / "ngsst_torchviz"
 graph.format = "pdf"
 graph.render(str(render_path), cleanup=True)
 logging.info("Torchviz autograd graph written to %s.pdf", render_path)
 return render_path.with_suffix(".pdf")

def export_onnx_graph(
 wrapper: ForwardWrapper,
 video: torch.Tensor,
 poses: Optional[torch.Tensor],
 out_dir: Path,
) -> Optional[Path]:
 onnx_path = out_dir / "ngsst.onnx"
 dummy_poses = poses
 if dummy_poses is None:
 batch, frames, _, _, _ = video.shape
 identity = torch.eye(4, device=video.device)
 dummy_poses = identity.unsqueeze(0).unsqueeze(0).repeat(batch, frames, 1, 1)
 logging.info("Generated identity camera poses for ONNX export.")
 try:
 torch.onnx.export(
 wrapper,
 (video, dummy_poses),
 onnx_path,
 opset_version=17,
 do_constant_folding=True,
 input_names=["video", "camera_poses"],
 output_names=["logits"],
 dynamic_axes={
 "video": {0: "batch", 1: "frames", 2: "height", 3: "width"},
 "camera_poses": {0: "batch", 1: "frames"},
 "logits": {0: "batch"},
 },
 )
 logging.info("ONNX graph written to %s", onnx_path)
 return onnx_path
 except Exception as exc: # pragma: no cover - runtime export errors
 logging.error("ONNX export failed: %s", exc)
 return None

def main() -> None:
 setup_logging()
 args = parse_args()
 args.out_dir.mkdir(parents=True, exist_ok=True)

 device = torch.device(args.device if torch.cuda.is_available() else "cpu")
 if args.device == "cuda" and device.type != "cuda":
 logging.warning("CUDA requested but not available; falling back to CPU.")

 config = NGSSTConfig()
 model = NGSST(config).to(device)
 wrapper = ForwardWrapper(model)

 video, poses = build_dummy_inputs(
 batch=args.batch,
 frames=args.frames,
 height=args.height,
 width=args.width,
 device=str(device),
 include_poses=args.include_poses,
 )

 with torch.no_grad():
 _ = wrapper(video, poses)

 export_fx_graph(wrapper, args.out_dir)

 if args.torchviz:
 export_torchviz_graph(wrapper, video, poses, args.out_dir)

 if args.onnx:
 export_onnx_graph(wrapper, video, poses, args.out_dir)

if __name__ == "__main__":
 main()
