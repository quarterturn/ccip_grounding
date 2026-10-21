"""OpenVINO iGPU acceleration for the CCIP grounding tools (chibi, feat/igpu-ov).

Why: imgutils opens every ONNX model through onnxruntime via one function
(imgutils.utils.onnxruntime._open_onnx_model), and on this box the ORT
OpenVINO EP silently runs the YOLO detectors on CPU (its auto device is
CPU, and its GPU mode miscompiles dynamic-shape graphs -- facesaver's
OVFaceDetector lesson; the 1.24.1 EP doesn't even accept static_layers).
Measured on the i3-N305 iGPU feeding OpenVINO DIRECTLY: person detect
131 ms @640 vs 2233 ms ORT-CPU @1216 (>> @640); CCIP feat 231 ms/crop
batched vs 544 ms. Detection dominates the grounder's inner loop, so raw
OV is the win.

This patch replaces _open_onnx_model's return value with OVSession: a tiny
ORT-shaped facade (run / get_modelmeta) over a compiled openvino model.
Shape policy per model, mirroring what imgutils feeds it:
  * spatial dims already static (CCIP 384x384): keep them, batch dynamic.
  * spatial dims dynamic (YOLO detectors): compile at a STATIC square from
    the model's own imgsz metadata (imgutils always resizes to that square;
    its fallback default is 640, and OV GPU needs static spatial shapes --
    mandatory, not an optimization).
Fallback: any OV failure prints a loud WARNING and loads the original ORT
CPU path; results stay correct when the GPU says no.

Enable explicitly (never implicit): --ov-gpu on the CLI, or env CCIP_OV_GPU=1
(orchestrated runs inherit it through ground_all's subprocesses).
"""
import json
import os
import sys

_OV_TAG = "[ovshim]"


def _read_meta(ckpt):
    """custom_metadata_map from the ONNX protobuf (same source ORT's
    ModelMeta reads), so imgutils' imgsz/names parsing is unchanged."""
    import onnx
    m = onnx.load(ckpt, load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


class OVSession:
    """ORT InferenceSession-shaped facade over an OpenVINO GPU model."""

    def __init__(self, ckpt, core):
        import openvino as ov
        self._ckpt = ckpt
        self._meta = _read_meta(ckpt)

        model = core.read_model(ckpt)
        iname = model.input().get_any_name()
        dims = list(model.input().get_partial_shape())
        if not dims:
            raise RuntimeError(f"scalar input? shape {dims}")
        # spatial dims of a 4-D NCHW input; anything else (e.g. the CCIP
        # metric model, [?, 768]) keeps its non-batch dims as-is.
        spatial = dims[2:4] if len(dims) == 4 else []
        if len(dims) != 4 or all(d.is_static for d in spatial):
            # keep every non-batch dim, make batch dynamic
            shape = [ov.Dimension()] + [
                d.get_length() if d.is_static else d for d in dims[1:]]
            model.reshape({iname: ov.PartialShape(shape)})
        else:
            # dynamic spatial (YOLO detectors): static square from imgsz
            size = 640
            if "imgsz" in self._meta:
                wh = json.loads(self._meta["imgsz"])
                assert wh[0] == wh[1], f"non-square imgsz {wh}"
                size = int(wh[0])
            model.reshape({iname: [1, 3, size, size]})
        self._model = core.compile_model(model, "GPU")
        self._req = self._model.create_infer_request()

    # ---- ORT InferenceSession facade -------------------------------
    def run(self, output_names, input_feed, run_options=None):
        # imgutils always feeds exactly one input and unpacks outputs
        # positionally, in model order.
        (_key, value), = input_feed.items()
        return list(self._req.infer([value]).values())

    def get_modelmeta(self):
        outer = self

        class _M:
            custom_metadata_map = outer._meta
        return _M()


def _install():
    import openvino as ov
    import imgutils.utils.onnxruntime as iuo

    _orig = iuo._open_onnx_model
    _core = ov.Core()

    def _open_patched(ckpt, provider, use_cpu=True):
        try:
            sess = OVSession(ckpt, _core)
            parts = ckpt.split("/")
            label = "/".join(parts[-3:-1]) if len(parts) >= 3 else ckpt
            print(f"{_OV_TAG} OV-GPU: {label}", file=sys.stderr, flush=True)
            return sess
        except Exception as exc:
            print(f"WARNING {_OV_TAG}: OpenVINO GPU failed for {ckpt!r}: "
                  f"{type(exc).__name__}: {exc}\n  Falling back to the "
                  f"ONNX-Runtime CPU path for this model -- results stay "
                  f"correct, speed will not.", file=sys.stderr, flush=True)
            return _orig(ckpt, "CPUExecutionProvider", use_cpu)

    iuo._open_onnx_model = _open_patched


def enable_if_requested(cli_flag=False):
    """Activate the OV-GPU shim when the CLI flag or CCIP_OV_GPU env is set.

    Must be called BEFORE any model is loaded (same contract as --gpu).
    Returns True when the shim is active.
    """
    if not (cli_flag or os.environ.get("CCIP_OV_GPU")):
        return False
    try:
        import openvino  # noqa: F401
        import onnx  # noqa: F401
    except ImportError:
        sys.exit("ERROR: --ov-gpu requested but openvino/onnx are missing. "
                 "pip install openvino onnx -- onnxruntime-openvino alone is "
                 "NOT enough (its EP runs the detectors on CPU; this shim "
                 "talks to OpenVINO directly).")
    _install()
    print(f"{_OV_TAG} ONNX models will run on the Intel iGPU (OpenVINO GPU; "
          f"static-shape detectors, batch-dynamic CCIP)", file=sys.stderr,
          flush=True)
    return True
