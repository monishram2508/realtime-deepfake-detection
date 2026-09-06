"""Static (calibrated) int8 quantization of the YOLOv8 ONNX model.

Dynamic quantization computes activation scales at runtime, which costs more than
it saves on this CPU (see outputs/benchmark.json). Static quantization runs real
frames through the fp32 model first, records the range of every activation
tensor, and bakes those scales into the graph - so nothing is measured at
inference time.

    python src/quantize.py                      # QDQ, per-channel, minmax
    python src/quantize.py --calib-method entropy --samples 128
    python src/quantize.py --format qoperator --no-per-channel

The calibration frames come from the demo videos, preprocessed through the exact
same letterbox path the detector uses. Calibrating on data that does not look
like deployment data is the classic way to get a fast model that detects nothing.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from config import data_path, models_path, model_name, input_size, outputs_path, get_logger
from detection import preprocess_frame
from utils import save_json

logger=get_logger(__name__)


def sample_frames(video_paths,num_samples):
    """Evenly sample frames across every camera clip, so calibration sees the
    whole scene rather than the first second of one video."""
    frames=[]
    per_video=max(1,num_samples//max(len(video_paths),1))
    for path in video_paths:
        capture=cv2.VideoCapture(str(path))
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        stride=max(1,total//per_video)
        index=0
        taken=0
        while taken<per_video:
            capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            read,frame=capture.read()
            if not read:
                break
            frames.append(frame)
            taken+=1
            index+=stride
        capture.release()
        logger.info(f"{path.name}: {taken} calibration frames")
    return frames


class videocalibrationreader(CalibrationDataReader):
    """Feeds preprocessed frames to the calibrator, one batch at a time."""

    def __init__(self,frames,input_name):
        self.input_name=input_name
        self.batches=[preprocess_frame(f,input_size)[0] for f in frames]
        self.index=0

    def get_next(self):
        if self.index>=len(self.batches):
            return None # None ends calibration
        batch=self.batches[self.index]
        self.index+=1
        return {self.input_name:batch}

    def rewind(self):
        self.index=0


def nodes_matching(onnx_path,pattern):
    """Node names containing `pattern` - used to keep sensitive layers in fp32."""
    import onnx
    model=onnx.load(str(onnx_path))
    return [n.name for n in model.graph.node if pattern in n.name]


def build_static_model(onnx_path,out_path,videos,samples=64,calib_method="minmax",
                       quant_format="qdq",per_channel=True,exclude_pattern="/model.22/"):
    """Calibrate on real frames and write a statically quantized int8 model.

    exclude_pattern keeps every node whose name contains it in fp32. The YOLOv8
    detection head ("/model.22/") is the usual culprit: its classification branch
    produces a handful of large logits among 8400 mostly-negative ones, and a
    single uint8 activation scale over that range rounds every positive logit to
    the same bucket - the model then confidently detects nothing at all.
    """
    methods={"minmax":CalibrationMethod.MinMax,
             "entropy":CalibrationMethod.Entropy,
             "percentile":CalibrationMethod.Percentile}
    formats={"qdq":QuantFormat.QDQ,"qoperator":QuantFormat.QOperator}

    # 1. shape inference + graph cleanup. quantize_static is much less effective
    #    (and sometimes fails outright) on a graph with unknown shapes.
    with tempfile.TemporaryDirectory() as tmp:
        prepped=Path(tmp)/"prepped.onnx"
        logger.info("pre-processing graph (symbolic shape inference)")
        quant_pre_process(str(onnx_path),str(prepped),skip_symbolic_shape=False)

        frames=sample_frames(videos,samples)
        logger.info(f"calibrating on {len(frames)} frames from {len(videos)} camera(s)")
        reader=videocalibrationreader(frames,"images")

        excluded=nodes_matching(prepped,exclude_pattern) if exclude_pattern else []
        if exclude_pattern:
            logger.info(f"keeping {len(excluded)} nodes matching '{exclude_pattern}' in fp32")
        logger.info(f"quantizing: format={quant_format} method={calib_method} "
                    f"per_channel={per_channel}")
        quantize_static(
            str(prepped),
            str(out_path),
            calibration_data_reader=reader,
            quant_format=formats[quant_format],
            per_channel=per_channel,
            activation_type=QuantType.QUInt8, # unsigned activations suit ORT's CPU kernels
            weight_type=QuantType.QInt8,      # signed, symmetric weights
            nodes_to_exclude=excluded,
            calibrate_method=methods[calib_method],
            extra_options={"ActivationSymmetric":False,"WeightSymmetric":True},
        )

    fp32_mb=onnx_path.stat().st_size/1e6
    int8_mb=out_path.stat().st_size/1e6
    logger.info(f"wrote {out_path} ({int8_mb:.2f} MB, {fp32_mb/int8_mb:.1f}x smaller than fp32)")
    return out_path


def ablation(onnx_path,videos,samples,frames_for_eval=24):
    """Build every quantization variant and score each against the fp32 model.

    This is the experiment behind the quantization section of the README: it is
    the only way to notice that the fastest variant is the one that detects
    nothing at all.
    """
    import time
    from accuracy import compare, sample_frames as sample_eval_frames
    from detection import yolov8detector

    variants=[
        ("static / whole graph",         dict(exclude_pattern=None)),
        ("static / entropy calibration", dict(exclude_pattern=None,calib_method="entropy")),
        ("static / cls branch in fp32",  dict(exclude_pattern="/model.22/cv3")),
        ("static / head in fp32",        dict(exclude_pattern="/model.22/")),
    ]

    eval_frames=sample_eval_frames(videos,frames_for_eval)
    logger.info(f"scoring on {len(eval_frames)} frames")
    reference_detector=yolov8detector(onnx_path,provider="CPUExecutionProvider")
    reference=[reference_detector.detect(f) for f in eval_frames]
    ref_count=sum(len(d) for d in reference)

    rows=[{"variant":"fp32 (reference)","detections":ref_count,"f1":1.0,
           "latency_ms":reference_detector.metrics.get_stats()["avg_latency"],
           "size_mb":round(onnx_path.stat().st_size/1e6,2)}]

    with tempfile.TemporaryDirectory() as tmp:
        for label,kwargs in variants:
            out=Path(tmp)/f"{label.replace('/','_').replace(' ','_')}.onnx"
            logger.info(f"building: {label}")
            build_static_model(onnx_path,out,videos,samples=samples,**kwargs)
            detector=yolov8detector(out,provider="CPUExecutionProvider")
            candidate=[detector.detect(f) for f in eval_frames]
            result=compare(reference,candidate,0.5)
            rows.append({"variant":label,
                         "detections":sum(len(d) for d in candidate),
                         "f1":result["f1"],
                         "mean_iou":result["mean_iou_of_matches"],
                         "latency_ms":detector.metrics.get_stats()["avg_latency"],
                         "size_mb":round(out.stat().st_size/1e6,2)})
            logger.info(f"  {rows[-1]['detections']} detections | F1 {rows[-1]['f1']:.3f} | "
                        f"{rows[-1]['latency_ms']:.1f} ms | {rows[-1]['size_mb']:.1f} MB")

    save_json({"samples":samples,"eval_frames":len(eval_frames),"results":rows},
              outputs_path/"quantization_ablation.json")
    # NOTE: these latencies are measured in-process, one variant after another, so
    # they are indicative only. benchmark.py runs each model in its own process,
    # and those are the numbers to quote.
    lines=["| Variant | Detections | F1 vs fp32 | Latency (indicative) | Size |",
           "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['detections']} | {r['f1']:.3f} | "
                     f"{r['latency_ms']:.1f} ms | {r['size_mb']:.1f} MB |")
    table="\n".join(lines)
    (outputs_path/"quantization_ablation.md").write_text(table+"\n")
    logger.info(f"\n{table}")


def main():
    parser=argparse.ArgumentParser(description="Static int8 quantization")
    parser.add_argument("--samples",type=int,default=64,help="calibration frames")
    parser.add_argument("--calib-method",choices=["minmax","entropy","percentile"],default="minmax")
    parser.add_argument("--format",choices=["qdq","qoperator"],default="qdq")
    parser.add_argument("--no-per-channel",action="store_true",
                        help="one scale per tensor instead of one per output channel")
    parser.add_argument("--exclude",default="/model.22/",metavar="PATTERN",
                        help="keep nodes whose name contains PATTERN in fp32 "
                             "(default: the detection head, which does not survive int8)")
    parser.add_argument("--quantize-head",action="store_true",
                        help="quantize everything, including the head - reproduces the "
                             "silent failure documented in the README")
    parser.add_argument("--ablation",action="store_true",
                        help="build every variant and score each one against fp32")
    parser.add_argument("--out",default=None)
    args=parser.parse_args()

    onnx_path=models_path/f"{model_name}.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"{onnx_path} missing - run: python src/benchmark.py")
    videos=sorted((data_path/"videos").glob("*.mp4"))
    if not videos:
        raise SystemExit("no calibration videos - run: python src/create_video.py")

    if args.ablation:
        ablation(onnx_path,videos,args.samples)
        return

    out_path=Path(args.out) if args.out else models_path/f"{model_name}_int8_static.onnx"
    build_static_model(onnx_path,out_path,videos,samples=args.samples,
                       calib_method=args.calib_method,quant_format=args.format,
                       per_channel=not args.no_per_channel,
                       exclude_pattern=None if args.quantize_head else (args.exclude or None))
    logger.info("next: python src/benchmark.py --compare-all   and   python src/accuracy.py")


if __name__=="__main__":
    main()
