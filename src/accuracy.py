"""How much detection quality does each optimization actually cost?

A speed benchmark on its own is meaningless: an int8 model that is 2x faster and
finds half the objects is not an improvement. This script runs the fp32 CPU model
as the reference, runs each candidate over the same frames, and measures how far
the candidate's detections drift.

    python src/accuracy.py                      # all variants, 40 frames
    python src/accuracy.py --frames 80 --iou 0.5

Reported against the fp32 reference, NOT against COCO ground truth - this
measures fidelity to the unquantized model, which is the question quantization
raises. Real mAP would need a labelled dataset.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from config import data_path, models_path, model_name, outputs_path, get_logger
from detection import yolov8detector
from utils import save_json, iou_matrix

logger=get_logger(__name__)


def match(reference,candidate,iou_thresh):
    """Greedy highest-IoU matching, class-aware. Returns (pairs, unmatched_ref, unmatched_cand)."""
    if not reference or not candidate:
        return [],len(reference),len(candidate)

    ious=iou_matrix([[d['x'],d['y'],d['w'],d['h']] for d in reference],
                    [[d['x'],d['y'],d['w'],d['h']] for d in candidate])
    # a match across different classes is not a match
    for i,r in enumerate(reference):
        for j,c in enumerate(candidate):
            if r['class_id']!=c['class_id']:
                ious[i,j]=0.0

    pairs=[]
    used_ref,used_cand=set(),set()
    order=np.dstack(np.unravel_index(np.argsort(ious,axis=None)[::-1],ious.shape))[0]
    for i,j in order:
        if ious[i,j]<iou_thresh:
            break
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
        pairs.append((int(i),int(j),float(ious[i,j])))
    return pairs,len(reference)-len(used_ref),len(candidate)-len(used_cand)


def sample_frames(videos,count):
    frames=[]
    per_video=max(1,count//max(len(videos),1))
    for path in videos:
        capture=cv2.VideoCapture(str(path))
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        stride=max(1,total//per_video)
        for k in range(per_video):
            capture.set(cv2.CAP_PROP_POS_FRAMES,k*stride)
            read,frame=capture.read()
            if not read:
                break
            frames.append(frame)
        capture.release()
    return frames


def compare(reference_dets,candidate_dets,iou_thresh):
    """Aggregate agreement over a whole run of frames."""
    matched=missed=extra=0
    ious=[]
    conf_deltas=[]
    for ref,cand in zip(reference_dets,candidate_dets):
        pairs,unmatched_ref,unmatched_cand=match(ref,cand,iou_thresh)
        matched+=len(pairs)
        missed+=unmatched_ref
        extra+=unmatched_cand
        for i,j,iou in pairs:
            ious.append(iou)
            conf_deltas.append(cand[j]['conf']-ref[i]['conf'])

    recall=matched/(matched+missed) if matched+missed else 1.0
    precision=matched/(matched+extra) if matched+extra else 1.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {
        "matched":matched,
        "missed_vs_reference":missed,   # reference found it, candidate did not
        "spurious_vs_reference":extra,  # candidate invented it
        "recall":round(recall,4),
        "precision":round(precision,4),
        "f1":round(f1,4),
        "mean_iou_of_matches":round(float(np.mean(ious)),4) if ious else 0.0,
        "mean_conf_delta":round(float(np.mean(conf_deltas)),4) if conf_deltas else 0.0,
        "max_conf_drop":round(float(-min(conf_deltas)),4) if conf_deltas else 0.0,
    }


def run_model(path,provider,frames):
    detector=yolov8detector(path,provider=provider)
    return [detector.detect(f) for f in frames],detector.metrics.get_stats()


def main():
    parser=argparse.ArgumentParser(description="Detection fidelity vs the fp32 model")
    parser.add_argument("--frames",type=int,default=40)
    parser.add_argument("--iou",type=float,default=0.5,help="IoU needed to call it the same box")
    args=parser.parse_args()

    videos=sorted((data_path/"videos").glob("*.mp4"))
    if not videos:
        raise SystemExit("no videos - run: python src/create_video.py")
    frames=sample_frames(videos,args.frames)
    logger.info(f"comparing on {len(frames)} frames from {len(videos)} camera(s)")

    fp32=models_path/f"{model_name}.onnx"
    candidates=[
        ("fp32 / CoreML",fp32,"CoreMLExecutionProvider"),
        ("int8_dynamic / CPU",models_path/f"{model_name}_int8.onnx","CPUExecutionProvider"),
        ("int8_static / CPU",models_path/f"{model_name}_int8_static.onnx","CPUExecutionProvider"),
    ]

    logger.info("running fp32 / CPU reference")
    reference,ref_stats=run_model(fp32,"CPUExecutionProvider",frames)
    ref_count=sum(len(d) for d in reference)
    logger.info(f"reference: {ref_count} detections, {ref_stats['avg_latency']:.1f} ms/frame")

    report={"frames":len(frames),"iou_threshold":args.iou,
            "reference":{"model":"fp32","provider":"CPU","detections":ref_count,
                         "avg_latency_ms":ref_stats["avg_latency"]},
            "candidates":{}}

    for label,path,provider in candidates:
        if not Path(path).exists():
            logger.warning(f"skipping {label} - {path} not found")
            continue
        logger.info(f"running {label}")
        try:
            dets,stats=run_model(path,provider,frames)
        except Exception as e:
            logger.warning(f"{label} failed: {e}")
            continue
        result=compare(reference,dets,args.iou)
        result["detections"]=sum(len(d) for d in dets)
        result["avg_latency_ms"]=stats["avg_latency"]
        report["candidates"][label]=result
        logger.info(f"  {result['detections']} detections | F1 {result['f1']:.3f} vs fp32 | "
                    f"mean IoU {result['mean_iou_of_matches']:.3f} | "
                    f"conf delta {result['mean_conf_delta']:+.3f} | "
                    f"{result['avg_latency_ms']:.1f} ms/frame")

    save_json(report,outputs_path/"accuracy.json")

    lines=["| Variant | Detections | F1 vs fp32 | Mean IoU | Missed | Spurious | Mean conf delta |",
           "|---|---|---|---|---|---|---|",
           f"| fp32 / CPU (reference) | {ref_count} | 1.000 | 1.000 | 0 | 0 | +0.000 |"]
    for label,r in report["candidates"].items():
        lines.append(f"| {label} | {r['detections']} | {r['f1']:.3f} | {r['mean_iou_of_matches']:.3f} | "
                     f"{r['missed_vs_reference']} | {r['spurious_vs_reference']} | "
                     f"{r['mean_conf_delta']:+.3f} |")
    table="\n".join(lines)
    (outputs_path/"accuracy.md").write_text(table+"\n")
    logger.info(f"\n{table}")


if __name__=="__main__":
    main()
