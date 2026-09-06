"""One-command showcase: test videos -> synchronized batches -> ONNX YOLOv8 -> JSON.

    python src/demo.py                                    # 2 cameras, full clip
    python src/demo.py --batches 20                       # stop early
    python src/demo.py --workers 2 --save-video           # a session per camera
    python src/demo.py --model models/yolov8n_int8_static.onnx
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import onnxruntime as rt

from config import data_path, models_path, outputs_path, get_logger
from detection import detectionpipe
from utils import save_json

logger=get_logger(__name__)


def check_prereqs(model_path):
    ok=True
    videos=sorted((data_path/"videos").glob("*.mp4"))
    if not videos:
        logger.error("no test videos - run: python src/create_video.py")
        ok=False
    if not Path(model_path).exists():
        logger.error(f"missing model {model_path} - run: python src/benchmark.py")
        ok=False
    return ok,videos


def scaling_study(model,batches):
    """Sweep workers x provider end-to-end, each config in a fresh process.

    Runs unpaced (--no-realtime) so the pipeline is measured at the throughput it
    can reach, not at the frame rate of the source clips.
    """
    providers=[p for p in ["CPUExecutionProvider","CoreMLExecutionProvider",
                           "CUDAExecutionProvider"] if p in rt.get_available_providers()]
    rows=[]
    for provider in providers:
        for workers in (1,2):
            logger.info(f"scaling: workers={workers} provider={provider}")
            subprocess.run([sys.executable,str(Path(__file__).resolve()),
                            "--model",str(model),"--batches",str(batches),"--no-realtime",
                            "--workers",str(workers),"--provider",provider,"--loop"],
                           capture_output=True,text=True)
            stats=json.loads((outputs_path/"metrics.json").read_text())
            pipeline=stats["pipeline"]
            rows.append({"provider":provider.replace("ExecutionProvider",""),
                         "workers":workers,
                         "batch_latency_ms":pipeline["avg_latency"],
                         "p95_ms":pipeline["p95_latency"],
                         "wait_for_frames_ms":pipeline["avg_wait_for_frames_ms"],
                         "batches_per_sec":pipeline["fps"],
                         "camera_fps":round(pipeline["fps"]*len(stats["cameras"]),1)})
            logger.info(f"  {rows[-1]['batch_latency_ms']:.1f} ms/batch  "
                        f"{rows[-1]['batches_per_sec']:.1f} batches/s")

    save_json({"batches":batches,"model":Path(model).name,"results":rows},
              outputs_path/"scaling.json")
    lines=["| Provider | Workers | Batch latency | p95 | Waiting on frames | Batches/s | Camera-fps |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['provider']} | {r['workers']} | {r['batch_latency_ms']:.1f} ms | "
                     f"{r['p95_ms']:.1f} ms | {r['wait_for_frames_ms']:.1f} ms | "
                     f"{r['batches_per_sec']:.1f} | {r['camera_fps']:.1f} |")
    table="\n".join(lines)
    (outputs_path/"scaling.md").write_text(table+"\n")
    logger.info(f"\n{table}")


def main():
    parser=argparse.ArgumentParser(description="Multi-camera detection demo")
    parser.add_argument("--model",default=str(models_path/"yolov8n.onnx"))
    parser.add_argument("--batches",type=int,default=None,help="stop after N synchronized batches")
    parser.add_argument("--save-video",action="store_true",help="write annotated mp4s to outputs/")
    parser.add_argument("--no-realtime",action="store_true",
                        help="read the files as fast as possible instead of pacing at their fps")
    parser.add_argument("--provider",default=None,
                        help="force an execution provider, e.g. CPUExecutionProvider "
                             "(default: auto - accelerator for float models, CPU for int8)")
    parser.add_argument("--workers",type=int,default=0,
                        help="cameras to run concurrently, one ONNX session each "
                             "(0 = auto: 1 on CPU, one per camera on an accelerator)")
    parser.add_argument("--no-track",action="store_true",help="detections only, no track ids")
    parser.add_argument("--loop",action="store_true",
                        help="restart clips at EOF so the source never starves the pipeline")
    parser.add_argument("--scaling",action="store_true",
                        help="sweep workers x provider end-to-end, write outputs/scaling.json")
    args=parser.parse_args()

    ok,videos=check_prereqs(args.model)
    if not ok:
        sys.exit(1)

    if args.scaling:
        scaling_study(args.model,args.batches or 40)
        return

    video_paths={v.stem:v for v in videos}
    logger.info(f"cameras: {list(video_paths)}")

    pipe=detectionpipe(video_paths,model_path=args.model,save_video=args.save_video,
                       realtime=not args.no_realtime,provider=args.provider,
                       workers=args.workers,track=not args.no_track,loop=args.loop)
    results=pipe.run(max_batches=args.batches)

    total=sum(c["detection_count"] for r in results for c in r["cameras"].values())
    tracks=sum(t["total_tracks_created"] for t in pipe.summary()["tracking"].values())
    logger.info(f"done - {len(results)} batches, {total} detections"
                +(f", {tracks} tracks" if not args.no_track else ""))
    logger.info(f"results: {outputs_path/'detections.json'}, metrics: {outputs_path/'metrics.json'}")


if __name__=="__main__":
    main()
