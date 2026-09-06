"""Benchmark every model variant against every available execution provider.

    python src/benchmark.py                 # full sweep, writes outputs/benchmark.{json,md}
    python src/benchmark.py --runs 30
    python src/benchmark.py --no-isolate    # faster, less trustworthy (see below)

Each (model, provider) pair is timed in its own subprocess by default. Sessions
built in one process compete for threads and leave the CPU warm, which was enough
to move fp32 CPU numbers by 50% between runs during development. Isolation costs
a few seconds of process startup and buys numbers that reproduce.

The sweep also records how CoreML partitions each graph, which is the whole
explanation for why quantized models are slower on the ANE/GPU than on the CPU.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as rt

from config import models_path, outputs_path, data_path, model_name, input_size, get_logger
from utils import save_json

logger=get_logger(__name__)

# ORT prints CoreML's partition decision as a warning; it is the most useful
# number in the whole sweep, so parse it back out instead of silencing it.
partition_re=re.compile(r"number of partitions supported by CoreML: (\d+) "
                        r"number of nodes in the graph: (\d+) "
                        r"number of nodes supported by CoreML: (\d+)")


def bench(fn,runs,warmup=3):
    """Time a callable, discarding warm-up runs. Returns (mean, p50, p95) in ms."""
    for _ in range(warmup):
        fn()
    times=[]
    for _ in range(runs):
        start=time.perf_counter()
        fn()
        times.append((time.perf_counter()-start)*1000)
    times.sort()
    return (sum(times)/len(times),
            times[len(times)//2],
            times[int(0.95*(len(times)-1))])


def bench_onnx(path,provider,runs):
    sesh=rt.InferenceSession(str(path),providers=[provider])
    name=sesh.get_inputs()[0].name
    dummy=np.random.randn(1,3,input_size,input_size).astype(np.float32)
    return bench(lambda:sesh.run(None,{name:dummy}),runs)


def bench_pytorch(pt_path,runs):
    from ultralytics import YOLO
    model=YOLO(str(pt_path))
    dummy=np.zeros((input_size,input_size,3),dtype=np.uint8)
    return bench(lambda:model(dummy,verbose=False),runs)


def run_isolated(target,provider,runs):
    """Re-invoke this file in --single mode and read one JSON line back."""
    proc=subprocess.run(
        [sys.executable,str(Path(__file__).resolve()),"--single",str(target),provider,
         "--runs",str(runs)],
        capture_output=True,text=True,
    )
    partitions=None
    match=partition_re.search(proc.stderr)
    if match:
        partitions={"coreml_partitions":int(match.group(1)),
                    "graph_nodes":int(match.group(2)),
                    "coreml_nodes":int(match.group(3))}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            result=json.loads(line)
            if partitions:
                result.update(partitions)
            return result
    return {"error":proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "no result"}


def single(target,provider,runs):
    """Child-process entry point: benchmark one config, print one JSON line."""
    if target=="pytorch":
        mean,p50,p95=bench_pytorch(models_path/f"{model_name}.pt",runs)
        size=(models_path/f"{model_name}.pt").stat().st_size/1e6
    else:
        mean,p50,p95=bench_onnx(target,provider,runs)
        size=Path(target).stat().st_size/1e6
    print(json.dumps({"mean_ms":round(mean,2),"p50_ms":round(p50,2),"p95_ms":round(p95,2),
                      "fps":round(1000/mean,1),"size_mb":round(size,2)}))


def ensure_models():
    """Export fp32, quantize dynamically, quantize statically - whatever is missing."""
    pt_path=models_path/f"{model_name}.pt"
    onnx_path=models_path/f"{model_name}.onnx"
    dyn_path=models_path/f"{model_name}_int8.onnx"
    static_path=models_path/f"{model_name}_int8_static.onnx"

    if not onnx_path.exists():
        import shutil
        from ultralytics import YOLO
        logger.info("exporting fp32 onnx")
        exported=YOLO(str(pt_path)).export(format="onnx",opset=17,imgsz=input_size)
        if Path(exported)!=onnx_path:
            shutil.move(str(exported),str(onnx_path))

    if not dyn_path.exists():
        from onnxruntime.quantization import quantize_dynamic, QuantType
        logger.info("dynamic int8 quantization")
        quantize_dynamic(str(onnx_path),str(dyn_path),weight_type=QuantType.QUInt8)

    if not static_path.exists():
        videos=sorted((data_path/"videos").glob("*.mp4"))
        if videos:
            from quantize import build_static_model
            logger.info("static int8 quantization (calibrating on demo frames)")
            build_static_model(onnx_path,static_path,videos,samples=64)
        else:
            logger.warning("no videos for calibration - skipping static int8 "
                           "(run python src/create_video.py first)")
    return {"fp32":onnx_path,"int8_dynamic":dyn_path,"int8_static":static_path}


def markdown_table(results,baseline_ms):
    lines=["| Model | Provider | Mean | p95 | FPS | Size | vs PyTorch |",
           "|---|---|---|---|---|---|---|"]
    for row in results:
        if "error" in row:
            lines.append(f"| {row['model']} | {row['provider']} | failed: {row['error'][:40]} | | | | |")
            continue
        speedup=f"{baseline_ms/row['mean_ms']:.2f}x" if baseline_ms else "-"
        lines.append(f"| {row['model']} | {row['provider']} | {row['mean_ms']:.1f} ms | "
                     f"{row['p95_ms']:.1f} ms | {row['fps']:.1f} | {row['size_mb']:.1f} MB | {speedup} |")
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description="Model x provider benchmark sweep")
    parser.add_argument("--single",nargs=2,metavar=("MODEL","PROVIDER"),
                        help=argparse.SUPPRESS) # internal: child process mode
    parser.add_argument("--runs",type=int,default=20)
    parser.add_argument("--no-isolate",action="store_true",
                        help="run every config in this process instead of a subprocess")
    args=parser.parse_args()

    if args.single:
        single(args.single[0],args.single[1],args.runs)
        return

    models=ensure_models()
    providers=[p for p in ["CPUExecutionProvider","CoreMLExecutionProvider",
                           "CUDAExecutionProvider","TensorrtExecutionProvider"]
               if p in rt.get_available_providers()]
    logger.info(f"providers on this machine: {providers}")

    configs=[("pytorch","-")]
    for label,path in models.items():
        if path.exists():
            configs+= [(label,provider) for provider in providers]

    results=[]
    for label,provider in configs:
        target="pytorch" if label=="pytorch" else str(models[label])
        logger.info(f"benchmarking {label} on {provider}")
        if args.no_isolate and label!="pytorch":
            mean,p50,p95=bench_onnx(target,provider,args.runs)
            row={"mean_ms":round(mean,2),"p50_ms":round(p50,2),"p95_ms":round(p95,2),
                 "fps":round(1000/mean,1),"size_mb":round(Path(target).stat().st_size/1e6,2)}
        else:
            row=run_isolated(target,provider,args.runs)
        row.update({"model":label,"provider":provider.replace("ExecutionProvider","")})
        results.append(row)
        if "error" in row:
            logger.warning(f"  failed: {row['error'][:120]}")
        else:
            extra=""
            if "coreml_partitions" in row:
                extra=(f"  [CoreML took {row['coreml_nodes']}/{row['graph_nodes']} nodes "
                       f"in {row['coreml_partitions']} partitions]")
            logger.info(f"  {row['mean_ms']:7.2f} ms  {row['fps']:6.1f} fps  "
                        f"{row['size_mb']:6.2f} MB{extra}")

    baseline=next((r["mean_ms"] for r in results if r["model"]=="pytorch" and "mean_ms" in r),None)
    fastest=min((r for r in results if "mean_ms" in r),key=lambda r:r["mean_ms"])
    payload={"runs":args.runs,"input_size":input_size,"isolated":not args.no_isolate,
             "providers_available":rt.get_available_providers(),
             "baseline_pytorch_ms":baseline,
             "fastest":{"model":fastest["model"],"provider":fastest["provider"],
                        "mean_ms":fastest["mean_ms"]},
             "results":results}
    save_json(payload,outputs_path/"benchmark.json")
    table=markdown_table(results,baseline)
    (outputs_path/"benchmark.md").write_text(table+"\n")
    logger.info(f"\n{table}")
    logger.info(f"fastest: {fastest['model']} on {fastest['provider']} "
                f"({fastest['mean_ms']:.1f} ms)")


if __name__=="__main__":
    main()
