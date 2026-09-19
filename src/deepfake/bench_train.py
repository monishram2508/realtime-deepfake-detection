"""How fast can this machine actually train the classifier, and at what batch size?

    python src/deepfake/bench_train.py                    # batch sweep on the best device
    python src/deepfake/bench_train.py --devices mps cpu  # device comparison
    python src/deepfake/bench_train.py --models legacy_xception resnet50

This exists because a training run that is 90x too slow does not announce itself.
Inference benchmarks in this repo measure a model that either works or does not;
a training loop past its memory budget still produces correct gradients, correct
loss curves and correct checkpoints - just over days instead of hours. The only
symptom is the clock, so the clock is what gets measured, before a real run.

The headline result on the machine this was written on (Apple M2, 8 GB):

    batch  4      311.6 ms/step    12.84 img/s
    batch  8      832.6 ms/step     9.61 img/s
    batch 12     1127.7 ms/step    10.64 img/s
    batch 16   130794.1 ms/step     0.12 img/s   <- 89x collapse

That is not an out-of-memory error. macOS pages MPS buffers to disk instead of
raising, so the run keeps going and silently costs ~90x more. `train.py` picks its
default batch size from this table rather than from a constant.

Timing note: MPS and CUDA both queue work asynchronously, so every step is
synchronized before the clock is read. Without that this measures how fast Python
can enqueue kernels, which is a much nicer and completely fictional number.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import outputs_path, get_logger
from utils import save_json

logger=get_logger(__name__)


def synchronize(device):
    if device.type=="mps":
        torch.mps.synchronize()
    elif device.type=="cuda":
        torch.cuda.synchronize()


def release(device):
    gc.collect()
    if device.type=="mps":
        torch.mps.empty_cache()
    elif device.type=="cuda":
        torch.cuda.empty_cache()


def available_devices():
    devices=[]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def bench_step(model_name,device,batch,size,iters,warmup):
    """One (model, device, batch) measurement. Returns ms/step and img/s."""
    import timm
    dev=torch.device(device)
    model=timm.create_model(model_name,pretrained=False,num_classes=2).to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    criterion=nn.CrossEntropyLoss()
    images=torch.randn(batch,3,size,size,device=dev)
    labels=torch.randint(0,2,(batch,),device=dev)

    def step():
        optimizer.zero_grad(set_to_none=True)
        criterion(model(images),labels).backward()
        optimizer.step()
        synchronize(dev)

    try:
        for _ in range(warmup):
            step()
        start=time.perf_counter()
        for _ in range(iters):
            step()
        elapsed=(time.perf_counter()-start)/iters
    finally:
        del model,optimizer,images,labels
        release(dev)
    return elapsed*1000,batch/elapsed


def sweep(models,devices,batches,size,iters,warmup,cliff_ratio=0.5):
    rows=[]
    for model_name in models:
        for device in devices:
            previous=None
            for batch in batches:
                try:
                    ms,ips=bench_step(model_name,device,batch,size,iters,warmup)
                except Exception as e:
                    logger.warning(f"{model_name} {device} batch {batch}: "
                                   f"{type(e).__name__}: {str(e)[:90]}")
                    rows.append({"model":model_name,"device":device,"batch":batch,
                                 "ms_per_step":None,"img_per_sec":None,
                                 "failed":type(e).__name__})
                    break
                cliff=bool(previous and ips<previous*cliff_ratio)
                rows.append({"model":model_name,"device":device,"batch":batch,
                             "ms_per_step":round(ms,1),"img_per_sec":round(ips,2),
                             "cliff":cliff})
                logger.info(f"{model_name:18s} {device:4s} batch {batch:3d}  "
                            f"{ms:9.1f} ms/step  {ips:7.2f} img/s"
                            +("   <- CLIFF" if cliff else ""))
                previous=ips
                if cliff:
                    logger.warning(f"throughput collapsed past batch {batch} - "
                                   "stopping this sweep, the machine is paging")
                    break
    return rows


def estimate(rows,crops_per_epoch,epochs,runs):
    """Turn img/s into the number that actually decides local vs rented GPU."""
    best={}
    for row in rows:
        if not row.get("img_per_sec"):
            continue
        key=(row["model"],row["device"])
        if row["img_per_sec"]>best.get(key,(0,))[0]:
            best[key]=(row["img_per_sec"],row["batch"])
    out=[]
    for (model_name,device),(ips,batch) in sorted(best.items()):
        epoch_s=crops_per_epoch/ips
        out.append({"model":model_name,"device":device,
                    "best_batch":batch,"img_per_sec":round(ips,2),
                    "hours_per_epoch":round(epoch_s/3600,2),
                    "hours_per_run":round(epoch_s*epochs/3600,1),
                    "hours_for_study":round(epoch_s*epochs*runs/3600,1)})
    return out


def main():
    parser=argparse.ArgumentParser(description="Training throughput and the memory cliff")
    parser.add_argument("--models",nargs="*",default=["legacy_xception"])
    parser.add_argument("--devices",nargs="*",default=None)
    parser.add_argument("--batches",nargs="*",type=int,default=[4,8,12,16,24,32])
    parser.add_argument("--size",type=int,default=299)
    parser.add_argument("--iters",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=2)
    parser.add_argument("--crops-per-epoch",type=int,default=115200,
                        help="FF++ train crops at 32 frames/video over 3600 videos")
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--runs",type=int,default=5,
                        help="training runs in the generalization study")
    args=parser.parse_args()

    devices=args.devices or available_devices()
    logger.info(f"torch {torch.__version__} | devices {devices} | {args.size}x{args.size} "
                f"fwd+bwd, {args.iters} timed iters after {args.warmup} warm-up")

    rows=sweep(args.models,devices,args.batches,args.size,args.iters,args.warmup)
    if not rows:
        raise SystemExit("no measurements")

    projections=estimate(rows,args.crops_per_epoch,args.epochs,args.runs)

    lines=["| Model | Device | Batch | ms/step | img/s |","|---|---|---|---|---|"]
    for row in rows:
        if row.get("ms_per_step") is None:
            lines.append(f"| {row['model']} | {row['device']} | {row['batch']} | "
                         f"{row.get('failed','failed')} | — |")
            continue
        mark=" **cliff**" if row.get("cliff") else ""
        lines.append(f"| {row['model']} | {row['device']} | {row['batch']} | "
                     f"{row['ms_per_step']}{mark} | {row['img_per_sec']} |")

    lines+=["","| Model | Device | Best batch | img/s | h/epoch | h/run | h/study |",
            "|---|---|---|---|---|---|---|"]
    for p in projections:
        lines.append(f"| {p['model']} | {p['device']} | {p['best_batch']} | "
                     f"{p['img_per_sec']} | {p['hours_per_epoch']} | "
                     f"{p['hours_per_run']} | {p['hours_for_study']} |")
    table="\n".join(lines)
    logger.info(f"\n{table}")

    save_json({"torch":torch.__version__,"size":args.size,
               "crops_per_epoch":args.crops_per_epoch,"epochs":args.epochs,
               "runs_in_study":args.runs,
               "measurements":rows,"projections":projections},
              outputs_path/"train_bench.json")
    (outputs_path/"train_bench.md").write_text(table+"\n")


if __name__=="__main__":
    main()
