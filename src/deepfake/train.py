"""Fine-tune Xception on face crops, with the class imbalance and the leakage handled.

    # prove the loop learns, on the fabricated dataset, no FF++ needed
    python src/deepfake/train.py --manifest outputs/crops_standin.json --epochs 3

    # the real thing
    python src/deepfake/train.py --manifest outputs/crops_manifest.json --epochs 12
    python src/deepfake/train.py --exclude-method NeuralTextures   # generalization run

Design decisions that are not the default and are deliberate:

**`legacy_xception` from timm**, not a modern backbone. It is the FF++ baseline
architecture, so the numbers here are comparable to the published ones. Swapping in
an EfficientNet would probably score better and would make the comparison useless.

**Class-weighted loss, not resampling.** Train is 4:1 fake:real because there are
four manipulation methods per original. Undersampling the fakes throws away three
quarters of the manipulations - including whichever one the model is worst at.
Weighting the loss keeps every sample and corrects the gradient instead.

**Validation selects on video-level AUC**, not frame accuracy. Frame accuracy at a
fixed threshold rewards a model that memorizes easy frames of easy videos; the
question being asked is whether a clip is manipulated.

**`--exclude-method` is the point of the whole project.** Training with a method
held out and testing on it is the honest measure of whether the detector found
*manipulation artefacts* or just the fingerprint of four specific tools. Expect the
number to fall hard. That fall is the result, not a bug to tune away.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import models_path, outputs_path, get_logger
from utils import save_json
from deepfake.metrics import best_threshold, by_group, evaluate, roc_auc

logger=get_logger(__name__)

imagenet_mean=np.array([0.485,0.456,0.406],dtype=np.float32)
imagenet_std=np.array([0.229,0.224,0.225],dtype=np.float32)


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


safe_batch={"cuda":32,"mps":4,"cpu":4}


def pick_batch_size(device,requested=None):
    kind=device.type
    limit=safe_batch.get(kind,8)
    if requested is None:
        logger.info(f"batch-size=auto -> {limit} for {kind}")
        return limit
    if kind=="mps" and requested>limit:
        logger.warning(f"batch {requested} is past the measured MPS memory cliff "
                       f"({limit}); expect a ~90x slowdown, not an OOM. "
                       f"Reproduce with src/deepfake/bench_train.py")
    return requested


class cropdataset(Dataset):
    """Face crops from the manifest. Augmentation is deliberately mild.

    Horizontal flip is safe - a mirrored face is still a face, and manipulation
    artefacts are not chirality-dependent. Heavy colour or blur augmentation is
    *not* applied: those destroy exactly the high-frequency compression artefacts
    the classifier is supposed to key on, so augmenting them away trains the model
    to ignore its own signal.
    """

    def __init__(self,rows,size=299,train=False):
        self.rows=rows
        self.size=size
        self.train=train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        import cv2
        row=self.rows[index]
        image=cv2.imread(row["path"])
        if image is None:
            raise FileNotFoundError(row["path"])
        if image.shape[0]!=self.size or image.shape[1]!=self.size:
            image=cv2.resize(image,(self.size,self.size),interpolation=cv2.INTER_AREA)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if self.train and np.random.rand()<0.5:
            image=np.ascontiguousarray(image[:,::-1])
        tensor=(image.astype(np.float32)/255.0-imagenet_mean)/imagenet_std
        return torch.from_numpy(tensor.transpose(2,0,1)),int(row["label"])


def load_rows(manifest_path,split,exclude_method=None,only_method=None,limit=None):
    payload=json.loads(Path(manifest_path).read_text())
    rows=[r for r in payload["crops"] if r["split"]==split]
    if exclude_method:
        rows=[r for r in rows if r["method"]=="original" or r["method"] not in exclude_method]
    if only_method:
        rows=[r for r in rows if r["method"]=="original" or r["method"] in only_method]
    return rows[:limit] if limit else rows


def class_weights(rows,device):
    """Inverse-frequency weights, normalized to mean 1 so the LR stays comparable."""
    counts=np.array([sum(1 for r in rows if r["label"]==c) for c in (0,1)],dtype=np.float64)
    if (counts==0).any():
        return None
    weights=counts.sum()/(2.0*counts)
    weights=weights/weights.mean()
    logger.info(f"class counts real/fake = {counts.astype(int).tolist()}, "
                f"loss weights = {np.round(weights,3).tolist()}")
    return torch.tensor(weights,dtype=torch.float32,device=device)


def build_model(name="legacy_xception",pretrained=True,dropout=0.5):
    import timm
    model=timm.create_model(name,pretrained=pretrained,num_classes=2,drop_rate=dropout)
    return model


@torch.no_grad()
def predict(model,rows,device,size,batch_size,workers):
    model.eval()
    loader=DataLoader(cropdataset(rows,size,train=False),batch_size=batch_size,
                      shuffle=False,num_workers=workers)
    scores=[]
    for images,_ in loader:
        logits=model(images.to(device))
        scores.append(torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


def train_one_epoch(model,loader,criterion,optimizer,device,scaler=None):
    model.train()
    running=0.0
    seen=0
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(images)
        loss=criterion(logits,labels)
        loss.backward()
        optimizer.step()
        running+=loss.item()*len(labels)
        seen+=len(labels)
    return running/max(seen,1)


def main():
    parser=argparse.ArgumentParser(description="Fine-tune Xception on FF++ face crops")
    parser.add_argument("--manifest",default=str(outputs_path/"crops_manifest.json"))
    parser.add_argument("--model",default="legacy_xception")
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--batch-size",type=int,default=None,
                        help="default: auto from the device (see safe_batch)")
    parser.add_argument("--lr",type=float,default=2e-4)
    parser.add_argument("--weight-decay",type=float,default=1e-5)
    parser.add_argument("--size",type=int,default=299)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--device",default=None,help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--patience",type=int,default=3,help="epochs without val AUC gain")
    parser.add_argument("--exclude-method",nargs="*",default=None,
                        help="hold these manipulations out of training entirely")
    parser.add_argument("--limit",type=int,default=None,help="cap rows per split (smoke test)")
    parser.add_argument("--no-pretrained",action="store_true")
    parser.add_argument("--tag",default=None,help="name for the checkpoint and report")
    args=parser.parse_args()

    device=pick_device(args.device)
    batch_size=pick_batch_size(device,args.batch_size)
    tag=args.tag or ("heldout_"+"_".join(args.exclude_method) if args.exclude_method else "all")
    logger.info(f"device={device} model={args.model} tag={tag}")

    train_rows=load_rows(args.manifest,"train",args.exclude_method,limit=args.limit)
    val_rows=load_rows(args.manifest,"val",args.exclude_method,limit=args.limit)
    if not train_rows or not val_rows:
        raise SystemExit(f"no rows in {args.manifest} - run src/deepfake/dataset.py first")

    train_ids={r["identity"] for r in train_rows}|{r["source"] for r in train_rows if r["source"]}
    val_ids={r["identity"] for r in val_rows}|{r["source"] for r in val_rows if r["source"]}
    leaked=train_ids&val_ids
    if leaked:
        raise SystemExit(f"identity leak between train and val: {sorted(leaked)[:10]}")
    logger.info(f"train {len(train_rows)} crops / {len(train_ids)} identities, "
                f"val {len(val_rows)} crops / {len(val_ids)} identities, no overlap")
    if args.exclude_method:
        logger.info(f"holding out {args.exclude_method} - unseen at training time")

    model=build_model(args.model,pretrained=not args.no_pretrained).to(device)
    criterion=nn.CrossEntropyLoss(weight=class_weights(train_rows,device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)

    loader=DataLoader(cropdataset(train_rows,args.size,train=True),
                      batch_size=batch_size,shuffle=True,
                      num_workers=args.workers,drop_last=len(train_rows)>batch_size)

    history=[]
    best_auc=-1.0
    best_epoch=-1
    stale=0
    checkpoint=models_path/f"xception_{tag}.pt"

    for epoch in range(1,args.epochs+1):
        start=time.perf_counter()
        loss=train_one_epoch(model,loader,criterion,optimizer,device)
        scheduler.step()

        scores=predict(model,val_rows,device,args.size,batch_size,args.workers)
        report=evaluate(val_rows,scores)
        video_auc=report["videos"]["auc"]
        frame_auc=report["frames"]["auc"]
        elapsed=time.perf_counter()-start

        history.append({"epoch":epoch,"loss":round(loss,4),
                        "val_frame_auc":frame_auc,"val_video_auc":video_auc,
                        "val_frame_acc":report["frames"]["accuracy"],
                        "seconds":round(elapsed,1)})
        logger.info(f"epoch {epoch}/{args.epochs} loss {loss:.4f} | "
                    f"val frame AUC {frame_auc:.4f} acc {report['frames']['accuracy']:.4f} | "
                    f"val video AUC {video_auc:.4f} | {elapsed:.1f}s")

        selector=video_auc if not np.isnan(video_auc) else frame_auc
        if selector>best_auc:
            best_auc,best_epoch,stale=selector,epoch,0
            torch.save({"model":args.model,"state_dict":model.state_dict(),
                        "epoch":epoch,"val_auc":selector,"tag":tag,
                        "excluded":args.exclude_method,"size":args.size},checkpoint)
            logger.info(f"  new best ({selector:.4f}) -> {checkpoint.name}")
        else:
            stale+=1
            if stale>=args.patience:
                logger.info(f"no gain for {args.patience} epochs - stopping at {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint,map_location=device)["state_dict"])
    scores=predict(model,val_rows,device,args.size,batch_size,args.workers)
    threshold,_=best_threshold([r["label"] for r in val_rows],scores)
    final=evaluate(val_rows,scores,threshold)
    logger.info(f"best epoch {best_epoch}, val threshold fitted at {threshold:.4f}")
    logger.info(f"val frames: {final['frames']}")
    logger.info(f"val videos: {final['videos']}")

    save_json({"tag":tag,"model":args.model,"device":str(device),
               "excluded_methods":args.exclude_method,
               "epochs_run":len(history),"best_epoch":best_epoch,
               "val_threshold":threshold,
               "checkpoint":str(checkpoint),
               "history":history,
               "val":final,
               "val_by_method":by_group(val_rows,scores,threshold=threshold)},
              outputs_path/f"train_{tag}.json")


if __name__=="__main__":
    main()
