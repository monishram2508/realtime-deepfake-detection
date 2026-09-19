"""Videos -> face crops on disk, plus the crop manifest the trainer reads.

    # once FF++ has downloaded
    python src/deepfake/dataset.py --split train --frames 32
    python src/deepfake/dataset.py --split all -c c40 --workers 4

    # today, with no dataset: a learnable stand-in built from the demo clips
    python src/deepfake/dataset.py --standin --frames 40

Sampling is **evenly spaced across each clip**, not the first N frames. FF++ clips
open on a mostly-static talking head, so the first N frames of a 500-frame video
are near-duplicates: the effective dataset is far smaller than the file count
suggests and the model overfits to a handful of poses.

One face per frame (the largest). FF++ clips are single-subject, and taking every
detection would pull bystanders in the background - unmanipulated faces - into the
*fake* class and teach the classifier to label real faces fake.

Crops are written as JPEG q95. That is lossy on a task that keys on compression
artefacts, but it is what every FF++ baseline does, and the alternative (PNG) is
~8x the disk for a distinction that c23/c40 have already destroyed upstream.

The `--standin` mode exists so train.py can be proven to learn *before* any GPU
time is spent on the real thing. It fabricates a known-learnable two-class problem
(clean crops vs recompressed-and-resampled crops) from whatever videos are
already on disk. It is a smoke test for the training loop, not an experiment.
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import data_path, outputs_path, get_logger
from utils import save_json
from deepfake.faces import crop_face, facedetector, ffpp_crop_scale, largest_face, xception_input
from deepfake.ffpp import build_manifest, check_leakage

logger=get_logger(__name__)

crops_path=data_path/"crops"
label_names={0:"real",1:"fake"}


def sample_indices(total,count):
    """Evenly spaced frame indices across a clip, without duplicates."""
    if total<=0:
        return []
    if total<=count:
        return list(range(total))
    return sorted(set(np.linspace(0,total-1,count,dtype=int).tolist()))


def extract_video(row,detector,frames,out_root,scale,size,overwrite=False):
    """One video -> crops on disk. Returns the crop manifest rows it produced."""
    path=Path(row["path"])
    if not path.exists():
        return [],{"missing":1}

    out_dir=(Path(out_root)/row["split"]/label_names[row["label"]]/row["method"])
    out_dir.mkdir(parents=True,exist_ok=True)
    stem=path.stem

    capture=cv2.VideoCapture(str(path))
    total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted=sample_indices(total,frames)
    rows=[]
    stats=Counter()

    for index in wanted:
        out_path=out_dir/f"{stem}_f{index:05d}.jpg"
        if out_path.exists() and not overwrite:
            # resumable: a run over 5000 videos will get interrupted
            rows.append(_crop_row(row,out_path,index))
            stats["skipped"]+=1
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES,int(index))
        read,frame=capture.read()
        if not read:
            stats["unreadable"]+=1
            continue
        face=largest_face(detector.detect(frame))
        if face is None:
            stats["no_face"]+=1
            continue
        crop=crop_face(frame,face,scale=scale,size=size)
        if crop is None:
            stats["bad_box"]+=1
            continue
        cv2.imwrite(str(out_path),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
        rows.append(_crop_row(row,out_path,index,face["conf"]))
        stats["written"]+=1

    capture.release()
    return rows,stats


def _crop_row(row,out_path,frame_index,conf=None):
    return {"path":str(out_path),
            "label":row["label"],
            "method":row["method"],
            "identity":row["identity"],
            "source":row["source"],
            "split":row["split"],
            "video":Path(row["path"]).stem,
            "frame":int(frame_index),
            "face_conf":conf}


def extract(rows,frames=32,out_root=None,scale=ffpp_crop_scale,size=xception_input,
            workers=4,overwrite=False,limit=None):
    """Run extraction over manifest rows, threaded.

    cv2 releases the GIL inside VideoCapture.read and inside the YuNet forward
    pass, so threads genuinely overlap here - same reason the multicam pipeline
    gets away with a ThreadPoolExecutor. Each worker gets its own detector
    because cv2.FaceDetectorYN holds per-instance input-size state.
    """
    out_root=Path(out_root or crops_path)
    rows=rows[:limit] if limit else rows
    detectors={}
    produced=[]
    totals=Counter()

    def work(row):
        key=id(__import__("threading").current_thread())
        if key not in detectors:
            detectors[key]=facedetector(name=f"facedetector-{len(detectors)}")
        return extract_video(row,detectors[key],frames,out_root,scale,size,overwrite)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done,(crop_rows,stats) in enumerate(pool.map(work,rows),1):
            produced.extend(crop_rows)
            totals.update(stats)
            if done%100==0 or done==len(rows):
                logger.info(f"{done}/{len(rows)} videos - {len(produced)} crops "
                            f"({dict(totals)})")
    return produced,totals


def standin_rows(videos,frames):
    """A fabricated two-class problem from arbitrary videos.

    class 0 - the crop as extracted.
    class 1 - the same crop pushed through an aggressive recompress-and-resample
              cycle, which is a crude stand-in for a manipulation artefact.

    The point is only to prove the training loop can drive loss down and AUC up
    on a signal that is definitely present. Any result from this data says
    nothing whatsoever about deepfake detection.
    """
    rows=[]
    for video in videos:
        stem=Path(video).stem
        for label in (0,1):
            rows.append({"path":str(video),
                         "label":label,
                         "method":"standin_clean" if label==0 else "standin_artefact",
                         "identity":stem,
                         "source":None,
                         "split":"train",
                         "compression":"standin"})
    return rows


def degrade(crop):
    """The synthetic 'manipulation': recompress hard, downsample, upsample back."""
    ok,buf=cv2.imencode(".jpg",crop,[cv2.IMWRITE_JPEG_QUALITY,25])
    if not ok:
        return crop
    out=cv2.imdecode(buf,cv2.IMREAD_COLOR)
    small=cv2.resize(out,(out.shape[1]//3,out.shape[0]//3),interpolation=cv2.INTER_AREA)
    return cv2.resize(small,(out.shape[1],out.shape[0]),interpolation=cv2.INTER_CUBIC)


def build_standin(videos,frames,out_root,size=xception_input):
    """Extract once per video, then write a clean and a degraded copy of each crop."""
    out_root=Path(out_root)
    detector=facedetector()
    produced=[]
    for video in videos:
        stem=Path(video).stem
        capture=cv2.VideoCapture(str(video))
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in sample_indices(total,frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES,int(index))
            read,frame=capture.read()
            if not read:
                continue
            face=largest_face(detector.detect(frame))
            if face is None:
                continue
            crop=crop_face(frame,face,size=size)
            if crop is None:
                continue
            # the same face lands in both classes, so the classifier cannot win
            # by memorizing identity - it has to find the artefact
            for label,image in ((0,crop),(1,degrade(crop))):
                method="standin_clean" if label==0 else "standin_artefact"
                # hold out whole videos, exactly as the real splits do
                split="train" if stem.endswith(("1","3","5","7","9")) else "val"
                out_dir=out_root/split/label_names[label]/method
                out_dir.mkdir(parents=True,exist_ok=True)
                out_path=out_dir/f"{stem}_f{index:05d}.jpg"
                cv2.imwrite(str(out_path),image,[cv2.IMWRITE_JPEG_QUALITY,95])
                produced.append({"path":str(out_path),"label":label,"method":method,
                                 "identity":stem,"source":None,"split":split,
                                 "video":stem,"frame":int(index),
                                 "face_conf":face["conf"]})
        capture.release()
    return produced


def crop_summary(rows):
    out={}
    for split in sorted({r["split"] for r in rows}):
        subset=[r for r in rows if r["split"]==split]
        out[split]={"crops":len(subset),
                    "real":sum(1 for r in subset if r["label"]==0),
                    "fake":sum(1 for r in subset if r["label"]==1),
                    "videos":len({r["video"] for r in subset}),
                    "identities":len({r["identity"] for r in subset}),
                    "per_method":dict(Counter(r["method"] for r in subset))}
    return out


def main():
    parser=argparse.ArgumentParser(description="Extract face crops for training")
    parser.add_argument("--split",default="train",choices=["train","val","test","all"])
    parser.add_argument("-c","--compression",default="c23",choices=["raw","c23","c40"])
    parser.add_argument("--methods",nargs="*",default=None)
    parser.add_argument("--frames",type=int,default=32,help="frames sampled per video")
    parser.add_argument("--scale",type=float,default=ffpp_crop_scale)
    parser.add_argument("--size",type=int,default=xception_input)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--limit",type=int,default=None,help="first N videos (smoke test)")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--out",default=None)
    parser.add_argument("--standin",action="store_true",
                        help="build the fabricated dataset from data/videos/*.mp4")
    args=parser.parse_args()

    out_root=Path(args.out or crops_path)

    if args.standin:
        videos=sorted((data_path/"videos").glob("*.mp4"))
        if not videos:
            raise SystemExit("no videos - run: python src/create_video.py")
        logger.info(f"stand-in dataset from {len(videos)} clip(s) - smoke test only")
        rows=build_standin(videos,args.frames,out_root/"standin",args.size)
    else:
        manifest=build_manifest(compression=args.compression,use_methods=args.methods)
        _,problems=check_leakage(manifest)
        if problems:
            raise SystemExit(f"refusing to extract from a leaking split: {problems}")
        splits=["train","val","test"] if args.split=="all" else [args.split]
        source=[r for s in splits for r in manifest["splits"][s]]
        logger.info(f"{len(source)} videos across {splits} at {args.compression}")
        rows,totals=extract(source,args.frames,out_root,args.scale,args.size,
                            args.workers,args.overwrite,args.limit)
        logger.info(f"extraction totals: {dict(totals)}")
        if totals.get("missing"):
            logger.warning(f"{totals['missing']} videos not on disk - "
                           "run src/deepfake/ffpp.py --audit")

    if not rows:
        raise SystemExit("no crops produced")

    summary=crop_summary(rows)
    for split,s in summary.items():
        logger.info(f"{split}: {s['crops']} crops from {s['videos']} videos, "
                    f"{s['identities']} identities ({s['real']} real, {s['fake']} fake)")

    name="crops_standin.json" if args.standin else "crops_manifest.json"
    save_json({"summary":summary,"crops":rows},outputs_path/name)


if __name__=="__main__":
    main()
