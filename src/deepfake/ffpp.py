"""FaceForensics++ layout, official splits, and the manifest everything downstream reads.

    python src/deepfake/ffpp.py --manifest              # build outputs/ffpp_manifest.json
    python src/deepfake/ffpp.py --manifest -c c40       # low-quality variant
    python src/deepfake/ffpp.py --audit                 # what is actually on disk

This module exists for one reason: **the split**. FF++ ships 1000 real videos and,
per manipulation method, one fake video for each ordered (target, source) pair,
named `<target>_<source>.mp4`. Splitting frames randomly - or even splitting
*videos* randomly - leaks identity. Frames of person 071 land in both train and
test, the classifier learns the face instead of the artefact, and the 99% test
accuracy that comes out is meaningless.

The official splits in `dataset/splits/*.json` are lists of *pairs*, not single
ids, precisely so that a fake video and both originals it was built from always
land in the same split. Verified on the real files: 720 / 140 / 140 unique
identities, zero overlap between any two splits (see tests/test_ffpp.py).

Those split files are public - they are in the FaceForensics repo, not behind the
access form - so the manifest can be built and tested before the dataset itself
has finished downloading.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import data_path, outputs_path, get_logger
from utils import save_json

logger=get_logger(__name__)

ffpp_path=data_path/"ffpp"
splits_path=ffpp_path/"splits"

methods=["Deepfakes","Face2Face","FaceSwap","NeuralTextures"]
compressions=["raw","c23","c40"]

real_label=0
fake_label=1


def original_dir(root,compression):
    return Path(root)/"original_sequences"/"youtube"/compression/"videos"


def manipulated_dir(root,method,compression):
    return Path(root)/"manipulated_sequences"/method/compression/"videos"


def load_split(name,splits_dir=None):
    """Read one official split file. Returns a list of (target, source) id pairs."""
    path=Path(splits_dir or splits_path)/f"{name}.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "fetch the official splits (they are public, no dataset access needed):\n"
            "  curl -sSL -o data/ffpp/splits/train.json "
            "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json"
        )
    return [(str(t),str(s)) for t,s in json.loads(path.read_text())]


def split_identities(pairs):
    """Every original-video id referenced by a split."""
    return {i for pair in pairs for i in pair}


def entries_for_split(name,root=None,compression="c23",use_methods=None,splits_dir=None):
    """Every video belonging to one split, as manifest rows.

    Each pair (t, s) contributes:
      - two originals, `t.mp4` and `s.mp4`
      - two fakes per method, `t_s.mp4` and `s_t.mp4` - the manipulation is run in
        both directions, which is how 500 pairs become 1000 videos per method.

    Paths are returned whether or not the file exists; `audit()` is what checks
    the download. That keeps the manifest a pure function of the split files, so
    it can be built and tested with no dataset on disk.
    """
    root=Path(root or ffpp_path)
    use_methods=use_methods or methods
    pairs=load_split(name,splits_dir)
    rows=[]

    for identity in sorted(split_identities(pairs)):
        rows.append({
            "path":str(original_dir(root,compression)/f"{identity}.mp4"),
            "label":real_label,
            "method":"original",
            "identity":identity,
            "source":None,
            "split":name,
            "compression":compression,
        })

    for method in use_methods:
        directory=manipulated_dir(root,method,compression)
        for target,source in pairs:
            for a,b in ((target,source),(source,target)):
                rows.append({
                    "path":str(directory/f"{a}_{b}.mp4"),
                    "label":fake_label,
                    "method":method,
                    "identity":a,
                    "source":b,
                    "split":name,
                    "compression":compression,
                })
    return rows


def build_manifest(root=None,compression="c23",use_methods=None,splits_dir=None):
    manifest={"compression":compression,
              "methods":list(use_methods or methods),
              "root":str(root or ffpp_path),
              "splits":{}}
    for name in ("train","val","test"):
        manifest["splits"][name]=entries_for_split(
            name,root,compression,use_methods,splits_dir)
    return manifest


def check_leakage(manifest):
    """No identity may appear in more than one split.

    This is the assertion the whole project rests on, so it is checked rather
    than assumed - and checked on the manifest, not on the split files, so that
    a bug in entries_for_split is caught too.
    """
    seen={}
    for name,rows in manifest["splits"].items():
        ids=set()
        for row in rows:
            ids.add(row["identity"])
            if row["source"]:
                ids.add(row["source"])
        seen[name]=ids

    problems=[]
    names=list(seen)
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            overlap=seen[names[i]]&seen[names[j]]
            if overlap:
                problems.append((names[i],names[j],sorted(overlap)))
    return seen,problems


def summarize(manifest):
    rows=[]
    for name,entries in manifest["splits"].items():
        counts=Counter(e["method"] for e in entries)
        real=sum(1 for e in entries if e["label"]==real_label)
        fake=sum(1 for e in entries if e["label"]==fake_label)
        rows.append({"split":name,
                     "videos":len(entries),
                     "real":real,
                     "fake":fake,
                     "ratio":round(fake/real,2) if real else 0,
                     "per_method":dict(counts)})
    return rows


def audit(manifest):
    """Which manifest rows actually exist on disk. Run this after downloading."""
    report={}
    for name,entries in manifest["splits"].items():
        present=[e for e in entries if Path(e["path"]).exists()]
        missing=[e for e in entries if not Path(e["path"]).exists()]
        by_method=Counter(e["method"] for e in missing)
        report[name]={"expected":len(entries),
                      "present":len(present),
                      "missing":len(missing),
                      "missing_by_method":dict(by_method),
                      "examples":[e["path"] for e in missing[:3]]}
    return report


def main():
    parser=argparse.ArgumentParser(description="FF++ splits, manifest and download audit")
    parser.add_argument("--root",default=None,help=f"dataset root (default {ffpp_path})")
    parser.add_argument("-c","--compression",default="c23",choices=compressions)
    parser.add_argument("--methods",nargs="*",default=None,choices=methods)
    parser.add_argument("--manifest",action="store_true",help="write outputs/ffpp_manifest.json")
    parser.add_argument("--audit",action="store_true",help="report what is missing on disk")
    args=parser.parse_args()

    manifest=build_manifest(args.root,args.compression,args.methods)

    seen,problems=check_leakage(manifest)
    logger.info("identities per split: "+", ".join(f"{k}={len(v)}" for k,v in seen.items()))
    if problems:
        for a,b,overlap in problems:
            logger.error(f"IDENTITY LEAK between {a} and {b}: {len(overlap)} shared ids "
                         f"(e.g. {overlap[:5]})")
        raise SystemExit("refusing to continue with a leaking split")
    logger.info("no identity overlap between splits")

    lines=["| Split | Videos | Real | Fake | Fake:Real |","|---|---|---|---|---|"]
    for row in summarize(manifest):
        logger.info(f"{row['split']}: {row['videos']} videos "
                    f"({row['real']} real, {row['fake']} fake, {row['ratio']}:1)")
        lines.append(f"| {row['split']} | {row['videos']} | {row['real']} | "
                     f"{row['fake']} | {row['ratio']}:1 |")
    logger.info("\n"+"\n".join(lines))

    if args.audit:
        report=audit(manifest)
        for name,r in report.items():
            logger.info(f"{name}: {r['present']}/{r['expected']} on disk, {r['missing']} missing")
            if r["missing"]:
                logger.warning(f"  missing by method: {r['missing_by_method']}")
                logger.warning(f"  e.g. {r['examples']}")
        save_json(report,outputs_path/"ffpp_audit.json")

    if args.manifest:
        save_json(manifest,outputs_path/"ffpp_manifest.json")


if __name__=="__main__":
    main()
