"""Binary classification metrics, and the frame -> video aggregation that matters most.

Pure numpy on purpose. sklearn would do all of this, but it is a heavy dependency
for four functions, and keeping it out means these run in CI next to the geometry
tests without installing a training stack.

Three things here are worth reading rather than skimming:

**AUC is reported alongside accuracy, not instead of it.** FF++ train splits run
4:1 fake:real, so a model that answers "fake" unconditionally scores 80% accuracy.
Accuracy at a fixed 0.5 threshold is close to meaningless on this data; AUC is
threshold-free and does not move when the class balance does.

**The threshold is chosen on validation, never on test.** `best_threshold` exists
so that the operating point is a fitted parameter with a stated provenance,
rather than an unexamined 0.5 that happens to flatter the numbers.

**Video-level scores are the headline.** A deepfake detector that classifies one
frame is an image classifier; the question anyone actually asks is whether a
*clip* is manipulated. Averaging frame probabilities over a video is the cheapest
honest way to answer that, and it is the same operation `serve.py` performs over a
face track - so the offline metric and the online behaviour agree by construction.
"""
import numpy as np


def roc_auc(labels,scores):
    """Area under the ROC curve, via the Mann-Whitney U identity.

    Ties get averaged ranks, which matters here: a quantized model can emit the
    same score for many frames, and counting ties as wins would inflate AUC.
    """
    labels=np.asarray(labels)
    scores=np.asarray(scores,dtype=np.float64)
    positives=int((labels==1).sum())
    negatives=int((labels==0).sum())
    if positives==0 or negatives==0:
        return float("nan")  # undefined with one class present - say so, don't return 0.5

    order=np.argsort(scores,kind="mergesort")
    ranked=scores[order]
    ranks=np.empty(len(scores),dtype=np.float64)
    i=0
    while i<len(ranked):
        j=i
        while j+1<len(ranked) and ranked[j+1]==ranked[i]:
            j+=1
        ranks[order[i:j+1]]=(i+j)/2.0+1.0  # average rank over the tied block
        i=j+1

    rank_sum=ranks[labels==1].sum()
    return float((rank_sum-positives*(positives+1)/2.0)/(positives*negatives))


def binary_metrics(labels,scores,threshold=0.5):
    """Confusion counts and the usual derived rates at one operating point.

    Positive class is 1 = fake, so recall is "of the manipulated clips, how many
    did we catch" and precision is "of the clips we flagged, how many were fake".
    """
    labels=np.asarray(labels)
    predictions=(np.asarray(scores)>=threshold).astype(int)
    tp=int(((predictions==1)&(labels==1)).sum())
    fp=int(((predictions==1)&(labels==0)).sum())
    tn=int(((predictions==0)&(labels==0)).sum())
    fn=int(((predictions==0)&(labels==1)).sum())

    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    total=tp+fp+tn+fn
    return {"threshold":round(float(threshold),4),
            "accuracy":round((tp+tn)/total,4) if total else 0.0,
            "precision":round(precision,4),
            "recall":round(recall,4),
            "f1":round(f1,4),
            "tp":tp,"fp":fp,"tn":tn,"fn":fn}


def best_threshold(labels,scores,metric="f1"):
    """Operating point maximizing f1 (or Youden's J) over the observed scores.

    Fit this on validation and carry it to test. Fitting it on test is how a
    project reports a number it could not reproduce on new data.
    """
    labels=np.asarray(labels)
    scores=np.asarray(scores,dtype=np.float64)
    candidates=np.unique(scores)
    if len(candidates)==0:
        return 0.5,{}
    # midpoints between distinct scores, plus the extremes, so every distinct
    # split of the data is actually reachable
    if len(candidates)>1:
        candidates=np.concatenate([[candidates[0]-1e-6],
                                   (candidates[:-1]+candidates[1:])/2.0,
                                   [candidates[-1]+1e-6]])
    best=(-1.0,0.5,{})
    for threshold in candidates:
        stats=binary_metrics(labels,scores,threshold)
        if metric=="youden":
            tnr=stats["tn"]/(stats["tn"]+stats["fp"]) if stats["tn"]+stats["fp"] else 0.0
            value=stats["recall"]+tnr-1.0
        else:
            value=stats[metric]
        if value>best[0]:
            best=(value,float(threshold),stats)
    return best[1],best[2]


def aggregate_by_video(rows,scores,reduction="mean"):
    """Frame scores -> one score per video.

    `rows` are crop-manifest rows (each carrying `video` and `label`), aligned
    positionally with `scores`. Returns (video_ids, labels, scores).

    mean is the default because it is what FF++ baselines report. median is
    steadier when the face detector occasionally hands over a bad crop, which is
    the realistic failure mode on c40.
    """
    buckets={}
    for row,score in zip(rows,scores):
        key=row["video"]
        if key not in buckets:
            buckets[key]={"label":row["label"],"scores":[]}
        buckets[key]["scores"].append(float(score))

    reduce_fn={"mean":np.mean,"median":np.median,"max":np.max}[reduction]
    videos=sorted(buckets)
    labels=np.array([buckets[v]["label"] for v in videos])
    values=np.array([reduce_fn(buckets[v]["scores"]) for v in videos])
    return videos,labels,values


def evaluate(rows,scores,threshold=0.5,reduction="mean"):
    """Frame-level and video-level metrics in one report."""
    frame_labels=np.array([r["label"] for r in rows])
    videos,video_labels,video_scores=aggregate_by_video(rows,scores,reduction)
    return {
        "frames":{"n":len(rows),
                  "auc":round(roc_auc(frame_labels,scores),4),
                  **binary_metrics(frame_labels,scores,threshold)},
        "videos":{"n":len(videos),
                  "reduction":reduction,
                  "auc":round(roc_auc(video_labels,video_scores),4),
                  **binary_metrics(video_labels,video_scores,threshold)},
    }


def by_group(rows,scores,key="method",threshold=0.5):
    """Per-manipulation-method breakdown.

    The headline number hides the thing worth knowing: NeuralTextures is
    consistently the hardest of the four, and a single averaged F1 lets a model
    that is blind to it look fine.
    """
    out={}
    groups=sorted({r[key] for r in rows})
    for group in groups:
        pairs=[(r,s) for r,s in zip(rows,scores) if r[key]==group]
        if not pairs:
            continue
        group_rows=[r for r,_ in pairs]
        group_scores=[s for _,s in pairs]
        labels=np.array([r["label"] for r in group_rows])
        stats=binary_metrics(labels,group_scores,threshold)
        # AUC needs both classes; a single-method slice is all-fake by construction,
        # so report recall there and leave AUC undefined rather than faking it
        stats["auc"]=round(roc_auc(labels,group_scores),4)
        stats["n"]=len(group_rows)
        out[group]=stats
    return out
