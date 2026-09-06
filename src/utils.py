import time
import json

import numpy as np
from config import get_logger
from contextlib import contextmanager

logger=get_logger(__name__)

@contextmanager
def timer(name,log=True):
    start=time.perf_counter()
    holder={}
    try:
        yield holder
    finally:
        runtime=(time.perf_counter()-start)*1000
        holder["ms"]=runtime
        if log:
            logger.debug(f"[TIMER] - {name}:{runtime:.2f} ms")


# creates a timing contextmanager which we use as below:
# with timer("inference") as t:
#   model(dummy)
# print(t["ms"])
#
# output: [TIMER] - inference: 42.32 ms

class perfmetrics:
    def __init__(self,name):
        self.name=name
        self.frame_count=0
        self.dropped_frames=0
        self.desync_count=0
        self.times=[]
        self.st_time=time.time()

    def reset(self):
        """Restart the clock. Called once real work begins, so that one-off setup
        (CoreML compiling a graph takes seconds) does not get averaged into fps."""
        self.frame_count=0
        self.dropped_frames=0
        self.desync_count=0
        self.times=[]
        self.st_time=time.time()

    def record_frame(self,latency):
        "record a frame's latency"
        self.frame_count+=1
        self.times.append(latency)

    def record_drop(self):
        self.dropped_frames+=1

    def record_desync(self):
        self.desync_count+=1

    def get_stats(self):
        if not self.times:
            return {"name":self.name,"frames_processed":0,"frames_dropped":self.dropped_frames}
        runtime=time.time()-self.st_time
        ordered=sorted(self.times)
        avg_latency=sum(self.times)/len(self.times)
        fps=self.frame_count/runtime if runtime>0 else 0
        return {
            "name":self.name,
            "fps":round(fps,2),
            "avg_latency":round(avg_latency,2),
            "p95_latency":round(ordered[int(0.95*(len(ordered)-1))],2),
            "max_latency":round(ordered[-1],2),
            "frames_processed":self.frame_count,
            "frames_dropped":self.dropped_frames,
            "desyncs":self.desync_count,
            "elapsed":round(runtime,1)
        }

    def log_stats(self):
        stats=self.get_stats()
        logger.info(f"[METRICS] - {self.name}:{stats}")
        return stats


def save_json(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f:
        json.dump(obj,f,indent=2)
    logger.info(f"wrote {path} ({path.stat().st_size/1e3:.1f} KB)")


def iou_matrix(boxes_a,boxes_b):
    """Pairwise IoU between two sets of [x, y, w, h] boxes."""
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a),len(boxes_b)))
    a=np.array(boxes_a,dtype=np.float32)[:,None,:] # (A, 1, 4)
    b=np.array(boxes_b,dtype=np.float32)[None,:,:] # (1, B, 4)

    ax1,ay1,ax2,ay2=a[...,0],a[...,1],a[...,0]+a[...,2],a[...,1]+a[...,3]
    bx1,by1,bx2,by2=b[...,0],b[...,1],b[...,0]+b[...,2],b[...,1]+b[...,3]

    inter_w=np.clip(np.minimum(ax2,bx2)-np.maximum(ax1,bx1),0,None)
    inter_h=np.clip(np.minimum(ay2,by2)-np.maximum(ay1,by1),0,None)
    inter=inter_w*inter_h
    union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return np.where(union>0,inter/np.maximum(union,1e-9),0.0)
