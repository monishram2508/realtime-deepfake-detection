"""ONNX YOLOv8 inference and the multi-camera pipeline that drives it."""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as rt

from config import (
    models_path, outputs_path, get_logger, conf_thresh, iou_thresh, input_size,
    coco_classes, provider_preference
)
from frame_loader import multicamloader
from tracker import ioutracker
from utils import perfmetrics, timer, save_json

logger=get_logger(__name__)


def available_providers():
    return rt.get_available_providers()


def pick_provider(model_path):
    """Choose an execution provider based on what the benchmark actually showed.

    Accelerators want a graph they can swallow whole. A QDQ-quantized model is
    full of Quantize/Dequantize pairs that CoreML cannot take, so the graph gets
    split into a hundred-plus partitions and the copies between them cost more
    than the accelerator saves (measured: 127 ms on CoreML vs 32 ms on CPU).
    So: int8 stays on the CPU, float models go to an accelerator if there is one.
    """
    name=str(model_path).lower()
    if "int8" in name or "quant" in name:
        return "CPUExecutionProvider"
    for candidate in provider_preference:
        if candidate in available_providers():
            return candidate
    return "CPUExecutionProvider"


def model_label(path):
    """Short display name for a model path."""
    return Path(str(path)).name


def letterbox(image,new_size):
    """Resize keeping aspect ratio, pad the rest with grey.

    Returns the padded image plus the (scale, pad_x, pad_y) needed to map
    boxes back onto the original frame.
    """
    h,w=image.shape[:2]
    scale=min(new_size/h,new_size/w)
    nh,nw=int(round(h*scale)),int(round(w*scale))
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((new_size,new_size,3),114,dtype=np.uint8)
    pad_x,pad_y=(new_size-nw)//2,(new_size-nh)//2
    canvas[pad_y:pad_y+nh,pad_x:pad_x+nw]=resized
    return canvas,scale,pad_x,pad_y


def preprocess_frame(image,size):
    """BGR frame (HWC, uint8) -> NCHW float32 batch in [0,1], plus the letterbox map.

    Shared with the calibration reader in quantize.py so that the statistics
    collected there match what the detector actually sees at inference time.
    """
    padded,scale,pad_x,pad_y=letterbox(image,size)
    img_rgb=cv2.cvtColor(padded,cv2.COLOR_BGR2RGB)
    img_norm=img_rgb.astype(np.float32)/255.0
    img_chw=np.transpose(img_norm,(2,0,1))
    img_batch=np.expand_dims(img_chw,0)
    return img_batch,scale,pad_x,pad_y


class yolov8detector:
    """One ONNX Runtime session plus the pre/post processing around it."""

    def __init__(self,model_path,provider=None,num_threads=None,name="yolov8detector"):
        self.provider=provider or pick_provider(model_path)
        if self.provider not in available_providers():
            logger.warning(f"{self.provider} unavailable - falling back to CPU")
            self.provider="CPUExecutionProvider"

        opts=rt.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads=num_threads
        # always keep CPU as the fallback so unsupported nodes still run
        providers=[self.provider] if self.provider=="CPUExecutionProvider" \
            else [self.provider,"CPUExecutionProvider"]
        self.session=rt.InferenceSession(str(model_path),opts,providers=providers)
        self.input_name=self.session.get_inputs()[0].name
        self.output_names=[o.name for o in self.session.get_outputs()]
        self.input_size=input_size
        self.metrics=perfmetrics(name)
        self.stage_ms={"preprocess":0.0,"inference":0.0,"postprocess":0.0}
        logger.info(f"Loaded {model_label(model_path)} on {self.provider}")

    def preprocess(self,image):
        return preprocess_frame(image,self.input_size)

    def _postprocess(self,output,scale,pad_x,pad_y,og_shape):
        """Parse raw YOLOv8 output -> list of detections in original-frame pixels.

        YOLOv8 emits (1, 84, 8400): 4 box values + 80 class scores per anchor.
        There is no separate objectness score (that was YOLOv5), so the class
        score IS the confidence.
        """
        preds=output[0]
        if preds.shape[0]<preds.shape[1]: # (84, 8400) -> (8400, 84)
            preds=preds.transpose(1,0)

        scores_all=preds[:,4:]
        class_ids=np.argmax(scores_all,axis=1)
        confs=scores_all[np.arange(len(scores_all)),class_ids]
        keep=confs>conf_thresh
        if not np.any(keep):
            return []

        boxes_xywh=preds[keep,:4]
        confs=confs[keep]
        class_ids=class_ids[keep]

        # centre-xywh in letterboxed space -> top-left xywh in original frame
        cx,cy,w,h=boxes_xywh[:,0],boxes_xywh[:,1],boxes_xywh[:,2],boxes_xywh[:,3]
        x=(cx-w/2-pad_x)/scale
        y=(cy-h/2-pad_y)/scale
        w=w/scale
        h=h/scale

        boxes=np.stack([x,y,w,h],axis=1)
        idxs=cv2.dnn.NMSBoxes(boxes.tolist(),confs.tolist(),conf_thresh,iou_thresh)
        if len(idxs)==0:
            return []
        idxs=np.array(idxs).flatten()

        oh,ow=og_shape[:2]
        detections=[]
        for i in idxs:
            bx,by,bw,bh=boxes[i]
            detections.append({
                'x':round(float(max(0,bx)),1),
                'y':round(float(max(0,by)),1),
                'w':round(float(min(bw,ow-max(0,bx))),1),
                'h':round(float(min(bh,oh-max(0,by))),1),
                'conf':round(float(confs[i]),3),
                'class_id':int(class_ids[i]),
                'class_name':coco_classes[int(class_ids[i])] if int(class_ids[i])<len(coco_classes) else 'unknown'
            })
        return detections

    def detect(self,image):
        start=time.perf_counter()
        with timer("preprocess") as t_pre:
            img_batch,scale,pad_x,pad_y=self.preprocess(image)
        with timer("inference") as t_inf:
            outputs=self.session.run(self.output_names,{self.input_name:img_batch})
        with timer("postprocess") as t_post:
            detections=self._postprocess(outputs[0],scale,pad_x,pad_y,image.shape)

        self.stage_ms["preprocess"]+=t_pre["ms"]
        self.stage_ms["inference"]+=t_inf["ms"]
        self.stage_ms["postprocess"]+=t_post["ms"]
        self.metrics.record_frame((time.perf_counter()-start)*1000)
        return detections

    def stage_breakdown(self):
        n=max(self.metrics.frame_count,1)
        return {k:round(v/n,2) for k,v in self.stage_ms.items()}


def draw(image,detections):
    """Annotate a copy of the frame with boxes, labels and track ids."""
    out=image.copy()
    for det in detections:
        if det.get('confirmed') is False: # unconfirmed track - probably a blip
            continue
        x,y,w,h=int(det['x']),int(det['y']),int(det['w']),int(det['h'])
        track_id=det.get('track_id')
        # stable colour per track so identity is visible at a glance
        colour=(0,255,0) if track_id is None else (
            int(37*track_id%255),int(17*track_id%255+90),int(97*track_id%255))
        cv2.rectangle(out,(x,y),(x+w,y+h),colour,2)
        label=f"{det['class_name']} {det['conf']:.2f}"
        if track_id is not None:
            label=f"#{track_id} {label}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        # a box near the top of the frame has no room for a label above it,
        # so tuck the label inside the box instead of drawing it off-frame
        top=y-th-6
        if top<0:
            top,baseline=y+2,y+th+6
        else:
            baseline=y
        cv2.rectangle(out,(x,top),(x+tw+4,baseline),colour,-1)
        cv2.putText(out,label,(x+2,baseline-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    return out


class detectionpipe:
    """Ties the loaders, the detector(s) and the trackers together."""

    def __init__(self,video_paths,model_path=None,save_video=False,realtime=True,
                 provider=None,workers=1,track=True,loop=False):
        self.model_path=model_path or (models_path/"yolov8n.onnx")
        self.loader=multicamloader(video_paths,realtime=realtime,loop=loop)
        self.cam_ids=list(video_paths)
        self.save_video=save_video
        self.track=track
        self.results=[]
        self.writers={}
        self.metrics=perfmetrics("detectionpipe")
        self.wait_ms=0.0 # time the pipeline sat idle waiting for a full batch

        # workers=1: one shared session, cameras processed one after another.
        # workers>1: one session per camera, run concurrently. ONNX Runtime
        # releases the GIL inside session.run, so Python threads really do
        # overlap here - see --workers in demo.py and the scaling table.
        #
        # workers=0 means "decide from the provider". The CPU provider already
        # parallelizes one inference across every core, so a second session just
        # oversubscribes the machine (measured: 2 workers made it 1.9x SLOWER).
        # An accelerator runs one graph at a time and leaves the CPU free for
        # pre/post-processing, so there overlapping cameras is a real win.
        if workers==0:
            probe=provider or pick_provider(self.model_path)
            workers=1 if probe=="CPUExecutionProvider" else len(self.cam_ids)
            logger.info(f"workers=auto -> {workers} (provider {probe})")
        self.workers=max(1,min(workers,len(self.cam_ids)))
        if self.workers>1:
            self.detectors={cam:yolov8detector(self.model_path,provider=provider,
                                               name=f"detector-{cam}")
                            for cam in self.cam_ids}
            self.pool=ThreadPoolExecutor(max_workers=self.workers)
        else:
            shared=yolov8detector(self.model_path,provider=provider)
            self.detectors={cam:shared for cam in self.cam_ids}
            self.pool=None
        self.provider=next(iter(self.detectors.values())).provider
        self.trackers={cam:ioutracker() for cam in self.cam_ids} if track else {}

    def _detect_one(self,cam_id,frame):
        detections=self.detectors[cam_id].detect(frame["image"])
        if self.track:
            detections=self.trackers[cam_id].update(detections)
        return cam_id,detections

    def _write_frame(self,cam_id,frame):
        if cam_id not in self.writers:
            h,w=frame.shape[:2]
            path=outputs_path/f"{cam_id}_annotated.mp4"
            self.writers[cam_id]=cv2.VideoWriter(
                str(path),cv2.VideoWriter_fourcc(*"mp4v"),15,(w,h)
            )
        self.writers[cam_id].write(frame)

    def run(self,max_batches=None,timeout=2.0):
        logger.info(f"pipeline: {len(self.cam_ids)} camera(s), {self.workers} worker(s), "
                    f"{model_label(self.model_path)} on {self.provider}, "
                    f"tracking {'on' if self.track else 'off'}")
        self.loader.start()
        self.metrics.reset() # exclude session construction from the fps figure
        batch_index=0
        try:
            while max_batches is None or batch_index<max_batches:
                wait_start=time.perf_counter()
                batch=self.loader.get_frame_batch(timeout=timeout)
                if batch is None:
                    break
                start=time.perf_counter()
                self.wait_ms+=(start-wait_start)*1000

                if self.pool:
                    outcomes=list(self.pool.map(lambda kv:self._detect_one(*kv),batch.items()))
                else:
                    outcomes=[self._detect_one(cam,frame) for cam,frame in batch.items()]

                record={"batch_index":batch_index,"timestamp":time.time(),"cameras":{}}
                for cam_id,detections in outcomes:
                    record["cameras"][cam_id]={
                        "frame_index":batch[cam_id]["frame_index"],
                        "detection_count":len(detections),
                        "detections":detections
                    }
                    if self.save_video:
                        self._write_frame(cam_id,draw(batch[cam_id]["image"],detections))

                self.metrics.record_frame((time.perf_counter()-start)*1000)
                self.results.append(record)
                if batch_index%25==0:
                    counts=", ".join(f"{c}: {r['detection_count']}" for c,r in record["cameras"].items())
                    logger.info(f"batch {batch_index} - {counts} detections")
                batch_index+=1
        except KeyboardInterrupt:
            logger.warning("interrupted - shutting down cleanly")
        finally:
            self.stop()
        return self.results

    def summary(self):
        detectors={}
        for cam,detector in self.detectors.items():
            detectors[detector.metrics.name]={**detector.metrics.get_stats(),
                                              "provider":detector.provider,
                                              "stages_ms":detector.stage_breakdown()}
        return {
            "model":model_label(self.model_path),
            "provider":self.provider,
            "workers":self.workers,
            "pipeline":{**self.metrics.get_stats(),
                        "avg_wait_for_frames_ms":round(self.wait_ms/max(self.metrics.frame_count,1),2)},
            "detectors":detectors,
            "cameras":{cid:l.metrics.get_stats() for cid,l in self.loader.loaders.items()},
            "sync":self.loader.sync_metrics.get_stats(),
            "tracking":{cam:t.stats() for cam,t in self.trackers.items()},
        }

    def stop(self):
        self.loader.stop()
        if self.pool:
            self.pool.shutdown(wait=True)
        for writer in self.writers.values():
            writer.release()
        if self.writers:
            logger.info(f"annotated video(s) in {outputs_path}")

        stats=self.summary()
        self.metrics.log_stats()
        for detector in set(self.detectors.values()):
            detector.metrics.log_stats()
        save_json(self.results,outputs_path/"detections.json")
        save_json(stats,outputs_path/"metrics.json")
        return stats
