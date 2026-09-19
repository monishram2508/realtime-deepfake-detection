"""Face detection and the crop convention the classifier is trained on.

    python src/deepfake/faces.py --video data/videos/cam_1.mp4 --save

Two jobs, and the second one matters more than it looks.

**Detection.** YuNet (232 KB, from the OpenCV Zoo) instead of the dlib detector the
original FF++ baseline used. dlib's `get_frontal_face_detector` is a HOG cascade
from 2014: it misses profile views, occlusion and anything under heavy
compression - which is exactly the footage this project cares about - and it is
painful to build on Apple Silicon. YuNet is a small CNN, ships as ONNX, gives five
landmarks for free, and OpenCV already decodes its three-stride output.

`facedetector.detect` returns the **same dictionary shape** `yolov8detector.detect`
returns - x, y, w, h, conf, class_id, class_name - which is the whole integration
claim: `ioutracker` then tracks faces across frames with no changes at all. The
detection contract is the interface, not the model.

**Cropping.** The classifier never sees a frame, only a crop, so the crop
convention *is* part of the model. FF++ baselines use a square box centred on the
face and scaled by 1.3, and a classifier trained on 1.3 crops degrades if it is
served 1.0 crops. `crop_face` implements that convention exactly, and `serve`
has to call the same function - which is why it lives here and not in dataset.py.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from config import models_path, outputs_path, get_logger
from utils import perfmetrics, timer

logger=get_logger(__name__)

yunet_name="face_detection_yunet_2023mar.onnx"
yunet_url=("https://github.com/opencv/opencv_zoo/raw/main/models/"
           "face_detection_yunet/face_detection_yunet_2023mar.onnx")

ffpp_crop_scale=1.3
xception_input=299


def model_file(path=None):
    path=Path(path or models_path/yunet_name)
    if not path.exists():
        raise SystemExit(f"missing {path}\n  curl -sSL -o {path} {yunet_url}")
    return path


class facedetector:
    """YuNet wrapped to speak the pipeline's detection dict.

    Args:
        conf_thresh: score floor. 0.6 is deliberately below YuNet's usual 0.9 -
            c40 footage is heavily compressed and faces score lower on it, and a
            missed face is a dropped training sample.
        input_size: YuNet is fully convolutional, so it is re-sized per frame to
            the frame's own dimensions rather than letterboxed. That avoids the
            padding round trip entirely; boxes come back in frame pixels already.
    """

    def __init__(self,model_path=None,conf_thresh=0.6,nms_thresh=0.3,top_k=5000,
                 name="facedetector"):
        self.model_path=model_file(model_path)
        self.conf_thresh=conf_thresh
        self.detector=cv2.FaceDetectorYN.create(
            str(self.model_path),"",(320,320),conf_thresh,nms_thresh,top_k)
        self._size=(320,320)
        self.metrics=perfmetrics(name)
        self.stage_ms={"preprocess":0.0,"inference":0.0,"postprocess":0.0}
        logger.info(f"loaded {self.model_path.name} (conf>{conf_thresh})")

    def detect(self,image):
        """One frame -> list of face detections in original-frame pixels."""
        h,w=image.shape[:2]
        with timer("preprocess") as t_pre:
            if self._size!=(w,h):
                self.detector.setInputSize((w,h))
                self._size=(w,h)
        with timer("inference") as t_inf:
            _,faces=self.detector.detect(image)
        with timer("postprocess") as t_post:
            detections=self._to_detections(faces,w,h)

        for stage,t in (("preprocess",t_pre),("inference",t_inf),("postprocess",t_post)):
            self.stage_ms[stage]+=t["ms"]
        self.metrics.record_frame(t_pre["ms"]+t_inf["ms"]+t_post["ms"])
        return detections

    def _to_detections(self,faces,frame_w,frame_h):
        if faces is None:
            return []
        out=[]
        for face in faces:
            x,y,w,h=face[:4]
            x,y=max(0.0,float(x)),max(0.0,float(y))
            w=min(float(w),frame_w-x)
            h=min(float(h),frame_h-y)
            if w<=1 or h<=1:
                continue
            out.append({
                'x':round(x,1),'y':round(y,1),'w':round(w,1),'h':round(h,1),
                'conf':round(float(face[14]),3),
                'class_id':0,
                'class_name':'face',
                'landmarks':[[round(float(face[4+2*i]),1),round(float(face[5+2*i]),1)]
                             for i in range(5)],
            })
        return out

    def stage_breakdown(self):
        n=max(self.metrics.frame_count,1)
        return {k:round(v/n,2) for k,v in self.stage_ms.items()}


def square_box(detection,frame_shape,scale=ffpp_crop_scale):
    """The FF++ crop box: square, centred on the face, scaled, clamped to frame.

    Ported from the FF++ baseline's get_boundingbox so crops match the published
    setup. Clamping shrinks the box at frame edges rather than shifting it, which
    keeps the face centred at the cost of a little context.
    """
    h,w=frame_shape[:2]
    x,y,bw,bh=detection['x'],detection['y'],detection['w'],detection['h']
    size=int(max(bw,bh)*scale)
    cx,cy=int(x+bw/2),int(y+bh/2)
    x1=max(cx-size//2,0)
    y1=max(cy-size//2,0)
    size=min(w-x1,size)
    size=min(h-y1,size)
    return x1,y1,size


def crop_face(image,detection,scale=ffpp_crop_scale,size=xception_input):
    """Frame + detection -> square face crop at the classifier's input size.

    Returns None when the box collapses (a face detected right on the frame edge),
    which the caller must treat as "no usable sample" rather than as a black image.
    """
    x1,y1,box=square_box(detection,image.shape,scale)
    if box<16:
        return None
    crop=image[y1:y1+box,x1:x1+box]
    if crop.size==0:
        return None
    interp=cv2.INTER_AREA if box>size else cv2.INTER_CUBIC
    return cv2.resize(crop,(size,size),interpolation=interp)


def largest_face(detections):
    """FF++ clips are single-subject, so the biggest face is the subject.

    Multi-face frames are a serving-time concern, not a training-time one - there
    the tracker keeps identities apart and every track is classified separately.
    """
    return max(detections,key=lambda d:d['w']*d['h']) if detections else None


def main():
    parser=argparse.ArgumentParser(description="Face detection smoke test")
    parser.add_argument("--video",default=None,help="video to scan")
    parser.add_argument("--image",default=None,help="single image to scan")
    parser.add_argument("--frames",type=int,default=30,help="frames to read from a video")
    parser.add_argument("--conf",type=float,default=0.6)
    parser.add_argument("--scale",type=float,default=ffpp_crop_scale)
    parser.add_argument("--save",action="store_true",help="write crops to outputs/faces/")
    args=parser.parse_args()

    if not args.video and not args.image:
        raise SystemExit("pass --video or --image")

    detector=facedetector(conf_thresh=args.conf)
    out_dir=outputs_path/"faces"
    if args.save:
        out_dir.mkdir(parents=True,exist_ok=True)

    frames=[]
    if args.image:
        image=cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"could not read {args.image}")
        frames=[(0,image)]
    else:
        capture=cv2.VideoCapture(str(args.video))
        for index in range(args.frames):
            read,frame=capture.read()
            if not read:
                break
            frames.append((index,frame))
        capture.release()
        if not frames:
            raise SystemExit(f"no frames read from {args.video}")

    total=0
    for index,frame in frames:
        detections=detector.detect(frame)
        total+=len(detections)
        face=largest_face(detections)
        if face and args.save:
            crop=crop_face(frame,face,scale=args.scale)
            if crop is not None:
                cv2.imwrite(str(out_dir/f"frame_{index:04d}.jpg"),crop)

    stats=detector.metrics.get_stats()
    logger.info(f"{len(frames)} frames, {total} faces, "
                f"{stats['avg_latency']:.2f} ms/frame avg, "
                f"stages {detector.stage_breakdown()}")
    if args.save:
        logger.info(f"crops in {out_dir}")


if __name__=="__main__":
    main()
