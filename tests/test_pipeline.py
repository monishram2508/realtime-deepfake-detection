"""Unit tests for the pieces that are easy to get quietly wrong:
geometry, output parsing, threading, and metrics."""
import numpy as np
import pytest

from config import models_path, input_size
from detection import letterbox, yolov8detector
from frame_loader import frameloader, multicamloader
from utils import perfmetrics

onnx_model=models_path/"yolov8n.onnx"
needs_model=pytest.mark.skipif(not onnx_model.exists(),
                               reason="run python src/benchmark.py first")


def test_letterbox_preserves_aspect_ratio():
    img=np.zeros((480,640,3),dtype=np.uint8)
    padded,scale,pad_x,pad_y=letterbox(img,input_size)
    assert padded.shape==(input_size,input_size,3)
    assert scale==pytest.approx(input_size/640)
    assert pad_y>0 and pad_x==0 # wide image -> padding top and bottom only


def test_letterbox_maps_a_point_back_to_the_original():
    img=np.zeros((480,640,3),dtype=np.uint8)
    _,scale,pad_x,pad_y=letterbox(img,input_size)
    # a point at the original (100, 200) lands here after letterboxing...
    lx,ly=100*scale+pad_x,200*scale+pad_y
    # ...and the inverse mapping used in _postprocess must bring it back
    assert (lx-pad_x)/scale==pytest.approx(100)
    assert (ly-pad_y)/scale==pytest.approx(200)


@needs_model
def test_detector_output_contract():
    det=yolov8detector(onnx_model)
    frame=np.zeros((480,640,3),dtype=np.uint8)
    batch,scale,pad_x,pad_y=det.preprocess(frame)
    assert batch.shape==(1,3,input_size,input_size)
    assert batch.dtype==np.float32
    assert 0.0<=batch.min() and batch.max()<=1.0


@needs_model
def test_postprocess_finds_a_planted_box():
    det=yolov8detector(onnx_model)
    # fake YOLOv8 output: (1, 84, 8400), one anchor with a confident "person"
    raw=np.zeros((1,84,8400),dtype=np.float32)
    raw[0,:4,0]=[320,320,64,64] # centre-xywh in letterboxed space
    raw[0,4,0]=0.9              # class 0 score
    _,scale,pad_x,pad_y=letterbox(np.zeros((480,640,3),np.uint8),input_size)
    out=det._postprocess(raw,scale,pad_x,pad_y,(480,640,3))
    assert len(out)==1
    box=out[0]
    assert box['class_name']=='person' and box['conf']==pytest.approx(0.9,abs=1e-3)
    assert box['w']==pytest.approx(64/scale,abs=1.0)


@needs_model
def test_postprocess_drops_low_confidence():
    det=yolov8detector(onnx_model)
    raw=np.zeros((1,84,8400),dtype=np.float32)
    raw[0,:4,0]=[320,320,64,64]
    raw[0,4,0]=0.1 # below conf_thresh
    assert det._postprocess(raw,1.0,0,0,(640,640,3))==[]


def test_frameloader_reads_a_clip(tmp_path):
    import cv2
    path=tmp_path/"clip.mp4"
    out=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),30,(64,64))
    for _ in range(10):
        out.write(np.zeros((64,64,3),dtype=np.uint8))
    out.release()

    loader=frameloader(path,"cam_test",queue_size=32,realtime=False)
    loader.start()
    frames=[]
    while True:
        frame=loader.get_frame(timeout=2)
        if frame is None:
            break
        frames.append(frame)
    loader.stop()

    assert len(frames)==10
    assert [f["frame_index"] for f in frames]==list(range(10))
    assert loader.is_exhausted()


def test_frameloader_handles_a_missing_file(tmp_path):
    loader=frameloader(tmp_path/"nope.mp4","cam_missing",realtime=False)
    loader.start()
    assert loader.get_frame(timeout=1) is None
    loader.stop()
    assert loader.is_exhausted()


def test_multicam_batch_is_none_when_a_camera_ends(tmp_path):
    import cv2
    paths={}
    for cam,n in [("a",4),("b",2)]: # cameras of unequal length
        p=tmp_path/f"{cam}.mp4"
        out=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*"mp4v"),30,(64,64))
        for _ in range(n):
            out.write(np.zeros((64,64,3),dtype=np.uint8))
        out.release()
        paths[cam]=p

    loader=multicamloader(paths,realtime=False)
    loader.start()
    batches=0
    while loader.get_frame_batch(timeout=2) is not None:
        batches+=1
    loader.stop()
    assert batches==2 # limited by the shorter camera


def test_metrics_percentiles():
    m=perfmetrics("t")
    for value in range(1,101):
        m.record_frame(value)
    m.record_drop()
    stats=m.get_stats()
    assert stats["frames_processed"]==100
    assert stats["frames_dropped"]==1
    assert stats["avg_latency"]==pytest.approx(50.5)
    assert stats["max_latency"]==100
    assert 90<=stats["p95_latency"]<=100
