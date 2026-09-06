"""Provider selection, calibration plumbing, and the fidelity metric."""
import numpy as np
import pytest

from accuracy import compare, match
from config import models_path, input_size
from detection import pick_provider

onnx_model=models_path/"yolov8n.onnx"


def det(x,y,w=50,h=50,class_id=0,conf=0.9):
    return {'x':float(x),'y':float(y),'w':float(w),'h':float(h),
            'conf':conf,'class_id':class_id,'class_name':'person'}


def test_int8_models_stay_on_cpu():
    # quantized graphs fragment on accelerators - this is the rule that encodes it
    assert pick_provider("models/yolov8n_int8.onnx")=="CPUExecutionProvider"
    assert pick_provider("models/yolov8n_int8_static.onnx")=="CPUExecutionProvider"


def test_float_models_get_an_available_provider():
    import onnxruntime as rt
    chosen=pick_provider("models/yolov8n.onnx")
    assert chosen in rt.get_available_providers()


def test_identical_detections_score_perfectly():
    frames=[[det(10,10),det(200,200)],[det(12,10)]]
    result=compare(frames,frames,0.5)
    assert result["f1"]==1.0
    assert result["mean_iou_of_matches"]==pytest.approx(1.0)
    assert result["missed_vs_reference"]==0 and result["spurious_vs_reference"]==0


def test_a_missed_object_lowers_recall():
    reference=[[det(10,10),det(200,200)]]
    candidate=[[det(10,10)]] # found one of the two
    result=compare(reference,candidate,0.5)
    assert result["recall"]==pytest.approx(0.5)
    assert result["precision"]==pytest.approx(1.0)
    assert result["missed_vs_reference"]==1


def test_an_invented_object_lowers_precision():
    reference=[[det(10,10)]]
    candidate=[[det(10,10),det(400,400)]]
    result=compare(reference,candidate,0.5)
    assert result["precision"]==pytest.approx(0.5)
    assert result["spurious_vs_reference"]==1


def test_confidence_drift_is_signed():
    reference=[[det(10,10,conf=0.9)]]
    candidate=[[det(10,10,conf=0.8)]]
    result=compare(reference,candidate,0.5)
    assert result["mean_conf_delta"]==pytest.approx(-0.1,abs=1e-6)
    assert result["max_conf_drop"]==pytest.approx(0.1,abs=1e-6)


def test_matching_is_class_aware():
    pairs,missed,extra=match([det(10,10,class_id=0)],[det(10,10,class_id=5)],0.5)
    assert pairs==[] and missed==1 and extra==1


def test_matching_is_greedy_on_best_overlap():
    reference=[det(0,0,100,100)]
    candidate=[det(50,0,100,100),det(2,0,100,100)] # second is the better match
    pairs,_,_=match(reference,candidate,0.3)
    assert len(pairs)==1
    assert pairs[0][1]==1


@pytest.mark.skipif(not onnx_model.exists(),reason="run python src/benchmark.py first")
def test_calibration_reader_yields_model_shaped_batches():
    from quantize import videocalibrationreader
    frames=[np.zeros((480,640,3),dtype=np.uint8) for _ in range(3)]
    reader=videocalibrationreader(frames,"images")
    batch=reader.get_next()
    assert batch["images"].shape==(1,3,input_size,input_size)
    assert batch["images"].dtype==np.float32
    assert reader.get_next() is not None and reader.get_next() is not None
    assert reader.get_next() is None # exhausted -> calibration stops
    reader.rewind()
    assert reader.get_next() is not None
