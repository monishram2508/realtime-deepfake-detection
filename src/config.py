import logging as lg
from pathlib import Path

root=Path(__file__).parent.parent
data_path=root/"data"
models_path=root/"models"
logs_path=root/"logs"
outputs_path=root/"outputs"
for path in [data_path,models_path,logs_path,outputs_path]:
    path.mkdir(parents=True,exist_ok=True)

model_name="yolov8n"
conf_thresh=0.5
iou_thresh=0.45

input_size=640
batch_size=1
max_threads=2

target_fps=10
target_latency=100

# tried in order for float models; int8 models are pinned to CPU (see
# pick_provider in detection.py - quantized graphs fragment on accelerators)
provider_preference=["CUDAExecutionProvider","CoreMLExecutionProvider","CPUExecutionProvider"]

log_level=lg.INFO
log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_logger(name):
    logger=lg.getLogger(name)
    logger.setLevel(log_level)
    # ultralytics and onnxruntime call logging.basicConfig(), which puts a handler
    # on the root logger. Without this, every line would be printed twice.
    logger.propagate=False
    if logger.handlers: # already configured - don't stack duplicate handlers
        return logger

    fh=lg.FileHandler(logs_path/f'{name}.log')
    fh.setLevel(log_level)
    fh.setFormatter(lg.Formatter(log_format))

    ch=lg.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(lg.Formatter(log_format))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# COCO class names, in the order YOLOv8 emits them
coco_classes=[
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]
