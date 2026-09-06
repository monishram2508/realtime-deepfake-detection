import time
from ultralytics import YOLO
from pathlib import Path
import logging
import numpy as np

root=Path(__file__).parent.parent
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__) #creates a logger object for this file
#__name__ is usually "main"
logger.info("loading yolov8 nano") # logs an INFO level message
model=YOLO(root/"models"/"yolov8n.pt")
logger.info("make dummy image + inference")

dummy=np.zeros((640,640,3),dtype=np.uint8) #(h,w,c)
_=model(dummy,verbose=False) #verbose=False means run inference quietly
# above line runs one inference before benchmarking
times=[]
for i in range(10):
    start=time.perf_counter()
    results=model(dummy,verbose=False)
    elapsed=time.perf_counter()-start
    times.append(elapsed*1000)
avg=sum(times)/len(times)
logger.info(f"pytorch inference = {avg:.2f} ms/frame")
logger.info(f"fps = {1000/avg:.1f}")
# (export + full benchmark now live in benchmark.py)
logger.info("done")

