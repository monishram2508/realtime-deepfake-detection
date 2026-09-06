# NOTES 

## Frame Loading Process - [frame_loader.py](../src/frame_loader.py)

To my understanding,:
- Creates a class called frameloader which is an object basically reads a video path using a cv2 function (VideoCapture()), and returns frames according to the function call.
- There are two situations:
  1. Video files are in the hard drive, already pre-recorded (testing case; usually not the case in most real-life implementations)
  2. Live video feed from multiple cameras
- Implementation v1 encompasses the former of the above points, where pre-recorded videos are fed to the system pipeline.
- One frameloader object is created for each of the camera inputs. These frameloaders create a [thread](definitions.md#thread) which performs the queuing of frames without blocking inference.
- Multiple frameloader threads can be started to read frames from different video sources simultaneously.
- The multicamloader class accomodates multiple cameras, by just including multiple frameloader objects, and reading each camera sequentially, adding it to an empty dictionary if and only if all the frames are within the [timeout](definitions.md#timeout) duration of each other.

Now, this is not the case for a live feed scenario:

- If we loop through the frames sequentially, there will be timing issues in the case of actual camera-sourced video feeds. Since video files are processed and have technically the exact same FPS, the frames come in at zero offset.
- To solve this problem, we use a timestamp validation technique, i.e. if the frames' timestamps are within a certain threshold (e.g. 10ms), the batch is taken, else, dropped.

## Detection Pipeline - [detection.py](../src/detection.py)

- We have a yolov8detector class, which opens the [ONNX](definitions.md#ONNX) model using CPU providers. This object has the following methods:
  1. preprocess() - carries out resizing, normalization (0-255 to 0-1), color format conversion (BGR to RGB), HWC to CHW conversion, and addition of batch dimension (CHW to BCHW)
  2. _postprocess() - parses raw ONNX model output and extracts detections. 
    - ONNX outputs shape (1,8400,84) which needs transposing. (1,8400,84) is a tensor shape of the format (batch dimension,number of predictions,values per prediction). So, this is the dimension of the output array.
    - Looping through all 8400 predictions:
      1. the first four elements of each subarray are the coordinates x,y,w, and h, where (x,y) represent the center of the bounding box and (w,h) represent the width and the height. 
      2. output[4] gives the confidence of the model for that particular prediction.
      3. output[5:] represent 80 COCO (common objects in context) class probabilities.
      4. The highest probable class is then found, and the final confidence is returned as the product of the class confidence and the detection confidence. 
      5. The detected object is recorded if the confidence exceeds a certain threshold.

## Creating Test Videos - [create_video.py](../src/create_video.py)

- Setting dimensions as 640x480, with fps=30 and duration=10 (seconds) as default values.
- Creates VideoWriter object. Internally OpenCV checks if [codec](definitions.md#codec) exists on the system, if the system contains the mp4v codec library. Initiliazes the codec: loads the "mp4v" encoder, configures it with the settings in point 1. Returns object ready to accept frames.
- Writes frames to the object; takes raw frame, encodes with encoder, writes compressed data to file, update frame_count in metadata
- Release and finalize. 

## Corrections after finishing the pipeline

Things the first draft of `detection.py` got wrong, and why they matter:

1. **YOLOv8 has no objectness column.** I had written `conf=det[4]` and `classes=det[5:]`, then
   multiplied the two. That is the YOLOv5 layout. YOLOv8's 84 values are 4 box values + 80 class
   scores, so `scores=det[4:]` and the max class score *is* the confidence. The old version halved
   every confidence, so almost nothing survived the 0.5 threshold.
2. **Plain resize distorts boxes.** Resizing 640x480 to 640x640 stretches the frame vertically, and
   the boxes come back stretched too. [Letterboxing](definitions.md#letterbox) scales by the smaller
   ratio and pads the rest, and the (scale, pad) pair is what maps boxes back to original pixels.
3. **Confidence filtering is not NMS.** 8400 anchors mean the same person gets a dozen overlapping
   boxes above threshold. `cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)` keeps the
   highest-scoring box in each overlapping cluster.
4. **`cv2.COLOR_RGB2BGR` where I wanted `BGR2RGB`.** OpenCV decodes as BGR; the model expects RGB.
   The two constants happen to swap the same channels, so this one worked by accident.
5. **Reading files as fast as the disk allows is not a simulation of a camera.** Unpaced, the queue
   fills in a fraction of a second and the drop counter reports hundreds of drops that mean nothing.
   Pacing each reader at the clip's own fps makes the drop counter measure what it claims to.

## Measuring - what the numbers actually said

- Inference is ~96% of per-frame cost (38.7 ms of 40.3 ms). Optimizing preprocessing would be
  wasted effort.
- Dynamic int8 quantization made things **slower** on the M2 (57.5 ms vs 39.0 ms fp32), while
  shrinking the model 3.7x. Dynamic quantization computes activation scales at runtime; that
  overhead beat the savings here. Static/calibrated quantization is the thing to try next.
- Two cameras cost roughly two times one camera, because they share a single ONNX session
  sequentially.

## Optimization - [benchmark.py](../src/benchmark.py), [quantize.py](../src/quantize.py), [accuracy.py](../src/accuracy.py)

### Measurement methodology first

Before trusting any number: each (model, provider) pair now runs in its own **subprocess**. Two
sessions built in one process share thread pools and leave the CPU warm - that alone moved my fp32
CPU number from 39 ms to 61 ms between runs. Warm-up runs are discarded, and I report mean *and*
p95, because a mean hides the stalls that a real-time pipeline actually feels.

### Execution providers, and why quantized models hate them

An [execution provider](definitions.md#execution-provider) is the backend ONNX Runtime dispatches
nodes to. On this Mac: CPU and CoreML (which reaches the GPU/Neural Engine).

- fp32 on CoreML: **10.7 ms** vs 38.5 ms on CPU. A 4x win for one constructor argument.
- int8 on CoreML: **~105 ms**, i.e. far *worse* than CPU.

The reason is in ORT's own log line, which I now parse instead of ignoring:

```
number of partitions supported by CoreML: 128
number of nodes in the graph: 934
number of nodes supported by CoreML: 157
```

A [QDQ](definitions.md#qdq) graph is stuffed with Quantize/Dequantize pairs that CoreML cannot
execute, so the graph gets chopped into 128 chunks and the data is copied between CPU and
accelerator at every boundary. The fp32 graph: 11 partitions, 221 of 233 nodes accepted. Lesson:
**accelerators want a graph they can swallow whole.** Fragmentation costs more than the acceleration
buys.

### Static quantization, and the failure worth remembering

[Dynamic quantization](definitions.md#dynamic-vs-static-quantization) computes activation scales at
runtime, which is why it was slower than fp32. Static quantization records those ranges up front by
running real frames (calibration) and bakes them in.

My first static model ran 32 ms - and detected **nothing at all**. The diagnosis:

```
fp32:         box range (2.8, 640.8)   max class score 0.9024
static int8:  box range (2.5, 641.3)   max class score 0.0000
```

Box regression was perfect. Classification was annihilated. Why: the classification branch produces
a few large logits among 8400 mostly-negative ones. One uint8 scale over that range rounds every
positive logit into the same bucket, and after the sigmoid everything reads as zero. The model was
confidently certain the frame was empty.

Things that did NOT fix it: entropy calibration, percentile calibration, excluding just the
classification branch (`/model.22/cv3`). What did: keeping the **entire detection head**
(`/model.22/`, 66 nodes) in fp32 - and that model is 41 ms, no faster than plain fp32 on CPU.

Conclusion: on Apple silicon there is no int8 win available for this model. The fast version is
broken and the working version is not fast. Quantization is still the right tool for a
memory-constrained device (2-3.7x smaller) or hardware with real int8 kernels.

### Why a speed benchmark alone is dangerous

A benchmark said the broken model was the fastest one. Only [accuracy.py](../src/accuracy.py) -
matching each candidate's detections against the fp32 model's with class-aware greedy IoU - showed
F1 = 0.000. **Never ship an optimization measured only in milliseconds.**

### Threading: more workers is not more throughput

Giving each camera its own ONNX session and running them in a thread pool works in Python because
ONNX Runtime releases the GIL inside `session.run`. But:

- CPU provider: 2 workers made the pipeline **1.6x slower** (110 -> 176 ms/batch). ORT's CPU
  provider already parallelizes one inference across all 8 cores; a second session spawns a second
  thread pool and they oversubscribe the machine.
- CoreML: 2 workers made it **1.5x faster** (29 -> 20 ms/batch). The accelerator runs one graph at a
  time, leaving the CPU free to decode/letterbox/NMS the other camera concurrently.

So the right worker count is a property of the provider, which is why the default is `auto`.

## Tracking - [tracker.py](../src/tracker.py)

Detection is per-frame and stateless; tracking adds identity. The greedy IoU tracker: for every new
frame, compute IoU between existing tracks and new detections, match highest overlap first, skipping
any track or detection already used.

The two parameters that carry the weight are not the matching itself:

- `max_age=5` - a track survives a few frames with no match, so one missed detection or a brief
  occlusion does not renumber the object.
- `min_hits=2` - a track is not reported until seen twice, which drops single-frame false positives.

Matching is class-aware: the same box called `person` then `car` is not one object. What is missing
is a motion model - with no velocity estimate, a fast-moving object's new box may not overlap its old
one at all, and it gets a new id. That is what SORT's Kalman filter fixes.
