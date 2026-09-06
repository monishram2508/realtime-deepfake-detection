# Definitions

## Thread

- New path of execution that can run concurrently with the main program

```python
thread=threading.Thread(target=self._load_loop,daemon=True)
```

- target -> what to do when the thread starts
  (notice how the function is just passed, and not called (_load_loop()), since we don't want it to execute it while defining the thread)
- daemon -> kill thread when main program ends

## Timeout

- Maximum duration in seconds to wait before giving up
- For example:
```python
get_frame_batch(self,timeout)
```

will loop through each camera and get a frame from each one; if any camera does NOT return a frame within *timeout* seconds, the whole batch will be dropped.

## ONNX

- Open Neural Network Exchange
- PyTorch carries all the training infrastructure that we don't need during inference; the problem arises when we train a model in PyTorch and deploy somewhere else, resulting in a crash if the system doesn't have the hardware power for running the required framework.
- ONNX is a standardized model format that converts the model weights (filters, biases, layer weights), operator implementation (convolution, matrix multiplication, normalization, activation, etc.), and tensor execution engine, into a universal format for all systems to access.
- Converts a typical PyTorch model into a standardized computation graph, operator descriptions, and serialized weights.

## Codec

- Software that compresses/decompresses video.
- Fourcc - four character code used to identify a codec. (mp4v, XVID, MJPG, H264, etc.)

## Letterbox

- Resizing that preserves aspect ratio: scale by `min(target/h, target/w)`, then pad the leftover
  space (grey, value 114 by convention) to reach a square input.
- The `(scale, pad_x, pad_y)` used on the way in is exactly what you need on the way out:

```python
x_original = (x_letterboxed - pad_x) / scale
```

## NMS (non-maximum suppression)

- Given many overlapping boxes for the same object, keep the highest-confidence box and discard any
  box overlapping it by more than an [IoU](#iou) threshold. Repeat for what's left.

## IoU

- Intersection over Union: area of overlap divided by area of union of two boxes. 1.0 = identical,
  0 = disjoint. The NMS threshold (0.45 here) is how much overlap is tolerated before two boxes are
  considered the same detection.

## Execution provider

- The backend ONNX Runtime dispatches graph nodes to: `CPUExecutionProvider`,
  `CoreMLExecutionProvider` (Apple GPU / Neural Engine), `CUDAExecutionProvider`,
  `TensorrtExecutionProvider`.
- Providers are given as an ordered list; ORT asks each in turn which nodes it can take, and
  anything unclaimed falls back to the CPU:

```python
rt.InferenceSession(path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
```

- Every switch between providers inside one graph costs a memory copy, so a graph split into many
  **partitions** can easily run slower than staying on the CPU.

## Dynamic vs static quantization

- **Dynamic**: weights are converted to int8 ahead of time; activation scales are computed at
  runtime, on every inference. No calibration data needed, but that runtime work costs latency.
- **Static**: real data is run through the fp32 model first (**calibration**) to record the min/max
  range of each activation tensor. Those scales are baked into the graph, so nothing is computed at
  runtime - faster, but only as good as the calibration data, and some layers do not survive it.

## QDQ

- Quantize-DeQuantize: a graph format where explicit `QuantizeLinear`/`DequantizeLinear` node pairs
  surround each quantized operation, rather than the operations themselves being replaced
  (`QOperator` format).
- More portable, but the extra node pairs are what accelerators refuse to execute, causing the
  graph fragmentation described in [notes.md](notes.md).

## Calibration

- Feeding representative data through the fp32 model to record activation ranges before quantizing.
- Must look like deployment data. Calibrating on frames that never occur in production produces
  scales that clip real activations.
- Methods: **MinMax** (observed extremes), **Entropy** (KL-divergence-optimal clipping),
  **Percentile** (ignores the tails).
