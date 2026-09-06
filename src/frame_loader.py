import cv2
import threading
import queue
import time
from config import get_logger
from utils import perfmetrics

logger=get_logger(__name__)

class frameloader():
    """Reads one video source in a background thread and buffers frames in a queue."""

    def __init__(self,video_path,cam_id,queue_size=32,realtime=True,loop=False):
        self.video_path=video_path
        self.cam_id=cam_id
        # realtime: pace reads at the clip's own fps so a file behaves like a
        # live camera. Without it the file is read as fast as the disk allows
        # and the queue overflows immediately, which tells you nothing.
        self.realtime=realtime
        # loop: restart the clip at EOF. Used by the throughput study, where the
        # source must never be the bottleneck, otherwise "frames per second"
        # measures how fast the file ran out.
        self.loop=loop
        self.frame_queue=queue.Queue(maxsize=queue_size)
        self.running=False
        self.finished=False # source exhausted (or failed to open)
        self.thread=None # Only create thread when needed
        self.metrics=perfmetrics(f"frameloader - {cam_id}")

    def start(self):
        if self.running:
            logger.warning(f"{self.cam_id} already running")
            return
        self.running=True
        self.finished=False
        self.metrics.reset()
        self.thread=threading.Thread(target=self._load_loop,daemon=True)
        self.thread.start()
        logger.info(f"frameloader started: {self.cam_id}")

    def _load_loop(self):
        capture=cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            logger.error(f"Failed to open {self.video_path}")
            self.running=False
            self.finished=True
            return
        src_fps=capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval=1.0/src_fps if self.realtime else 0.0
        next_due=time.perf_counter()
        frame_index=0
        try:
            while self.running:
                if frame_interval:
                    sleep_for=next_due-time.perf_counter()
                    if sleep_for>0:
                        time.sleep(sleep_for)
                    next_due+=frame_interval
                start=time.perf_counter()
                read,frame=capture.read()
                if not read:
                    if self.loop and frame_index>0:
                        capture.set(cv2.CAP_PROP_POS_FRAMES,0)
                        continue
                    logger.info(f"{self.cam_id} reached end of clip")
                    break
                read_ms=(time.perf_counter()-start)*1000
                try:
                    self.frame_queue.put_nowait({
                            "cam_id":self.cam_id,
                            "frame_index":frame_index,
                            "image":frame,
                            "timestamp":time.time()
                        })
                    self.metrics.record_frame(read_ms)
                    frame_index+=1
                except queue.Full:
                    # inference is slower than I/O - drop the newest frame
                    # rather than block the reader (keeps latency bounded)
                    self.metrics.record_drop()
                    logger.debug(f"{self.cam_id} dropped frame {frame_index}")
                    frame_index+=1
        finally:
            capture.release()
            self.running=False
            self.finished=True

    def get_frame(self,timeout=1):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_exhausted(self):
        """True once the source ended AND the buffer has been drained."""
        return self.finished and self.frame_queue.empty()

    def stop(self):
        self.running=False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(f"frameloader stopped: {self.cam_id}")
        self.metrics.log_stats()


class multicamloader():
    """Fans out to one frameloader per camera and hands back synchronized batches."""

    def __init__(self,video_paths,queue_size=32,max_skew=0.050,realtime=True,loop=False): # creates one frameloader for each camera
        self.loaders={
            cam_id:frameloader(path,cam_id,queue_size,realtime,loop)
            for cam_id,path in video_paths.items()
        }
        self.max_skew=max_skew # seconds; batches wider than this are counted as desynced
        self.sync_metrics=perfmetrics("multicamsync")

    def start(self):
        for loader in self.loaders.values():
            loader.start()
        self.sync_metrics.reset()

    def get_frame_batch(self,timeout=1.0):
        """One frame per camera, or None once any camera runs dry."""
        batch={}
        start=time.perf_counter()
        for cam_id,loader in self.loaders.items():
            frame=loader.get_frame(timeout=timeout)
            if frame is None:
                if loader.is_exhausted():
                    logger.info(f"{cam_id} exhausted - ending batch stream")
                else:
                    logger.warning(f"timeout waiting for {cam_id}")
                    self.sync_metrics.record_drop()
                return None
            batch[cam_id]=frame

        skew=max(f["timestamp"] for f in batch.values())-min(f["timestamp"] for f in batch.values())
        if skew>self.max_skew:
            # kept, not dropped: on file sources skew is an artifact of queue draining.
            # On live cameras this is the signal that a feed is lagging.
            self.sync_metrics.record_desync()
            logger.debug(f"batch skew {skew*1000:.1f} ms > {self.max_skew*1000:.0f} ms")
        self.sync_metrics.record_frame((time.perf_counter()-start)*1000)
        return batch

    def stop(self):
        for loader in self.loaders.values():
            loader.stop()
        self.sync_metrics.log_stats()
