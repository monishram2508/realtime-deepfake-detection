"""Generate the test camera feeds used by the demo.

Two flavours:
  * "scene"     - pans a crop window across a real photo, so the detector has
                  actual people/vehicles to find. Needs one download the first time.
  * "synthetic" - drawn shapes only. No network needed, but YOLO will (correctly)
                  find nothing in it, so it only exercises the plumbing.
"""
import urllib.request
import cv2
import numpy as np
from config import data_path, get_logger

logger=get_logger(__name__)

sources={
    "cam_1":"https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
    "cam_2":"https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg",
}
w,h=640,480


def _fetch(url,dest):
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True,exist_ok=True)
    logger.info(f"downloading {url}")
    urllib.request.urlretrieve(url,dest)
    return dest


def _writer(filename,fps):
    video_path=data_path/"videos"/filename
    video_path.parent.mkdir(parents=True,exist_ok=True)
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    return video_path,cv2.VideoWriter(str(video_path),fourcc,fps,(w,h))


def _stamp(frame,cam_id,index):
    """Burn the camera id and frame number in at the bottom: detections tend to
    start near the top of a frame, and overlapping labels are unreadable."""
    y=frame.shape[0]-15
    cv2.putText(frame,f"{cam_id} | frame {index}",(10,y),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),4)   # dark outline
    cv2.putText(frame,f"{cam_id} | frame {index}",(10,y),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)


def create_scene_vid(cam_id,url,duration=10,fps=15):
    """Pan a 640x480 window across a real photo to fake a moving camera."""
    src=_fetch(url,data_path/"source"/f"{cam_id}.jpg")
    image=cv2.imread(str(src))
    if image is None:
        raise RuntimeError(f"could not read {src}")

    # scale so the photo is comfortably larger than the output window
    scale=max(w*1.4/image.shape[1],h*1.4/image.shape[0])
    image=cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)))
    max_x,max_y=image.shape[1]-w,image.shape[0]-h

    filename=f"{cam_id}.mp4"
    video_path,out=_writer(filename,fps)
    total_frames=duration*fps
    logger.info(f"Creating {filename} ({total_frames} frames) from a real scene...")
    for index in range(total_frames):
        # ping-pong the crop window so the scene drifts back and forth
        t=index/max(total_frames-1,1)
        phase=1-abs(2*t-1)
        x=int(max_x*phase) if max_x>0 else 0
        y=int(max_y*phase*0.5) if max_y>0 else 0
        frame=image[y:y+h,x:x+w].copy()
        _stamp(frame,cam_id,index)
        out.write(frame)
    out.release()
    logger.info(f"{filename} created ({video_path.stat().st_size/1e6:.2f} MB)")


def create_simple_vid(filename,duration=10,fps=15):
    """Synthetic fallback: a green box sliding across a grey background."""
    video_path,out=_writer(filename,fps)
    total_frames=duration*fps
    logger.info(f"Creating {filename} ({total_frames} frames)...")
    for index in range(total_frames):
        frame=np.ones((h,w,3),dtype=np.uint8)*200
        x=int(index/total_frames*w)
        cv2.rectangle(frame,(x,100),(x+50,150),(0,255,0),-1)
        _stamp(frame,filename[:-4],index)
        out.write(frame)
    out.release()
    logger.info(f"{filename} created ({video_path.stat().st_size/1e6:.2f} MB)")


def main():
    for cam_id,url in sources.items():
        try:
            create_scene_vid(cam_id,url)
        except Exception as e:
            logger.warning(f"scene video failed for {cam_id} ({e}) - falling back to synthetic")
            create_simple_vid(f"{cam_id}.mp4")
    logger.info(f"Test videos ready at {data_path/'videos'}")


if __name__=="__main__":
    main()
