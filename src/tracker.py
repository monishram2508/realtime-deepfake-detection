"""Minimal IoU tracker: turns per-frame detections into objects with identity.

Detection alone answers "what is in this frame". Tracking answers "is that the
same person as last frame", which is what every downstream question (counting,
dwell time, cross-camera hand-off) actually needs.

The algorithm is deliberately the simplest thing that works on a static camera:
greedily match new detections to existing tracks by IoU, age out tracks that go
unmatched. No motion model, no appearance embedding - both are listed in the
roadmap because both are what you would add for crowded or moving scenes.
"""
import itertools

from utils import iou_matrix


class track:
    def __init__(self,track_id,detection,frame_index):
        self.track_id=track_id
        self.detection=detection
        self.class_id=detection['class_id']
        self.class_name=detection['class_name']
        self.first_seen=frame_index
        self.last_seen=frame_index
        self.hits=1
        self.age=0 # frames since last match

    def update(self,detection,frame_index):
        self.detection=detection
        self.last_seen=frame_index
        self.hits+=1
        self.age=0

    def box(self):
        d=self.detection
        return [d['x'],d['y'],d['w'],d['h']]


class ioutracker:
    """One instance per camera.

    Args:
        iou_thresh: overlap needed to consider a detection the same object.
        max_age: frames a track survives unmatched before it is dropped
                 (covers brief occlusion and single-frame detector misses).
        min_hits: matches needed before a track is reported, which suppresses
                 one-frame false positives.
    """

    def __init__(self,iou_thresh=0.3,max_age=5,min_hits=2):
        self.iou_thresh=iou_thresh
        self.max_age=max_age
        self.min_hits=min_hits
        self.tracks=[]
        self._ids=itertools.count(1)
        self.frame_index=0
        self.total_tracks=0

    def update(self,detections):
        """Feed one frame of detections, get back detections annotated with track_id."""
        self.frame_index+=1

        matched_tracks={}
        if self.tracks and detections:
            ious=iou_matrix([t.box() for t in self.tracks],
                            [[d['x'],d['y'],d['w'],d['h']] for d in detections])
            for i,t in enumerate(self.tracks):
                for j,d in enumerate(detections):
                    if t.class_id!=d['class_id']:
                        ious[i,j]=0.0

            # greedy: highest-overlap pair first, then remove that row and column
            used_tracks,used_dets=set(),set()
            order=sorted(((ious[i,j],i,j) for i in range(len(self.tracks))
                          for j in range(len(detections))),reverse=True)
            for score,i,j in order:
                if score<self.iou_thresh:
                    break
                if i in used_tracks or j in used_dets:
                    continue
                used_tracks.add(i)
                used_dets.add(j)
                self.tracks[i].update(detections[j],self.frame_index)
                matched_tracks[j]=self.tracks[i]

        for index,detection in enumerate(detections):
            if index not in matched_tracks: # unmatched detection -> new track
                new=track(next(self._ids),detection,self.frame_index)
                self.tracks.append(new)
                matched_tracks[index]=new
                self.total_tracks+=1

        for t in self.tracks:
            if t.last_seen!=self.frame_index:
                t.age+=1
        self.tracks=[t for t in self.tracks if t.age<=self.max_age]

        out=[]
        for index,detection in enumerate(detections):
            t=matched_tracks[index]
            enriched=dict(detection)
            enriched['track_id']=t.track_id
            enriched['track_hits']=t.hits
            enriched['confirmed']=t.hits>=self.min_hits
            out.append(enriched)
        return out

    def stats(self):
        confirmed=[t for t in self.tracks if t.hits>=self.min_hits]
        return {"active_tracks":len(self.tracks),
                "confirmed_tracks":len(confirmed),
                "total_tracks_created":self.total_tracks,
                "mean_track_length":round(
                    sum(t.hits for t in self.tracks)/len(self.tracks),1) if self.tracks else 0}
