"""Tracker behaviour: identity has to survive motion, and not leak across objects."""
import numpy as np
import pytest

from tracker import ioutracker
from utils import iou_matrix


def det(x,y,w=50,h=50,class_id=0,conf=0.9):
    return {'x':float(x),'y':float(y),'w':float(w),'h':float(h),
            'conf':conf,'class_id':class_id,'class_name':'person'}


def test_iou_matrix_values():
    ious=iou_matrix([[0,0,10,10]],[[0,0,10,10],[5,5,10,10],[100,100,10,10]])
    assert ious[0,0]==pytest.approx(1.0)
    assert ious[0,1]==pytest.approx(25/175,abs=1e-4) # 5x5 overlap, 175 union
    assert ious[0,2]==0.0


def test_iou_matrix_handles_empty():
    assert iou_matrix([],[[0,0,1,1]]).shape==(0,1)
    assert iou_matrix([[0,0,1,1]],[]).shape==(1,0)


def test_id_is_stable_across_small_movements():
    tracker=ioutracker(iou_thresh=0.3)
    ids=[]
    for step in range(6):
        out=tracker.update([det(100+step*5,100)]) # drifts 5 px per frame
        ids.append(out[0]['track_id'])
    assert len(set(ids))==1 # one object, one id


def test_a_jump_creates_a_new_id():
    tracker=ioutracker(iou_thresh=0.3)
    first=tracker.update([det(100,100)])[0]['track_id']
    second=tracker.update([det(400,400)])[0]['track_id'] # no overlap at all
    assert first!=second


def test_two_objects_keep_separate_ids():
    tracker=ioutracker(iou_thresh=0.3)
    out=tracker.update([det(100,100),det(300,300)])
    ids={d['track_id'] for d in out}
    assert len(ids)==2
    out2=tracker.update([det(105,100),det(305,300)])
    assert {d['track_id'] for d in out2}==ids # same two objects, same two ids


def test_class_change_is_not_the_same_object():
    tracker=ioutracker(iou_thresh=0.3)
    first=tracker.update([det(100,100,class_id=0)])[0]['track_id']
    second=tracker.update([det(100,100,class_id=2)])[0]['track_id'] # same box, car not person
    assert first!=second


def test_track_survives_a_missed_frame_then_ages_out():
    tracker=ioutracker(iou_thresh=0.3,max_age=2)
    first=tracker.update([det(100,100)])[0]['track_id']
    tracker.update([])          # detector missed it once
    again=tracker.update([det(100,100)])[0]['track_id']
    assert again==first         # recovered

    for _ in range(4):          # gone for longer than max_age
        tracker.update([])
    assert tracker.stats()["active_tracks"]==0
    new=tracker.update([det(100,100)])[0]['track_id']
    assert new!=first


def test_confirmation_requires_min_hits():
    tracker=ioutracker(min_hits=2)
    assert tracker.update([det(100,100)])[0]['confirmed'] is False
    assert tracker.update([det(101,100)])[0]['confirmed'] is True


def test_stats_report_track_counts():
    tracker=ioutracker()
    tracker.update([det(100,100),det(300,300)])
    tracker.update([det(102,100),det(302,300)])
    stats=tracker.stats()
    assert stats["active_tracks"]==2
    assert stats["confirmed_tracks"]==2
    assert stats["total_tracks_created"]==2
