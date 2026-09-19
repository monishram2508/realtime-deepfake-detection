"""Binary metrics, threshold fitting and frame->video aggregation.

Pure numpy, no torch, so these run in CI alongside the geometry tests.
"""
import numpy as np
import pytest

from deepfake.metrics import (
    aggregate_by_video, best_threshold, binary_metrics, by_group, evaluate, roc_auc,
)


def crop(video,label,method="Deepfakes",identity=None):
    return {"video":video,"label":label,"method":method,
            "identity":identity or video,"source":None,"split":"test"}


def test_perfect_separation_scores_one():
    assert roc_auc([0,0,1,1],[0.1,0.2,0.8,0.9])==1.0


def test_inverted_scores_give_zero():
    assert roc_auc([0,0,1,1],[0.9,0.8,0.2,0.1])==0.0


def test_interleaved_ordering_sits_at_a_half():
    # positives at 0.2/0.3 straddled by negatives at 0.1/0.4: the positives win
    # exactly half the pairwise comparisons
    assert roc_auc([0,1,1,0],[0.1,0.2,0.3,0.4])==0.5


def test_partial_ordering_scores_between():
    # positives at 0.2/0.4 beat 3 of the 4 pairings against negatives 0.1/0.3
    assert roc_auc([0,1,0,1],[0.1,0.2,0.3,0.4])==0.75


def test_all_scores_tied_is_exactly_a_half():
    # a collapsed model (every frame identical) must not look better than chance -
    # this is the quantization failure mode from the multicam study
    assert roc_auc([0,0,1,1],[0.5,0.5,0.5,0.5])==0.5


def test_auc_is_undefined_with_one_class_present():
    assert np.isnan(roc_auc([1,1,1],[0.2,0.5,0.9]))
    assert np.isnan(roc_auc([0,0,0],[0.2,0.5,0.9]))


def test_auc_ignores_class_balance():
    # 4:1 imbalance, same ranking - AUC must not move, which is why it is the
    # selection metric rather than accuracy
    balanced=roc_auc([0,1],[0.2,0.8])
    skewed=roc_auc([0,1,1,1,1],[0.2,0.8,0.8,0.8,0.8])
    assert balanced==skewed==1.0


def test_binary_metrics_counts_and_rates():
    stats=binary_metrics([1,1,0,0],[0.9,0.4,0.6,0.1],threshold=0.5)
    assert (stats["tp"],stats["fn"],stats["fp"],stats["tn"])==(1,1,1,1)
    assert stats["precision"]==0.5
    assert stats["recall"]==0.5
    assert stats["accuracy"]==0.5


def test_majority_class_guessing_is_caught_by_auc_not_accuracy():
    # 4 fake, 1 real - answering "fake" always
    labels=[1,1,1,1,0]
    scores=[0.99]*5
    assert binary_metrics(labels,scores,0.5)["accuracy"]==0.8   # looks fine
    assert roc_auc(labels,scores)==0.5                           # is not fine


def test_best_threshold_finds_a_separating_cut():
    labels=[0,0,1,1]
    scores=[0.1,0.2,0.8,0.9]
    threshold,stats=best_threshold(labels,scores)
    assert 0.2<threshold<0.8
    assert stats["f1"]==1.0


def test_best_threshold_beats_a_bad_default():
    # every score sits under 0.5, so the default threshold calls nothing fake
    labels=[0,0,1,1]
    scores=[0.01,0.02,0.30,0.35]
    assert binary_metrics(labels,scores,0.5)["f1"]==0.0
    _,stats=best_threshold(labels,scores)
    assert stats["f1"]==1.0


def test_video_aggregation_collapses_frames():
    rows=[crop("v1",1),crop("v1",1),crop("v2",0),crop("v2",0)]
    videos,labels,scores=aggregate_by_video(rows,[0.8,0.6,0.2,0.4])
    assert videos==["v1","v2"]
    assert labels.tolist()==[1,0]
    assert scores.tolist()==[pytest.approx(0.7),pytest.approx(0.3)]


def test_video_aggregation_rescues_a_noisy_frame():
    # one badly-cropped frame scores wrong; the clip verdict should survive it
    rows=[crop("v1",1)]*4
    frame_scores=[0.9,0.85,0.05,0.95]
    assert binary_metrics([1,1,1,1],frame_scores,0.5)["recall"]==0.75
    _,labels,scores=aggregate_by_video(rows,frame_scores)
    assert binary_metrics(labels,scores,0.5)["recall"]==1.0


def test_median_reduction_is_steadier_than_mean():
    rows=[crop("v1",1)]*5
    scores=[0.9,0.9,0.9,0.9,0.0]  # one catastrophic frame
    _,_,mean=aggregate_by_video(rows,scores,"mean")
    _,_,median=aggregate_by_video(rows,scores,"median")
    assert median[0]>mean[0]


def test_evaluate_reports_both_levels():
    rows=[crop("v1",1),crop("v1",1),crop("v2",0),crop("v2",0)]
    report=evaluate(rows,[0.9,0.8,0.1,0.2])
    assert report["frames"]["n"]==4
    assert report["videos"]["n"]==2
    assert report["frames"]["auc"]==1.0
    assert report["videos"]["auc"]==1.0


def test_by_group_separates_the_hard_method():
    rows=[crop("v1",1,"Deepfakes"),crop("v2",1,"NeuralTextures"),crop("v3",0,"original")]
    # NeuralTextures is missed, Deepfakes is caught
    stats=by_group(rows,[0.95,0.10,0.05],threshold=0.5)
    assert stats["Deepfakes"]["recall"]==1.0
    assert stats["NeuralTextures"]["recall"]==0.0
    assert stats["original"]["n"]==1
