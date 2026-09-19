"""The FF++ split contract: no identity may cross a split boundary.

Everything the deepfake classifier reports depends on this being true, so it is
tested rather than trusted. These tests need only the committed split files, not
the dataset, so they run in CI.
"""
import pytest

from deepfake.ffpp import (
    build_manifest, check_leakage, entries_for_split, load_split,
    methods, split_identities, summarize,
)


@pytest.fixture(scope="module")
def manifest():
    return build_manifest()


def test_official_splits_have_the_documented_sizes():
    # FF++ ships 500 (target, source) pairs over 1000 original videos
    assert len(load_split("train"))==360
    assert len(load_split("val"))==70
    assert len(load_split("test"))==70


def test_splits_partition_all_thousand_identities():
    train,val,test=(split_identities(load_split(n)) for n in ("train","val","test"))
    assert (len(train),len(val),len(test))==(720,140,140)
    assert len(train|val|test)==1000


def test_no_identity_appears_in_two_splits(manifest):
    seen,problems=check_leakage(manifest)
    assert problems==[],f"identity leak: {problems}"
    assert {k:len(v) for k,v in seen.items()}=={"train":720,"val":140,"test":140}


def test_leak_guard_catches_a_planted_leak(manifest):
    # the guard is only worth having if it fires, so plant one and check
    poisoned={"splits":{k:list(v) for k,v in manifest["splits"].items()}}
    stolen=poisoned["splits"]["train"][0]
    poisoned["splits"]["test"]=poisoned["splits"]["test"]+[stolen]
    _,problems=check_leakage(poisoned)
    assert problems,"check_leakage missed an identity present in both train and test"


def test_both_manipulation_directions_are_present():
    # each pair is manipulated both ways, which is how 500 pairs become 1000
    # videos per method - miss this and half the fake data silently disappears
    rows=entries_for_split("test",use_methods=["Deepfakes"])
    names={r["path"].rsplit("/",1)[-1] for r in rows if r["label"]==1}
    target,source=load_split("test")[0]
    assert f"{target}_{source}.mp4" in names
    assert f"{source}_{target}.mp4" in names


def test_fake_rows_never_borrow_an_identity_from_another_split(manifest):
    for name,rows in manifest["splits"].items():
        allowed=split_identities(load_split(name))
        for row in rows:
            assert row["identity"] in allowed
            if row["source"]:
                assert row["source"] in allowed,(
                    f"{row['path']} pulls source {row['source']} from outside {name}")


def test_class_balance_is_four_to_one(manifest):
    # 1 original vs 4 manipulation methods - the trainer has to correct for this
    for row in summarize(manifest):
        assert row["ratio"]==len(methods)


def test_manifest_is_built_without_the_dataset_on_disk(manifest):
    # paths are derived from the splits, not globbed, so the manifest exists
    # before the download finishes and audit() is what checks reality
    paths=[r["path"] for r in manifest["splits"]["train"]]
    assert len(paths)==720+720*len(methods)
    assert all(p.endswith(".mp4") for p in paths)


def test_compression_flows_into_every_path():
    for compression in ("c23","c40"):
        rows=entries_for_split("val",compression=compression,use_methods=["FaceSwap"])
        assert all(f"/{compression}/" in r["path"] for r in rows)
