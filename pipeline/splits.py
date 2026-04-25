import pandas as pd
from sklearn.model_selection import train_test_split


def filter_loewe(df, drug1_in=None, drug1_not_in=None, drug2_in=None, drug2_not_in=None, cell_in=None, cell_not_in=None):
    mask = pd.Series(True, index=df.index)

    if drug1_in is not None:
        mask &= df["Drug1"].isin(drug1_in)
    if drug1_not_in is not None:
        mask &= ~df["Drug1"].isin(drug1_not_in)
    if drug2_in is not None:
        mask &= df["Drug2"].isin(drug2_in)
    if drug2_not_in is not None:
        mask &= ~df["Drug2"].isin(drug2_not_in)
    if cell_in is not None:
        mask &= df["Cell line"].isin(cell_in)
    if cell_not_in is not None:
        mask &= ~df["Cell line"].isin(cell_not_in)

    return df.loc[mask].reset_index(drop=True)


def concat_frames(*frames):
    return pd.concat(list(frames), ignore_index=True)


def split_named_frames(frame_map, test_size=0.5):
    val_splits = {}
    test_splits = {}

    for name, frame in frame_map.items():
        val_frame, test_frame = train_test_split(frame, test_size=test_size)
        val_splits[f"{name}_val"] = val_frame.reset_index(drop=True)
        test_splits[f"{name}_test"] = test_frame.reset_index(drop=True)

    return val_splits, test_splits


def build_split_frames(train_dc, val_splits, test_splits):
    split_frames = {
        "train_dc": train_dc.reset_index(drop=True),
        "val_dc": concat_frames(*val_splits.values()),
        "test_dc": concat_frames(*test_splits.values()),
    }
    split_frames.update(val_splits)
    split_frames.update(test_splits)
    return split_frames


def apply_to_split_frames(split_frames, func):
    return {name: func(frame.copy()) for name, frame in split_frames.items()}
