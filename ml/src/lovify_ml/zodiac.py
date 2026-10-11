"""Zodiac taste classification -- and an honest accounting of what it can do.

The 2023 deck reported 76% accuracy on 12 classes with a decision tree, and
14% with a random forest. A random forest is an average of many decision trees
with extra randomisation; across 12 balanced classes it essentially never
loses to a single unpruned tree by 62 points. When you see that gap, the tree
is not smarter -- it is reading something the forest's feature subsampling
happens to dilute.

The culprit is duplicate tracks. The scrape took the top playlists for each
sign, every curator reached for the same hits, and so one track appears under
five signs with five different labels. Split those rows at random and the
model meets the SAME SONG in train and test, memorises its feature vector,
and 'predicts' the label it already saw. Splitting by position -- the deck's
'top 30 rows from each sign' -- is worse still: those 30 rows are one
contiguous playlist, so the test set is not even representative.

This module runs both protocols side by side so the gap is visible, not
asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import schema as S


# ── Splits ────────────────────────────────────────────────────────────────────

def naive_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0):
    """The ORIGINAL protocol: shuffle rows, slice. Duplicate tracks leak."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_size))
    return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()


def contiguous_split(df: pd.DataFrame, n_per_sign: int = 30):
    """The deck's stated protocol: 'top 30 rows from each sign' as validation.

    Reproduced so the notebook can show why it is not a valid holdout.
    """
    test = df.groupby(S.STAR_SIGN, group_keys=False).head(n_per_sign)
    train = df.drop(index=test.index)
    return train.copy(), test.copy()


def deduplicate(df: pd.DataFrame, strategy: str = "majority") -> pd.DataFrame:
    """One row per track.

    strategy='majority' -- keep each track's most common sign (ties broken
        deterministically). Retains data, but the label is genuinely noisy.
    strategy='unanimous' -- drop any track that curators labelled with more
        than one sign. Smaller and cleaner; what is left is the subset where
        curators actually agreed, which is the only subset a label can mean
        anything on.
    """
    counts = df.groupby([S.TRACK_ID, S.STAR_SIGN]).size().rename("n").reset_index()
    if strategy == "unanimous":
        n_signs = counts.groupby(S.TRACK_ID)[S.STAR_SIGN].nunique()
        keep = n_signs[n_signs == 1].index
        counts = counts[counts[S.TRACK_ID].isin(keep)]
    counts = counts.sort_values(["n", S.STAR_SIGN], ascending=[False, True])
    best = counts.groupby(S.TRACK_ID).first().reset_index()[[S.TRACK_ID, S.STAR_SIGN]]
    first_row = df.drop(columns=[S.STAR_SIGN]).groupby(S.TRACK_ID, as_index=False).first()
    return first_row.merge(best, on=S.TRACK_ID, how="inner")


def honest_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0):
    """Group-aware split: a track is entirely in train or entirely in test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df[S.TRACK_ID]))
    return df.iloc[tr].copy(), df.iloc[te].copy()


# ── Models ────────────────────────────────────────────────────────────────────

def model_zoo(seed: int = 0) -> dict:
    """The deck's four algorithms, plus the two baselines it was missing.

    DummyClassifier is the one that matters most: on 12 classes it scores
    ~8.3%, and ANY result must be read as a distance from that number.
    """
    return {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=seed),
        "Logistic regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, random_state=seed)),
        "Decision tree": DecisionTreeClassifier(random_state=seed),
        "Decision tree (depth 8)": DecisionTreeClassifier(max_depth=8, random_state=seed),
        "Random forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
        # CalibratedClassifierCV gives SVC a predict_proba without the
        # deprecated probability=True path -- taste_mix() needs probabilities.
        "SVM (rbf)": make_pipeline(StandardScaler(),
                                   CalibratedClassifierCV(SVC(random_state=seed), ensemble=False)),
        "KNN (k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    }


def baselines(y) -> dict[str, float]:
    """The two floors an accuracy figure has to be read against.

    `uniform` is 1/12 and assumes balanced classes. `majority` is the rate you
    get by always predicting the commonest sign, and it is the real floor the
    moment classes are uneven.

    This is not hypothetical here: the title filter removes far more Cancer
    and Leo playlists than any other sign, because those are the two sign
    names that are also ordinary English words. We create the imbalance
    ourselves. Measured on a filter-shaped imbalance, the floor rises from
    0.0833 to 0.0950 -- which turns an apparent 1.53x chance into 1.34x, an
    overstatement of 14% from using the wrong denominator.
    """
    vc = pd.Series(y).value_counts(normalize=True)
    return {"uniform": S.CHANCE_ACCURACY, "majority": float(vc.iloc[0])}


def benchmark(train: pd.DataFrame, test: pd.DataFrame,
              features: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    """Fit every model on train, score on test. Returns a tidy results frame.

    Reports balanced accuracy alongside raw accuracy. Balanced accuracy is the
    macro-average of per-class recall, so its chance level stays at 1/12 no
    matter how uneven the classes are -- which makes it the metric to quote
    once the real, filtered scrape lands.
    """
    from sklearn.metrics import balanced_accuracy_score

    features = features or S.AUDIO_FEATURES
    Xtr, ytr = train[features].to_numpy(float), train[S.STAR_SIGN].to_numpy()
    Xte, yte = test[features].to_numpy(float), test[S.STAR_SIGN].to_numpy()
    floors = baselines(ytr)

    rows = []
    for name, model in model_zoo(seed).items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = float((pred == yte).mean())
        rows.append({
            "model": name,
            "accuracy": acc,
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "vs_chance": acc / S.CHANCE_ACCURACY,
            # The honest multiplier when classes are uneven.
            "vs_majority": acc / floors["majority"],
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def grouped_cv_score(df: pd.DataFrame, model, features: list[str] | None = None,
                     n_splits: int = 5) -> np.ndarray:
    """Cross-validation that never lets a track straddle the fold boundary."""
    features = features or S.AUDIO_FEATURES
    return cross_val_score(
        model, df[features].to_numpy(float), df[S.STAR_SIGN].to_numpy(),
        groups=df[S.TRACK_ID].to_numpy(), cv=GroupKFold(n_splits=n_splits), n_jobs=-1,
    )


# ── Turning a classifier into the product feature ─────────────────────────────

def taste_mix(model, user_tracks: pd.DataFrame, features: list[str] | None = None) -> pd.Series:
    """The pie chart, done properly.

    The deck computed it by hard-classifying each song and counting labels.
    Averaging the model's PROBABILITIES instead keeps the information in a
    song the model thought was 51/49, rather than rounding it to a vote.
    """
    features = features or S.AUDIO_FEATURES
    P = model.predict_proba(user_tracks[features].to_numpy(float))
    return pd.Series(P.mean(axis=0), index=model.classes_).sort_values(ascending=False)


def base_rate(model, reference_tracks: pd.DataFrame,
              features: list[str] | None = None) -> pd.Series:
    """What the model says about an AVERAGE listener.

    This is the number the deck was missing. If the model calls everyone 11.9%
    Capricorn, then telling one user they are 11.9% Capricorn is telling them
    nothing about themselves.
    """
    return taste_mix(model, reference_tracks, features)


def lift(mix: pd.Series, base: pd.Series) -> pd.Series:
    """How much more X this listener is than the average listener.

    This is the shippable version of the feature: 'you are 2.3x more Libra
    than average' is a statement about the user. '9.5% Pisces' is mostly a
    statement about the class prior.
    """
    return (mix / base.reindex(mix.index)).sort_values(ascending=False)
