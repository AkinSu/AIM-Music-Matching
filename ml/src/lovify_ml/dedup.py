"""Playlist independence: how many opinions a scrape actually contains.

THE PROBLEM THIS MODULE EXISTS FOR.

Suppose "Supermassive Black Hole" shows up in nine Aries playlists and one
Leo playlist. The obvious reading is 9-1 in favour of Aries, and the 2023
pipeline read it that way: every (track, playlist) pair became a labelled row,
so popular tracks got nine rows of evidence.

But playlists copy each other. If eight of those nine Aries playlists were
cloned from the first -- same 50 tracks, new title, new owner -- then there
are not nine Aries opinions. There is ONE Aries opinion, duplicated, and one
Leo opinion. The honest score is 1-1: a coin flip, not 90% confidence.

Statistically this is the difference between n=10 and n=2. Sample size is the
count of INDEPENDENT observations, and nine copies of one observation are
still one observation. Everything here is machinery for estimating that
number instead of assuming it.

HOW WE DETECT COPIES. A playlist is its set of track ids. Jaccard similarity
(|A n B| / |A u B|) over those sets is high exactly when two playlists hold
the same music, which is what copying looks like from the outside -- and it
survives renaming, reordering and a copier adding a track or two of their own.
Playlists above a Jaccard threshold are joined into clusters, and one cluster
counts as one vote.

WHY SINGLE LINKAGE (connected components) RATHER THAN SOMETHING TIGHTER.
Components are transitive: if J(A,B)=0.6 and J(B,C)=0.6 then A, B and C share
a cluster even when J(A,C) is small. That over-merges, which means it
UNDER-counts independent evidence. We choose that direction deliberately: the
headline claim of this scrape is "there is less evidence here than it looks
like", and that claim should be made with the estimator that is conservative
about evidence, not the one that flatters it. `cluster_playlists` takes a
`linkage` argument so the stricter alternative can be reported alongside.

THRESHOLD. 0.5 is the default: half the tracks shared. Real unrelated
playlists for the same sign overlap because everyone reaches for the same
hits, and that overlap sits well below 0.5 in practice; deliberate copies sit
above 0.85. The gap is wide, but it is an empirical claim about a particular
scrape -- check it with `threshold_sweep` before trusting any one number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S


# ── track sets ────────────────────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    """|A n B| / |A u B|; 0.0 for two empty sets rather than a ZeroDivision."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def playlist_track_sets(df: pd.DataFrame) -> dict[str, set]:
    """playlist_id -> set of its track ids. Duplicates within a playlist
    collapse, which is correct: a playlist holding a track twice is not twice
    as much evidence."""
    grouped = df.groupby(S.PLAYLIST_ID, sort=True)[S.TRACK_ID]
    return {str(pid): set(vals.dropna().astype(str)) for pid, vals in grouped}


def _incidence(df: pd.DataFrame):
    """Sparse playlists x tracks boolean matrix, plus the playlist index.

    Sparse because the dense version is unaffordable at scrape scale: 2,400
    playlists x ~80,000 distinct tracks is ~190M cells, where the sparse form
    holds only the ~120,000 actual memberships.
    """
    from scipy import sparse

    sub = df[[S.PLAYLIST_ID, S.TRACK_ID]].dropna().astype(str).drop_duplicates()
    playlists = pd.Index(sorted(sub[S.PLAYLIST_ID].unique()), name=S.PLAYLIST_ID)
    tracks = pd.Index(sorted(sub[S.TRACK_ID].unique()))
    rows = playlists.get_indexer(sub[S.PLAYLIST_ID])
    cols = tracks.get_indexer(sub[S.TRACK_ID])
    M = sparse.csr_matrix(
        (np.ones(len(sub), dtype=np.int32), (rows, cols)),
        shape=(len(playlists), len(tracks)),
    )
    return M, playlists


def playlist_jaccard_matrix(df: pd.DataFrame, min_tracks: int = 1) -> pd.DataFrame:
    """Pairwise Jaccard over playlists' track sets, as a square DataFrame.

    `min_tracks` drops tiny playlists first. A 2-track playlist is Jaccard-1.0
    with any other 2-track playlist holding the same pair, so it manufactures
    spurious clusters while contributing almost no data.
    """
    if min_tracks > 1:
        sizes = df.groupby(S.PLAYLIST_ID)[S.TRACK_ID].nunique()
        keep = sizes[sizes >= min_tracks].index
        df = df[df[S.PLAYLIST_ID].isin(keep)]
    if df.empty:
        return pd.DataFrame(dtype=float)

    M, playlists = _incidence(df)
    inter = np.asarray((M @ M.T).todense(), dtype=float)
    sizes = np.asarray(M.sum(axis=1)).ravel().astype(float)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    # Build the array fully before wrapping it: in pandas 3.0 a DataFrame's
    # .to_numpy() is a read-only copy, so np.fill_diagonal on it raises.
    np.fill_diagonal(J, 1.0)
    return pd.DataFrame(J, index=playlists, columns=playlists)


# ── clustering ────────────────────────────────────────────────────────────────

def cluster_playlists(df: pd.DataFrame, threshold: float = 0.5,
                      min_tracks: int = 1, linkage: str = "single") -> pd.Series:
    """Group near-duplicate playlists. Returns playlist_id -> cluster id.

    linkage='single'   connected components of the J >= threshold graph.
                       Transitive, so it over-merges and under-counts
                       evidence -- the conservative direction (see module
                       docstring).
    linkage='complete' agglomerative clustering on the precomputed distance
                       1 - J, cut at 1 - threshold. Every pair inside a
                       cluster is within the threshold, so nothing is merged
                       through a chain. Use it as the optimistic bound.

    Cluster ids are stable: 0, 1, 2 ... ordered by sorted playlist id, so two
    runs on the same frame label things the same way.
    """
    J = playlist_jaccard_matrix(df, min_tracks=min_tracks)
    if J.empty:
        return pd.Series(dtype=int, name=S.PLAYLIST_CLUSTER)

    if linkage == "single":
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        A = sparse.csr_matrix(J.to_numpy() >= threshold)
        _, labels = connected_components(A, directed=False)
    elif linkage == "complete":
        from sklearn.cluster import AgglomerativeClustering

        D = 1.0 - J.to_numpy()
        np.fill_diagonal(D, 0.0)
        D = np.clip((D + D.T) / 2.0, 0.0, 1.0)  # enforce exact symmetry
        if len(D) == 1:
            labels = np.zeros(1, dtype=int)
        else:
            labels = AgglomerativeClustering(
                n_clusters=None, metric="precomputed", linkage="complete",
                distance_threshold=1.0 - threshold,
            ).fit_predict(D)
    else:
        raise ValueError(f"unknown linkage {linkage!r}; use 'single' or 'complete'")

    out = pd.Series(labels, index=J.index, name=S.PLAYLIST_CLUSTER)
    # Relabel so ids run 0..k-1 in order of first appearance.
    order = {old: new for new, old in enumerate(dict.fromkeys(out.to_list()))}
    return out.map(order)


def cluster_sizes(clusters: pd.Series) -> pd.Series:
    """Playlists per cluster, largest first. A cluster of 14 is 13 copies."""
    return clusters.value_counts().sort_values(ascending=False)


def attach_clusters(df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Add the playlist_cluster column. Rows whose playlist was filtered out
    of the matrix (too few tracks) get their own singleton ids, so they are
    never silently merged with anything."""
    out = df.copy()
    mapped = out[S.PLAYLIST_ID].astype(str).map(clusters)
    if mapped.isna().any():
        start = int(clusters.max()) + 1 if len(clusters) else 0
        unknown = sorted(out.loc[mapped.isna(), S.PLAYLIST_ID].astype(str).unique())
        extra = {pid: start + i for i, pid in enumerate(unknown)}
        mapped = mapped.fillna(out[S.PLAYLIST_ID].astype(str).map(extra))
    out[S.PLAYLIST_CLUSTER] = mapped.astype(int)
    return out


# ── label confidence ──────────────────────────────────────────────────────────

def label_confidence(df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Per (track, sign): how much of the independent evidence backs that sign.

        confidence = clusters of THIS sign containing the track
                     ------------------------------------------
                     clusters of ANY sign containing the track

    One cluster is one vote regardless of how many copies it holds, so nine
    cloned Aries playlists and one Leo playlist give Aries 0.5, not 0.9.

    Confidence 1.0 means every independent curator who placed this track
    agreed. That subset -- not the whole scrape -- is the only place a zodiac
    label can carry information, and it is what a classifier should be trained
    and evaluated on. The playlist-count columns are kept alongside so the
    inflation is visible: n_playlists_sign=9 next to n_clusters_sign=1 is the
    finding, stated per row.
    """
    tagged = attach_clusters(df, clusters)[
        [S.TRACK_ID, S.STAR_SIGN, S.PLAYLIST_ID, S.PLAYLIST_CLUSTER]
    ].astype({S.TRACK_ID: str}).drop_duplicates()

    per_sign = (tagged.groupby([S.TRACK_ID, S.STAR_SIGN])
                .agg(**{S.N_CLUSTERS_SIGN: (S.PLAYLIST_CLUSTER, "nunique"),
                        "n_playlists_sign": (S.PLAYLIST_ID, "nunique")})
                .reset_index())
    per_track = (tagged.groupby(S.TRACK_ID)
                 .agg(**{S.N_CLUSTERS_TOTAL: (S.PLAYLIST_CLUSTER, "nunique"),
                         "n_playlists_total": (S.PLAYLIST_ID, "nunique"),
                         "n_signs": (S.STAR_SIGN, "nunique")})
                 .reset_index())

    out = per_sign.merge(per_track, on=S.TRACK_ID, how="left")
    out[S.LABEL_CONFIDENCE] = out[S.N_CLUSTERS_SIGN] / out[S.N_CLUSTERS_TOTAL]
    # The naive version, for the side-by-side: this is what counting playlists
    # instead of clusters would have told you.
    out["naive_confidence"] = out["n_playlists_sign"] / out["n_playlists_total"]
    return out.sort_values([S.LABEL_CONFIDENCE, S.N_CLUSTERS_SIGN],
                           ascending=[False, False]).reset_index(drop=True)


def attach_confidence(df: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    """Merge label_confidence back onto the scrape rows, for the output CSV."""
    cols = [S.TRACK_ID, S.STAR_SIGN, S.N_CLUSTERS_SIGN, S.N_CLUSTERS_TOTAL,
            S.LABEL_CONFIDENCE]
    out = df.copy()
    out[S.TRACK_ID] = out[S.TRACK_ID].astype(str)
    return out.merge(conf[cols].astype({S.TRACK_ID: str}),
                     on=[S.TRACK_ID, S.STAR_SIGN], how="left")


def confident_labels(conf: pd.DataFrame, min_confidence: float = 1.0,
                     min_clusters: int = 2) -> pd.DataFrame:
    """The trainable subset: tracks whose label independent curators agree on.

    `min_clusters=2` is the part that is easy to get wrong. A track in exactly
    one cluster has confidence 1.0 by construction -- unanimous because only
    one curator was asked. Requiring two independent clusters is what makes
    agreement mean agreement.
    """
    keep = conf[(conf[S.LABEL_CONFIDENCE] >= min_confidence)
                & (conf[S.N_CLUSTERS_TOTAL] >= min_clusters)]
    return keep.reset_index(drop=True)


# ── effective sample size ─────────────────────────────────────────────────────

def effective_sample_size(df: pd.DataFrame, clusters: pd.Series) -> dict:
    """How much the apparent evidence shrinks once copies collapse.

    This is the headline number of the whole scrape, so be precise about what
    is counted:

      apparent_votes   distinct (track, playlist) pairs -- one per row the old
                       pipeline would have trained on.
      effective_votes  distinct (track, cluster) pairs -- copies collapsed.
      ess_ratio        effective / apparent. 1.0 would mean every playlist is
                       an independent opinion; 0.4 means 60% of the evidence
                       was the same opinion counted again.
      inflation        apparent / effective, i.e. "the old count overstated
                       the evidence by this factor".

    Also reported: `rows`, which exceeds apparent_votes whenever the same
    track sits in the same playlist twice, and `independent_label_rate`, the
    fraction of (track, sign) pairs backed by at least two clusters -- the
    share of the data on which "curators agree" is even a meaningful sentence.

    NOT reported, deliberately: the mean of naive_confidence next to the mean
    of label_confidence. Both normalise per track -- each track's confidences
    sum to 1 across signs either way -- so their means are identical by
    construction and comparing them says nothing. The per-row shift is the
    real comparison: `confidence_shift` is the mean |cluster - naive| over
    (track, sign) pairs, and `share_overstated` is the fraction of pairs whose
    confidence the playlist count had inflated.
    """
    tagged = attach_clusters(df, clusters)
    tagged[S.TRACK_ID] = tagged[S.TRACK_ID].astype(str)

    apparent = len(tagged[[S.TRACK_ID, S.PLAYLIST_ID]].drop_duplicates())
    effective = len(tagged[[S.TRACK_ID, S.PLAYLIST_CLUSTER]].drop_duplicates())
    n_playlists = tagged[S.PLAYLIST_ID].nunique()
    n_clusters = tagged[S.PLAYLIST_CLUSTER].nunique()

    conf = label_confidence(df, clusters)
    multi = int((conf[S.N_CLUSTERS_TOTAL] >= 2).sum())
    shift = conf[S.LABEL_CONFIDENCE] - conf["naive_confidence"]

    return {
        "rows": int(len(tagged)),
        "playlists": int(n_playlists),
        "clusters": int(n_clusters),
        "playlist_collapse": float(n_clusters / n_playlists) if n_playlists else 0.0,
        "unique_tracks": int(tagged[S.TRACK_ID].nunique()),
        "apparent_votes": int(apparent),
        "effective_votes": int(effective),
        "ess_ratio": float(effective / apparent) if apparent else 0.0,
        "inflation": float(apparent / effective) if effective else 0.0,
        "track_sign_pairs": int(len(conf)),
        "independent_label_rate": float(multi / len(conf)) if len(conf) else 0.0,
        "mean_label_confidence": float(conf[S.LABEL_CONFIDENCE].mean()) if len(conf) else 0.0,
        "confidence_shift": float(shift.abs().mean()) if len(conf) else 0.0,
        "share_overstated": float((shift < 0).mean()) if len(conf) else 0.0,
        "share_changed": float((shift != 0).mean()) if len(conf) else 0.0,
    }


def per_sign_ess(df: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """effective_sample_size broken out by sign.

    Worth looking at: a sign whose playlists are mostly clones of one editorial
    playlist has far less evidence behind it than the row count suggests, and
    averaging over signs hides that.
    """
    rows = []
    for sign, sub in df.groupby(S.STAR_SIGN, sort=True):
        ess = effective_sample_size(sub, clusters)
        rows.append({S.STAR_SIGN: sign, **{k: ess[k] for k in
                     ("playlists", "clusters", "unique_tracks",
                      "apparent_votes", "effective_votes", "ess_ratio")}})
    return pd.DataFrame(rows)


def threshold_sweep(df: pd.DataFrame,
                    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    linkage: str = "single") -> pd.DataFrame:
    """Cluster count and ESS across thresholds -- a sensitivity check.

    If the headline shrink factor swings wildly between 0.4 and 0.6, the
    number is an artefact of the threshold and must be reported as a range.
    If it is flat, the clusters are real structure.
    """
    rows = []
    for t in thresholds:
        clusters = cluster_playlists(df, threshold=t, linkage=linkage)
        ess = effective_sample_size(df, clusters)
        rows.append({"threshold": t, "linkage": linkage,
                     "clusters": ess["clusters"],
                     "largest_cluster": int(cluster_sizes(clusters).iloc[0])
                     if len(clusters) else 0,
                     "effective_votes": ess["effective_votes"],
                     "ess_ratio": ess["ess_ratio"]})
    return pd.DataFrame(rows)


def confidence_distribution(conf: pd.DataFrame,
                            bins: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> pd.DataFrame:
    """Histogram of label confidence, as counts and shares.

    Read the top bucket first: that is the fraction of the dataset on which a
    zodiac label is not simply contested.
    """
    if conf.empty:
        return pd.DataFrame(columns=["bucket", "n", "share"])
    cut = pd.cut(conf[S.LABEL_CONFIDENCE], bins=list(bins),
                 include_lowest=True, right=True)
    counts = cut.value_counts().sort_index()
    return pd.DataFrame({
        # include_lowest makes the first interval open at -0.001; print 0.0.
        "bucket": [f"{max(iv.left, 0.0):.1f}-{iv.right:.1f}" for iv in counts.index],
        "n": counts.to_numpy(int),
        "share": counts.to_numpy(float) / len(conf),
    })
