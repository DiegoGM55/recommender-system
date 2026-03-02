import numpy as np
from typing import Dict, Iterable, List, Optional, Set

from scipy.sparse import csr_matrix


def build_interaction_matrix(
    train_df,
    user_map: Dict[int, int],
    item_map: Dict[str, int],
    weight: str = "log",  # one of: "count", "log"
    alpha: float = 40.0,
) -> csr_matrix:
    """
    Build a CSR user-item matrix from transactions with optional confidence weighting.

    - weight="count": raw counts
    - weight="log":   C = 1 + alpha * log(1 + count)
    - weight="capped_count": counts capped at 50

    Returns a CSR matrix with shape (num_users, num_items).
    """
    grouped = (
        train_df.groupby(["CustomerID", "StockCode"]).size().reset_index(name="count")
    )
    user_idx = grouped["CustomerID"].map(user_map).astype(int)
    item_idx = grouped["StockCode"].map(item_map).astype(int)
    counts = grouped["count"].astype(float).to_numpy()

    if weight == "log":
        data = 1.0 + alpha * np.log1p(counts)
    elif weight == "count":
        data = counts
    elif weight == "capped_count":
        # Teto de 50. 
        # Motivo: A mediana do dataset é 54. 
        # Um peso de 50 já é "gosto muito".
        data = np.clip(counts, 0, 50)
    else:
        raise ValueError("weight must be one of: 'count', 'log', 'capped_count'")

    num_users = len(user_map)
    num_items = len(item_map)

    return csr_matrix((data, (user_idx, item_idx)), shape=(num_users, num_items))


def apply_bm25_or_tfidf(
    user_item_csr: csr_matrix,
    method: Optional[str] = None,  # one of: None, "bm25", "tfidf"
    bm25_K1: float = 100.0,
    bm25_B: float = 0.8,
) -> csr_matrix:
    """
    Optionally apply BM25 or TF-IDF weighting recommended by the 'implicit' library.

    Note: 'implicit' expects an item-user matrix for fitting; we apply weighting on the
    item-user orientation (transpose), then transpose back for convenience.
    """
    if method is None:
        return user_item_csr

    try:
        from implicit.nearest_neighbours import bm25_weight, tfidf_weight
    except Exception as e:
        raise RuntimeError(
            "The 'implicit' package is required for BM25/TF-IDF weighting."
        ) from e

    item_user = user_item_csr.T.tocsr()
    if method == "bm25":
        item_user_w = bm25_weight(item_user, K1=bm25_K1, B=bm25_B)
    elif method == "tfidf":
        item_user_w = tfidf_weight(item_user)
    else:
        raise ValueError("method must be one of: None, 'bm25', 'tfidf'")

    return item_user_w.T.tocsr()


def train_als(
    user_item_csr: csr_matrix,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 20,
    use_gpu: bool = False,
    random_state: Optional[int] = 42,
):
    """
    Train implicit ALS. Important: implicit expects an item-user matrix for fitting.
    """
    import implicit

    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=use_gpu,
        random_state=random_state,
    )

    # Fit directly on the user x item matrix
    model.fit(user_item_csr)
    return model


def recommend_user(
    model,
    user_id: int,
    user_map: Dict[int, int],
    inverse_item_map: Dict[int, str],
    user_item_csr: csr_matrix,
    K: int = 10,
    exclude: Optional[Iterable[str]] = None,
    filter_history: bool = True,
) -> List[str]:
    """
    Recommend top-K item external IDs for the given user.

    Uses `model.recommend` and manually filters both the items the user already
    interacted with and any additional exclusions passed through `exclude`.
    """
    user_idx = user_map.get(user_id)
    if user_idx is None:
        return []

    if filter_history:
        # Pega o histórico do treino para filtrar
        filter_indices: Set[int] = set(user_item_csr[user_idx].indices.tolist())
    else:
        # Começa limpo (permite recomendar itens do treino/recompra)
        filter_indices: Set[int] = set()

    if exclude:
        item_map: Dict[str, int] = {ext: idx for idx, ext in inverse_item_map.items()}
        for ext_id in exclude:
            idx = item_map.get(ext_id)
            if idx is not None:
                filter_indices.add(idx)

    filter_items_idx: Optional[np.ndarray] = None
    if filter_indices:
        filter_items_idx = np.asarray(sorted(filter_indices), dtype=np.int64)

    rec_indices, _ = model.recommend(
        user_idx,
        None,
        N=K,
        filter_already_liked_items=False,
        filter_items=filter_items_idx,
        recalculate_user=False,
    )

    return [inverse_item_map[i] for i in rec_indices]
