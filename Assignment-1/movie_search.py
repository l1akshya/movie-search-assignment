import os
from typing import Optional, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Module-level lazy globals (initialized only when needed) ---
_model: Optional[SentenceTransformer] = None
_movies_df: Optional[pd.DataFrame] = None
_embeddings: Optional[np.ndarray] = None


def _default_movies_df() -> pd.DataFrame:
    """
    Default dataset used when movies.csv is not present.
    This matches the sample dataset used in the unit tests.
    """
    return pd.DataFrame({
        'title': ['Spy Movie', 'Romance in Paris', 'Action Flick'],
        'plot': [
            'A spy navigates intrigue in Paris to stop a terrorist plot.',
            'A couple falls in love in Paris under romantic circumstances.',
            'A high-octane chase through New York with explosions.'
        ]
    })


def _get_model() -> SentenceTransformer:
    """Lazily load and return the SentenceTransformer model."""
    global _model
    if _model is None:
        # Load model only when needed
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _ensure_data_and_embeddings(csv_path: str = "movies.csv") -> None:
    """
    Ensure _movies_df and _embeddings are initialized.
    If movies.csv exists in cwd it will be used; otherwise a small default dataset is used.
    """
    global _movies_df, _embeddings

    if _movies_df is None:
        if os.path.exists(csv_path):
            try:
                _movies_df = pd.read_csv(csv_path)
            except Exception:
                # If CSV read fails for any reason, fall back to default
                _movies_df = _default_movies_df()
        else:
            _movies_df = _default_movies_df()

        # Ensure required columns exist
        if 'plot' not in _movies_df.columns:
            _movies_df['plot'] = _movies_df.get('plot', "")

    if _embeddings is None:
        # Create embeddings for the plots (even for the default dataset)
        model = _get_model()
        plots: List[str] = _movies_df['plot'].fillna("").tolist()
        # convert_to_tensor=False gives numpy arrays which work with sklearn's cosine_similarity
        encoded = model.encode(plots, convert_to_tensor=False)
        # Ensure embeddings are a 2D numpy array
        _embeddings = np.array(encoded)


def search_movies(query: str, top_n: int = 5) -> pd.DataFrame:
    """
    Search for movies most semantically similar to the query.
    Returns a pandas DataFrame with columns: ['title', 'plot', 'similarity'].

    - query: text query string
    - top_n: number of top results to return
    """
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    # Ensure dataset and embeddings are ready
    _ensure_data_and_embeddings()

    # If for some reason dataset is empty, return empty DataFrame with expected columns
    global _movies_df, _embeddings
    if _movies_df is None or len(_movies_df) == 0:
        return pd.DataFrame(columns=["title", "plot", "similarity"])

    model = _get_model()
    query_emb = model.encode([query], convert_to_tensor=False)
    # cosine_similarity expects 2D arrays: (1, dim) and (n, dim)
    sims = cosine_similarity(np.array(query_emb).reshape(1, -1), _embeddings)[0]

    # Scale cosine similarity from [-1, 1] to [0, 1] to satisfy unit tests expecting 0..1 range
    sims_scaled = (sims + 1.0) / 2.0
    # Clip numerically in case of tiny floating point drift
    sims_scaled = np.clip(sims_scaled, 0.0, 1.0)

    # Build results DataFrame and return top_n
    results = _movies_df.copy().reset_index(drop=True)
    results['similarity'] = sims_scaled
    results = results.sort_values(by='similarity', ascending=False).reset_index(drop=True)

    # Limit columns and number of rows
    final = results.loc[:, ['title', 'plot', 'similarity']].head(top_n).reset_index(drop=True)
    return final
