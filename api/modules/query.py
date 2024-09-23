from typing import List, Tuple
import numpy as np
from modules.data_transform import make_dense_embedding, make_sparse_embedding
from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest, NamedVector, NamedSparseVector, SparseVector, ScoredPoint
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

def search(client, collection_name: str, query_text: str):
    # Generar vectores esparcidos (sparse) y densos (dense)
    query_sparse_vectors: List[SparseEmbedding] = make_sparse_embedding([query_text])
    query_dense_vector: List[np.ndarray] = make_dense_embedding([query_text])

    # Verificar que los vectores no estén vacíos
    if not query_sparse_vectors or not query_dense_vector:
        raise ValueError("Los vectores generados están vacíos.")

    # Ejecutar la búsqueda con los vectores generados
    search_results = client.search_batch(
        collection_name=collection_name,
        requests=[
            SearchRequest(
                vector=NamedVector(
                    name="text-dense",
                    vector=query_dense_vector[0].tolist(),  # Convertir el vector en una lista
                ),
                limit=10,
                with_payload=True,
            ),
            SearchRequest(
                vector=NamedSparseVector(
                    name="text-sparse",
                    vector=SparseVector(
                        indices=query_sparse_vectors[0].indices.tolist(),  # Convertir índices en una lista
                        values=query_sparse_vectors[0].values.tolist(),    # Convertir valores en una lista
                    ),
                ),
                limit=10,
                with_payload=True,
            ),
        ],
    )

    return search_results


def rrf(rank_lists, alpha=60, default_rank=1000):
    """
    Optimized Reciprocal Rank Fusion (RRF) using NumPy for large rank lists.

    :param rank_lists: A list of rank lists. Each rank list should be a list of (item, rank) tuples.
    :param alpha: The parameter alpha used in the RRF formula. Default is 60.
    :param default_rank: The default rank assigned to items not present in a rank list. Default is 1000.
    :return: Sorted list of items based on their RRF scores.
    """
    # Consolidate all unique items from all rank lists
    all_items = set(item for rank_list in rank_lists for item, _ in rank_list)

    # Create a mapping of items to indices
    item_to_index = {item: idx for idx, item in enumerate(all_items)}

    # Initialize a matrix to hold the ranks, filled with the default rank
    rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)

    # Fill in the actual ranks from the rank lists
    for list_idx, rank_list in enumerate(rank_lists):
        for item, rank in rank_list:
            rank_matrix[item_to_index[item], list_idx] = rank

    # Calculate RRF scores using NumPy operations
    rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)

    # Sort items based on RRF scores
    sorted_indices = np.argsort(-rrf_scores)  # Negative for descending order

    # Retrieve sorted items
    sorted_items = [(list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices]

    return sorted_items


def rank_list(search_result: List[ScoredPoint]):
    return [(point.id, rank + 1) for rank, point in enumerate(search_result)]

def find_point_by_id(
    client: QdrantClient, collection_name: str, rrf_rank_list: List[Tuple[int, float]]
):
    return client.retrieve(
        collection_name=collection_name, ids=[item[0] for item in rrf_rank_list]
    )




def query_hybrid(client, collection_name: str, query_text: str):
    search_results = search(client, collection_name, query_text)
    dense_rank_list, sparse_rank_list = rank_list(search_results[0]), rank_list(search_results[1])
    rrf_rank_list = rrf([dense_rank_list, sparse_rank_list])
    result = find_point_by_id(client, collection_name, rrf_rank_list)
    return result