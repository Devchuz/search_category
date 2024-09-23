import pandas as pd
from qdrant_client.models import PointStruct, SparseVector
from typing import List


def make_points(df: pd.DataFrame) -> List[PointStruct]:
    sparse_vectors = df["sparse_embedding"].tolist()
    product_texts = df["combined_text"].tolist()
    dense_vectors = df["dense_embedding"].tolist()
    rows = df.to_dict(orient="records")
    points = []
    for idx, (text, sparse_vector, dense_vector) in enumerate(
        zip(product_texts, sparse_vectors, dense_vectors)
    ):
        sparse_vector = SparseVector(
            indices=sparse_vector.indices.tolist(), values=sparse_vector.values.tolist()
        )
        point = PointStruct(
            id=idx,
            payload={
                "text": text,
                "id": rows[idx]["id"],
            },  # Add any additional payload if necessary
            vector={
                "text-sparse": sparse_vector,
                "text-dense": dense_vector.tolist(),
            },
        )
        points.append(point)
    return points
