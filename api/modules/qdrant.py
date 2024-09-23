from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
def init_collection(client: QdrantClient, collection_name: str, vector_size: int = 1024, on_disk: bool = False):
    """
    Inicializa una colección en Qdrant si no existe.

    Parámetros:
    - client (QdrantClient): Cliente de Qdrant.
    - collection_name (str): Nombre de la colección.
    - vector_size (int): Tamaño del vector denso (por defecto 1024).
    - on_disk (bool): Si los vectores esparcidos deben almacenarse en disco (por defecto False).
    """
    try:
        # Intentar obtener la colección
        client.get_collection(collection_name)
        print(f"La colección '{collection_name}' ya existe.")
    except ValueError as e:
        if "not found" in str(e):
            # Crear la colección si no existe
            client.create_collection(
                collection_name,
                vectors_config={
                    "text-dense": VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "text-sparse": SparseVectorParams(
                        index=SparseIndexParams(
                            on_disk=on_disk,
                        )
                    )
                },
            )
            print(f"Se ha creado la colección '{collection_name}'.")
        else:
            raise  # Si es otro error, relanzarlo
