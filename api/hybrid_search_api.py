from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from modules.qdrant import init_collection
from modules.data_load import load_data
from modules.data_transform import data_transform
from modules.upload_data import make_points
from modules.query import query_hybrid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastapi.middleware.cors import CORSMiddleware
import os

# Inicializa la aplicación FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Permitir solo este origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todas las cabeceras
)
# Inicializa el cliente de Qdrant en memoria
client = QdrantClient(":memory:")
collection_name = "hybrid_search"
init_collection(client, collection_name)
# Inicializa la colección y carga los datos una sola vez
path = os.path.join('', 'data')
df_category = load_data(path)
sampled_df = data_transform(df_category)
points: List[PointStruct] = make_points(sampled_df)
client.upsert(collection_name, points)

# Define el modelo de entrada para la API
class SearchQuery(BaseModel):
    query_text: str

# Ruta de búsqueda híbrida
@app.post("/search/")
def search(query: SearchQuery):
    try:
        search_results = query_hybrid(client, collection_name, query.query_text)
        return {"results": search_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para verificar el estado de la API
@app.get("/health_check/")
def health_check():
    return {"status": "API is running"}
