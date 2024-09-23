import os
import pandas as pd



def load_data(path: str, max_per_category: int = 20, sample_size: int = 200) -> pd.DataFrame:
    """
    Carga archivos CSV desde una carpeta, concatena los DataFrames, filtra por categorías y devuelve una muestra.

    Parámetros:
    - path (str): Ruta de la carpeta que contiene los archivos CSV.
    - max_per_category (int): Número máximo de muestras por categoría (por defecto 20).
    - sample_size (int): Tamaño de la muestra final si el DataFrame tiene más de 1000 filas (por defecto 200).

    Retorna:
    - pd.DataFrame: DataFrame concatenado y filtrado por categoría.
    """
    # Verificar si la ruta existe y es una carpeta
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError(f"La ruta '{path}' no existe o no es una carpeta válida.")

    # Lista para almacenar los DataFrames
    dataframes = []

    # Recorrer la carpeta y leer archivos CSV
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith('.csv') and os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenar todos los DataFrames en uno solo
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    # Asegurarse de que el DataFrame tiene una columna de categoría
    if 'main_category' not in concatenated_df.columns:
        raise ValueError("'main_category' no se encuentra en las columnas del DataFrame.")

    # Filtrar las filas con un máximo de max_per_category por categoría
    grouped = concatenated_df.groupby('main_category')
    df_category = grouped.apply(lambda x: x.sample(min(len(x), max_per_category))).reset_index(drop=True)

    # Si hay más de 1000 filas, tomar una muestra de sample_size
    if len(df_category) > 1000:
        df_category = df_category.sample(sample_size).reset_index(drop=True)

    return df_category