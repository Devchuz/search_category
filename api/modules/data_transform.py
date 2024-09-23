import pandas as pd
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from transformers import AutoTokenizer
from typing import List, Tuple

sparse_model_name = "prithvida/Splade_PP_en_v1"
dense_model_name = "BAAI/bge-large-en-v1.5"
# This triggers the model download
sparse_model = SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)
dense_model = TextEmbedding(model_name=dense_model_name, batch_size=32)

# Generate sparse embedding

def make_sparse_embedding(texts: List[str]):
    return list(sparse_model.embed(texts, batch_size=32))

# generate dense embedding

def make_dense_embedding(texts: List[str]):
    return list(dense_model.embed(texts))



def get_tokens_and_weights(sparse_embedding, model_name):
    # Find the tokenizer for the model
    tokenizer_source = None
    for model_info in SparseTextEmbedding.list_supported_models():
        if model_info["model"].lower() == model_name.lower():
            tokenizer_source = model_info["sources"]["hf"]
            break
        else:
            raise ValueError(f"Model {model_name} not found in the supported models.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    token_weight_dict = {}
    for i in range(len(sparse_embedding.indices)):
        token = tokenizer.decode([sparse_embedding.indices[i]])
        weight = sparse_embedding.values[i]
        token_weight_dict[token] = weight

    # Sort the dictionary by weights
    token_weight_dict = dict(
        sorted(token_weight_dict.items(), key=lambda item: item[1], reverse=True)
    )
    return token_weight_dict


def generate_embeddings(df_samples):
    product_texts = df_samples["combined_text"].tolist()
    df_samples["sparse_embedding"] = make_sparse_embedding(product_texts)
    df_samples["dense_embedding"] = make_dense_embedding(product_texts)
    return df_samples

def data_transform(df_category):
    sampled_df = pd.DataFrame()
    df_category['combined_text'] = (df_category['main_category'] + "\n"+ df_category['sub_category'])
    sampled_df['combined_text'] = df_category['combined_text'].unique()
    sampled_df['id'] = sampled_df.index
    sampled_df = generate_embeddings(sampled_df)
    return sampled_df