# utils.py
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.decomposition import PCA
import joblib
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "123456"),
    verify_certs=False     
)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
df = pd.read_csv('Dataset-final.csv')

def process_binary_hashes(binary_hash_str):
    return np.array(list(map(int, binary_hash_str.strip())), dtype=np.uint8)

df['binary_code'] = df['binary_hash'].apply(process_binary_hashes)
binary_codes = np.vstack(df['binary_code'].values)

def pack_binary_codes(binary_codes):
    binary_codes_packed = np.packbits(binary_codes, axis=1)
    return binary_codes_packed

binary_codes_packed = pack_binary_codes(binary_codes)

num_bits = binary_codes.shape[1]
index = faiss.IndexBinaryFlat(num_bits)
index.add(binary_codes_packed)


pca = joblib.load('pca_model.joblib')
rotation_matrix = np.load('rotation_matrix.npy')
embeddings_mean = np.load('embeddings_mean.npy')
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def query_to_hashed_vector(query):
    query_embedding = embedding_model.encode(query)
    query_embedding_pca = pca.transform([query_embedding])[0]
    centered_query_embedding = query_embedding_pca - embeddings_mean
    rotated_query_embedding = np.dot(centered_query_embedding, rotation_matrix)
    binary_code = (rotated_query_embedding > 0).astype(int)
    return binary_code.tolist()

def keyword_search(query_text, index_name):
    keyword_query = {
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "product_name",
                    "product_description^3"
                ],
                "type": "best_fields"
            }
        }
    }
    keyword_response = es.search(index=index_name, body=keyword_query)
    return keyword_response

def get_all_products(index_name, size=1000):
    query = {
        "query": {
            "match_all": {}
        },
        "size": size
    }
    try:
        response = es.search(index=index_name, body=query)
        products = response['hits']['hits']
        product_data = [
            {
                'product_id': hit['_source'].get('product_id', ''),
                'product_name': hit['_source'].get('product_name', ''),
                'product_description': hit['_source'].get('product_description', ''),
                'product_price': hit['_source'].get('price', 0),
                'product_url': hit['_source'].get('product_url', ''),
                'product_images': hit['_source'].get('images', []),
                'keyword_score': hit['_score']
            }
            for hit in products
        ]
        products_df = pd.DataFrame(product_data)
        return products_df

    except Exception as e:
        print(f"Error fetching products: {e}")
        return pd.DataFrame()

def semantic_search_hash(query, index1, df, k=10):
    query_hash = query_to_hashed_vector(query)
    query_hash = np.array(query_hash, dtype=np.uint8)
    query_hash_packed = np.packbits(query_hash)
    query_hash_packed = np.expand_dims(query_hash_packed, axis=0)
    distances, indices = index.search(query_hash_packed, k)
    results = df.iloc[indices[0]].copy()
    results['hamming_distance'] = distances[0]
    max_distance = num_bits
    results['similarity'] = 1 - (results['hamming_distance'] / max_distance)
    return results

def hybrid_search_hash(query, index, df, k=10, alpha=0.7):
    semantic_results = semantic_search_hash(query, index, df, k)
    semantic_results['semantic_score'] = semantic_results['similarity']
    semantic_results = semantic_results.rename(columns={'Producturl': 'product_url', 'name': 'product_name', 'price': 'product_price', 'description': 'product_description', 'images': 'product_images'})
    
    keyword_results = keyword_search(query, index)
    keyword_hits = keyword_results['hits']['hits']
    keyword_data = [
        {
            'product_id': hit['_source']['product_id'],
            'product_url': hit['_source']['product_url'],
            'product_price': hit['_source']['price'],
            'product_name': hit['_source'].get('product_name', ''),
            'product_description': hit['_source'].get('product_description', ''),
            'product_images': hit['_source'].get('images', []),
            'keyword_score': hit['_score']
        } 
        for hit in keyword_hits
    ]
    keyword_df = pd.DataFrame(keyword_data)
    combined = pd.merge(semantic_results, keyword_df, on='product_id', how='outer')
    combined['semantic_score'] = combined['semantic_score'].fillna(0)
    combined['keyword_score'] = combined['keyword_score'].fillna(0)
    combined['combined_score'] = alpha * combined['semantic_score'] + (1 - alpha) * combined['keyword_score']
    combined = combined.sort_values(by='combined_score', ascending=False)
    return combined.head(k)

def rerank_with_cross_encoder(query, candidates):
    cross_encoder_input = [(query, row['product_description_y']) for _, row in candidates.iterrows()]
    relevance_scores = cross_encoder.predict(cross_encoder_input)
    candidates['cross_encoder_score'] = relevance_scores
    reranked_results = candidates.sort_values(by='cross_encoder_score', ascending=False)
    return reranked_results
