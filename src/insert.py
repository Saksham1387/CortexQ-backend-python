import pandas as pd
from elasticsearch import Elasticsearch

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "123456"),
    verify_certs=False   
)

final_df = pd.read_csv('Dataset-final.csv')
index_name = 'products2'

def binary_hash_to_list(binary_hash):
    return [int(bit) for bit in binary_hash]


final_df['hashed_vector'] = final_df['binary_hash'].apply(binary_hash_to_list)


synonyms = [
    "laptop, notebook",
    "tv, television",
    "cellphone, smartphone, mobile phone",
    "headphones, headset",
    "camera, camcorder",
]

if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

index_settings = {
    "settings": {
        "analysis": {
            "filter": {
                "english_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                },
                "english_stemmer": {
                    "type": "stemmer",
                    "language": "english"
                },
                "english_possessive_stemmer": {
                    "type": "stemmer",
                    "language": "possessive_english"
                },
                "synonym_filter": {
                    "type": "synonym",
                    "synonyms": synonyms
                }
            },
            "analyzer": {
                "custom_english_analyzer": {
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "english_possessive_stemmer",
                        "english_stop",
                        "english_stemmer",
                        "synonym_filter"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "product_id": {"type": "keyword"},
           "hashed_vector": {
              "type": "dense_vector",
              "dims": 64  
            },
            "product_url": {"type": "keyword"},
            "product_description": {
                "type": "text",
                "analyzer": "custom_english_analyzer"
            },
            "product_name": {
                "type": "text",
                "analyzer": "custom_english_analyzer"
            },
            "images": {"type": "keyword"},
            "price": {"type": "float"}
        }
    }
}

es.indices.create(index=index_name, body=index_settings)

def index_documents(df, index_name):
    print(f"Starting the indexing process for index: {index_name}")
    success_count = 0
    error_count = 0
    for _, row in df.iterrows():
        images = row['images'].split(' | ') if isinstance(row['images'], str) else []

        document = {
            "product_id": row['product_id'],
            "hashed_vector": row['hashed_vector'],  
            "product_url": row['Producturl'],
            "product_description": row['description'],
            "product_name": row['name'],
            "images": images, 
            "price": row['price']
        }
        try:
            es.index(index=index_name, body=document)
            success_count += 1
        except Exception as e:
            print(f"Error indexing document {row['product_id']}: {e}")
            error_count += 1

    print(f"Indexing completed. Successfully indexed {success_count} documents. {error_count} errors occurred.")


index_documents(final_df, index_name)