from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import time
import logging
import pandas as pd
from utils import (
    es, index, cross_encoder, df, 
    get_all_products, hybrid_search_hash, rerank_with_cross_encoder
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True  

def fetch_all_products(index_name):
    all_products = get_all_products(index_name)
    if not all_products.empty:
        all_products.rename(columns={
            "product_url": "product_url_y",
            "product_name": "product_name_y",
            "product_description": "product_description_y",
            "product_images": "product_images_y",
            "product_price": "product_price_y"
        }, inplace=True)
        columns_to_keep = [
            "product_id", "product_url_y", "product_name_y", "product_description_y",
            "product_images_y", "keyword_score", "product_price_y"
        ]
        return all_products[columns_to_keep].to_dict(orient='records')
    return []

@app.route('/', methods=['POST'])
def home():
    # Endpoint to fetch all the products in the specified index
    start_time = time.time()
    
    # Parse the request body to get the index_name
    try:
        data = request.get_json()
        index_name = data.get("index_name")
        
        if not index_name:
            return jsonify({"error": "index_name is required"}), 400

        products_data = fetch_all_products(index_name)
        response_data = json.dumps(products_data, indent=4)
    
    except Exception as e:
        logger.error("Error fetching products: %s", e)
        return jsonify({"error": "Unable to retrieve products"}), 500
    
    finally:
        logger.info("Time taken to serve / request: %.2f seconds", time.time() - start_time)
    
    return response_data

@app.route('/api/query', methods=['POST'])
def query_process():
    # Endpoint to process the query of the user
    start_time = time.time()
    data = request.get_json()
    query = request.args.get('text', default="", type=str)
    if not query:
        return jsonify({"error": "Query text required"}), 400
    index_name = data.get('index_name')
    if not index_name:
        return jsonify({"error": "index_name required"}), 400
    
    try:
        initial_results = hybrid_search_hash(query, index_name, df, k=10, alpha=0.7)
        reranked_results = rerank_with_cross_encoder(query, initial_results)
        
        columns_to_keep = [
            "product_id", "product_url_y", "product_name_y", "product_description_y",
            "product_images_y", "keyword_score", "combined_score", "cross_encoder_score", "product_price_y"
        ]
        results_data = reranked_results[columns_to_keep].to_dict(orient='records')
        response_data = json.dumps(results_data, indent=4)
    except Exception as e:
        logger.error("Error processing query: %s", e)
        return jsonify({"error": "Unable to process query"}), 500
    finally:
        logger.info("Time taken to serve /api/query request: %.2f seconds", time.time() - start_time)

    return response_data

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    
    
    
    
