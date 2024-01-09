from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import firestore
from google.cloud import bigquery
from time import time
from datetime import timedelta
import sqlparse
import nl2sql


client = bigquery.Client()

global app
app = Flask(__name__)
cors = CORS(app)

global CURRENT_PROJECT_ID
CURRENT_PROJECT_ID = "brands-cdp-demo"

# Initialize Firestore client
global db
db = firestore.Client(CURRENT_PROJECT_ID)

def init():
    app.run(host="0.0.0.0", port=8080)

def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj

@app.route("/publishers", methods=["GET"])
def get_publishers():

    publishers = []
    # Get the document reference
    docs = db.collection("publishers").stream()

    for doc in docs:
        dict = doc.to_dict()
        publishers.append(dict)

    return jsonify(publishers), 200

@app.route("/publishers/<publisher_name>", methods=["GET"])
def get_publisher(publisher_name):

    # Get the document reference
    doc_ref = db.collection("publishers").document(publisher_name)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Publisher " + publisher_name + " does not exist"}), 400
    else:
        return jsonify(doc.to_dict()), 200

@app.route("/publishers/<publisher_name>", methods=["POST", "UPDATE"])
def set_publishers(publisher_name):

    request_data = request.get_json()
    if request_data:
        if 'dataset' in request_data:
            dataset = request_data['dataset']
        else:
            dataset = publisher_name + "_dataset"
        if 'project_id' in request_data:
            project_id = request_data['project_id']
        else:
            project_id = CURRENT_PROJECT_ID
    else:
        dataset = publisher_name + "_dataset"
        project_id = CURRENT_PROJECT_ID
    
    publisher_doc_ref = db.collection("publishers").document(publisher_name)
    publisher_doc = publisher_doc_ref.get()
    if request.method == 'POST':
        if not publisher_doc.exists:
            publisher_doc_ref.set({'dataset': dataset, 'project_id': project_id})
            return jsonify({"success": "Creating new publisher " + publisher_name + " with dataset " + dataset + " and project_id " + project_id}), 200
        else:
            return jsonify({"error": "Publisher " + publisher_name + " already exists - Please use UPDATE method"}), 400
    else:
        if not publisher_doc.exists:
            return jsonify({"error": "Publisher " + publisher_name + " does not exist. Please create it using the POST method"}), 400
        else:
            publisher_doc_ref.update({'dataset': dataset, 'project_id': project_id})
            return jsonify({"success": "Updating publisher " + publisher_name + " with dataset " + dataset + " and project_id " + project_id}), 200

@app.route("/publishers/<publisher_name>/segments/query", methods=["GET"])
def query_segments(publisher_name):

    args = request.args
    query = args.get("q")

    print("Incoming query: ", query)

    start_time = time()
    generated_query = nl2sql.call_gen_sql(query)

    return generated_query

@app.route("/brands", methods=["GET"])
def get_brands():

    brands = []
    # Get the document reference
    docs = db.collection("brands").stream()

    for doc in docs:
        dict = doc.to_dict()
        brands.append(dict)

    return jsonify(brands), 200

@app.route("/brands/<brand_name>", methods=["GET"])
def get_brand(brand_name):

    # Get the document reference
    doc_ref = db.collection("brands").document(brand_name)
    doc = doc_ref.get()

    if not doc.exists:
        return jsonify({"error": "Brand " + brand_name + " does not exist"}), 400
    else:
        return jsonify(doc.to_dict()), 200

@app.route("/brands/<brand_name>", methods=["POST", "UPDATE"])
def set_brands(brand_name):

    request_data = request.get_json()
    if request_data:
        if 'dataset' in request_data:
            dataset = request_data['dataset']
        else:
            dataset = brand_name + "_dataset"
        if 'project_id' in request_data:
            project_id = request_data['project_id']
        else:
            project_id = CURRENT_PROJECT_ID
    else:
        dataset = brand_name + "_dataset"
        project_id = CURRENT_PROJECT_ID
    
    brand_doc_ref = db.collection("brands").document(brand_name)
    brand_doc = brand_doc_ref.get()
    if request.method == 'POST':
        if not brand_doc.exists:
            brand_doc_ref.set({'dataset': dataset, 'project_id': project_id})
            return jsonify({"success": "Creating new brand " + brand_name + " with dataset " + dataset + " and project_id " + project_id}), 200
        else:
            return jsonify({"error": "Brand " + brand_name + " already exists - Please use UPDATE method"}), 400
    else:
        if not brand_doc.exists:
            return jsonify({"error": "Brand " + brand_name + " does not exist. Please create it using the POST method"}), 400
        else:
            brand_doc_ref.update({'dataset': dataset, 'project_id': project_id})
            return jsonify({"success": "Updating brand " + brand_name + " with dataset " + dataset + " and project_id " + project_id}), 200
        

@app.route("/publishers/<publisher_name>/brands/<brand_name>", methods=["POST"])
def create_brand(publisher_name, brand_name):

    # Check if brand name is provided
    if not publisher_name or not brand_name:
        return jsonify({"error": "Either publisher or brand name not provided"}), 400

    # Get the document reference
    doc_ref = db.collection("publishers").document(publisher_name)
    if not doc_ref.get().exists:
        return jsonify({"error": "Publisher " + publisher_name + " does not exist. Please create it first using the POST method before creating a mapping with a brand"}), 400
    doc_ref = doc_ref.collection("mappings").document(brand_name)

    doc_ref.set({"common_users": -1})
    
    return jsonify({"success": "Publisher added to existing brand"}), 200

@app.route("/publishers/<publisher_name>/brands/<brand_name>", methods=["DELETE"])
def delete_brand(publisher_name, brand_name):

    # Check if brand name is provided
    if not publisher_name or not brand_name:
        return jsonify({"error": "Missing either brand or publisher name"}), 400

    # Get the document reference
    doc_ref = db.collection("publishers").document(publisher_name)
    doc_ref = doc_ref.collection("mappings").document(brand_name)
    doc_ref.delete()