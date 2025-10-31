from flask import Flask,jsonify,request
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
from pymongo.collection import Collection
from scipy.stats import zscore

app = Flask(__name__)

load_dotenv()

#reading from the .env file
MONGO_USER = os.environ.get("MONGO_USER")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")
MONGO_CLUSTER_URL = os.environ.get("MONGO_CLUSTER_URL")
db_Name = "water_quality_data"
collection_Name = "asv_1"

url = (f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER_URL}/?retryWrites=true&w=majority&appName=FirstCluster")

client = MongoClient(url, server_api=ServerApi('1'))
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e) 
    

try:
    db = client.get_database(f"{db_Name}")
    collection = db.get_collection(f"{collection_Name}")
    print(f"You successfully got the database : {db_Name} and Collection : {collection_Name}!")
except Exception as e:
    print(e) 


# creating the index route that just returns a json dictionary of the routes to be available.
@app.route("/")
def index():
    return jsonify({
        "Home Page":{
            "/class": "CIS3590 w/ Gregory Reis",
            "/people": "Rajiv Chevannes, Tobi, Ayeean"
        }
    })

@app.route("/api/health")
def api_health():
    return jsonify({"status":"ok"})

@app.route("/api/observations")
def api_observations():
    # Get pagination parameters from query string
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 100))
    
    # Calculate skip value
    skip = (page - 1) * per_page
    
    # Query with pagination
    items = list(collection.find()
                .skip(skip)
                .limit(per_page))
    
    # Get total count for metadata
    total = collection.count_documents({})
    
    # Convert ObjectId to string for JSON serialization
    for item in items:
        item['_id'] = str(item['_id'])
    
    return jsonify({
        'items': items,
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': (total + per_page - 1) // per_page
    })
    # if db is None or collection is None:
    #     return jsonify({"error": "Database connection not available."}), 500
    
    # try: 
    #     observationsDict = {
    #             "first" : getFirstFromDB(),
    #             "last" : getLastFromDB(),
    #             "minTemperature" : getminTempFromDB(),
    #             "maxTemperature" : getmaxTempFromDB(),
    #             "minSalinity" : getminSalFromDB(),
    #             "minSalinity" : getmaxSalFromDB(),
    #             "minOdo" : getminOdoFromDB(),
    #             "maxOdo" : getmaxOdoFromDB(),
    #         }
    #     return jsonify(observationsDict),200
    
    # except Exception as e:
    #     print()
    #     return jsonify({"error": "{e}"}) ,400

@app.route("/api/stats")
def api_stats():
    # count, mean, min, max, and percentiles (25%, 50%, 75%).
    if db is None or collection is None:
        return jsonify({"error": "Database connection not available."}), 500
    
    try: 
        statsDict = {
                "temperature" : {
                    "minTemperature" : getminTempFromDB(),
                    "TemperaturePercentiles" : getPercentiles("temp_z-score"),
                    "maxTemperature" : getmaxTempFromDB(),
                    "averageTemperature" : getMean("temp_z-score")
                    },
                "Salinity" : {
                    "minSalinity" : getminSalFromDB(),
                    "SalinityPercentiles" : getPercentiles("salinity_z-score"),
                    "minSalinity" : getmaxSalFromDB(),
                    # "averageSalinity" : getMean("salinity_z-score")
                },
                "Odo" : {
                    "minOdo" : getminOdoFromDB(),
                    "odoPercentiles" : getPercentiles("odo_z-score"),
                    "maxOdo" : getmaxOdoFromDB(),
                    # "averageOdo" : getMean("odo_z-score")
                },
            }
        return jsonify(statsDict),200
    
    except Exception as e:
        print()
        return jsonify({"error": "{e}"}) ,400
    # return jsonify(db.head(10).to_dict(orient="records"))


#Finds the 25th, 50th, and 75th percentiles of a specified numeric column 
# in a MongoDB collection using the $percentile aggregation operator.
def getPercentiles(columnName : str) -> list:
    # making an aggregation pipeline for the computations
    print(type(collection))
    print(f"Using DB: {collection.database.name}, Collection: {collection.name}")
    print(f"The columnName before the aggregation :{columnName}.")
    converted_input = { "$toDouble": f"${columnName}" }
    pipeline = [
        # Match only documents where the columnName field is a number (optional but good practice)
        # {
        #     "$match": {
        #         columnName: {}
        #     }
        # },
        
        # Calculate the percentiles across all matched documents
        {
            "$group": {
                "_id": None, # Grouping all documents together
                "percentiles": {
                    "$percentile": {
                        # The field to calculate percentiles on, dynamically using columnName
                        "input": converted_input, 
                        # The percentiles to calculate (25th, 50th, 75th)
                        "p": [0.25, 0.5, 0.75],
                        # Method to use for interpolation (e.g., 'r-7' for nearest rank)
                        "method": "approximate" 
                    }
                }
            }
        }
    ]

    # Execute the aggregation pipeline
    try:
        # get the column from the db, find the percentiles
        # Use collection.aggregate(pipeline) to execute the pipeline
        cursor = collection.aggregate(pipeline)
        
        # Convert the cursor to a list and return the result
        result = list(cursor)
        print(f"The result from the aggregation",result)
        # The result will typically be a list like:
        # [{'_id': None, 'percentiles': [2.1, 5.0, 7.9]}]
        return result

    except Exception as e:
        print(f"An error occurred during aggregation: {e}")
        # Return an empty list or an error response as appropriate for your API
        return [f"Error, please try again. {e}"]

def getMean(columnName: str) :
    pipeline = [
        {
            # Stage 1: Add a new field with the converted numeric value
            '$addFields': {
                'converted_value': {
                    '$convert': {
                        'input': f'${columnName}', # Use the input column
                        'to': 'double',             # Convert to the 'double' BSON type
                        'onError': None,            # Set to None if conversion fails
                        'onNull': None              # Set to None if input is null
                    }
                }
            }
        },
        {
            # Stage 2: Group all documents and calculate the average
            '$group': {
                '_id': None,  # Group all documents into a single group
                'mean': { '$avg': '$converted_value' } # Calculate the average of the new field
            }
        }
    ]
    print(f"Column Name is {columnName}")
    print(f"Pipeline Name is {pipeline}")


    try:
        result = list(collection.aggregate(pipeline))
        if result and 'mean' in result[0]:
            return result[0]['mean']
        else:
            # Handle cases where no documents were processed (e.g., all were non-numeric)
            return None
    
    except Exception as e:
        print(f"An error occurred during mean aggregation: {e}")
        # Return an empty list or an error response as appropriate for your API
        return [f"Error, please try again. {e}"]

@app.route("/api/outliers")
def api_outliers():
    # Allow re-checking outliers on demand using IQR or z-score.
    try:
        field = request.args.get("field","temperature")
        method = request.args.get("method","iqr").lower()
        k = float(request.args.get("k",1.5))

        if field not in ["temperature","salinity","odo"]:
            return jsonify({"error":"field must be: temperature|salinity|odo"}),400
        
        data = list(collection.find())
        # print(data)
        
        if not data:
            return jsonify({"message": "No data found", "removed": 0}), 200

        # TODO  : doc[field] should not get the right information as the field values are only the z scores.  They need to include the actual values as well.
        data = [doc for doc in data if field in doc and doc[field] is not None]
        
        outlier_ids = []

        for doc in data:
            if doc[field] > abs(3):
                outlier_ids.append(doc["_id"])

        # if there are outliers
        if outlier_ids:
            result = collection.delete_many({"_id": {"$in": outlier_ids}})
            removed_count = result.deleted_count
        else:
            removed_count = 0

        return jsonify({
            "message": "Outliers removed successfully",
            "field": field,
            "method": "zscore",
            "removed": removed_count,
            "removed Ids" : outlier_ids
        }), 200

        return jsonify()
    except Exception as e:
        return jsonify({"error":f"{e}"}),500

def getminSalFromDB():
    minSalinity = collection.find_one(sort=[("salinity_z-score", -1)])
    
    if minSalinity is None:
        return jsonify({"error": "No observations found in database."}), 404
    
    minSalinity["_id"] = str(minSalinity["_id"])

    return minSalinity


def getmaxSalFromDB():
    maxSalinity = collection.find_one(sort=[("salinity_z-score", 1)])
    
    if maxSalinity is None:
        return jsonify({"error": "No observations found in database."}), 404
    
    maxSalinity["_id"] = str(maxSalinity["_id"])

    return maxSalinity


def getminOdoFromDB():
    minOdo = collection.find_one(sort=[("odo_z-score", -1)])
    
    if minOdo is None:
        return jsonify({"error": "No observations found in database."}), 404
    
    minOdo["_id"] = str(minOdo["_id"])

    return minOdo


def getmaxOdoFromDB():
    maxOdo = collection.find_one(sort=[("odo_z-score", 1)])
    
    if maxOdo is None:
        return jsonify({"error": "No observations found in database."}), 404
    
    maxOdo["_id"] = str(maxOdo["_id"])

    return maxOdo

def getminTempFromDB() :
    minTemperature = collection.find_one(sort=[("temp_z-score", -1)])
    
    if minTemperature is None:
        return jsonify({"error" : "No observations found in database."}), 404
    
    minTemperature["_id"] = str(minTemperature["_id"])

    return minTemperature

def getmaxTempFromDB() :
    maxTemperature = collection.find_one(sort=[("temp_z-score", 1)])
    
    if maxTemperature is None:
        return jsonify({"error" : "No observations found in database."}), 404
    
    maxTemperature["_id"] = str(maxTemperature["_id"])

    return maxTemperature

def getFirstFromDB() :
    earliest_observation = collection.find_one(sort=[("Date", 1), ("Time", 1)])
        
    if earliest_observation is None:
        return jsonify({"error": "No observations found in database."}), 404
        
    # Convert ObjectId to string for JSON serialization
    earliest_observation["_id"] = str(earliest_observation["_id"])
    
    return earliest_observation

def getLastFromDB() :
    latest_observation = collection.find_one(sort=[("Date", -1), ("Time", -1)])
        
    if latest_observation is None:
        return jsonify({"error": "No observations found in database."}), 404
        
    # Convert ObjectId to string for JSON serialization
    latest_observation["_id"] = str(latest_observation["_id"])
    
    return latest_observation

# actually runs the api
if __name__ == '__main__':
    app.run(debug=True, port=5050)