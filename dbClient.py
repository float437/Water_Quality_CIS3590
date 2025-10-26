from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
import os

load_dotenv()

# reading from my .env file
MONGO_USER = os.environ.get("MONGO_USER")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")
MONGO_CLUSTER_URL = os.environ.get("MONGO_CLUSTER_URL")
csvFileName = "cleanedOctober7"

url = (f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER_URL}/?retryWrites=true&w=majority&appName=WaterQuality")

client = MongoClient(url)



try:
    # rawFilesDB = client["rawCSV"]
    # cleanFilesDB = client["cleanedCSV"]

    cleanFilesDB = client.get_database("water_quality_data")

    clean_CSV = pd.read_csv(f"data/Clean_CSV/{csvFileName}.csv")
    clean_CSV = clean_CSV.to_dict('records')
    result = cleanFilesDB.collection.insert_many(clean_CSV)
    print(f"Successfully inserted {len(result.inserted_ids)} documents.")
except Exception as e:
    raise Exception("Unable to find the document due to the following error: ", e)

# TODO: exporting Raw_CSV's into the DB 
# TODO: exporting Cleaned CSV's into the DB