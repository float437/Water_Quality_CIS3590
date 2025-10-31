from dotenv import load_dotenv
from pymongo import MongoClient
import csv
import pandas as pd
import os

load_dotenv()

# reading from my .env file
MONGO_USER = os.environ.get("MONGO_USER")
MONGO_PASSWORD = os.environ.get("MONGO_PASSWORD")
MONGO_CLUSTER_URL = os.environ.get("MONGO_CLUSTER_URL")
csvFileName = "cleanedOctober7.csv"

url = (f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER_URL}/?retryWrites=true&w=majority&appName=WaterQuality")

client = MongoClient(url)

cleanFilesDB = client.get_database("water_quality_data")
collection = cleanFilesDB["asv_1"]
nameOfCSVFile = "cleanedOctober21.csv"
csv_file = f"data/Clean_CSV/{nameOfCSVFile}"

try:
    # Open the CSV Files
    with open(csv_file, "r") as file:
        reader = csv.DictReader(file)
        # Iterate over each row in the csv file.
        for row in reader:
            collection.insert_one(row)

    print(f"Successfully inserted {collection.count_documents({})} documents.")
    # clean_CSV = pd.read_csv(f"data/Clean_CSV/{csvFileName}.csv")
    # clean_CSV = clean_CSV.to_dict('records')
    # result = cleanFilesDB.collection.insert_many(clean_CSV)
    # print(f"Successfully inserted {len(result.inserted_ids)} documents.")
except Exception as e:
    raise Exception("Unable to find the document due to the following error: ", e)