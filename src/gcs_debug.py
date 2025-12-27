from google.cloud import storage
import os

# 1. HARDCODED PATHS - NO CONFIG FILES
KEY_PATH = r"C:\Users\user\Desktop\ML Ops Project\strategic-reef-479708-u4-ee7574828564.json"
BUCKET_NAME = 'paul_mlops_project1' # PUT YOUR BUCKET NAME HERE
FILE_NAME = 'Hotel_Reservations.csv'             # PUT YOUR FILE NAME HERE

print(f"Checking if key exists at: {KEY_PATH}")
if os.path.exists(KEY_PATH):
    print("✅ Key file found on disk.")
else:
    print("❌ Key file NOT found. Check the path again.")
    exit()

try:
    print("Attempting to connect to Google Cloud...")
    # This specific line IGNORES environment variables
    client = storage.Client.from_service_account_json(KEY_PATH)
    
    bucket = client.get_bucket(BUCKET_NAME)
    print(f"✅ Successfully connected to bucket: {BUCKET_NAME}")
    
    blob = bucket.blob(FILE_NAME)
    if blob.exists():
        print(f"✅ File {FILE_NAME} found in bucket!")
    else:
        print(f"❌ File {FILE_NAME} NOT found in bucket.")
        
except Exception as e:
    print(f"❌ CONNECTION FAILED: {e}")