import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase with your service account key
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# Example client update
client_data = {
    "name": "Daniel Park",
    "lastWorkout": "2025-11-08",
    "lastRecommendation": "Focus on bicep relaxation",
    "progress": 85,
    "notes": "Good EMG control today."
}

# Write to Firestore
db.collection("clients").document("client001").set(client_data)

print("Data uploaded successfully!")
