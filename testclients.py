import firebase_admin
from firebase_admin import credentials, firestore
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
fake = Faker()

# Configuration
NUM_CLIENTS = 5          # number of fake clients
MAX_WORKOUTS = 5         # max workouts per client

# Generate fake clients
for i in range(NUM_CLIENTS):
    client_name = fake.name()
    client_email = fake.email()
    
    # Reference to this client in Firestore
    client_ref = db.collection("clients").document(client_email)
    
    # Generate workouts for this client
    workouts = []
    for j in range(random.randint(1, MAX_WORKOUTS)):
        workout_date = datetime.now() - timedelta(days=random.randint(0, 30))
        recommendation = random.choice([
            "Increase reps", 
            "Focus on form", 
            "Rest more", 
            "Add resistance band", 
            "Light stretching"
        ])
        notes = fake.sentence(nb_words=10)
        
        workout = {
            "date": workout_date.strftime("%Y-%m-%d %H:%M:%S"),
            "recommendation": recommendation,
            "notes": notes
        }
        workouts.append(workout)
    
    # Set client document in Firestore
    client_ref.set({
        "name": client_name,
        "email": client_email,
        "last_workout": workouts[-1]["date"],
        "workouts": workouts
    })
    
    print(f"Added client: {client_name} with {len(workouts)} workouts")

print("Finished adding fake clients!")
