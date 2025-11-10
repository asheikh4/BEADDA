"""
ml_trainer.py
This file trains machine learning models to recognize good exercise form.
It reads CSV files of good reps and creates models that can predict form quality.

HOW TO USE:
1. Perform exercises using main.py in "collect" mode to gather good reps
2. Run this script: python ml_trainer.py
3. Models will be saved to the models/ folder
4. Use main.py in "feedback" mode to get AI suggestions

WHAT IT DOES:
- Loads good reps from CSV files
- Trains Random Forest models on the data
- Saves trained models for live feedback
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

class ExerciseMLTrainer:
    """
    Trains ML models for exercise form detection.
    Think of this as a teacher that learns what good form looks like.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
    
    def load_good_reps(self, exercise_name):
        """
        Loads good rep data from CSV file.
        
        Args:
            exercise_name: Name of exercise (e.g., 'bicep_curl', 'squat', 'wall_sit')
        
        Returns:
            DataFrame with all the good rep data
        """
        csv_path = f'data/{exercise_name}_good_reps.csv'
        
        if not os.path.exists(csv_path):
            print(f"❌ No data found for {exercise_name}")
            print(f"   Expected file: {csv_path}")
            print(f"   Please perform some reps first to collect data!")
            return None
        
        try:
            data = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(data)} good reps for {exercise_name}")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def prepare_features(self, data, exercise_name):
        """
        Prepares the data for machine learning.
        This converts raw angles and positions into features the model can learn from.
        
        Args:
            data: DataFrame with exercise data
            exercise_name: Name of the exercise
        
        Returns:
            Features (X) ready for training
        """
        # Remove non-feature columns
        exclude_cols = ['timestamp', 'rep_number']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Handle missing values by filling with column mean
        data_clean = data[feature_cols].fillna(data[feature_cols].mean())
        
        X = data_clean.values
        
        # Store feature names for later use
        self.feature_names[exercise_name] = feature_cols
        
        print(f"📊 Prepared {len(feature_cols)} features for {exercise_name}")
        print(f"   Features: {', '.join(feature_cols[:5])}... (showing first 5)")
        
        return X
    
    def train_model(self, exercise_name):
        """
        Trains a machine learning model for one exercise.
        
        Args:
            exercise_name: Name of exercise to train
        
        Returns:
            True if training successful, False otherwise
        """
        print(f"\n🏋️  Training model for {exercise_name}...")
        print("=" * 50)
        
        # Load data
        data = self.load_good_reps(exercise_name)
        if data is None or len(data) < 10:
            print(f"❌ Need at least 10 good reps to train. You have {len(data) if data is not None else 0}")
            return False
        
        # Prepare features
        X = self.prepare_features(data, exercise_name)
        
        # Create labels (all are good reps, labeled as 1)
        y = np.ones(len(X))
        
        # Scale the features (this normalizes the data for better learning)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model
        # RandomForest is like having multiple decision trees vote on the answer
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees in the forest
            max_depth=10,      # How deep each tree can grow
            random_state=42    # For reproducible results
        )
        
        model.fit(X_scaled, y)
        
        # Save the model and scaler
        self.models[exercise_name] = model
        self.scalers[exercise_name] = scaler
        
        print(f"✅ Model trained successfully!")
        print(f"   Model accuracy: {model.score(X_scaled, y) * 100:.1f}%")
        
        return True
    
    def save_models(self):
        """
        Saves trained models to disk so we can use them later.
        """
        os.makedirs('models', exist_ok=True)
        
        for exercise_name in self.models.keys():
            model_path = f'models/{exercise_name}_model.pkl'
            scaler_path = f'models/{exercise_name}_scaler.pkl'
            features_path = f'models/{exercise_name}_features.pkl'
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[exercise_name], f)
            
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[exercise_name], f)
            
            # Save feature names
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_names[exercise_name], f)
            
            print(f"💾 Saved model for {exercise_name}")
    
    def train_all_exercises(self):
        """
        Trains models for all three exercises.
        """
        exercises = ['bicep_curl', 'squat', 'wall_sit']
        
        print("\n" + "=" * 50)
        print("🚀 TRAINING ALL EXERCISE MODELS")
        print("=" * 50)
        
        trained_count = 0
        for exercise in exercises:
            if self.train_model(exercise):
                trained_count += 1
        
        if trained_count > 0:
            self.save_models()
            print("\n" + "=" * 50)
            print(f"✅ Successfully trained {trained_count}/{len(exercises)} models")
            print("=" * 50)
        else:
            print("\n❌ No models were trained. Collect more data first!")


def main():
    """
    Main function to run training.
    """
    print("\n👨‍⚕️ PHYSIOTHERAPY ML TRAINER")
    print("This program trains AI models to recognize good exercise form\n")
    
    trainer = ExerciseMLTrainer()
    trainer.train_all_exercises()
    
    print("\n✨ Training complete! You can now use live feedback during exercises.")
    print("   Run main.py to start exercising with AI feedback!\n")


if __name__ == "__main__":
    main()