"""
feedback_system.py
This file provides real-time feedback on exercise form using trained ML models.
It analyzes your current pose and tells you how to improve.

HOW IT WORKS:
1. Loads trained ML models
2. Extracts features from your current pose
3. Compares to good reps to give quality score
4. Provides specific feedback on how to improve
5. Logs all feedback for session review
"""

import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime
from exercises_CSVLogger import angle_to_vertical_2d, angle_3pt, to_xy

class FeedbackSystem:
    """
    Provides live feedback on exercise form using trained ML models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.feedback_log = []
        self.loaded_exercises = []
    
    def load_model(self, exercise_name):
        """
        Loads a trained model for an exercise.
        
        Args:
            exercise_name: Name of exercise (e.g., 'bicep_curl')
        
        Returns:
            True if successful, False otherwise
        """
        model_path = f'models/{exercise_name}_model.pkl'
        scaler_path = f'models/{exercise_name}_scaler.pkl'
        features_path = f'models/{exercise_name}_features.pkl'
        
        if not os.path.exists(model_path):
            print(f"⚠️  No trained model found for {exercise_name}")
            print(f"   Run ml_trainer.py first to train models!")
            return False
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.models[exercise_name] = pickle.load(f)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scalers[exercise_name] = pickle.load(f)
            
            # Load feature names
            with open(features_path, 'rb') as f:
                self.feature_names[exercise_name] = pickle.load(f)
            
            self.loaded_exercises.append(exercise_name)
            print(f"✅ Loaded model for {exercise_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_all_models(self):
        """
        Loads models for all exercises.
        """
        exercises = ['bicep_curl', 'squat', 'wall_sit']
        for exercise in exercises:
            self.load_model(exercise)
    
    def extract_features_from_landmarks(self, lm, w, h, exercise_name, mp_pose):
        """
        Extracts features from MediaPipe landmarks.
        
        Args:
            lm: MediaPipe landmarks
            w, h: Image width and height
            exercise_name: Name of current exercise
            mp_pose: MediaPipe pose object
        
        Returns:
            Dictionary with feature names and values
        """
        features = {}
        
        if exercise_name == 'bicep_curl':
            # Extract bicep curl features (both arms)
            P = mp_pose.PoseLandmark
            
            # Left arm
            left_shoulder = to_xy(lm, P.LEFT_SHOULDER.value, w, h)
            left_elbow = to_xy(lm, P.LEFT_ELBOW.value, w, h)
            left_wrist = to_xy(lm, P.LEFT_WRIST.value, w, h)
            
            # Right arm
            right_shoulder = to_xy(lm, P.RIGHT_SHOULDER.value, w, h)
            right_elbow = to_xy(lm, P.RIGHT_ELBOW.value, w, h)
            right_wrist = to_xy(lm, P.RIGHT_WRIST.value, w, h)
            
            # Calculate angles
            if min(left_shoulder + left_elbow + left_wrist) > 0:
                features['left_elbow_angle'] = angle_3pt(left_shoulder, left_elbow, left_wrist)
                features['upper_arm_vertical_deg_left'] = angle_to_vertical_2d(left_shoulder, left_elbow)
            else:
                features['left_elbow_angle'] = 0
                features['upper_arm_vertical_deg_left'] = 0
            
            if min(right_shoulder + right_elbow + right_wrist) > 0:
                features['right_elbow_angle'] = angle_3pt(right_shoulder, right_elbow, right_wrist)
                features['upper_arm_vertical_deg_right'] = angle_to_vertical_2d(right_shoulder, right_elbow)
            else:
                features['right_elbow_angle'] = 0
                features['upper_arm_vertical_deg_right'] = 0
            
            # Add coordinates
            features['left_shoulder_x'], features['left_shoulder_y'] = left_shoulder
            features['left_elbow_x'], features['left_elbow_y'] = left_elbow
            features['left_wrist_x'], features['left_wrist_y'] = left_wrist
            features['right_shoulder_x'], features['right_shoulder_y'] = right_shoulder
            features['right_elbow_x'], features['right_elbow_y'] = right_elbow
            features['right_wrist_x'], features['right_wrist_y'] = right_wrist
        
        elif exercise_name == 'squat':
            # Extract squat features
            P = mp_pose.PoseLandmark
            
            # Get joints
            l_hip = to_xy(lm, P.LEFT_HIP.value, w, h)
            l_knee = to_xy(lm, P.LEFT_KNEE.value, w, h)
            l_ankle = to_xy(lm, P.LEFT_ANKLE.value, w, h)
            l_shoulder = to_xy(lm, P.LEFT_SHOULDER.value, w, h)
            
            r_hip = to_xy(lm, P.RIGHT_HIP.value, w, h)
            r_knee = to_xy(lm, P.RIGHT_KNEE.value, w, h)
            r_ankle = to_xy(lm, P.RIGHT_ANKLE.value, w, h)
            r_shoulder = to_xy(lm, P.RIGHT_SHOULDER.value, w, h)
            
            # Calculate angles
            if min(l_hip + l_knee + l_ankle) > 0:
                features['left_knee_angle'] = angle_3pt(l_hip, l_knee, l_ankle)
                features['left_hip_angle'] = angle_3pt(l_shoulder, l_hip, l_knee)
            else:
                features['left_knee_angle'] = 0
                features['left_hip_angle'] = 0
            
            if min(r_hip + r_knee + r_ankle) > 0:
                features['right_knee_angle'] = angle_3pt(r_hip, r_knee, r_ankle)
                features['right_hip_angle'] = angle_3pt(r_shoulder, r_hip, r_knee)
            else:
                features['right_knee_angle'] = 0
                features['right_hip_angle'] = 0
            
            # Torso metrics (use whichever side is more visible)
            if min(l_hip + l_shoulder) > 0:
                features['torso_lean_deg'] = angle_to_vertical_2d(l_hip, l_shoulder)
                tibia_angle = angle_to_vertical_2d(l_ankle, l_knee) if min(l_ankle + l_knee) > 0 else 0
                features['trunk_tibia_diff'] = abs(features['torso_lean_deg'] - tibia_angle)
            elif min(r_hip + r_shoulder) > 0:
                features['torso_lean_deg'] = angle_to_vertical_2d(r_hip, r_shoulder)
                tibia_angle = angle_to_vertical_2d(r_ankle, r_knee) if min(r_ankle + r_knee) > 0 else 0
                features['trunk_tibia_diff'] = abs(features['torso_lean_deg'] - tibia_angle)
            else:
                features['torso_lean_deg'] = 0
                features['trunk_tibia_diff'] = 0
            
            # Add coordinates
            features['left_hip_x'], features['left_hip_y'] = l_hip
            features['left_knee_x'], features['left_knee_y'] = l_knee
            features['left_ankle_x'], features['left_ankle_y'] = l_ankle
            features['left_shoulder_x'], features['left_shoulder_y'] = l_shoulder
            features['right_hip_x'], features['right_hip_y'] = r_hip
            features['right_knee_x'], features['right_knee_y'] = r_knee
            features['right_ankle_x'], features['right_ankle_y'] = r_ankle
            features['right_shoulder_x'], features['right_shoulder_y'] = r_shoulder
        
        elif exercise_name == 'wall_sit':
            # Extract wall sit features
            P = mp_pose.PoseLandmark
            
            l_hip = to_xy(lm, P.LEFT_HIP.value, w, h)
            l_knee = to_xy(lm, P.LEFT_KNEE.value, w, h)
            l_ankle = to_xy(lm, P.LEFT_ANKLE.value, w, h)
            l_shoulder = to_xy(lm, P.LEFT_SHOULDER.value, w, h)
            
            r_hip = to_xy(lm, P.RIGHT_HIP.value, w, h)
            r_knee = to_xy(lm, P.RIGHT_KNEE.value, w, h)
            r_ankle = to_xy(lm, P.RIGHT_ANKLE.value, w, h)
            r_shoulder = to_xy(lm, P.RIGHT_SHOULDER.value, w, h)
            
            # Calculate angles
            if min(l_hip + l_knee + l_ankle) > 0:
                features['left_knee_angle'] = angle_3pt(l_hip, l_knee, l_ankle)
                features['left_hip_angle'] = angle_3pt(l_shoulder, l_hip, l_knee)
                knee_forward = float(l_knee[0] - l_ankle[0])
                features['knee_over_toe_norm_left'] = max(0, knee_forward / w) if w > 0 else 0
            else:
                features['left_knee_angle'] = 0
                features['left_hip_angle'] = 0
                features['knee_over_toe_norm_left'] = 0
            
            if min(r_hip + r_knee + r_ankle) > 0:
                features['right_knee_angle'] = angle_3pt(r_hip, r_knee, r_ankle)
                features['right_hip_angle'] = angle_3pt(r_shoulder, r_hip, r_knee)
                knee_forward = float(r_knee[0] - r_ankle[0])
                features['knee_over_toe_norm_right'] = max(0, knee_forward / w) if w > 0 else 0
            else:
                features['right_knee_angle'] = 0
                features['right_hip_angle'] = 0
                features['knee_over_toe_norm_right'] = 0
            
            # Torso vertical
            if min(l_hip + l_shoulder) > 0:
                features['torso_vertical_deg'] = angle_to_vertical_2d(l_hip, l_shoulder)
            elif min(r_hip + r_shoulder) > 0:
                features['torso_vertical_deg'] = angle_to_vertical_2d(r_hip, r_shoulder)
            else:
                features['torso_vertical_deg'] = 0
            
            features['hold_duration'] = 0  # Will be updated externally
            
            # Add coordinates
            features['left_hip_x'], features['left_hip_y'] = l_hip
            features['left_knee_x'], features['left_knee_y'] = l_knee
            features['left_ankle_x'], features['left_ankle_y'] = l_ankle
            features['right_hip_x'], features['right_hip_y'] = r_hip
            features['right_knee_x'], features['right_knee_y'] = r_knee
            features['right_ankle_x'], features['right_ankle_y'] = r_ankle
        
        return features
    
    def get_feedback(self, lm, exercise_name, rep_number, w=1280, h=720):
        """
        Gets feedback on current form.
        
        Args:
            lm: MediaPipe landmarks
            exercise_name: Name of current exercise
            rep_number: Current rep number
            w, h: Image dimensions
        
        Returns:
            Feedback message and quality score
        """
        if exercise_name not in self.loaded_exercises:
            return "Model not loaded", 0.0
        
        # Import mp_pose here to avoid circular imports
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # Extract features from current pose
        features_dict = self.extract_features_from_landmarks(lm, w, h, exercise_name, mp_pose)
        
        # Convert to array in correct order, handling missing features
        feature_values = []
        for name in self.feature_names[exercise_name]:
            if name in features_dict:
                feature_values.append(features_dict[name])
            else:
                feature_values.append(0)  # Default value for missing features
        
        X = np.array(feature_values).reshape(1, -1)
        
        # Handle NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        X_scaled = self.scalers[exercise_name].transform(X)
        
        # Get prediction probability
        probs = self.models[exercise_name].predict_proba(X_scaled)[0]
        if len(probs) == 1:
            probability = 1.0  # or 0.5, depending how you want to handle single-class
        else:
            probability = probs[1]

        #probability = self.models[exercise_name].predict_proba(X_scaled)[0][1]
        
        # Generate feedback based on probability
        feedback_msg, feedback_tag = self.generate_feedback(
            probability, features_dict, exercise_name
        )
        
        # Log feedback
        self.log_feedback(exercise_name, rep_number, probability, feedback_tag, features_dict)
        
        return feedback_msg, probability
    
    def generate_feedback(self, probability, features, exercise_name):
        """
        Generates specific feedback based on form quality.
        
        Args:
            probability: Quality score (0-1)
            features: Dictionary of current features
            exercise_name: Name of exercise
        
        Returns:
            Feedback message and tag
        """
        if probability > 0.85:
            return "✅ Excellent form!", "excellent"
        
        elif probability > 0.70:
            # Good form with minor suggestions
            if exercise_name == 'bicep_curl':
                left_angle = features.get('left_elbow_angle', 0)
                right_angle = features.get('right_elbow_angle', 0)
                
                if abs(left_angle - right_angle) > 15:
                    return "👍 Good! Keep elbows even", "good_uneven_elbows"
                else:
                    return "👍 Good form!", "good"
            
            elif exercise_name == 'squat':
                return "👍 Good! Go slightly lower", "good_depth"
            
            elif exercise_name == 'wall_sit':
                return "👍 Good! Hold steady", "good"
        
        else:
            # Needs improvement
            if exercise_name == 'bicep_curl':
                left_angle = features.get('left_elbow_angle', 0)
                right_angle = features.get('right_elbow_angle', 0)
                
                if left_angle < 40 or right_angle < 40:
                    return "⚠️  Curl higher", "low_curl"
                elif left_angle > 150 or right_angle > 150:
                    return "⚠️  Don't fully extend", "over_extension"
                else:
                    return "⚠️  Keep elbows stable", "unstable_elbows"
            
            elif exercise_name == 'squat':
                knee_angle = min(features.get('left_knee_angle', 180), 
                               features.get('right_knee_angle', 180))
                
                if knee_angle > 120:
                    return "⚠️  Squat deeper", "shallow_squat"
                else:
                    return "⚠️  Keep back straight", "back_angle"
            
            elif exercise_name == 'wall_sit':
                knee_angle = min(features.get('left_knee_angle', 180),
                               features.get('right_knee_angle', 180))
                
                if knee_angle < 80:
                    return "⚠️  Raise hips slightly", "too_low"
                elif knee_angle > 100:
                    return "⚠️  Lower hips more", "too_high"
                else:
                    return "⚠️  Adjust position", "adjust_position"
        
        return "👍 Keep going!", "neutral"
    
    def log_feedback(self, exercise_name, rep_number, quality_score, feedback_tag, features):
        """
        Logs feedback for later analysis.
        """
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'exercise': exercise_name,
            'rep_number': rep_number,
            'quality_score': quality_score,
            'feedback_tag': feedback_tag
        }
        
        # Add key features
        if exercise_name == 'bicep_curl':
            log_entry['left_elbow_angle'] = features.get('left_elbow_angle', 0)
            log_entry['right_elbow_angle'] = features.get('right_elbow_angle', 0)
        elif exercise_name == 'squat':
            log_entry['left_knee_angle'] = features.get('left_knee_angle', 0)
            log_entry['right_knee_angle'] = features.get('right_knee_angle', 0)
        elif exercise_name == 'wall_sit':
            log_entry['left_knee_angle'] = features.get('left_knee_angle', 0)
            log_entry['right_knee_angle'] = features.get('right_knee_angle', 0)
        
        self.feedback_log.append(log_entry)
    
    def save_session_log(self):
        """
        Saves all feedback from this session to a CSV file.
        """
        if not self.feedback_log:
            return
        
        os.makedirs('logs', exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(self.feedback_log)
        
        # Generate filename with timestamp
        filename = f"logs/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"\n💾 Session log saved: {filename}")
        
        # Print summary
        print("\n📊 SESSION SUMMARY:")
        print("=" * 50)
        for exercise in df['exercise'].unique():
            exercise_data = df[df['exercise'] == exercise]
            avg_quality = exercise_data['quality_score'].mean()
            rep_count = len(exercise_data)
            print(f"   {exercise}: {rep_count} reps, avg quality: {avg_quality:.1%}")
        print("=" * 50)