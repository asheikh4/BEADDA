"""
Flask/Socket.IO server with CV integration points for Mistral AI coaching.
Handles real-time metrics, coaching tips, and end-of-session patient charts.
"""

import time
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from llm import live_coach_tip, patient_chart_json

# Initialize Flask and Socket.IO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
socketio = SocketIO(app, cors_allowed_origins="*")

# Data structures
class MetricEvent:
    """One frame's CV metrics."""
    def __init__(self, **kwargs):
        self.timestamp = kwargs.get('timestamp', time.time())
        self.exercise_type = kwargs.get('exercise_type', 'squat')
        self.reps = kwargs.get('reps', 0)
        self.phase = kwargs.get('phase', 'top')
        
        # Angles (exercise-specific)
        self.knee_angle = kwargs.get('knee_angle', None)
        self.elbow_angle = kwargs.get('elbow_angle', None)
        self.torso_lean = kwargs.get('torso_lean', None)
        self.lumbar_excursion = kwargs.get('lumbar_excursion', None)
        
        # Zones (quality indicators)
        self.zones = kwargs.get('zones', {})  # e.g., {"depth_zone": "good", "torso_lean_zone": "acceptable"}
        
        # Form cues/flags
        self.cues = kwargs.get('cues', [])  # List of form issues
        
        # Additional metrics
        self.metrics = kwargs.get('metrics', {})  # Any other metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "exercise_type": self.exercise_type,
            "reps": self.reps,
            "phase": self.phase,
            "knee_angle": self.knee_angle,
            "elbow_angle": self.elbow_angle,
            "torso_lean": self.torso_lean,
            "lumbar_excursion": self.lumbar_excursion,
            "zones": self.zones,
            "cues": self.cues,
            "metrics": self.metrics
        }


class MLTag:
    """ML classifier output."""
    def __init__(self, label: str, score: float, timestamp: Optional[float] = None):
        self.label = label
        self.score = score
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "score": self.score,
            "timestamp": self.timestamp
        }


# Session state
class SessionState:
    """Tracks current session state."""
    def __init__(self):
        self.is_active = False
        self.start_time = None
        self.exercise_type = "squat"
        self.recent_events: List[MetricEvent] = []  # Last 10 events for context
        self.all_events: List[MetricEvent] = []  # All events in session
        self.ml_tags: List[MLTag] = []  # ML classifier outputs
        self.max_recent_events = 10
    
    def start(self, exercise_type: str = "squat"):
        """Start a new session."""
        self.is_active = True
        self.start_time = time.time()
        self.exercise_type = exercise_type
        self.recent_events.clear()
        self.all_events.clear()
        self.ml_tags.clear()
    
    def end(self) -> Dict[str, Any]:
        """End session and return metadata."""
        duration = time.time() - self.start_time if self.start_time else 0
        metadata = {
            "session_id": f"session_{int(time.time())}",
            "user_id": "user_001",  # TODO: Get from authentication
            "exercise_type": self.exercise_type,
            "duration": int(duration),
            "timestamp": datetime.now().isoformat(),
            "total_events": len(self.all_events),
            "total_ml_tags": len(self.ml_tags)
        }
        self.is_active = False
        return metadata
    
    def add_event(self, event: MetricEvent):
        """Add a metric event to the session."""
        if not self.is_active:
            return
        
        self.all_events.append(event)
        self.recent_events.append(event)
        
        # Keep only recent events
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events.pop(0)
    
    def add_ml_tag(self, tag: MLTag):
        """Add an ML tag to the session."""
        if not self.is_active:
            return
        self.ml_tags.append(tag)


# Global session state
session_state = SessionState()


# ==================== CV INTEGRATION POINTS ====================

def on_metrics_from_cv(metrics: Dict[str, Any]):
    """
    Call this from your CV loop each frame with current metrics.
    
    Example:
        metrics = {
            "exercise_type": "squat",
            "reps": 5,
            "phase": "down",
            "knee_angle": 95.0,
            "torso_lean": 25.0,
            "zones": {
                "depth_zone": "good",
                "torso_lean_zone": "acceptable"
            },
            "cues": ["Go deeper"],
            "lumbar_excursion": 12.0
        }
        on_metrics_from_cv(metrics)
    """
    # Create MetricEvent
    event = MetricEvent(**metrics)
    
    # Add to session
    session_state.add_event(event)
    
    # Emit metrics to frontend for overlays
    socketio.emit("metrics", event.to_dict())
    
    # Get coaching tip (with debouncing handled in llm.py)
    current_dict = event.to_dict()
    recent_dicts = [e.to_dict() for e in session_state.recent_events[-5:]]  # Last 5 for context
    
    tip = live_coach_tip(
        current_event=current_dict,
        recent_events=recent_dicts,
        exercise_type=session_state.exercise_type
    )
    
    if tip:
        # Emit coaching tip to frontend
        socketio.emit("coach_tip", {
            "tip": tip,
            "timestamp": time.time(),
            "exercise_type": session_state.exercise_type
        })


def on_ml_tag(label: str, score: float):
    """
    Call this when you get ML classifier output.
    
    Example:
        on_ml_tag("poor_form", 0.85)
        on_ml_tag("good_depth", 0.92)
    """
    tag = MLTag(label=label, score=score)
    session_state.add_ml_tag(tag)
    
    # Optionally emit to frontend
    socketio.emit("ml_tag", tag.to_dict())


# ==================== API ENDPOINTS ====================

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Start a new session."""
    data = request.get_json() or {}
    exercise_type = data.get('exercise_type', 'squat')
    
    session_state.start(exercise_type=exercise_type)
    
    return jsonify({
        "status": "success",
        "message": f"Session started for {exercise_type}",
        "session_id": session_state.start_time
    })


@app.route('/api/end_session', methods=['POST'])
def end_session():
    """
    End the current session and generate patient chart.
    Returns JSON patient chart.
    """
    if not session_state.is_active:
        return jsonify({
            "status": "error",
            "message": "No active session"
        }), 400
    
    # Get session metadata
    metadata = session_state.end()
    
    # Convert events and tags to dicts
    all_event_dicts = [e.to_dict() for e in session_state.all_events]
    ml_tag_dicts = [t.to_dict() for t in session_state.ml_tags]
    
    # Generate patient chart
    chart = patient_chart_json(
        session_metadata=metadata,
        all_events=all_event_dicts,
        ml_tags=ml_tag_dicts,
        exercise_type=metadata["exercise_type"]
    )
    
    # Add metadata to chart
    chart["session_metadata"] = metadata
    
    return jsonify({
        "status": "success",
        "chart": chart
    })


@app.route('/api/session_status', methods=['GET'])
def session_status():
    """Get current session status."""
    return jsonify({
        "is_active": session_state.is_active,
        "exercise_type": session_state.exercise_type,
        "event_count": len(session_state.all_events),
        "ml_tag_count": len(session_state.ml_tags),
        "duration": int(time.time() - session_state.start_time) if session_state.start_time else 0
    })


# ==================== SOCKET.IO EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("Client connected")
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")


@socketio.on('start_session')
def handle_start_session(data):
    """Handle session start from frontend."""
    exercise_type = data.get('exercise_type', 'squat')
    session_state.start(exercise_type=exercise_type)
    emit('session_started', {
        'exercise_type': exercise_type,
        'timestamp': time.time()
    })


@socketio.on('end_session')
def handle_end_session(data):
    """Handle session end from frontend."""
    if not session_state.is_active:
        emit('error', {'message': 'No active session'})
        return
    
    metadata = session_state.end()
    all_event_dicts = [e.to_dict() for e in session_state.all_events]
    ml_tag_dicts = [t.to_dict() for t in session_state.ml_tags]
    
    chart = patient_chart_json(
        session_metadata=metadata,
        all_events=all_event_dicts,
        ml_tags=ml_tag_dicts,
        exercise_type=metadata["exercise_type"]
    )
    
    chart["session_metadata"] = metadata
    
    emit('session_ended', {'chart': chart})


# ==================== EXAMPLE CV INTEGRATION ====================

def example_cv_integration():
    """
    Example of how to integrate this with your CV loop.
    Call on_metrics_from_cv() from your frame processing function.
    """
    # This is just an example - integrate this into your actual CV loop
    pass
    # Example metrics dict:
    # metrics = {
    #     "exercise_type": "squat",
    #     "reps": 5,
    #     "phase": "down",
    #     "knee_angle": 95.0,
    #     "torso_lean": 25.0,
    #     "zones": {
    #         "depth_zone": "good",
    #         "torso_lean_zone": "acceptable",
    #         "depth_symmetry_zone": "perfect"
    #     },
    #     "cues": [],
    #     "lumbar_excursion": 12.0,
    #     "metrics": {
    #         "knee_angle": 95.0,
    #         "torso_lean": 25.0
    #     }
    # }
    # on_metrics_from_cv(metrics)


# ==================== MAIN ====================

if __name__ == '__main__':
    print("Starting Flask server with Mistral AI integration...")
    print("CV Integration:")
    print("  - Call on_metrics_from_cv(metrics_dict) from your CV loop")
    print("  - Call on_ml_tag(label, score) when ML classifier outputs")
    print("Socket.IO Events:")
    print("  - 'metrics' → live overlays")
    print("  - 'coach_tip' → coaching tips")
    print("  - 'ml_tag' → ML classifier outputs")
    print("API Endpoints:")
    print("  - POST /api/start_session")
    print("  - POST /api/end_session → returns patient chart JSON")
    print("  - GET /api/session_status")
    
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')

