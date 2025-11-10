# Mistral AI Integration Guide

## Overview

This skeleton provides:
1. **Real-time AI coaching tips** during workouts
2. **End-of-session patient charts** (JSON) for display/saving

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment:**
```bash
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

3. **Get Mistral API key:**
   - Sign up at https://console.mistral.ai/
   - Create an API key
   - Add it to `.env` file

## Integration Points

### 1. CV Loop Integration

In your CV processing loop (e.g., `thisone.py`), call `on_metrics_from_cv()` each frame:

```python
from app import on_metrics_from_cv

# In your frame processing function:
def process_frame(frame, pose_results):
    # ... your CV processing ...
    
    # Build metrics dict
    metrics = {
        "exercise_type": "squat",  # or "wallsit", "curl"
        "reps": squat_counter.reps,
        "phase": phase,  # "top", "down", "bottom", "up"
        "knee_angle": knee_angle,  # for squats/wallsits
        "elbow_angle": elbow_angle,  # for curls
        "torso_lean": torso_lean_deg,
        "lumbar_excursion": lumbar_excursion,
        "zones": {
            "depth_zone": m["depth_zone"],  # "perfect", "good", "acceptable", "poor"
            "torso_lean_zone": m["torso_lean_zone"],
            # ... other zones
        },
        "cues": cues,  # List of form issues, e.g., ["Go deeper", "Stay upright"]
        "metrics": {
            # Any additional metrics
        }
    }
    
    # Send to AI coaching system
    on_metrics_from_cv(metrics)
```

### 2. ML Tag Integration

When your ML classifier outputs a label, call `on_ml_tag()`:

```python
from app import on_ml_tag

# When ML classifier outputs:
on_ml_tag("poor_form", 0.85)
on_ml_tag("good_depth", 0.92)
on_ml_tag("risk_injury", 0.78)
```

### 3. Session Management

**Start Session:**
```javascript
// Frontend
socket.emit('start_session', { exercise_type: 'squat' });

// Or API
fetch('/api/start_session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ exercise_type: 'squat' })
});
```

**End Session (get patient chart):**
```javascript
// Frontend
socket.emit('end_session', {});

// Or API
fetch('/api/end_session', {
    method: 'POST'
})
.then(response => response.json())
.then(data => {
    const chart = data.chart;
    // Display chart in UI
    displayPatientChart(chart);
});
```

## Data Structures

### MetricEvent

```python
{
    "timestamp": 1234567890.123,
    "exercise_type": "squat",
    "reps": 5,
    "phase": "down",
    "knee_angle": 95.0,
    "torso_lean": 25.0,
    "lumbar_excursion": 12.0,
    "zones": {
        "depth_zone": "good",
        "torso_lean_zone": "acceptable"
    },
    "cues": ["Go deeper"],
    "metrics": {}
}
```

### MLTag

```python
{
    "label": "poor_form",
    "score": 0.85,
    "timestamp": 1234567890.123
}
```

### Patient Chart (JSON)

```json
{
    "overview": "Session summary...",
    "exercises": [
        {
            "name": "squat",
            "reps": 10,
            "quality": "good",
            "notes": "Good form overall"
        }
    ],
    "risk_flags": ["Slight forward lean"],
    "personalized_cues": [
        "Focus on depth",
        "Keep chest up"
    ],
    "follow_up_metrics": [
        "Track knee angle consistency",
        "Monitor torso lean"
    ],
    "session_metadata": {
        "session_id": "session_1234567890",
        "user_id": "user_001",
        "duration": 300,
        "timestamp": "2024-01-01T12:00:00"
    }
}
```

## API Endpoints

- `POST /api/start_session` - Start a new session
- `POST /api/end_session` - End session and get patient chart
- `GET /api/session_status` - Get current session status

## Socket.IO Events

**Client → Server:**
- `start_session` - Start session
- `end_session` - End session

**Server → Client:**
- `metrics` - Real-time metrics
- `coach_tip` - AI coaching tips
- `ml_tag` - ML classifier outputs
- `session_started` - Session started confirmation
- `session_ended` - Session ended with chart

## Running

```bash
python app.py
```

Server runs on `http://localhost:5000`

## Notes

- **Debouncing:** Coaching tips are debounced (800ms) to avoid spamming the LLM
- **Partial Data:** All fields are optional - works with partial data as ML/CV components are developed
- **Error Handling:** Falls back to safe JSON if LLM fails
- **Session State:** Maintains session state across CV loop calls

## Example Integration

See `app.py` for example CV integration in the `example_cv_integration()` function (commented out).

## Frontend Integration (Your Implementation)

Your frontend should listen to Socket.IO events and call API endpoints:

**Socket.IO Events (listen for):**
- `metrics` - Real-time metrics for overlays
- `coach_tip` - AI coaching tips
- `ml_tag` - ML classifier outputs
- `session_started` - Session started confirmation
- `session_ended` - Session ended with chart

**API Endpoints (call from frontend):**
- `POST /api/start_session` - Start session
- `POST /api/end_session` - End session and get chart
- `GET /api/session_status` - Get session status

