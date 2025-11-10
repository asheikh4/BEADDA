# API Reference

## Overview

This is a **backend-only API skeleton** for Mistral AI integration. No frontend is included - integrate with your own frontend.

## Integration Points

### 1. CV Loop Integration

**Function:** `on_metrics_from_cv(metrics: Dict)`

Call this from your CV processing loop each frame:

```python
from app import on_metrics_from_cv

# In your CV loop:
metrics = {
    "exercise_type": "squat",  # or "wallsit", "curl"
    "reps": 5,
    "phase": "down",  # "top", "down", "bottom", "up"
    "knee_angle": 95.0,  # for squats/wallsits
    "elbow_angle": 45.0,  # for curls
    "torso_lean": 25.0,
    "lumbar_excursion": 12.0,
    "zones": {
        "depth_zone": "good",  # "perfect", "good", "acceptable", "poor"
        "torso_lean_zone": "acceptable"
    },
    "cues": ["Go deeper"],  # List of form issues
    "metrics": {}  # Any additional metrics
}

on_metrics_from_cv(metrics)
```

### 2. ML Tag Integration

**Function:** `on_ml_tag(label: str, score: float)`

Call this when your ML classifier outputs:

```python
from app import on_ml_tag

on_ml_tag("poor_form", 0.85)
on_ml_tag("good_depth", 0.92)
```

## REST API Endpoints

### POST `/api/start_session`

Start a new exercise session.

**Request:**
```json
{
    "exercise_type": "squat"  // "squat", "wallsit", or "curl"
}
```

**Response:**
```json
{
    "status": "success",
    "message": "Session started for squat",
    "session_id": 1234567890.123
}
```

### POST `/api/end_session`

End the current session and get patient chart.

**Request:**
```json
{}
```

**Response:**
```json
{
    "status": "success",
    "chart": {
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
            "Track knee angle consistency"
        ],
        "session_metadata": {
            "session_id": "session_1234567890",
            "user_id": "user_001",
            "duration": 300,
            "timestamp": "2024-01-01T12:00:00"
        }
    }
}
```

### GET `/api/session_status`

Get current session status.

**Response:**
```json
{
    "is_active": true,
    "exercise_type": "squat",
    "event_count": 150,
    "ml_tag_count": 5,
    "duration": 45
}
```

## Socket.IO Events

### Client → Server

**`start_session`**
```javascript
socket.emit('start_session', { exercise_type: 'squat' });
```

**`end_session`**
```javascript
socket.emit('end_session', {});
```

### Server → Client

**`metrics`** - Real-time metrics for overlays
```javascript
socket.on('metrics', (data) => {
    // data contains: timestamp, exercise_type, reps, phase, angles, zones, cues, etc.
});
```

**`coach_tip`** - AI coaching tips (debounced ~800ms)
```javascript
socket.on('coach_tip', (data) => {
    // data.tip contains the coaching message
    // data.timestamp, data.exercise_type
});
```

**`ml_tag`** - ML classifier outputs
```javascript
socket.on('ml_tag', (data) => {
    // data.label, data.score, data.timestamp
});
```

**`session_started`** - Session started confirmation
```javascript
socket.on('session_started', (data) => {
    // data.exercise_type, data.timestamp
});
```

**`session_ended`** - Session ended with patient chart
```javascript
socket.on('session_ended', (data) => {
    // data.chart contains the full patient chart JSON
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
    "elbow_angle": None,  # or value for curls
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

### Patient Chart

```json
{
    "overview": "Session summary text...",
    "exercises": [
        {
            "name": "squat",
            "reps": 10,
            "quality": "good",
            "notes": "Good form overall"
        }
    ],
    "risk_flags": ["flag1", "flag2"],
    "personalized_cues": ["cue1", "cue2"],
    "follow_up_metrics": ["metric1", "metric2"],
    "session_metadata": {
        "session_id": "session_1234567890",
        "user_id": "user_001",
        "exercise_type": "squat",
        "duration": 300,
        "timestamp": "2024-01-01T12:00:00",
        "total_events": 150,
        "total_ml_tags": 5
    }
}
```

## Notes

- **All fields are optional** - Works with partial data as components develop
- **Debouncing:** Coaching tips are debounced (800ms) to avoid spamming LLM
- **Error handling:** Falls back to safe JSON if LLM fails
- **CORS:** Socket.IO is configured with `cors_allowed_origins="*"` - adjust for production

## Running

```bash
python app.py
```

Server runs on `http://localhost:5000`

