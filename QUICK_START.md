# Quick Start Guide

## Setup (5 minutes)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment:**
```bash
# Copy the template
cp env_template.txt .env

# Edit .env and add your Mistral API key
# Get it from: https://console.mistral.ai/
```

3. **Run the server:**
```bash
python app.py
```

Server will start on `http://localhost:5000`

## Integration with Your CV Loop

### Step 1: Import the function
```python
from app import on_metrics_from_cv
```

### Step 2: Call it from your frame processing
```python
# In your CV loop (e.g., thisone.py generate_frames function)
def generate_frames():
    while True:
        # ... your CV processing ...
        
        # After processing metrics, call:
        metrics = {
            "exercise_type": "squat",  # or "wallsit", "curl"
            "reps": squat_counter.reps,
            "phase": phase,  # "top", "down", "bottom", "up"
            "knee_angle": knee_angle,
            "torso_lean": torso_lean_deg,
            "zones": {
                "depth_zone": m["depth_zone"],
                "torso_lean_zone": m["torso_lean_zone"],
            },
            "cues": cues,  # List of form issues
        }
        
        on_metrics_from_cv(metrics)
```

### Step 3: Add ML tags (optional)
```python
from app import on_ml_tag

# When ML classifier outputs:
on_ml_tag("poor_form", 0.85)
```

## API Integration (For Your Frontend)

### Socket.IO Events (Listen For)

Your frontend should listen to these Socket.IO events:

```javascript
socket.on('metrics', (data) => {
    // Update your metrics display
    console.log('Metrics:', data);
});

socket.on('coach_tip', (data) => {
    // Show coaching tip to user
    showToast(data.tip);
});

socket.on('ml_tag', (data) => {
    // Handle ML tag
    console.log('ML Tag:', data);
});

socket.on('session_started', (data) => {
    // Session started
    console.log('Session started:', data);
});

socket.on('session_ended', (data) => {
    // Display patient chart
    console.log('Patient Chart:', data.chart);
});
```

### REST API Endpoints (Call From Frontend)

**Start Session:**
```javascript
fetch('/api/start_session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ exercise_type: 'squat' })
});
```

**End Session (get chart):**
```javascript
fetch('/api/end_session', {
    method: 'POST'
})
.then(response => response.json())
.then(data => {
    const chart = data.chart;
    // Display chart
});
```

## Testing

1. Start the server: `python app.py`
2. Server runs on `http://localhost:5000`
3. Integrate `on_metrics_from_cv()` into your CV loop
4. Your frontend can connect via Socket.IO and call API endpoints
5. Test with your frontend or use curl/Postman for API endpoints

## What You Get

1. **Real-time Coaching Tips:**
   - Debounced (800ms) to avoid spamming
   - Context-aware based on recent metrics
   - Short, actionable cues

2. **Patient Chart (JSON):**
   - Overview of session
   - Exercise summary (reps, quality)
   - Risk flags
   - Personalized cues
   - Follow-up metrics

## File Structure

```
.
├── app.py              # Flask server + API integration points
├── llm.py              # Mistral AI logic
├── requirements.txt    # Dependencies
├── env_template.txt    # Environment template
├── INTEGRATION_GUIDE.md  # Detailed guide
└── QUICK_START.md      # This file
```

## Next Steps

1. Integrate `on_metrics_from_cv()` into your CV loop
2. Connect your frontend to Socket.IO events and API endpoints
3. Add ML tag integration when your ML classifier is ready
4. Customize prompts in `llm.py` if needed

## Troubleshooting

**"MISTRAL_API_KEY not found":**
- Make sure you created `.env` file
- Add `MISTRAL_API_KEY=your_key_here` to `.env`

**No coaching tips appearing:**
- Check that `on_metrics_from_cv()` is being called
- Check console for errors
- Verify API key is valid

**Patient chart is empty:**
- Make sure session was started
- Check that metrics were sent during session
- Check console for errors

## Support

See `INTEGRATION_GUIDE.md` for detailed documentation.

