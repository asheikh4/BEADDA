"""
Mistral AI Integration for Real-time Coaching and Patient Charts
Handles all LLM logic for live coaching tips and end-of-session reports.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from mistralai import Mistral

# Load environment variables
load_dotenv()

# Initialize Mistral client
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env file")

client = Mistral(api_key=MISTRAL_API_KEY)

# Model names
MODEL_LIVE_COACH = "mistral-small-latest"  # Fast, for real-time tips
MODEL_PATIENT_CHART = "mistral-medium-latest"  # More capable, for detailed reports

# Debounce timer for live coaching (to avoid spamming LLM)
_last_coach_call_time = 0
COACH_DEBOUNCE_MS = 800  # milliseconds


def live_coach_tip(
    current_event: Dict[str, Any],
    recent_events: List[Dict[str, Any]],
    exercise_type: str = "squat"
) -> Optional[str]:
    """
    Generate real-time coaching tip from current and recent metrics.
    
    Args:
        current_event: Current frame's metrics (MetricEvent)
        recent_events: List of recent MetricEvent dicts (last 5-10 frames)
        exercise_type: Type of exercise ("squat", "wallsit", "curl")
    
    Returns:
        Short coaching tip (1-2 sentences) or None if debounced/error
    """
    global _last_coach_call_time
    
    # Debounce: only call if enough time has passed
    current_time_ms = time.time() * 1000
    if current_time_ms - _last_coach_call_time < COACH_DEBOUNCE_MS:
        return None
    
    _last_coach_call_time = current_time_ms
    
    try:
        # Build context from recent events
        context_summary = _build_context_summary(recent_events, current_event)
        
        # System prompt for coaching
        system_prompt = f"""You are a physiotherapy assistant providing real-time exercise coaching.
Provide 1-2 short, actionable cues to help the patient improve their {exercise_type} form.
Be specific, encouraging, and focus on the most important correction needed right now.
Keep responses under 50 words."""

        # User prompt with current metrics
        user_prompt = f"""Current {exercise_type} metrics:
{context_summary}

Provide a brief coaching tip based on the current form."""

        # Call Mistral
        response = client.chat.complete(
            model=MODEL_LIVE_COACH,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        tip = response.choices[0].message.content.strip()
        return tip if tip else None
        
    except Exception as e:
        print(f"Error generating coach tip: {e}")
        import traceback
        traceback.print_exc()
        return None


def patient_chart_json(
    session_metadata: Dict[str, Any],
    all_events: List[Dict[str, Any]],
    ml_tags: List[Dict[str, Any]],
    exercise_type: str = "squat"
) -> Dict[str, Any]:
    """
    Generate end-of-session patient chart as structured JSON.
    
    Args:
        session_metadata: Dict with user_id, session_id, duration, timestamp, etc.
        all_events: List of all MetricEvent dicts from the session
        ml_tags: List of MLTag dicts (label, score)
        exercise_type: Type of exercise performed
    
    Returns:
        Structured JSON with overview, exercises, risk_flags, personalized_cues, follow_up_metrics
    """
    try:
        # Aggregate metrics
        aggregates = _aggregate_metrics(all_events, exercise_type)
        
        # Build prompt
        system_prompt = """You are a physiotherapy AI assistant generating a patient session report.
Return ONLY valid JSON with this exact structure:
{
  "overview": "Brief summary of the session (2-3 sentences)",
  "exercises": [
    {
      "name": "exercise name",
      "reps": number,
      "quality": "excellent|good|fair|poor",
      "notes": "brief notes"
    }
  ],
  "risk_flags": ["flag1", "flag2"],
  "personalized_cues": ["cue1", "cue2", "cue3"],
  "follow_up_metrics": ["metric1 to track", "metric2 to track"]
}

Be specific, professional, and actionable. Return ONLY the JSON, no markdown formatting."""

        user_prompt = f"""Session Data:
- Exercise: {exercise_type}
- Duration: {session_metadata.get('duration', 'N/A')} seconds
- User: {session_metadata.get('user_id', 'Unknown')}

Aggregated Metrics:
{aggregates}

ML Tags:
{ml_tags}

Generate the patient chart JSON."""

        # Call Mistral Medium for detailed report
        response = client.chat.complete(
            model=MODEL_PATIENT_CHART,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent JSON
            max_tokens=1000
        )
        
        # Parse JSON response
        try:
            chart_json = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if chart_json.startswith("```"):
                parts = chart_json.split("```")
                if len(parts) > 1:
                    chart_json = parts[1]
                    if chart_json.startswith("json"):
                        chart_json = chart_json[4:]
                    chart_json = chart_json.strip()
            return json.loads(chart_json)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {response.choices[0].message.content[:200]}")
            # Fallback to safe JSON structure
            return _get_fallback_chart(session_metadata, aggregates, ml_tags)
            
    except Exception as e:
        print(f"Error generating patient chart: {e}")
        import traceback
        traceback.print_exc()
        return _get_fallback_chart(session_metadata, aggregates or {}, ml_tags)


def _build_context_summary(recent_events: List[Dict[str, Any]], current: Dict[str, Any]) -> str:
    """Build a compact summary of recent events for context."""
    if not recent_events and not current:
        return "No metrics available"
    
    summary_parts = []
    
    # Current event
    if current:
        if "knee_angle" in current:
            summary_parts.append(f"Knee angle: {current.get('knee_angle', 'N/A')}°")
        if "torso_lean" in current:
            summary_parts.append(f"Torso lean: {current.get('torso_lean', 'N/A')}°")
        if "elbow_angle" in current:
            summary_parts.append(f"Elbow angle: {current.get('elbow_angle', 'N/A')}°")
        if "reps" in current:
            summary_parts.append(f"Reps: {current.get('reps', 0)}")
        if "phase" in current:
            summary_parts.append(f"Phase: {current.get('phase', 'N/A')}")
        if "cues" in current:
            cues = current.get("cues", [])
            if cues:
                summary_parts.append(f"Form issues: {', '.join(cues[:3])}")
    
    # Recent trends (simplified)
    if len(recent_events) > 3:
        summary_parts.append(f"Recent trend: {len(recent_events)} frames analyzed")
    
    return "\n".join(summary_parts) if summary_parts else "No metrics"


def _aggregate_metrics(events: List[Dict[str, Any]], exercise_type: str) -> str:
    """Aggregate metrics from all events into a summary string."""
    if not events:
        return "No events recorded"
    
    aggregates = []
    
    # Count reps
    if events:
        last_event = events[-1]
        total_reps = last_event.get("reps", 0)
        aggregates.append(f"Total reps: {total_reps}")
    
    # Average angles (if available)
    angles = []
    for event in events:
        if "knee_angle" in event:
            angles.append(event["knee_angle"])
        elif "elbow_angle" in event:
            angles.append(event["elbow_angle"])
    
    if angles:
        avg_angle = sum(angles) / len(angles)
        angle_name = "knee" if "knee_angle" in events[0] else "elbow"
        aggregates.append(f"Average {angle_name} angle: {avg_angle:.1f}°")
    
    # Form quality (count poor zones)
    poor_count = sum(1 for e in events if any(
        zone in e.get("zones", {}) and e["zones"][zone] == "poor"
        for zone in e.get("zones", {})
    ))
    if poor_count > 0:
        aggregates.append(f"Form issues detected in {poor_count} frames")
    
    return "\n".join(aggregates) if aggregates else "Basic metrics collected"


def _get_fallback_chart(
    session_metadata: Dict[str, Any],
    aggregates: Dict[str, Any],
    ml_tags: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a safe fallback JSON structure if LLM fails."""
    return {
        "overview": f"Session completed. Duration: {session_metadata.get('duration', 'N/A')} seconds.",
        "exercises": [
            {
                "name": session_metadata.get("exercise_type", "unknown"),
                "reps": aggregates.get("reps", 0),
                "quality": "good",
                "notes": "Session data collected successfully."
            }
        ],
        "risk_flags": [],
        "personalized_cues": [
            "Continue practicing proper form",
            "Focus on consistent movement patterns"
        ],
        "follow_up_metrics": [
            "Track exercise consistency",
            "Monitor form improvements"
        ]
    }

