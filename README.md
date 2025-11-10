# BEADDA

**Team:** Bayan Shayab, Eshal Mir, Aahil Ansari, Daniel Tran, Dylan Sheen, Ayman Sheikh

**Project Description:**  
BEADDA is an AI-powered physiotherapy rehabilitation platform designed to remotely monitor, analyze, and optimize patient recovery. It integrates multimodal data streams, including EEG (brain activity), EMG (muscle activation), and computer-vision-based motion tracking, to assess both physical form and cognitive engagement during rehabilitation exercises.  

---

## Table of Contents
1. [Motivation](#motivation)  
2. [Features](#features)  
3. [Architecture & Data Flow](#architecture--data-flow)  
4. [Installation & Setup](#installation--setup)  
5. [Usage](#usage)  
6. [Challenges & Learnings](#challenges--learnings)  
7. [Future Work](#future-work)  
8. [Acknowledgements](#acknowledgements)  

---

## Motivation
Recovering from injuries often requires extensive physiotherapy, but patients struggle with adherence to home exercises. Clinicians lack objective ways to monitor performance remotely. BEADDA bridges this gap by providing real-time, actionable feedback and continuous tracking, accelerating recovery and enhancing patient engagement.

---

## Features
- **Real-time pose estimation** using MediaPipe for exercises like squats, curls, and wall-sits  
- **EMG monitoring** for muscle activation  
- **EEG monitoring** for neural engagement, focus, and fatigue  
- **AI-powered coaching feedback** using Mistral to summarize cues into actionable guidance  
- **Interactive dashboard** for clinicians to visualize patient progress and metrics  
- **Automatic set/session tracking** and history storage via Firebase  

---

## Architecture & Data Flow
**Multimodal streams are processed and synchronized before visualization:**  

- **EEG:** EEG → LSL Stream → Python Processing → Dashboard  
- **EMG:** EMG → LSL Stream → Python Processing → Dashboard  
- **Camera:** Camera → MediaPipe → Rep Detection → Dashboard  

**Overview Diagram:**  
[Patient Sensors] → [Python Processing & AI] → [Dashboard / Clinician Interface]

## Key Components
- Python backend using Flask and SocketIO  
- MediaPipe for pose detection and rep tracking  
- EMG/EEG signal processing using `pylsl` and NumPy  
- Firebase Firestore for data storage  
- Mistral AI for generating concise coaching feedback  

---
## Key Components
- Python backend using Flask and SocketIO  
- MediaPipe for pose detection and rep tracking  
- EMG/EEG signal processing using `pylsl` and NumPy  
- Firebase Firestore for data storage  
- Mistral AI for generating concise coaching feedback  

---

## Installation & Setup
1. **Clone the repository:**  
```bash
git clone <repo_url>
cd BEADDA
```
2. **Create a virtual environment and activate it:**
```
python -m venv venv
venv\Scripts\activate
```
3. **Install dependencies:**
```
pip install -r requirements.txt
```
4. **Setup environment variables:**
Create a .env file with:
```
MISTRAL_API_KEY=<your_api_key>
```
5. **Run the server:**
```
python thisone.py
```
---
## Usage
- Start the server and open the web interface
- Connect EEG/EMG devices via LSL streams
- Select an exercise (squat, curl, wallsit)
- Begin a session and follow on-screen coaching cues
- Real-time feedback and metrics appear on the clinician dashboard
- Finish sets and sessions to save data to Firebase

## Challenges & Learnings
- Integrating multiple hardware streams (EEG, EMG, Camera) and synchronizing them
- Maintaining real-time performance while processing high-frequency physiological data
- Designing an intuitive dashboard for non-technical users
- Signal processing and visualization for EMG and EEG in Python

## Future Work
- Expand to scalable clinical tool with personalized rehab recommendations
- Integrate with telehealth platforms
- Support elderly and pediatric users with adaptive exercises
- Incorporate additional AI insights for predictive recovery analytics
  
## Acknowledgements
- OpenBCI for EEG/EMG hardware
- MediaPipe for pose estimation
- Firebase for backend data storage
- Mistral AI for feedback summarization











---

