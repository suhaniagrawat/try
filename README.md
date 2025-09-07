
# The Route Cause - AI-Powered Traffic Management System  
**Submission for Smart India Hackathon 2025**  

**Team ID:** 6 | **Problem ID:** 25050 | **Theme:** Transportation & Logistics  

---

## The Problem We Solve  
Traditional traffic lights use static timers, leading to unnecessary congestion, long wait times, and an inability to respond to real-world events like emergency vehicles or sudden traffic jams.  
Our system solves this by bringing intelligent, adaptive control to urban intersections.  

---

## Our Multi-Layered AI Solution  

### Real-Time Perception (The Eyes)  
A YOLOv8 computer vision model provides real-time situational awareness, detecting and counting every vehicle and pedestrian.  

### Intelligent Decisions (The Brain)  
A trained Q-Learning Reinforcement Learning agent analyzes the live data to make strategic decisions on signal timing to minimize congestion.  

### Optimization & Safety (The Supervisor)  
A rule-based Optimization Engine supervises the AI's decisions, enforcing critical safety and fairness rules like emergency overrides and starvation prevention.  

This data is then streamed via a FastAPI backend to a live React dashboard for visualization.  

---

## Website & Dashboard Screenshots  

### Live Traffic Dashboard  
![Traffic Dashboard](images/screenshot-traffic-dashboard.png)  

### Performance Metrics  
![Performance Metrics](images/screenshot-performance-metrics.png)  

### Emergency Override Monitoring  
![Emergency Mode](images/screenshot-emergency-mode.png)  

### Real-Time Detection  
![System Status](images/screenshot-system-status.png)  

---

## Key Features & Innovations  

- **Adaptive AI Control**: Dynamically allocates green lights based on live traffic density.  
- **Emergency Vehicle Preemption**: Executes a professional-grade clearing sequence (yellow -> all-red -> dedicated green) for emergency vehicles.  
- **Dynamic Queue Management**: Intelligently overrides the cycle to service excessively long queues and prevent gridlock.  
- **Fairness Guarantee**: A "starvation" timer prevents any vehicle from waiting indefinitely.  
- **Real-Time Data & Transparency**: All perceptions and decisions are streamed via WebSockets to a live dashboard for complete system transparency.  

---

## Tech Stack  

- **AI & Computer Vision**: Python, OpenCV, PyTorch, YOLOv8, Q-Learning  
- **Backend & Data Pipeline**: Python, FastAPI, WebSockets  
- **Frontend & Visualization**: React.js, TypeScript, Vite, TailwindCSS, Recharts  
- **Deployment**: Vercel (Frontend), Railway (Backend)  

---

## How to Run This Project  

### Prerequisites  
- Python 3.10+  
- Node.js and npm/yarn  
- A video file (`my_video.mp4`) of traffic for the AI to process  

### Backend & AI Agent Setup  
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
python run_live_agent.py
```

### Frontend Setup  
```bash
cd frontend
npm install
npm run dev
```

---
