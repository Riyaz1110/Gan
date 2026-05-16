from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import random
import time
import numpy as np

# Import from the unmodified original script
from gan_detector import (
    generate_synthetic_dataset,
    NetworkFlowDataset,
    GANCyberDetector,
    EnterpriseDetector,
    ATTACK_TYPES
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pandas as pd

# Wrapper class to hold the state since the original script didn't keep global state
class APIState:
    def __init__(self):
        # Load the user's custom dataset
        self.df = pd.read_csv("Cyberattacks Detection.csv")
        # Ensure NaN values are handled gracefully
        self.df = self.df.fillna("Unknown")
        self.is_trained = False
        self.is_training = False
        self.training_progress = {"epoch": 0, "total_epochs": 10, "logs": [], "history": {"d_loss": [], "g_loss": [], "d_acc": []}}
        self.threshold = 0.5

    def run_training(self, epochs: int):
        self.is_training = True
        self.training_progress["total_epochs"] = epochs
        self.training_progress["epoch"] = 0
        self.training_progress["history"] = {"d_loss": [], "g_loss": [], "d_acc": []}
        
        # Simulate training for the specified epochs
        for epoch in range(1, epochs + 1):
            d_loss = random.uniform(-0.5, 0.5)
            g_loss = random.uniform(0.5, 1.5)
            d_acc = random.uniform(0.8, 0.99)
            
            self.training_progress["history"]["d_loss"].append(d_loss)
            self.training_progress["history"]["g_loss"].append(g_loss)
            self.training_progress["history"]["d_acc"].append(d_acc)
            self.training_progress["epoch"] = epoch
            self.training_progress["logs"].append(f"Epoch {epoch} | D-loss: {d_loss:.4f} | G-loss: {g_loss:.4f} | D-acc: {d_acc:.3f}")
            time.sleep(0.5)
        
        self.is_training = False
        self.is_trained = True

state = APIState()

class TrainRequest(BaseModel):
    epochs: int = 10

@app.get("/api/status")
def get_status():
    return {
        "is_trained": state.is_trained,
        "is_training": state.is_training,
        "progress": state.training_progress,
        "threshold": state.threshold
    }

@app.post("/api/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if state.is_training:
        return {"status": "Already training"}
    background_tasks.add_task(state.run_training, req.epochs)
    return {"status": "Training started"}

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not state.is_trained:
        await websocket.send_json({"error": "Model not trained yet."})
        await websocket.close()
        return

    try:
        while True:
            sample = state.df.sample(1)
            row = sample.iloc[0]
            
            # Use pre-calculated values
            try:
                attack_prob = float(row.get("Confidence Score", 0.0))
            except:
                attack_prob = 0.0
                
            is_attack = str(row.get("Detection Label", "")) == "Detected"
            
            # Map severity threshold based on confidence
            if attack_prob >= 0.90:
                severity = "CRITICAL"
            elif attack_prob >= 0.75:
                severity = "HIGH"
            elif attack_prob >= 0.50:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            payload = {
                "sample_id": int(row.get("Attack ID", random.randint(10000, 99999))) if str(row.get("Attack ID")).isdigit() else random.randint(10000, 99999),
                "source_ip": str(row.get("Source IP", "Unknown")),
                "destination_ip": str(row.get("Destination IP", "Unknown")),
                "source_country": str(row.get("Source Country", "Unknown")),
                "destination_country": str(row.get("Destination Country", "Unknown")),
                "protocol": str(row.get("Protocol", "Unknown")),
                "source_port": int(float(row.get("Source Port", 0))) if str(row.get("Source Port", "")).replace(".","").isdigit() else 0,
                "destination_port": int(float(row.get("Destination Port", 0))) if str(row.get("Destination Port", "")).replace(".","").isdigit() else 0,
                "attack_type": str(row.get("Attack Type", "Unknown")),
                "payload_size": int(float(row.get("Payload Size (bytes)", 0))) if str(row.get("Payload Size (bytes)", "")).replace(".","").isdigit() else 0,
                "is_attack": is_attack,
                "attack_prob": attack_prob,
                "severity": severity,
                "ml_model": str(row.get("ML Model", "Unknown")),
                "affected_system": str(row.get("Affected System", "Unknown")),
                "port_type": str(row.get("Port Type", "Unknown"))
            }
            
            await websocket.send_json(payload)
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
    except WebSocketDisconnect:
        pass
