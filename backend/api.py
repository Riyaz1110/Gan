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

# Wrapper class to hold the state since the original script didn't keep global state
class APIState:
    def __init__(self):
        self.df = generate_synthetic_dataset(n_samples=5000) # small for demo
        self.dataset = NetworkFlowDataset(self.df)
        self.model = GANCyberDetector()
        self.is_trained = False
        self.is_training = False
        self.training_progress = {"epoch": 0, "total_epochs": 10, "logs": [], "history": {"d_loss": [], "g_loss": [], "d_acc": []}}
        self.detector = None
        self.threshold = 0.5

    def run_training(self, epochs: int):
        self.is_training = True
        self.training_progress["total_epochs"] = epochs
        self.training_progress["epoch"] = 0
        self.training_progress["history"] = {"d_loss": [], "g_loss": [], "d_acc": []}
        
        train_loader, test_loader, normal_loader = self.dataset.get_loaders(batch_size=256)
        
        for epoch in range(1, epochs + 1):
            stats = self.model.train_epoch(train_loader, normal_loader)
            for k, v in stats.items():
                self.training_progress["history"][k].append(v)
            self.training_progress["epoch"] = epoch
            self.training_progress["logs"].append(f"Epoch {epoch} | D-loss: {stats['d_loss']:.4f} | G-loss: {stats['g_loss']:.4f}")
            time.sleep(0.1)

        # Setup detector using 0.5 threshold so probability comparison works correctly
        self.detector = EnterpriseDetector(self.model, self.dataset.scaler, 0.5)
        
        # We also compute the anomaly threshold purely for the stat display
        scores = self.model.anomaly_score(self.dataset.X_test)
        normal_scores = scores[self.dataset.y_test.astype(int) == 0]
        self.threshold = float(normal_scores.mean() + normal_scores.std())
        
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

    feature_cols = [c for c in state.df.columns if c.startswith("feature_")]
    
    try:
        while True:
            sample = state.df.sample(1)
            row = sample.iloc[0]
            raw = np.array([[row[c] for c in feature_cols]])
            
            res = state.detector.classify(raw)[0]
            
            payload = {
                "sample_id": random.randint(10000, 99999),
                "true_label": "ATTACK" if row["label"] == 1 else "NORMAL",
                "attack_type": row["attack_type"],
                "attack_prob": res["attack_prob"],
                "is_attack": res["is_attack"],
                "severity": res["severity"],
                "features_summary": {
                    "src_port": round(row["feature_0"]),
                    "dst_port": round(row["feature_1"]),
                    "bytes_sent": round(row["feature_2"])
                }
            }
            
            await websocket.send_json(payload)
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
    except WebSocketDisconnect:
        pass
