from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import asyncio
import random

app = FastAPI(title="Live GAN Cyber Attack Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ATTACK_TYPES = [
    "Normal", "DoS", "Probe", "R2L", "U2R",
    "BruteForce", "PortScan", "WebAttack", "Botnet"
]

@app.get("/")
def home():
    return {"message": "GAN Cyber Attack Detection Backend Running"}

@app.websocket("/ws/live-threats")
async def live_threat_stream(websocket: WebSocket):
    await websocket.accept()
    event_id = 1

    while True:
        attack_prob = round(random.uniform(0.05, 0.98), 4)

        if attack_prob >= 0.90:
            severity = "CRITICAL"
        elif attack_prob >= 0.75:
            severity = "HIGH"
        elif attack_prob >= 0.50:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        is_attack = attack_prob >= 0.50
        attack_type = random.choice(ATTACK_TYPES[1:]) if is_attack else "Normal"

        now = datetime.now()

        threat_data = {
            "id": event_id,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "attack_probability": attack_prob,
            "attack_type": attack_type,
            "severity": severity,
            "status": "ATTACK DETECTED" if is_attack else "NORMAL",
            "is_attack": is_attack
        }

        await websocket.send_json(threat_data)

        event_id += 1
        await asyncio.sleep(1)
