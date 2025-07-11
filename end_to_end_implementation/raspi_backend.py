from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import asyncio
import time
from raspi_inference import get_latest_result, start_recording, stop_recording, get_latest_spectrogram
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
#recording_active = False

app = FastAPI()

# Enable CORS to allow access f
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Track active connections
active_connections = []
streaming = False
stream_interval = 5.0

############ Test EndPoints###############
@app.get("/test/audio")
async def test_audio_module():
    """Test audio recording functionality""" 
    global recording_active, recording_thread, audio_buffer,latest_spectrogram
    try:
        # Test recording start/stop
        start_recording()
        await asyncio.sleep(5)  # Record for 2 seconds
        stop_recording()
        
        # Check if buffer has data
        
        result = get_latest_result()
        print(result)
        if result["label"] == "Initializing...":
            return {"status": "error", "message": "No audio data captured"}
        
        return {"status": "success", "message": "Audio module working", "sample_result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/test/inference")
async def test_inference():
    """Test inference functionality"""
    global recording_active, recording_thread, audio_buffer, latest_spectrogram, latest_result
    try:
        start_recording()
        await asyncio.sleep(5)  # Record for 2 seconds
        stop_recording()
        result = get_latest_result()
        if not result or "label" not in result:
            return {"status": "error", "message": "Inference not producing valid results"}
        
        return {
            "status": "success", 
            "inference_working": True,
            "sample_output": {
                "class": result["class"],
                "label": result["label"],
                "confidence": round(result.get("score", 0.0), 3)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.get("/test/spectrogram")
async def test_spectrogram():
    """Test spectrogram generation"""
    global recording_active, recording_thread, audio_buffer, latest_spectrogram
    try:
        start_recording()
        await asyncio.sleep(5)  # Record for 2 seconds
        stop_recording()
        spectrogram = get_latest_spectrogram()
        if not spectrogram or "data" not in spectrogram:
            return {"status": "error", "message": "Spectrogram generation failed"}
        
        return {
            "status": "success",
            "spectrogram_working": True,
            "data_points": len(spectrogram["data"]),
            "min_value": spectrogram["min"],
            "max_value": spectrogram["max"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.get("/test/websocket")
async def test_websocket():
    """Verify WebSocket connectivity (manual test required)"""
    return {
        "status": "manual_test_required",
        "instructions": "Connect to ws://[your-ip]:8000/ws to test WebSocket functionality",
        "expected_behavior": "Should receive periodic updates with inference results"
    }
# #####################################
# app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/spectrogram")
# def get_spectrogram():
#     return FileResponse("static/spectrogram.png", media_type="image/png")
@app.get("/")
async def root():

    return {"status": "ok", "message": "Audio Inference Service is running"}

@app.get("/inference")
def get_prediction():
    global recording_active, recording_thread, audio_buffer, latest_spectrogram
    result = get_latest_result()
    return {
        "class": result.get("class"),
        "label": result.get("label"),
        "confidence": round(result.get("score", 0.0), 3)
    }
@app.post("/start")
async def start_audio_recording():
    global recording_active, recording_thread, audio_buffer, latest_spectrogram
    global streaming
    start_recording()
    streaming = True
    return {"status": "recording_started"}

@app.post("/stop")
async def stop_audio_recording():
    global recording_active, recording_thread, audio_buffer, latest_spectrogram
    global streaming
    stop_recording()
    streaming = False
    return {"status": "recording_stopped"}

@app.post("/interval/{seconds}")
async def set_interval(seconds: float):
    global recording_active, recording_thread, audio_buffer, latest_spectrogram
    global stream_interval
    stream_interval = max(0.5, min(10.0, seconds))  # Limit between 0.5 and 10 seconds
    return {"status": "success", "interval": stream_interval}

@app.get("/spectrogram")
async def get_spectrogram():
    global recording_active, recording_thread, audio_buffer, latest_spectrogram
    latest_spectrogram = get_latest_spectrogram()
    if latest_spectrogram:
        print(latest_spectrogram)
        return latest_spectrogram
    return {"error": "No spectrogram data available"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Check if we should be streaming
            if streaming:
                # Get latest result
                result = get_latest_result()
                spectrogram=get_latest_spectrogram()
                # Handle None cases
                if result is None:
                    result = {"class": None, "label": "Processing", "score": 0.0}
                if spectrogram is None:
                    spectrogram = {"data": [], "min": 0, "max": 1}
                # Send result to client
                await websocket.send_json({
                    "class": result.get("class"),
                    "label": result.get("label","Unknown"),
                    "confidence": round(result.get("score", 0.0), 3),
                    "timestamp": time.time(),
                    "spectrogram": spectrogram
                })
            
            # Wait for the specified interval
            await asyncio.sleep(stream_interval)
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run("raspi_backend:app", host="0.0.0.0", port=8000, reload=False)