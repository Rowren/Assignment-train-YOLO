"""FastAPI webapp — PolyDetect (YOLO segmentation) + Tailwind landing page."""

import base64
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from ultralytics import YOLO

ROOT = Path(__file__).parent
_CANDIDATES = [
    ROOT / "best_poly.pt",
    ROOT / "best_poly_segment.pt",
    ROOT.parent / "weights" / "best_poly.pt",
    ROOT.parent / "runs" / "segment" / "animals_poly" / "weights" / "best.pt",
]
WEIGHTS = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

app = FastAPI(title="PolyDetect", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- model ----------
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS))
    return _model

CLASSES = ["Bird", "Cat", "Dog", "Horse", "Cow"]
COLORS_BGR = {
    "bird":  (255, 80, 80),
    "cat":   (60, 180, 75),
    "dog":   (255, 140, 0),
    "horse": (0, 130, 255),
    "cow":   (200, 50, 200),
}

def draw_masks(image_bgr, result, conf_thres, mode="both"):
    out = image_bgr.copy()
    overlay = out.copy()
    detections = []
    names = result.names
    draw_mask = mode in ("mask", "both")
    draw_box = mode in ("box", "both")
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            if conf < conf_thres:
                continue
            cls_id = int(box.cls[0])
            name = names.get(cls_id, str(cls_id))
            color = COLORS_BGR.get(name.lower(), (200, 200, 200))
            if draw_mask and result.masks is not None and i < len(result.masks.xy):
                pts = result.masks.xy[i].astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(out, [pts], True, color, 2)
            x0, y0, x1, y1 = map(int, box.xyxy[0])
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x0, max(0, y0 - th - 8)), (x0 + tw + 4, y0), color, -1)
            cv2.putText(out, label, (x0 + 2, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if draw_box:
                cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            detections.append({
                "class": name,
                "confidence": round(conf, 4),
                "bbox": [x0, y0, x1, y1]
            })
    if draw_mask:
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out, detections

# ---------- pages ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_path = ROOT / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

# ---------- API ----------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    mode: str = Form("both"),
):
    if mode not in ("box", "mask", "both"):
        mode = "both"
    contents = await file.read()
    img_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    model = get_model()
    result = model.predict(img, imgsz=640, device="cpu", verbose=False)[0]
    vis, detections = draw_masks(img, result, conf, mode)

    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return {
        "image": f"data:image/jpeg;base64,{b64}",
        "detections": detections,
        "count": len(detections),
    }

@app.get("/health")
async def health():
    return {"status": "ok", "weights": str(WEIGHTS), "weights_exists": WEIGHTS.exists()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
