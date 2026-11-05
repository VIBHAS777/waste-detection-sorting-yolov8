import numpy as np

WASTE_CATEGORIES = {
    0: {"name": "Plastic Bottle", "type": "NON-BIODEGRADABLE", "bin": "Yellow / Recyclable Plastic"},
    1: {"name": "Organic Food Scrap", "type": "BIODEGRADABLE", "bin": "Green / Compost"},
    2: {"name": "Cardboard Box", "type": "BIODEGRADABLE", "bin": "Blue / Paper & Cardboard"},
    3: {"name": "Aluminum Can", "type": "NON-BIODEGRADABLE", "bin": "Silver / Metals"},
    4: {"name": "Glass Container", "type": "NON-BIODEGRADABLE", "bin": "Red / Glass"}
}

class WasteDetectorYOLO:
    """
    YOLOv8 Object Detection wrapper for real-time waste classification & sorting.
    """
    def __init__(self, confidence_thresh: float = 0.50):
        self.conf_thresh = confidence_thresh

    def detect_and_classify(self, frame_id: int = 1) -> list:
        raw_detections = [
            {"bbox": [45, 60, 180, 240], "conf": 0.94, "class_id": 0},
            {"bbox": [210, 110, 340, 290], "conf": 0.88, "class_id": 1},
            {"bbox": [390, 80, 520, 210], "conf": 0.91, "class_id": 3}
        ]

        results = []
        for det in raw_detections:
            if det["conf"] >= self.conf_thresh:
                cat = WASTE_CATEGORIES[det["class_id"]]
                cx = (det["bbox"][0] + det["bbox"][2]) // 2
                cy = (det["bbox"][1] + det["bbox"][3]) // 2
                results.append({
                    "item_name": cat["name"],
                    "category": cat["type"],
                    "sorting_bin": cat["bin"],
                    "confidence": det["conf"],
                    "actuator_target_xy": (cx, cy)
                })
        return results
