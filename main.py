from src.detector import WasteDetectorYOLO

def main():
    print("=" * 70)
    print("  AI-Powered Waste Detection & Sorting using YOLOv8")
    print("  Publication: 4th IEEE ICSSAS 2026")
    print("=" * 70)

    detector = WasteDetectorYOLO(confidence_thresh=0.50)

    print("[*] Processing conveyor belt visual inspection stream (Frame #1042):\n")
    detections = detector.detect_and_classify(frame_id=1042)

    for i, item in enumerate(detections, 1):
        print(f"  [Detected Object #{i}]")
        print(f"    - Item Name         : {item['item_name']}")
        print(f"    - Classification    : {item['category']}")
        print(f"    - Confidence Score  : {item['confidence']*100:.1f}%")
        print(f"    - Actuator Target   : X={item['actuator_target_xy'][0]} px, Y={item['actuator_target_xy'][1]} px")
        print(f"    - Routed Sorter Bin : [{item['sorting_bin']}]\n")

    print("=" * 70)
    print("  PERFORMANCE BENCHMARKS (From Published Paper)")
    print("=" * 70)
    print("  Mean Average Precision (mAP@0.5) : 92.4%")
    print("  Inference Speed                  : 42 FPS on NVIDIA GPU (Real-Time)")
    print("  Sorting Accuracy                 : 95.8% Correct Bin Deflection")
    print("=" * 70)

if __name__ == "__main__":
    main()
