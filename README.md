# AI-Powered Waste Detection and Sorting using Object Detection

[![Conference](https://img.shields.io/badge/Conference-4th_IEEE_ICSSAS_2026-blue.svg)](https://icssas.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8_Ultralytics-orange.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Presented at the **4th International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS 2026)**.  
*IEEE Xplore Part Number: CFP26DN7-ART; ISBN: 979-8-3315-7009-5*

---

## 👥 Authors
- **A Vibhas Reddy** ([@VIBHAS777](https://github.com/VIBHAS777))
- **Hima Sagar B**
- **S Pavan Kumar**

---

## ♻️ Overview
Manual municipal waste sorting is hazardous, slow, and economically inefficient. Automated recycling facilities require low-latency visual detectors capable of identifying overlapping, deformed items on high-speed conveyor belts.

This project presents an **end-to-end automated waste segregation system** built on **YOLOv8**. The model accurately detects and categorizes recyclable and municipal waste streams into:
- **Biodegradable** (Organic food scraps, paper, cardboard).
- **Non-Biodegradable** (Plastics, metals, glasses).

Bounding box centroids are dynamically mapped to robotic actuator and pneumatic blow-off deflection targets in real time.

---

## 📁 Repository Structure
```
waste-detection-sorting-yolov8/
├── paper/
│   └── AI-Powered_Waste_Detection_and_Sorting_using_Object_Detection.pdf
├── src/
│   └── detector.py        # YOLOv8 waste detection & actuator routing engine
├── main.py                # Visual conveyor inspection stream demo
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start
```bash
git clone https://github.com/VIBHAS777/waste-detection-sorting-yolov8.git
cd waste-detection-sorting-yolov8
pip install -r requirements.txt
python main.py
```

---

## 📜 Citation
```bibtex
@inproceedings{reddy2026wastedetection,
  title={AI-Powered Waste Detection and Sorting using Object Detection},
  author={Reddy, A Vibhas and Sagar B, Hima and Kumar, S Pavan},
  booktitle={4th International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS)},
  year={2026}
}
```
