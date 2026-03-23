# PyTorch & YOLO Model Inspector

A lightweight Python utility to inspect, verify, and extract metadata from `.pt` (PyTorch) and YOLO model files. This tool helps developers quickly understand model architectures, check parameter counts, verify file integrity via SHA256, and list class names.

## 🚀 Features

* **Integrity Verification:** Generates a SHA256 hash for model version tracking.
* **Deep PyTorch Inspection:** Extracts `state_dict` keys, total parameter counts, and object types.
* **YOLO/Ultralytics Integration:** If `ultralytics` is installed, it fetches:
    * Class names and counts.
    * Model task (detect, segment, etc.).
    * Trainable vs. Total parameter counts.
* **Flexible Output:** Prints a human-readable summary and can export a detailed `.json` report.

## 📋 Prerequisites

* **Python 3.8+**
* **PyTorch:** For deep inspection of `.pt` files.
* **Ultralytics (Optional):** For YOLO-specific metadata.

```bash
pip install -r requirements.txt
