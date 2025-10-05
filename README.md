# 🚗 ADAS Lane Detection System

### Lane Detection System for Advanced Driver Assistance Systems (ADAS) using Python and OpenCV  
Includes **curved lane detection** and **ego-vehicle trajectory prediction line** for Lane Keep Assist visualization.

---

## 🧠 Project Overview

This project implements a real-time **Lane Detection System** for ADAS.  
It detects lane boundaries from a front-facing camera video, fits **curved lane polynomials**, and predicts the **vehicle’s center trajectory**.  
The system mimics functions used in:
- Lane Departure Warning (LDW)
- Lane Keep Assist (LKA)
- Autonomous driving perception modules

---

## ⚙️ Tech Stack

- **Language:** Python 3.x  
- **Libraries:** OpenCV, NumPy  
- **Dataset:** Custom dashcam video (`11367262-hd_1920_1080_60fps.mp4`)

---

## 🧩 Key Features

✅ Real-time lane detection using Canny edges  
✅ ROI (Region of Interest) masking to focus on road area  
✅ Polynomial fitting for curved lanes  
✅ Center trajectory prediction line (ego path)  
✅ Output video saved as `lane_output_centerline.mp4`

---

## 🪜 Steps to Run

```bash
# 1️⃣ Clone the repository
git clone https://github.com/abhishekjumale/ADAS-Lane-Detection.git
cd ADAS-Lane-Detection

# 2️⃣ Install dependencies
pip install opencv-python numpy

# 3️⃣ Run the lane detection script
python lane_detection_poly_centerline.py
