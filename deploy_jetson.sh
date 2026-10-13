#!/usr/bin/env bash
# ==============================================================================
# Jetson Hardware Setup and TensorRT Engine Compilation Script
# Target: Jetson Orin Nano / Orin NX / AGX Orin / Xavier NX (JetPack 5.x / 6.x)
# ==============================================================================

set -e

echo "===================================================================="
echo "    NVIDIA JETSON EMBEDDED SYSTEM CONFIGURATION (IP-MOT)"
echo "===================================================================="

# 1. Lock Jetson to Maximum Clock Frequencies (Removes latency jitter)
echo "[1/4] Setting Jetson Power Mode to MAXN & Locking Hardware Clocks..."
sudo nvpmodel -m 0          # MAXN power mode (maximum wattage)
sudo jetson_clocks          # Locks CPU, GPU, and EMC memory clocks to max freq

# 2. Install Jetson Python Dependencies
echo "[2/4] Installing Embedded Dependencies..."
pip3 install -r requirements.txt
pip3 install ultralytics

# 3. Compile YOLOv8 Detector to TensorRT FP16 Engine
echo "[3/4] Compiling YOLOv8 to TensorRT FP16 Engine (half precision)..."
if [ ! -f "yolov8n.engine" ]; then
    yolo export model=yolov8n.pt format=engine half=True device=0
    echo "TensorRT engine yolov8n.engine created successfully!"
else
    echo "yolov8n.engine already compiled."
fi

# 4. Optional: Compile Deep ReID Model with trtexec
# trtexec --onnx=models/reid_osnet.onnx --saveEngine=models/reid_osnet_fp16.engine --fp16 --workspace=1024

echo "[4/4] Hardware Setup Complete!"
echo "To run with USB Camera:           python3 scripts/jetson_pipeline.py --source 0"
echo "To run with Jetson CSI Camera:    python3 scripts/jetson_pipeline.py --csi"
echo "To run on Video Stream (Headless): python3 scripts/jetson_pipeline.py --source input.mp4 --headless --record output.mp4"
echo "===================================================================="
