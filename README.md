# Person_tracking_using_PTZ_camera

WHAT THIS SYSTEM ACHIEVES

Robust RTSP handling
Real-time person tracking
Cross-camera identity preservation
Works even with camera disconnection
Scalable to more cameras

COMPLETE DATA FLOW (ONE PERSON)
 
CAM1 frame
  ↓
YOLO detects person
  ↓
DeepSORT assigns local ID=4
  ↓
Extract embedding
  ↓
CrossTracker → Global ID=1
  ↓
Person exits CAM1
 
CAM2 frame
  ↓
YOLO detects same person
  ↓
DeepSORT assigns local ID=9
  ↓
Embedding similarity > threshold
  ↓
Mapped to Global ID=1
