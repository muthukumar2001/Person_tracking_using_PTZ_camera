import threading
import cv2
import time
import socket
import json
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from queue import Queue
from collections import defaultdict

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 7000
BUFFER_SIZE = 4096


class ActiveTarget:
    """Tracks the currently selected person across cameras"""
    def __init__(self):
        self.lock = threading.Lock()
        self.global_id = None
        self.last_seen_time = 0
        self.timeout = 10.0  # Increased timeout to 10 seconds
        self.confidence_scores = defaultdict(float)  # Track confidence per camera
    
    def set(self, global_id):
        with self.lock:
            self.global_id = global_id
            self.last_seen_time = time.time()
            self.confidence_scores.clear()
            print(f"🎯 Active target set to Global ID: {global_id}")
    
    def update_seen(self, global_id, confidence=1.0):
        """Update that this ID was seen with confidence score"""
        with self.lock:
            if self.global_id == global_id:
                self.last_seen_time = time.time()
                # Track confidence to handle occlusions
                self.confidence_scores[time.time()] = confidence
                # Keep only recent scores
                cutoff = time.time() - 2.0
                self.confidence_scores = {k: v for k, v in self.confidence_scores.items() if k > cutoff}
    
    def clear(self):
        with self.lock:
            old_id = self.global_id
            self.global_id = None
            self.confidence_scores.clear()
            if old_id:
                print(f"❌ Active target cleared (was Global ID: {old_id})")
    
    def get(self):
        with self.lock:
            # Check if target has timed out
            if self.global_id is not None:
                if time.time() - self.last_seen_time > self.timeout:
                    print(f"⏰ Target Global ID {self.global_id} timed out (not seen for {self.timeout}s)")
                    self.global_id = None
                    self.confidence_scores.clear()
            return self.global_id
    
    def get_avg_confidence(self):
        """Get average recent confidence"""
        with self.lock:
            if not self.confidence_scores:
                return 0.0
            return sum(self.confidence_scores.values()) / len(self.confidence_scores)
    
    def is_active(self):
        with self.lock:
            return self.global_id is not None


class ClickCoordinates:
    """Thread-safe storage for click coordinates"""
    def __init__(self):
        self.lock = threading.Lock()
        self.x = 0
        self.y = 0
        self.camera_ip = None
        self.timestamp = 0
        self.processed = False
    
    def update(self, x, y, camera_ip):
        with self.lock:
            self.x = x
            self.y = y
            self.camera_ip = camera_ip
            self.timestamp = time.time()
            self.processed = False
            print(f"📍 NEW CLICK: Camera={camera_ip}, X={x}, Y={y}")
    
    def get(self):
        with self.lock:
            return self.x, self.y, self.camera_ip, self.timestamp, self.processed
    
    def mark_processed(self):
        with self.lock:
            self.processed = True


class PTZSocketClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.last_send_time = {}
        self.debounce_interval = 0.3  # Reduced for more responsive tracking
        
        self.command_queue = Queue(maxsize=10)
        self.running = True
        
        self.worker_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.worker_thread.start()
        print(f"✅ PTZ client started for {server_ip}:{server_port}")
    
    def _process_commands(self):
        """Background thread that processes PTZ commands from queue"""
        while self.running:
            try:
                try:
                    command = self.command_queue.get(timeout=0.5)
                except:
                    continue
                
                x, y, zoom, global_id, detection_id = command
                
                # Debounce check
                if detection_id is not None:
                    current_time = time.time()
                    if detection_id in self.last_send_time:
                        if current_time - self.last_send_time[detection_id] < self.debounce_interval:
                            self.command_queue.task_done()
                            continue
                    self.last_send_time[detection_id] = current_time
                
                # Send command
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1.0)
                        s.connect((self.server_ip, self.server_port))
                        msg = f"{int(x)},{int(y)},{int(zoom)}\n"
                        s.sendall(msg.encode("utf-8"))
                        print(f"📤 PTZ [{self.server_ip}]: x={int(x)}, y={y}, zoom={zoom}, ID={global_id}")
                        try:
                            s.settimeout(0.1)
                            resp = s.recv(512)
                        except:
                            pass
                except Exception as e:
                    print(f"[PTZ] send error to {self.server_ip}: {e}")
                
                self.command_queue.task_done()
                
            except Exception as e:
                print(f"[PTZ] worker error: {e}")
    
    def send_click(self, x, y=360, zoom=0, global_id=None, detection_id=None):
        """Queue PTZ command for async sending"""
        if self.running:
            try:
                self.command_queue.put_nowait((x, y, zoom, global_id, detection_id))
            except:
                # Queue full, remove oldest and add new
                try:
                    self.command_queue.get_nowait()
                    self.command_queue.put_nowait((x, y, zoom, global_id, detection_id))
                except:
                    pass
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join(timeout=2.0)
        print(f"🛑 PTZ client stopped for {self.server_ip}")


class ClickListener:
    """Background thread that listens for click coordinates"""
    def __init__(self, click_coords, active_target, listen_ip=LISTEN_IP, listen_port=LISTEN_PORT):
        self.click_coords = click_coords
        self.active_target = active_target
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
    
    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.listen_ip, self.listen_port))
        sock.listen(5)
        sock.settimeout(1.0)

        print(f"✅ Listening for click coordinates on {self.listen_ip}:{self.listen_port}")

        while self.running:
            try:
                conn, addr = sock.accept()
                try:
                    data = conn.recv(BUFFER_SIZE)
                    if data:
                        message = data.decode('utf-8')
                        payload = json.loads(message)

                        camera_ip = payload.get("camera_ip")
                        click_x = payload.get("x")
                        click_y = payload.get("y")

                        if click_x is not None and click_y is not None:
                            if self.active_target.is_active():
                                self.active_target.clear()
                            
                            self.click_coords.update(click_x, click_y, camera_ip)

                except Exception as e:
                    print(f"Error processing click data: {e}")
                finally:
                    conn.close()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Socket error: {e}")
                    time.sleep(1)
        
        sock.close()
        print("🛑 Click listener stopped")
    
    def stop(self):
        self.running = False


class RTSPCamera:
    def __init__(self, url, reconnect_delay=3):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.cap = None
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        self.connect()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
    
    def connect(self):
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduced buffer for lower latency
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if self.cap.isOpened():
            print(f"✅ Connected to {self.url}")
        else:
            print(f"❌ Failed to connect to {self.url}")
    
    def update(self):
        consecutive_failures = 0
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"🔄 Reconnecting to {self.url}...")
                self.connect()
                time.sleep(self.reconnect_delay)
                continue
            
            ret, frame = self.cap.read()
            
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            
            if consecutive_failures > 30:
                print(f"⚠️ Too many read failures, reconnecting...")
                consecutive_failures = 0
                self.connect()
                time.sleep(self.reconnect_delay)
    
    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
    
    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


class CrossCameraTracker:
    """Manages person tracking across multiple cameras with improved matching"""
    def __init__(self, similarity_threshold=0.45):
        self.global_tracks = {}
        self.local_to_global = {}
        self.next_global_id = 1
        self.similarity_threshold = similarity_threshold
        self.max_age = 60  # Increased to handle temporary occlusions
        self.min_feature_count = 3  # Require more features for stability
        self.position_weight = 0.3  # Weight for position-based matching
        self.last_positions = {}  # Track last known positions
        
    def get_global_id(self, camera_id, local_id, features, bbox_center=None):
        key = (camera_id, local_id)
        current_time = time.time()
        
        self._clean_old_tracks(current_time)
        
        # Update position history
        if bbox_center is not None:
            self.last_positions[key] = {
                'position': bbox_center,
                'time': current_time
            }
        
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            if global_id in self.global_tracks:
                # Update track
                if features is not None:
                    self.global_tracks[global_id]['features'].append(features)
                    # Keep last 15 features for better averaging
                    if len(self.global_tracks[global_id]['features']) > 15:
                        self.global_tracks[global_id]['features'].pop(0)
                
                self.global_tracks[global_id]['last_seen'] = current_time
                self.global_tracks[global_id]['camera'] = camera_id
                
                if bbox_center is not None:
                    self.global_tracks[global_id]['last_position'] = bbox_center
            return global_id
        
        # Try to match with existing tracks
        best_match_id = None
        best_score = 0
        
        for global_id, track_data in self.global_tracks.items():
            # Skip tracks from same camera
            if track_data['camera'] == camera_id:
                continue
            
            # Only match if we have enough features
            if len(track_data['features']) < self.min_feature_count:
                continue
            
            # Calculate feature similarity
            feature_similarity = 0
            if features is not None and len(track_data['features']) > 0:
                similarities = []
                for stored_features in track_data['features'][-5:]:  # Use last 5 features
                    if stored_features is not None:
                        try:
                            if isinstance(features, np.ndarray) and isinstance(stored_features, np.ndarray):
                                if features.size > 0 and stored_features.size > 0:
                                    sim = 1 - cosine(features.flatten(), stored_features.flatten())
                                    if not np.isnan(sim) and not np.isinf(sim):
                                        similarities.append(sim)
                        except Exception as e:
                            continue
                
                if len(similarities) >= 2:
                    # Use median instead of mean to reduce outlier impact
                    feature_similarity = np.median(similarities)
            
            # Calculate position similarity (for temporal consistency)
            position_similarity = 0
            if bbox_center is not None and 'last_position' in track_data:
                last_pos = track_data['last_position']
                # Normalized position distance (assuming frame coordinates 0-640, 0-480)
                dist = np.sqrt((bbox_center[0] - last_pos[0])**2 + (bbox_center[1] - last_pos[1])**2)
                position_similarity = max(0, 1 - dist / 500.0)  # Normalize to 0-1
            
            # Combined score with feature dominance
            combined_score = (1 - self.position_weight) * feature_similarity + \
                           self.position_weight * position_similarity
            
            if combined_score > best_score:
                best_score = combined_score
                best_match_id = global_id
        
        # More conservative threshold for cross-camera matching
        threshold = max(self.similarity_threshold, 0.55)
        
        if best_match_id is not None and best_score > threshold:
            self.local_to_global[key] = best_match_id
            if features is not None:
                self.global_tracks[best_match_id]['features'].append(features)
            self.global_tracks[best_match_id]['last_seen'] = current_time
            self.global_tracks[best_match_id]['camera'] = camera_id
            if bbox_center is not None:
                self.global_tracks[best_match_id]['last_position'] = bbox_center
            print(f"🔗 Matched CAM{camera_id} ID:{local_id} -> Global ID:{best_match_id} (score: {best_score:.3f})")
            return best_match_id
        
        # Create new global ID
        global_id = self.next_global_id
        self.next_global_id += 1
        self.local_to_global[key] = global_id
        self.global_tracks[global_id] = {
            'features': [features] if features is not None else [],
            'last_seen': current_time,
            'camera': camera_id,
            'last_position': bbox_center
        }
        print(f"➕ New person: Global ID:{global_id} on CAM{camera_id}")
        return global_id
    
    def _clean_old_tracks(self, current_time):
        to_remove = []
        for global_id, track_data in self.global_tracks.items():
            if current_time - track_data['last_seen'] > self.max_age:
                to_remove.append(global_id)
        
        for global_id in to_remove:
            del self.global_tracks[global_id]
            keys_to_remove = [k for k, v in self.local_to_global.items() if v == global_id]
            for key in keys_to_remove:
                del self.local_to_global[key]
                if key in self.last_positions:
                    del self.last_positions[key]


# -------------------- CONFIG --------------------
RTSP_CAM1 = "rtsp://192.168.26.98:554/stream1"
RTSP_CAM2 = "rtsp://192.168.26.97:554/stream1"

# Initialize YOLO with lower confidence for better detection
model = YOLO("yolov8m.pt")

# Improved DeepSORT parameters for crowded scenes
tracker1 = DeepSort(
    max_age=60,  # Increased to handle occlusions
    n_init=12,  # Increased to reduce false positives
    nms_max_overlap=0.4,  # Slightly higher to handle overlapping people
    max_cosine_distance=0.35,  # More strict for feature matching
    nn_budget=100,
    embedder="clip_RN50",
    half=True,
    embedder_wts=None,
    polygon=False,
    today=None
)

tracker2 = DeepSort(
    max_age=60,
    n_init=12,
    nms_max_overlap=0.4,
    max_cosine_distance=0.35,
    nn_budget=100,
    embedder="clip_RN50",
    half=True,
    embedder_wts=None,
    polygon=False,
    today=None
)

cross_tracker = CrossCameraTracker(similarity_threshold=0.45)
cap1 = RTSPCamera(RTSP_CAM1, reconnect_delay=3)
cap2 = RTSPCamera(RTSP_CAM2, reconnect_delay=3)

click_coords = ClickCoordinates()
active_target = ActiveTarget()
click_listener = ClickListener(click_coords, active_target)

ptz_client_cam1 = PTZSocketClient("192.168.26.98", 9000)
ptz_client_cam2 = PTZSocketClient("192.168.26.97", 9000)


def resize(frame, width=640):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    ratio = width / w
    return cv2.resize(frame, (width, int(h * ratio)))


def validate_bbox(bbox, frame_shape):
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[:2]
    
    if w <= 0 or h <= 0:
        return None
    
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    
    # More lenient minimum size
    if w < 15 or h < 30:  # Reduced from 20x20 to catch distant people
        return None
    
    return [x, y, w, h]


def point_in_box(px, py, l, t, r, b, margin=10):
    """Check if point is inside box with margin for easier clicking"""
    return (l - margin) <= px <= (r + margin) and (t - margin) <= py <= (b + margin)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two boxes"""
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2
    
    # Calculate intersection
    x_left = max(l1, l2)
    y_top = max(t1, t2)
    x_right = min(r1, r2)
    y_bottom = min(b1, b2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (r1 - l1) * (b1 - t1)
    box2_area = (r2 - l2) * (b2 - t2)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0


def process_frame(frame, tracker, camera_id, camera_name, color, 
                  click_coords, active_target, ptz_client):
    """Process frame with improved handling of crowded scenes"""
    if frame is None:
        return None, []
    
    # Rotate 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    # Get click info
    click_x, click_y, camera_ip, click_time, processed = click_coords.get()
    click_is_recent = (time.time() - click_time) < 3.0  # Increased to 3 seconds
    
    # YOLO Detection with lower confidence threshold
    results = model(
        frame, 
        classes=[0],  # person class
        conf=0.5,  # REDUCED from 0.9 to catch more detections
        iou=0.4,  # Reduced for better separation in crowds
        verbose=False,
        imgsz=640
    )
    
    detections = []
    raw_boxes = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            bbox = [x1, y1, x2 - x1, y2 - y1]
            valid_bbox = validate_bbox(bbox, frame.shape)
            
            if valid_bbox is not None:
                detections.append((valid_bbox, conf, 'person'))
                raw_boxes.append((x1, y1, x2, y2, conf))
    
    # Tracking
    try:
        tracks = tracker.update_tracks(detections, frame=frame)
    except Exception as e:
        print(f"⚠️ Tracker error on {camera_name}: {e}")
        tracks = []
    
    track_info = []
    current_active_id = active_target.get()
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        local_id = track.track_id
        ltrb = track.to_ltrb()
        
        if len(ltrb) != 4:
            continue
            
        l, t, r, b = map(int, ltrb)
        
        if l >= r or t >= b:
            continue
        
        # Clamp coordinates
        l = max(0, min(l, frame.shape[1] - 1))
        t = max(0, min(t, frame.shape[0] - 1))
        r = max(0, min(r, frame.shape[1] - 1))
        b = max(0, min(b, frame.shape[0] - 1))
        
        # Get features
        features = None
        if hasattr(track, 'get_feature') and callable(track.get_feature):
            try:
                features = track.get_feature()
            except:
                pass
        elif hasattr(track, 'features') and track.features is not None:
            features = track.features
        
        # Calculate center for PTZ (in original resolution)
        center_x = ((l * 2) + (r * 2)) // 2
        center_y = ((t * 2) + (b * 2)) // 2
        bbox_center = (center_x, center_y)
        
        # Get global ID with position info
        global_id = cross_tracker.get_global_id(camera_id, local_id, features, bbox_center)
        
        # Check if clicked (with margin for easier selection)
        if click_is_recent and not processed and not active_target.is_active():
            scaled_click_x = click_x // 2
            scaled_click_y = click_y // 2
            
            if point_in_box(scaled_click_x, scaled_click_y, l, t, r, b, margin=15):
                active_target.set(global_id)
                current_active_id = global_id
                click_coords.mark_processed()
                print(f"🎯 Person clicked: Global ID {global_id} at ({scaled_click_x}, {scaled_click_y})")
        
        # Check if active target
        is_active = (current_active_id == global_id)
        
        if is_active:
            # Calculate confidence based on detection overlap
            detection_conf = 0.5
            for bx1, by1, bx2, by2, conf in raw_boxes:
                iou = calculate_iou((l, t, r, b), (bx1, by1, bx2, by2))
                if iou > 0.5:
                    detection_conf = conf
                    break
            
            active_target.update_seen(global_id, detection_conf)
            
            # Send PTZ command
            detection_id = f"{camera_id}_{global_id}"
            inverted_x = 1280 - center_x
            ptz_client.send_click(inverted_x, 360, 0, global_id=global_id, detection_id=detection_id)
            
            # Highlight active target
            thickness = 4 if detection_conf > 0.7 else 3
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), thickness)
            cv2.putText(frame, f"TRACKING ID:{global_id} ({detection_conf:.2f})", 
                       (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)           
            
            # Draw center point
            cv2.circle(frame, ((l+r)//2, (t+b)//2), 5, (0, 255, 255), -1)
        else:
            # Normal bounding box
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"ID:{global_id}", (l, t - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        track_info.append({
            'local_id': local_id,
            'global_id': global_id,
            'bbox': (l, t, r, b)
        })
    
    # Camera label
    cv2.putText(frame, camera_name, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Show detection count
    cv2.putText(frame, f"Detections: {len(detections)}", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame, track_info


print("✅ IMPROVED Cross-Camera Tracking System")
print("=" * 50)
print("🎯 Click on any person to start tracking")
print("📍 Lower detection threshold for better sensitivity")
print("🔄 Improved handling of crowded scenes")
print("⏱️  Longer timeout for handling occlusions")
print("Press 'Q' to exit, 'C' to clear active target")
print("=" * 50)

frame_count = 0
last_frame_time = time.time()
process_every_n_frames = 2  # Process every 2nd frame for balance

try:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if (not ret1 or frame1 is None) and (not ret2 or frame2 is None):
            print("⏳ Waiting for frames...")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Reduced frame skipping for better tracking
        if frame_count % process_every_n_frames != 0:
            continue
        
        # Resize
        if frame1 is not None:
            frame1 = resize(frame1)
        if frame2 is not None:
            frame2 = resize(frame2)
        
        # Process both cameras
        processed1, tracks1 = process_frame(
            frame1, tracker1, 1, "CAM1", (0, 255, 0),
            click_coords, active_target, ptz_client_cam1
        )
        processed2, tracks2 = process_frame(
            frame2, tracker2, 2, "CAM2", (255, 0, 255),
            click_coords, active_target, ptz_client_cam2
        )
        
        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
        last_frame_time = current_time
        
        # Display
        display_frames = []
        
        if processed1 is not None:
            cv2.putText(processed1, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(processed1, f"People: {len(tracks1)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            display_frames.append(processed1)
        
        if processed2 is not None:
            cv2.putText(processed2, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(processed2, f"People: {len(tracks2)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            display_frames.append(processed2)
        
        if len(display_frames) == 2:
            h1, w1 = display_frames[0].shape[:2]
            h2, w2 = display_frames[1].shape[:2]
            
            if h1 != h2:
                display_frames[1] = cv2.resize(display_frames[1], (w2, h1))
            
            combined = np.hstack(display_frames)
            
            # Status bar
            total_people = len(cross_tracker.global_tracks)
            active_id = active_target.get()
            
            if active_id:
                avg_conf = active_target.get_avg_confidence()
                status = f"Tracking ID:{active_id} (conf: {avg_conf:.2f})"
                status_color = (0, 255, 0) if avg_conf > 0.7 else (0, 165, 255)
            else:
                status = "No active target - Click to select"
                status_color = (255, 255, 0)
            
            cv2.putText(combined, f"Total People: {total_people} | {status}", 
                       (10, combined.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow("Cross-Camera Person Tracking", combined)
        elif len(display_frames) == 1:
            cv2.imshow("Cross-Camera Person Tracking", display_frames[0])
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            active_target.clear()
            print("🧹 Manually cleared active target")

finally:
    print("\n🛑 Shutting down...")
    click_listener.stop()
    ptz_client_cam1.stop()
    ptz_client_cam2.stop()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    print("✅ Cleanup complete")
