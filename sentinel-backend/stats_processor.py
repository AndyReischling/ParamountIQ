import cv2
import numpy as np
import json
from datetime import datetime
from collections import deque


class VideoStatisticsProcessor:
    """
    Processes video to extract ball and player movement statistics.
    Uses OpenCV for tracking and computes continuous time-series data.
    """
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.fps = 30
        self.duration = 0
        self.ball_data = []
        self.player_data = []
        
    def process_video(self, sample_rate=1):
        """
        Main processing function that extracts statistics from video.
        
        Args:
            sample_rate: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
        """
        print("📊 Starting statistics processing...")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = frame_count_total / self.fps
        
        print(f"📹 Video info: {self.duration:.1f}s @ {self.fps:.1f} fps")
        
        # Initialize trackers
        ball_tracker = BallTracker()
        player_tracker = PlayerTracker()
        
        frame_count = 0
        prev_frame = None
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on sample_rate
                if frame_count % sample_rate != 0:
                    frame_count += 1
                    continue
                
                timestamp = frame_count / self.fps
                
                # Track ball
                ball_info = ball_tracker.track(frame, timestamp)
                if ball_info:
                    self.ball_data.append(ball_info)
                
                # Track players
                player_info = player_tracker.track(frame, timestamp)
                if player_info:
                    self.player_data.append(player_info)
                
                prev_frame = frame
                frame_count += 1
                
                # Progress indicator
                if frame_count % (int(self.fps) * 5) == 0:  # Every 5 seconds
                    progress = (frame_count / frame_count_total) * 100
                    print(f"  Processing: {progress:.1f}% ({timestamp:.1f}s)")
        
        except Exception as e:
            print(f"⚠️ Error during processing: {e}")
            raise
        finally:
            cap.release()
        
        # Compute final statistics
        self._compute_statistics()
        
        print(f"✅ Statistics processing complete!")
        print(f"   Ball data points: {len(self.ball_data)}")
        print(f"   Player data points: {len(self.player_data)}")
        
        return self.get_statistics()
    
    def _compute_statistics(self):
        """Compute derived statistics from tracked data."""
        # Compute speeds and distances for ball
        if len(self.ball_data) > 1:
            prev_pos = None
            cumulative_distance = 0
            
            for i, ball_point in enumerate(self.ball_data):
                if prev_pos is not None and ball_point.get('x') is not None:
                    dx = ball_point['x'] - prev_pos['x']
                    dy = ball_point['y'] - prev_pos['y']
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Speed calculation (pixels per second)
                    time_diff = ball_point['timestamp'] - prev_pos['timestamp']
                    if time_diff > 0:
                        speed = distance / time_diff
                        ball_point['speed'] = speed
                        cumulative_distance += distance
                        ball_point['distance'] = cumulative_distance
                    else:
                        ball_point['speed'] = 0
                        ball_point['distance'] = cumulative_distance
                else:
                    ball_point['speed'] = 0
                    ball_point['distance'] = 0
                
                prev_pos = ball_point
        
        # Compute speeds for players
        if len(self.player_data) > 1:
            for i, player_point in enumerate(self.player_data):
                if i > 0 and player_point.get('positions'):
                    prev_point = self.player_data[i-1]
                    if prev_point.get('positions'):
                        speeds = []
                        for j, pos in enumerate(player_point['positions']):
                            if j < len(prev_point['positions']):
                                prev_pos = prev_point['positions'][j]
                                dx = pos[0] - prev_pos[0]
                                dy = pos[1] - prev_pos[1]
                                distance = np.sqrt(dx*dx + dy*dy)
                                time_diff = player_point['timestamp'] - prev_point['timestamp']
                                if time_diff > 0:
                                    speeds.append(distance / time_diff)
                                else:
                                    speeds.append(0)
                            else:
                                speeds.append(0)
                        player_point['speeds'] = speeds
    
    def get_statistics(self):
        """Return statistics in the format expected by the cache."""
        return {
            "ball": self.ball_data,
            "players": self.player_data,
            "metadata": {
                "processed": True,
                "processing_date": datetime.now().isoformat(),
                "fps": self.fps,
                "duration": self.duration,
                "ball_data_points": len(self.ball_data),
                "player_data_points": len(self.player_data)
            }
        }
    
    def get_statistics_at_timestamp(self, timestamp):
        """Get statistics for a specific timestamp (interpolated if needed)."""
        # Find closest ball data point
        ball_info = None
        if self.ball_data:
            closest_idx = min(range(len(self.ball_data)), 
                            key=lambda i: abs(self.ball_data[i]['timestamp'] - timestamp))
            ball_info = self.ball_data[closest_idx]
        
        # Find closest player data point
        player_info = None
        if self.player_data:
            closest_idx = min(range(len(self.player_data)),
                            key=lambda i: abs(self.player_data[i]['timestamp'] - timestamp))
            player_info = self.player_data[closest_idx]
        
        return {
            "ball": ball_info,
            "players": player_info
        }


class BallTracker:
    """Tracks ball movement using color detection and optical flow."""
    
    def __init__(self):
        self.prev_gray = None
        self.prev_ball_pos = None
        self.trajectory = deque(maxlen=30)  # Keep last 30 positions
        
    def track(self, frame, timestamp):
        """Track ball in a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Ball detection: look for white/yellow objects (typical soccer ball colors)
        # White ball range
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Yellow ball range
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_pos = None
        if contours:
            # Filter by size (ball should be relatively small)
            valid_contours = [c for c in contours if 50 < cv2.contourArea(c) < 5000]
            
            if valid_contours:
                # Get the largest valid contour (likely the ball)
                largest = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    ball_pos = (cx, cy)
                    self.trajectory.append(ball_pos)
        
        # Use optical flow if we have previous frame and no detection
        if ball_pos is None and self.prev_gray is not None and self.prev_ball_pos is not None:
            # Try to track using optical flow
            p0 = np.array([[self.prev_ball_pos]], dtype=np.float32)
            p1, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            if status[0][0] == 1:  # Tracking successful
                ball_pos = (int(p1[0][0][0]), int(p1[0][0][1]))
                self.trajectory.append(ball_pos)
        
        # Update state
        self.prev_gray = gray
        self.prev_ball_pos = ball_pos
        
        # Return ball data point
        if ball_pos:
            # Normalize coordinates to 0-100 scale (relative to frame)
            height, width = frame.shape[:2]
            normalized_x = (ball_pos[0] / width) * 100
            normalized_y = (ball_pos[1] / height) * 100
            
            # Determine team possession based on field position
            # Left side (0-50%) = Home team, Right side (50-100%) = Away team
            # This is a simple heuristic - can be enhanced with actual team detection
            possession_team = "Home" if normalized_x < 50 else "Away"
            
            return {
                "timestamp": timestamp,
                "x": normalized_x,
                "y": normalized_y,
                "pixel_x": ball_pos[0],
                "pixel_y": ball_pos[1],
                "detected": True,
                "possession_team": possession_team
            }
        else:
            # Return None position but still record timestamp
            return {
                "timestamp": timestamp,
                "x": None,
                "y": None,
                "pixel_x": None,
                "pixel_y": None,
                "detected": False,
                "possession_team": None
            }


class PlayerTracker:
    """Tracks player movement using blob detection and optical flow."""
    
    def __init__(self):
        self.prev_gray = None
        self.prev_positions = []
        self.max_players = 22  # Maximum players on field
        
    def track(self, frame, timestamp):
        """Track players in a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Player detection: look for non-green objects (players vs field)
        # Green field range (typical soccer field)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_not_green = cv2.bitwise_not(mask_green)
        
        # Also look for white/colored jerseys
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_not_green, mask_white)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        positions = []
        if contours:
            # Filter by size (players should be medium-sized objects)
            valid_contours = [c for c in contours if 500 < cv2.contourArea(c) < 50000]
            
            # Sort by area and take top N
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            valid_contours = valid_contours[:self.max_players]
            
            for contour in valid_contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Normalize coordinates
                    height, width = frame.shape[:2]
                    normalized_x = (cx / width) * 100
                    normalized_y = (cy / height) * 100
                    
                    positions.append([normalized_x, normalized_y])
        
        # Use optical flow to track existing players if detection fails
        if len(positions) < len(self.prev_positions) and self.prev_gray is not None:
            # Try to track previous positions
            if self.prev_positions:
                p0 = np.array([[pos] for pos in self.prev_positions], dtype=np.float32)
                p1, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, p0, None,
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                # Add successfully tracked positions
                tracked_positions = []
                for i, (st, pt) in enumerate(zip(status, p1)):
                    if st[0] == 1:  # Successfully tracked
                        height, width = frame.shape[:2]
                        px, py = pt[0]
                        normalized_x = (px / width) * 100
                        normalized_y = (py / height) * 100
                        tracked_positions.append([normalized_x, normalized_y])
                
                # Merge detected and tracked positions
                if tracked_positions:
                    positions = tracked_positions[:self.max_players]
        
        # Update state
        self.prev_gray = gray
        self.prev_positions = positions
        
        # Assign teams to players based on position
        # Left side players = Home, Right side players = Away
        # This is a simple heuristic - can be enhanced with jersey color detection
        player_teams = []
        for pos in positions:
            team = "Home" if pos[0] < 50 else "Away"
            player_teams.append(team)
        
        return {
            "timestamp": timestamp,
            "positions": positions,
            "count": len(positions),
            "teams": player_teams
        }

