import cv2
import numpy as np
import librosa
import json
import os
from scipy.signal import butter, lfilter


class SoccerEventDetector:

    def __init__(self, video_path):
        self.video_path = video_path
        self.events = []
        self.fps = 0
        self.duration = 0


    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a


    def detect_audio_events(self):
        print("🎧 Analyzing Audio (Whistles & Crowd)...")
        try:
            # Load audio (mono) - optimize by loading only first 5 mins if file is huge
            y, sr = librosa.load(self.video_path, sr=22050, mono=True)
            self.duration = librosa.get_duration(y=y, sr=sr)


            # 1. WHISTLE DETECTOR (Bandpass 3.5kHz - 4.5kHz)
            b, a = self.butter_bandpass(3500, 4500, sr, order=6)
            whistle_signal = lfilter(b, a, y)
            
            # RMS Energy
            hop_length = 512
            frame_length = 2048
            rms_whistle = librosa.feature.rms(y=whistle_signal, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Thresholding
            # Normalize first
            if np.max(rms_whistle) > 0:
                rms_whistle = rms_whistle / np.max(rms_whistle)
            
            whistle_peaks = np.where(rms_whistle > 0.4)[0] # 0.4 sensitivity
            
            curr_time = 0
            for frame in whistle_peaks:
                timestamp = librosa.frames_to_time(frame, sr=sr, hop_length=hop_length)
                if timestamp - curr_time > 8: # 8s debounce
                    self.events.append({
                        "seconds": round(timestamp, 2),
                        "type": "audio_whistle",
                        "desc": "High pitch audio spike (Potential Whistle/Foul)"
                    })
                    curr_time = timestamp
        except Exception as e:
            print(f"⚠️ Audio analysis skipped: {e}")


    def detect_visual_events(self):
        print("👀 Analyzing Video (Scene Cuts)...")
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if not self.fps: self.fps = 30
        
        prev_hist = None
        frame_count = 0
        skip_frames = int(self.fps) # Analyze 1 frame per second


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue


            current_seconds = frame_count / self.fps


            # Histogram Scene Change Detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)


            if prev_hist is not None:
                score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if score < 0.5: # Significant cut
                     self.events.append({
                        "seconds": round(current_seconds, 2),
                        "type": "visual_cut",
                        "desc": "Significant camera cut or replay"
                    })


            prev_hist = hist
            frame_count += 1


        cap.release()


    def generate_manifest(self):
        # Sort and return
        self.events.sort(key=lambda x: x['seconds'])
        # Deduplicate timestamps close together (within 5 seconds)
        unique_events = []
        if self.events:
            curr = self.events[0]
            unique_events.append(curr)
            for evt in self.events[1:]:
                if evt['seconds'] - curr['seconds'] > 5:
                    unique_events.append(evt)
                    curr = evt
        
        return unique_events

