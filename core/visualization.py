# utils/visualization.py
import cv2
import numpy as np
from typing import List, Dict, Tuple

class Visualizer:
    def __init__(self, colors_func):
        self.colors = colors_func

    def draw_boxes(self, frame: np.ndarray, detections: List[Dict], 
                  color_order: List[str]) -> np.ndarray:
        frame_copy = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            color = self.colors(color_order.index(det['label']), True)
            label = f"{det['label']} {det['confidence']:.2f}"
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame_copy

    def draw_fixed_labels(self, frame: np.ndarray, avg_scores: Dict[str, float], 
                         color_order: List[str]) -> np.ndarray:
        frame_copy = frame.copy()
        for i, color_name in enumerate(color_order):
            y_pos = 30 + i * 30
            score = avg_scores.get(color_name, 0)
            label_text = f"{color_name}: {score:.2f}" if score > 0 else f"{color_name}: 未檢測"
            color = self.colors(i, True)
            cv2.putText(frame_copy, label_text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame_copy

    def draw_progress(self, frame: np.ndarray, elapsed_time: float, 
                     max_time: float = 2) -> np.ndarray:
        frame_copy = frame.copy()
        progress = min(elapsed_time / max_time, 1.0)
        bar_width = 200
        bar_height = 20
        filled_width = int(bar_width * progress)
        
        cv2.rectangle(frame_copy, (10, 70), (10 + bar_width, 70 + bar_height), 
                     (0,0,0), 2)
        cv2.rectangle(frame_copy, (10, 70), (10 + filled_width, 70 + bar_height), 
                     (0,255,0), -1)
        return frame_copy