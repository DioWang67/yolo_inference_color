
import os
import pandas as pd
from datetime import datetime
import cv2
from typing import List, Dict
from .utils import ImageUtils, DetectionResults
import numpy as np
from yolov5.utils.plots import colors

class ResultHandler:
    def __init__(self, config, base_dir: str = "Result"):
        """
        初始化結果處理器
        
        Args:
            config: 配置對象
            base_dir: 基礎目錄路徑
        """
        self.base_dir = base_dir
        self.config = config
        self.colors = colors
        self.excel_path = os.path.join(self.base_dir, "results.xlsx")
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(config)
        
        os.makedirs(self.base_dir, exist_ok=True)
        if not os.path.exists(self.excel_path):
            self._initialize_excel()

    def _initialize_excel(self) -> None:
        """初始化 Excel 文件"""
        columns = [
            "時間戳記", "測試編號", "預期順序", "檢測順序",
            "結果", "信心分數", "錯誤訊息",
            "標註影像路徑", "原始影像路徑"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_excel(self.excel_path, index=False, engine='openpyxl')

    def _read_excel(self) -> pd.DataFrame:
        """讀取 Excel 文件"""
        try:
            if os.path.exists(self.excel_path):
                return pd.read_excel(self.excel_path, engine='openpyxl')
            return pd.DataFrame()
        except Exception as e:
            print(f"讀取 Excel 時發生錯誤: {str(e)}")
            return pd.DataFrame()

    def _append_to_excel(self, data: Dict) -> None:
        """
        將數據添加到 Excel
        
        Args:
            data: 要添加的數據
        """
        try:
            df = self._read_excel()
            if isinstance(data.get('時間戳記'), datetime):
                data['時間戳記'] = data['時間戳記'].strftime('%Y-%m-%d %H:%M:%S')
            new_row = pd.DataFrame([data])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(self.excel_path, index=False, engine='openpyxl')
        except Exception as e:
            print(f"寫入 Excel 時發生錯誤: {str(e)}")

    def _draw_detection_box(self, frame: np.ndarray, detection: Dict) -> None:
        """
        在圖像上繪製檢測框
        
        Args:
            frame: 要繪製的圖像
            detection: 檢測結果
        """
        x1, y1, x2, y2 = detection['box']
        color_name = detection['label']
        
        try:
            color_index = self.config.expected_color_order.index(color_name)
        except ValueError:
            color_index = 0
            
        color = self.colors(color_index, True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{color_name} {detection['confidence']:.2f}"
        self.image_utils.draw_label(frame, label, (x1, y1 - 10), color)

    def save_results(self, frame: np.ndarray, detections: List[Dict], 
                    status: str, detector) -> np.ndarray:
        """
        保存檢測結果
        
        Args:
            frame: 原始圖像
            detections: 檢測結果列表
            status: 狀態 (PASS/FAIL)
            detector: 檢測器實例
            
        Returns:
            np.ndarray: 標註後的圖像
        """
        try:
            result_dir, time_stamp, annotated_dir, original_dir = \
                self.image_utils.create_result_directories(self.base_dir, status)

            original_path = os.path.join(original_dir, f"{time_stamp}.jpg")
            cv2.imwrite(original_path, frame)

            annotated_frame = frame.copy()
            avg_scores = {det['label']: det['confidence'] for det in detections}
            detector.draw_fixed_labels(annotated_frame, avg_scores)
            
            status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
            self.image_utils.draw_label(annotated_frame, status, (230, 230), 
                                      status_color, font_scale=3, thickness=3)

            for det in detections:
                self._draw_detection_box(annotated_frame, det)

            annotated_path = os.path.join(annotated_dir, f"{time_stamp}.jpg")
            cv2.imwrite(annotated_path, annotated_frame)

            excel_data = self.detection_results.format_detection_data(
                detections, annotated_path, original_path)
            excel_data["測試編號"] = len(self._read_excel()) + 1
            self._append_to_excel(excel_data)

            return annotated_frame

        except Exception as e:
            print(f"保存結果時發生錯誤: {str(e)}")
            return frame