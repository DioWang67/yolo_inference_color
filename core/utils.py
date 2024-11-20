import cv2
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class ImageUtils:
    @staticmethod
    def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int], font_scale: float = 0.6, 
                   thickness: int = 2) -> None:
        """
        在圖像上繪製標籤
        
        Args:
            frame: 要繪製的圖像
            text: 標籤文字
            position: 標籤位置 (x, y)
            color: 顏色 (B, G, R)
            font_scale: 字體大小
            thickness: 線條粗細
        """
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, thickness)

    @staticmethod
    def create_result_directories(base_dir: str, status: str) -> Tuple[str, str, str, str]:
        """
        創建結果目錄結構
        
        Args:
            base_dir: 基礎目錄
            status: 狀態 (PASS/FAIL)
            
        Returns:
            Tuple[str, str, str, str]: 結果目錄, 時間戳, 標註目錄, 原始目錄
        """
        date_folder = datetime.now().strftime("%Y%m%d")
        time_stamp = datetime.now().strftime("%H%M%S")
        result_dir = os.path.join(base_dir, date_folder, status)
        
        annotated_dir = os.path.join(result_dir, "annotated")
        original_dir = os.path.join(result_dir, "original")
        
        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
        
        return result_dir, time_stamp, annotated_dir, original_dir

class DetectionResults:
    def __init__(self, config):
        """
        初始化檢測結果處理器
        
        Args:
            config: 配置對象
        """
        self.config = config
        
    def evaluate_detection(self, detections: List[Dict], required_score: float = 0.5) -> Tuple[str, str]:
        """
        評估檢測結果
        
        Args:
            detections: 檢測結果列表
            required_score: 最低要求分數
            
        Returns:
            Tuple[str, str]: 結果狀態和錯誤信息
        """
        detected_colors = [det['label'] for det in detections 
                         if det['confidence'] >= required_score]
        
        if detected_colors == self.config.expected_color_order:
            return "PASS", ""
        return "FAIL", "順序不正確"

    def format_detection_data(self, detections: List[Dict], 
                            annotated_path: str, original_path: str) -> Dict:
        """
        格式化檢測數據以供保存
        
        Args:
            detections: 檢測結果列表
            annotated_path: 標註影像路徑
            original_path: 原始影像路徑
            
        Returns:
            Dict: 格式化後的數據
        """
        detected_colors = [det['label'] for det in detections]
        confidence_scores = {det['label']: det['confidence'] for det in detections}
        
        result, error_message = self.evaluate_detection(detections)
        
        return {
            "時間戳記": datetime.now(),
            "測試編號": None,  # 將在 ResultHandler 中設置
            "預期順序": ", ".join(self.config.expected_color_order),
            "檢測順序": ", ".join(detected_colors),
            "結果": result,
            "信心分數": str(dict(confidence_scores)),
            "錯誤訊息": error_message,
            "標註影像路徑": annotated_path,
            "原始影像路徑": original_path,
        }