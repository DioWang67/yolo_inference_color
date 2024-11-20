
# main.py
import torch
import cv2
import time
import os
import pyfiglet
import numpy as np
from typing import Optional, Tuple

from core.detector import YOLODetector
from core.result_handler import ResultHandler
from core.logger import DetectionLogger
from core.config import DetectionConfig
from core.utils import ImageUtils, DetectionResults
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import scale_boxes
from MvImport.MvCameraControl_class import *
from MVS_camera_control import MVSCamera

class YOLOInference:
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化 YOLO 推論系統
        
        Args:
            config_path: 配置文件路徑
        """
        self.logger = DetectionLogger()
        self.config = DetectionConfig.from_yaml(config_path)
        self.model = self._load_model()
        self.detector = YOLODetector(self.model, self.config)
        self.result_handler = ResultHandler(self.config)
        self.camera = MVSCamera()
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(self.config)

    def _load_model(self) -> DetectMultiBackend:
        """
        載入 YOLO 模型
        
        Returns:
            DetectMultiBackend: 載入的模型實例
        """
        try:
            model = DetectMultiBackend(self.config.weights, device=self.config.device)
            self.logger.logger.info("模型載入成功")
            return model
        except Exception as e:
            self.logger.logger.error(f"模型載入錯誤: {str(e)}")
            raise

    def print_large_text(self, text: str) -> None:
        """
        使用 ASCII art 打印大型文字
        
        Args:
            text: 要打印的文字
        """
        ascii_art = pyfiglet.figlet_format(text)
        print(ascii_art)

    def handle_detection(self, frame: np.ndarray, detections: list, 
                        elapsed_time: float) -> Tuple[str, Optional[np.ndarray]]:
        """
        處理檢測結果
        
        Args:
            frame: 原始幀
            detections: 檢測結果列表
            elapsed_time: 已經過的時間
            
        Returns:
            Tuple[str, Optional[np.ndarray]]: 結果文字和標注後的幀
        """
        result, _ = self.detection_results.evaluate_detection(detections)
        
        if result == "PASS" or elapsed_time >= self.config.timeout:
            status = "PASS" if result == "PASS" else "FAIL"
            result_frame = self.result_handler.save_results(
                frame=frame,
                detections=detections,
                status=status,
                detector=self.detector
            )
            self.print_large_text(status)
            return status, result_frame
            
        return "", None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        處理單個幀
        
        Args:
            frame: 輸入幀
            
        Returns:
            Tuple[np.ndarray, list]: 處理後的幀和檢測結果
        """
        im = self.detector.preprocess_image(frame)
        with torch.no_grad():
            pred = self.model(im)
        return self.detector.process_detections(pred, im, frame)

    def run_inference(self) -> None:
        """執行推論主循環"""
        try:
            # 初始化相機
            if not self.camera.enum_devices():
                raise IOError("無法找到MVS相機")
            if not self.camera.connect_to_camera():
                raise IOError("無法連接MVS相機")

            # 狀態變量
            detecting = False
            wait_for_restart = False
            result_text = ""
            last_result_frame = None

            while True:
                # 獲取並處理幀
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                    
                frame = cv2.resize(frame, (640, 640))
                original_frame = frame.copy()

                # 處理按鍵輸入
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                # 處理檢測狀態
                if not detecting and not wait_for_restart and key == ord(' '):
                    detecting = True
                    wait_for_restart = False
                    start_time = time.time()
                    result_text = ""
                    last_result_frame = None
                    continue

                # 執行檢測
                if detecting:
                    elapsed_time = time.time() - start_time
                    result_frame, detections = self.process_frame(frame)
                    
                    result_text, new_result_frame = self.handle_detection(
                        original_frame, detections, elapsed_time)
                    
                    if result_text:
                        last_result_frame = new_result_frame
                        detecting = False
                        wait_for_restart = True
                    
                    display_frame = self.detector.draw_results(
                        frame, result_text, detections) if result_text else frame
                    cv2.imshow('YOLOv5 檢測', display_frame)

                # 處理等待重啟狀態
                elif wait_for_restart:
                    if key == ord(' '):
                        detecting = True
                        wait_for_restart = False
                        start_time = time.time()
                        result_text = ""
                        last_result_frame = None
                    cv2.imshow('YOLOv5 檢測', 
                             last_result_frame if last_result_frame is not None else frame)

                # 顯示當前幀
                else:
                    cv2.imshow('YOLOv5 檢測', 
                             last_result_frame if last_result_frame is not None else frame)

        except Exception as e:
            self.logger.logger.error(f"執行過程中發生錯誤: {str(e)}")
            raise

        finally:
            self.camera.close()
            cv2.destroyAllWindows()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        inference = YOLOInference(r"D:\Git\robotlearning\yolo_inference\config.yaml")
        inference.run_inference()
    except Exception as e:
        print(f"程序執行出錯: {str(e)}")