import os
import sys
import cv2
import torch
import numpy as np
from datetime import datetime
import time

class YOLOInference:
    def __init__(self, weights, device='cpu', conf_thres=0.25, iou_thres=0.45, color_ranges=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yolov5_path = os.path.join(current_dir, 'yolov5')
        sys.path.insert(0, yolov5_path)

        from yolov5.utils.general import non_max_suppression, scale_boxes
        from yolov5.models.common import DetectMultiBackend
        from yolov5.utils.plots import Annotator, colors

        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        self.DetectMultiBackend = DetectMultiBackend
        self.Annotator = Annotator
        self.colors = colors

        self.weights = weights
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.expected_color_order = ["Red", "Green", "Orange", "Yellow", "Black", "Black1"]
        self.color_ranges = {
            "Red": [[(0, 100, 100), (2, 255, 255)], [(90, 100, 100), (240, 255, 255)]],  # 紅色的兩段範圍
            "Green": [[(85, 65, 50), (105, 200, 130)]],
            "Orange": [[(5, 100, 100), (12, 255, 255)]],
            "Yellow": [[(12, 100, 100), (25, 255, 255)]],
            "Black": [[(0, 0, 0), (180, 255, 50)]],
            "Black1": [[(0, 0, 0), (180, 255, 50)]]
        }

        
        self.model, self.stride, self.names, self.imgsz = self.load_model()

    def load_model(self):
        model = self.DetectMultiBackend(self.weights, device=self.device)
        stride, names = model.stride, model.names
        imgsz = (1280, 1280)
        return model, stride, names, imgsz

    def preprocess_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(frame_rgb).to(self.device).permute(2, 0, 1).float().unsqueeze(0)
        im /= 255.0
        return im

    def check_color_in_range(self,image, box, color_name):
        x1, y1, x2, y2 = box
        region = image[y1:y2, x1:x2]  # 裁剪檢測框
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)  # 轉換為 HSV

        # 處理多段範圍的情況
        masks = []
        for lower, upper in self.color_ranges[color_name]:
            mask = cv2.inRange(hsv_region, np.array(lower), np.array(upper))
            masks.append(mask)

        # 合併所有範圍的遮罩
        full_mask = masks[0]
        for mask in masks[1:]:
            full_mask = cv2.bitwise_or(full_mask, mask)

        # 計算區域內匹配像素的比例
        match_ratio = cv2.countNonZero(full_mask) / (full_mask.size)
        return match_ratio > 0.2  # 調整此比例以改變檢測門檻


    def process_predictions(self, pred, im, frame):
        pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres)
        detections = []
        annotator = self.Annotator(frame, line_width=3, example=str(self.names))

        detected_colors = set()
        overlapping_threshold = 0.3

        for det in pred:
            if len(det):
                det[:, :4] = self.scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    color_name = self.names[int(cls)]
                    if color_name in detected_colors or color_name not in self.expected_color_order:
                        continue

                    box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    overlapping = any(self.iou(box, det_box['box']) > overlapping_threshold for det_box in detections)
                    if overlapping:
                        continue

                    if not self.check_color_in_range(frame, box, color_name):
                        continue

                    detections.append({'label': color_name, 'confidence': conf.item(), 'box': box})
                    color = self.colors(self.expected_color_order.index(color_name), True)
                    annotator.box_label(xyxy, f"{color_name} {conf:.2f}", color=color)
                    detected_colors.add(color_name)

        detections = sorted(detections, key=lambda d: self.expected_color_order.index(d['label']))
        return annotator.result(), detections

    @staticmethod
    def iou(box1, box2):
        x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection_area / float(box1_area + box2_area - intersection_area)

    def save_results(self, frame, detections, status):
        date_folder = datetime.now().strftime("%Y%m%d")
        time_stamp = datetime.now().strftime("%H%M%S")
        result_dir = f"Result/{date_folder}/{status}"
        os.makedirs(f"{result_dir}/annotated", exist_ok=True)
        os.makedirs(f"{result_dir}/original", exist_ok=True)

        # Save original image without annotations
        original_image_path = f"{result_dir}/original/{time_stamp}.jpg"
        cv2.imwrite(original_image_path, frame)  # 儲存原始影像（無標註）

        # Save annotated image with detection labels
        annotated_image_path = f"{result_dir}/annotated/{time_stamp}.jpg"
        annotated_frame = frame.copy()  # 創建帶標註的影像副本
        for det in detections:
            x1, y1, x2, y2 = det['box']
            color = self.colors(self.expected_color_order.index(det['label']), True)
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(annotated_image_path, annotated_frame)  # 儲存帶標註的影像

    def draw_fixed_labels(self,frame, avg_scores):
        # 在左邊固定位置畫出固定順序的標籤
        label_offset_y = 30  # 每個標籤之間的垂直間距
        label_start_y = 30    # 第一個標籤的初始 y 位置
        label_x_pos = 10      # 標籤放在圖片左側的 x 位置

        for i, color_name in enumerate(self.expected_color_order):
            label_y_pos = label_start_y + i * label_offset_y
            
            # 如果顏色已被檢測，則顯示分數，否則顯示 "未檢測"
            score = avg_scores.get(color_name, 0)
            if score > 0:
                label_text = f"{color_name}: {score:.2f}"
            else:
                label_text = f"{color_name}: 未檢測"
            
            # 使用該顏色顯示標籤
            color = self.colors(i, True)
            cv2.putText(frame, label_text, (label_x_pos, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def run_inference(self, source=0, required_score=0.5):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("無法打開攝像頭")
            return

        detecting = False
        wait_for_restart = False  # 等待空白鍵重啟檢測
        result_text = ""  # 初始化 result_text

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("無法捕捉影像幀")
                break

            # 等待按下空白鍵以開始或重新開始檢測
            if not detecting and not wait_for_restart and cv2.waitKey(1) == ord(' '):
                detecting = True
                wait_for_restart = False  # 重置等待標誌
                start_time = time.time()
                result_text = ""  # 重置 result_text 每次重新開始
                continue

            # 檢測過程
            if detecting:
                elapsed_time = time.time() - start_time
                im = self.preprocess_image(frame)
                with torch.no_grad():
                    pred = self.model(im)
                result_frame, detections = self.process_predictions(pred, im, frame)

                # 檢查當前幀中是否滿足所有顏色的順序和分數門檻
                detected_colors = []
                for color in self.expected_color_order:
                    color_detections = [det for det in detections if det['label'] == color and det['confidence'] >= required_score]
                    if color_detections:
                        detected_colors.append(color)

                # 判斷當前幀的顏色是否符合預期順序，並且所有顏色都達到分數
                if detected_colors == self.expected_color_order:
                    result_text = "PASS"
                    print(result_text)
                    self.save_results(result_frame, detections, status="PASS")
                    detecting = False  # 結束檢測
                    wait_for_restart = True  # 等待重新開始
                elif elapsed_time >= 2:
                    result_text = "FAIL"
                    print(result_text)
                    self.save_results(result_frame, detections, status="FAIL")
                    detecting = False  # 結束檢測
                    wait_for_restart = True  # 等待重新開始

                # 繪製左側固定順序的標籤
                avg_scores = {det['label']: det['confidence'] for det in detections}
                self.draw_fixed_labels(result_frame, avg_scores)

                # 顯示檢測結果
                cv2.putText(result_frame, result_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0) if result_text == "PASS" else (0, 0, 255), 3)
                cv2.imshow('YOLOv5 檢測', result_frame)
                cv2.waitKey(1)

            # 當前處於等待重新開始狀態
            elif wait_for_restart:
                # 顯示結果畫面並等待按下空白鍵以重新開始檢測
                cv2.putText(frame, result_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0) if result_text == "PASS" else (0, 0, 255), 3)
                cv2.imshow('YOLOv5 檢測', frame)
                # if cv2.waitKey(1) == ord(' '):
                wait_for_restart = False  # 重置等待標誌，準備重新開始

            # 顯示目前畫面
            cv2.imshow('YOLOv5 檢測', frame)

            # 按下 'q' 離開
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_inference = YOLOInference(
        weights="D:\\Git\\robotlearning\\yolo_inference\\best.pt", 
        device=torch.device('cpu'), 
        conf_thres=0.8, 
        iou_thres=0.3
    )
    yolo_inference.run_inference(source=0)
