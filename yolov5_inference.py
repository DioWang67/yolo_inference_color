import os
import sys
import cv2
import torch
import numpy as np 

# 根據當前檔案的相對路徑動態添加 yolov5 的路徑
current_dir = os.path.dirname(os.path.abspath(__file__))  # 取得當前檔案的路徑
yolov5_path = os.path.join(current_dir, 'yolov5')         # yolov5 資料夾的相對路徑
sys.path.insert(0, yolov5_path)                           # 將 yolov5 路徑添加到 sys.path

# 匯入 yolov5 模組
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.plots import Annotator, colors

# 絕對路徑指定 default.yaml 的位置
config_path = None

# 預期的顏色順序
expected_color_order = ["Red", "Green", "Orange", "Yellow", "Black", "Black1"]

def load_model(weights, device, config_path):
    
    model = DetectMultiBackend(weights, device=device, data=config_path)
    stride, names = model.stride, model.names
    imgsz = (640, 640)
    return model, stride, names, imgsz

def preprocess_image(frame, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = torch.from_numpy(frame_rgb).to(device).permute(2, 0, 1).float().unsqueeze(0)
    im /= 255.0
    return im

def process_predictions(pred, im, frame, names, conf_thres, iou_thres ):
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    detections = []
    annotator = Annotator(frame, line_width=3, example=str(names))
    
    detected_colors = set()  # 使用集合來儲存已檢測的顏色
    overlapping_threshold = 0.3  # 設置重疊的 IoU 閾值

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                color_name = names[int(cls)]
                
                # 如果該顏色已經出現過或不在預期順序中，則跳過
                if color_name in detected_colors or color_name not in expected_color_order:
                    continue

                # 檢查是否與已有的框重疊過多
                box = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]


                overlapping = False
                for det_box in detections:
                    if iou(box, det_box['box']) > overlapping_threshold:
                        overlapping = True
                        break


                # 如果重疊過多，則跳過該框
                if overlapping:
                    continue

                # 加入檢測結果
                detections.append({
                    'label': color_name,
                    'confidence': conf.item(),
                    'box': box
                })

                # 設置框的顏色和標籤顏色
                color = colors(expected_color_order.index(color_name), True)
                
                # 顯示檢測框和標籤
                annotator.box_label(xyxy, f" {conf:.2f}", color=color)

                # 記錄該顏色，避免重複標籤
                detected_colors.add(color_name)

    # 根據預期順序排序檢測結果
    detections = sorted(detections, key=lambda d: expected_color_order.index(d['label']))

    return annotator.result(), detections
def iou(box1, box2):
    """
    計算兩個框之間的 IoU 值 (Intersection over Union)
    box1 和 box2 格式為 [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 計算重疊部分的面積
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 計算兩個框的面積
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 計算 IoU
    iou_value = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou_value

def draw_fixed_labels(frame, avg_scores):
    # 在左邊固定位置畫出固定順序的標籤
    label_offset_y = 30  # 每個標籤之間的垂直間距
    label_start_y = 30    # 第一個標籤的初始 y 位置
    label_x_pos = 10      # 標籤放在圖片左側的 x 位置

    for i, color_name in enumerate(expected_color_order):
        label_y_pos = label_start_y + i * label_offset_y
        
        # 如果顏色已被檢測，則顯示分數，否則顯示 "未檢測"
        score = avg_scores.get(color_name, 0)
        if score > 0:
            label_text = f"{color_name}: {score:.2f}"
        else:
            label_text = f"{color_name}: 未檢測"
        
        # 使用該顏色顯示標籤
        color = colors(i, True)
        cv2.putText(frame, label_text, (label_x_pos, label_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def setup_camera(cap):
    # 設定影像的寬度為 640 像素
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    
    # 設定影像的高度為 640 像素
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    # 設定自動曝光 (0.25 表示手動模式)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    # 設定曝光值，負值表示較暗的場景
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)
    
    # 設定影像的亮度
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    
    # 設定影像的對比度
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    
    # 設定影像的飽和度
    cap.set(cv2.CAP_PROP_SATURATION, 32)
    
    # 設定影像的色調 (色相)
    cap.set(cv2.CAP_PROP_HUE, 0)
    
    # 設定影像的銳利度
    cap.set(cv2.CAP_PROP_SHARPNESS, 8)
    
    # 設定影像的伽馬值，控制亮度曲線
    cap.set(cv2.CAP_PROP_GAMMA, 64)


def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def check_color_order(detections):
    if len(detections) < 2:
        # 當檢測結果不足以判斷順序時，直接返回 False 或其他適當結果
        print("檢測結果不足，無法判斷顏色順序")
        return False

    # 計算每個檢測框的質心
    centroids = [get_centroid(d['box']) for d in detections]
    centroids = np.array(centroids)

    # 使用PCA找出主要方向
    mean = np.mean(centroids, axis=0)
    centered = centroids - mean
    cov_matrix = np.cov(centered.T)
    
    # 檢查協方差矩陣是否為0維或存在無效值
    if cov_matrix.ndim < 2 or np.isnan(cov_matrix).any():
        print("協方差矩陣無效，無法計算主成分")
        return False

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # 沿著主要方向投影質心，並根據投影結果排序
    projections = np.dot(centered, principal_axis)
    sorted_indices = np.argsort(projections)
    sorted_detections = [detections[i] for i in sorted_indices]

    # 比對排序結果是否符合預期順序
    detected_colors = [d['label'] for d in sorted_detections]
    return detected_colors == expected_color_order


def run_inference(weights='best.pt', source=0, device='cuda', conf_thres=0.25, iou_thres=0.45):
    model, stride, names, imgsz = load_model(weights, device, config_path)
    

    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("無法打開攝像頭")
            return
        
        setup_camera(cap)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("無法捕捉影像幀")
                break
            
            im = preprocess_image(frame, device)
            with torch.no_grad():
                pred = model(im)
            result_frame, detections = process_predictions(pred, im, frame, names, conf_thres, iou_thres )
            
            # 模擬多次檢測，這裡僅進行一次評估
            avg_scores = {det['label']: det['confidence'] for det in detections}
            
            # 繪製左側固定順序的標籤
            draw_fixed_labels(result_frame, avg_scores)
            
            # 檢查顏色順序
            is_correct_order = check_color_order(detections)
            order_text = "顏色順序正確" if is_correct_order else "顏色順序錯誤"

            # 顯示結果
            cv2.imshow('YOLOv5 檢測', result_frame)
            print(f"檢測結果: {detections}")
            print(f"顏色順序是否正確: {is_correct_order}")
            
            if is_correct_order:
                print("顏色順序正確，按下任意鍵繼續...")
                # 等待按下任意鍵後繼續
                cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference(weights=r'D:\Git\robotlearning\yolo_inference\best.pt', source=0, device=torch.device('cpu'), conf_thres=0.8, iou_thres=0.3)
