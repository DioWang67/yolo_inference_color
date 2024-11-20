# config.py
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class DetectionConfig:
    weights: str
    device: str = 'cpu'
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    color_match_threshold: float = 0.2
    imgsz: Tuple[int, int] = (1280, 1280)
    expected_color_order: List[str] = None
    color_ranges: Dict[str, List[List[Tuple[int, int, int]]]] = None
    timeout: int = 2

    @classmethod
    def from_yaml(cls, path: str) -> 'DetectionConfig':
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            print("Loaded YAML:", config_dict)  # 調試輸出
        return cls(**config_dict)


    def __post_init__(self):
        if self.color_ranges is None:
            self.color_ranges = {
                "Red": [[(0, 100, 100), (2, 255, 255)], 
                        [(90, 100, 100), (240, 255, 255)]],
                "Green": [[(85, 65, 50), (105, 200, 130)]],
                "Orange": [[(5, 100, 100), (12, 255, 255)]],
                "Yellow": [[(12, 100, 100), (25, 255, 255)]],
                "Black": [[(0, 0, 0), (180, 255, 50)]],
                "Black1": [[(0, 0, 0), (180, 255, 50)]]
            }
        if self.expected_color_order is None:
            self.expected_color_order = ["Red", "Green", "Orange", "Yellow", "Black", "Black1"]
