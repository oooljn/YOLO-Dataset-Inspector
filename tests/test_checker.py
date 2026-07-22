import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "src"))

import checker
import pytest

def test_parse_good_label():
    """测试 good.txt 是否能被正确解析"""
    label_path = BASE_DIR / "examples" / "demo_dataset" / "labels" / "good.txt"
    
    print("\n🔍 正在寻找文件:", label_path)
    print("📂 文件是否存在:", label_path.exists())
    
    result = checker.parse_label(str(label_path))

    assert result is not None, "解析失败，应该返回一个列表或字典"
    assert len(result) == 1, "应该解析出 1 个目标"
    assert result[0]['class_id'] == 0
    assert result[0]['x_center'] == 0.50
    assert result[0]['y_center'] == 0.50
    assert result[0]['width'] == 0.30
    assert result[0]['height'] == 0.20
    def test_out_of_bound_label():
        """测试越界坐标的检测"""
    label_path = BASE_DIR / "examples" / "demo_dataset" / "labels" / "out_of_bound.txt"
    result = checker.parse_label(str(label_path))
    assert result is not None
    assert len(result) == 0, "越界标签不应被解析为有效目标"

def test_bad_format_label():
    """测试格式错误的标签"""
    label_path = BASE_DIR / "examples" / "demo_dataset" / "labels" / "bad_format.txt"
    result = checker.parse_label(str(label_path))
    assert result is not None
    assert len(result) == 0, "格式错误不应解析出任何目标"

def test_missing_label():
    """测试标签文件缺失的情况"""
    fake_label_path = str(BASE_DIR / "examples" / "demo_dataset" / "labels" / "i_dont_exist.txt")
    result = checker.parse_label(fake_label_path)
    assert result is None, "缺失的标签文件应返回 None"