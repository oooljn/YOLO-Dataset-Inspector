import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from src.checker import inspect_dataset
from src.statistics import analyze_dataset
from src.visualizer import draw_annotations


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(dataset_path: str) -> list[str]:
    """
    从数据集目录中的 data.yaml 读取类别名称。

    兼容两种常见格式：

    names:
      - person
      - car

    或：

    names:
      0: person
      1: car
    """
    yaml_path = Path(dataset_path) / "data.yaml"

    if not yaml_path.is_file():
        return []

    try:
        with yaml_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return []

    if not isinstance(config, dict):
        return []

    names = config.get("names", [])

    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        try:
            sorted_items = sorted(
                names.items(),
                key=lambda item: int(item[0]),
            )
        except (TypeError, ValueError):
            sorted_items = list(names.items())

        return [str(name) for _, name in sorted_items]

    return []


def initialize_session_state() -> None:
    """初始化 Streamlit 会话状态。"""
    default_values: dict[str, Any] = {
        "dataset_path": "",
        "check_result": None,
        "stat_result": None,
        "class_names": [],
    }

    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_check_summary(check_data: dict[str, Any]) -> None:
    """展示数据集检查结果。"""
    st.header("2. 数据集检查报告")

    st.subheader("基础数据汇总")

    summary_dataframe = pd.DataFrame(
        [
            {
                "统计项": "图片总数量",
                "数值": check_data.get("image_count", 0),
            },
            {
                "统计项": "标签文件总数量",
                "数值": check_data.get("label_count", 0),
            },
            {
                "统计项": "标注框总数量",
                "数值": check_data.get("annotation_count", 0),
            },
        ]
    )

    st.dataframe(
        summary_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("异常清单")

    missing_labels = check_data.get("missing_labels", [])
    orphan_labels = check_data.get("orphan_labels", [])
    empty_labels = check_data.get("empty_labels", [])
    invalid_annotations = check_data.get("invalid_annotations", [])
    errors = check_data.get("errors", [])

    if missing_labels:
        st.error(f"缺失标签的图片：{len(missing_labels)} 个")
        st.write(missing_labels)
    else:
        st.success("无缺失标签的图片文件")

    if orphan_labels:
        st.error(f"无匹配图片的孤立标签：{len(orphan_labels)} 个")
        st.write(orphan_labels)
    else:
        st.success("无孤立标签文件")

    if empty_labels:
        st.warning(f"空标签文件：{len(empty_labels)} 个")
        st.write(empty_labels)
    else:
        st.success("无空标签文件")

    if invalid_annotations:
        st.error(
            f"格式或坐标非法的标注："
            f"{len(invalid_annotations)} 条"
        )

        invalid_dataframe = pd.DataFrame(invalid_annotations)

        st.dataframe(
            invalid_dataframe.head(50),
            use_container_width=True,
            hide_index=True,
        )

        if len(invalid_annotations) > 50:
            st.info("当前仅展示前 50 条非法标注记录。")
    else:
        st.success("全部标注格式和坐标数值合法")

    if errors:
        st.error("检测过程中发现以下数据集结构或读取错误：")

        for error in errors:
            st.write(f"- {error}")


def display_statistics(stat_data: dict[str, Any]) -> None:
    """展示数据集统计分析结果。"""
    st.header("3. 数据集统计分析")

    image_count = stat_data.get("image_count", 0)
    label_count = stat_data.get("label_count", 0)
    total_objects = stat_data.get("total_objects", 0)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("图片数量", image_count)
    metric_col2.metric("标签文件数量", label_count)
    metric_col3.metric("标注目标总数", total_objects)

    st.subheader("各类别目标数量分布")

    class_counts = stat_data.get("class_counts", {})
    class_names = stat_data.get("class_names", {})

    class_rows = []

    for class_id, object_count in sorted(class_counts.items()):
        class_name = class_names.get(class_id, f"class_{class_id}")

        class_rows.append(
            {
                "类别ID": class_id,
                "类别名称": class_name,
                "目标数量": object_count,
            }
        )

    class_dataframe = pd.DataFrame(class_rows)

    if not class_dataframe.empty:
        chart_dataframe = class_dataframe.set_index("类别名称")

        st.bar_chart(
            chart_dataframe["目标数量"],
            use_container_width=True,
        )

        st.dataframe(
            class_dataframe,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("没有统计到有效的类别标注。")

    st.subheader("图片分辨率分布")

    image_sizes = stat_data.get("image_sizes", {})

    size_dataframe = pd.DataFrame(
        [
            {
                "图片分辨率": resolution,
                "文件数量": count,
            }
            for resolution, count in sorted(image_sizes.items())
        ]
    )

    if not size_dataframe.empty:
        st.dataframe(
            size_dataframe,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("没有统计到有效的图片分辨率信息。")

    st.subheader("单张图片目标数量")

    objects_per_image = stat_data.get(
        "objects_per_image",
        {
            "min": 0,
            "max": 0,
            "avg": 0.0,
        },
    )

    object_col1, object_col2, object_col3 = st.columns(3)

    object_col1.metric(
        "最少目标数",
        objects_per_image.get("min", 0),
    )
    object_col2.metric(
        "最多目标数",
        objects_per_image.get("max", 0),
    )
    object_col3.metric(
        "平均目标数",
        objects_per_image.get("avg", 0.0),
    )

    st.subheader("边界框面积分布")

    bbox_distribution = stat_data.get(
        "bbox_area_dist",
        {
            "small": 0,
            "medium": 0,
            "large": 0,
        },
    )

    bbox_dataframe = pd.DataFrame(
        [
            {
                "目标尺度": "小目标",
                "数量": bbox_distribution.get("small", 0),
            },
            {
                "目标尺度": "中目标",
                "数量": bbox_distribution.get("medium", 0),
            },
            {
                "目标尺度": "大目标",
                "数量": bbox_distribution.get("large", 0),
            },
        ]
    )

    st.bar_chart(
        bbox_dataframe.set_index("目标尺度")["数量"],
        use_container_width=True,
    )

    st.dataframe(
        bbox_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    statistics_errors = stat_data.get("errors", [])

    if statistics_errors:
        st.warning("统计过程中存在以下提示：")

        for error in statistics_errors:
            st.write(f"- {error}")


def get_image_files(dataset_path: str) -> list[Path]:
    """获取数据集 images 目录中的所有支持图片。"""
    images_directory = Path(dataset_path) / "images"

    if not images_directory.is_dir():
        return []

    return sorted(
        [
            file_path
            for file_path in images_directory.iterdir()
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_IMAGE_EXTENSIONS
            )
        ],
        key=lambda file_path: file_path.name.lower(),
    )


def display_visualization(dataset_path: str) -> None:
    """展示标注可视化预览。"""
    st.header("4. 标注可视化预览")

    if not dataset_path:
        st.info("请先输入数据集路径并执行数据集检测。")
        return

    image_files = get_image_files(dataset_path)

    if not image_files:
        st.warning(
            "当前数据集的 images 目录中没有可用图片。"
        )
        return

    image_name_to_path = {
        image_path.name: image_path
        for image_path in image_files
    }

    selected_image_name = st.selectbox(
        "选择需要查看的图片",
        options=list(image_name_to_path.keys()),
    )

    selected_image_path = image_name_to_path[
        selected_image_name
    ]

    label_path = (
        Path(dataset_path)
        / "labels"
        / f"{selected_image_path.stem}.txt"
    )

    if not label_path.exists():
        st.warning(
            f"当前图片没有对应标签文件：{label_path.name}"
        )

    try:
        with st.spinner("正在渲染标注图像……"):
            preview_image = draw_annotations(
                image_path=str(selected_image_path),
                label_path=str(label_path),
                class_names=st.session_state.class_names,
            )

        st.image(
            preview_image,
            caption=f"标注预览：{selected_image_name}",
            use_container_width=True,
        )

    except (OSError, ValueError) as error:
        st.error(f"标注图像渲染失败：{error}")


def run_dataset_inspection(dataset_path: str) -> None:
    """运行数据集检查与统计。"""
    if not dataset_path:
        st.error("请输入数据集文件夹路径。")
        return

    normalized_path = os.path.abspath(
        os.path.expanduser(dataset_path)
    )

    if not os.path.isdir(normalized_path):
        st.error(
            "填写的数据集路径不存在或不是文件夹，"
            "请检查后重新输入。"
        )
        return

    try:
        with st.spinner("正在扫描并检测数据集……"):
            check_result = inspect_dataset(normalized_path)
            stat_result = analyze_dataset(normalized_path)
            class_names = load_class_names(normalized_path)

        st.session_state.dataset_path = normalized_path
        st.session_state.check_result = check_result
        st.session_state.stat_result = stat_result
        st.session_state.class_names = class_names

        st.success("数据集检测任务完成！")

    except Exception as error:
        st.session_state.check_result = None
        st.session_state.stat_result = None
        st.error(f"数据集检测失败：{error}")


def main() -> None:
    """Streamlit Web 应用入口。"""
    st.set_page_config(
        page_title="YOLO Dataset Inspector",
        page_icon="🔍",
        layout="wide",
    )

    initialize_session_state()

    st.title(
        "YOLO Dataset Inspector"
        " — YOLO 数据集检查与可视化工具"
    )

    st.markdown(
        """
        本工具用于检查 YOLO 格式目标检测数据集，能够完成：

        - 图片与标签文件匹配检查
        - 空标签和孤立标签检查
        - 非法类别编号与边界框坐标检查
        - 类别数量和图片分辨率统计
        - 标注框可视化预览
        """
    )

    st.divider()

    st.header("1. 输入数据集路径")

    input_path = st.text_input(
        "数据集文件夹路径",
        value=st.session_state.dataset_path,
        placeholder=(
            r"例如：D:\datasets\my_yolo_dataset"
        ),
        help=(
            "数据集目录中应包含 images、labels，"
            "并建议包含 data.yaml。"
        ),
    )

    st.caption(
        "预期目录结构："
        "`dataset/images/`、`dataset/labels/`、"
        "`dataset/data.yaml`"
    )

    button_column, _ = st.columns([1, 3])

    with button_column:
        run_button = st.button(
            "开始检测",
            type="primary",
            use_container_width=True,
        )

    if run_button:
        run_dataset_inspection(input_path)

    if st.session_state.check_result is not None:
        st.divider()
        display_check_summary(
            st.session_state.check_result
        )

    if st.session_state.stat_result is not None:
        st.divider()
        display_statistics(
            st.session_state.stat_result
        )

    st.divider()

    visualization_path = (
        st.session_state.dataset_path
        if st.session_state.check_result is not None
        else input_path
    )

    display_visualization(visualization_path)


if __name__ == "__main__":
    main()
