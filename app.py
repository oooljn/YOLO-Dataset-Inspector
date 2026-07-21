import streamlit as st
import os
import yaml
import pandas as pd
from PIL import Image
from src.checker import inspect_dataset
from src.visualizer import draw_annotations
from src.statistics import analyze_dataset


def main():
    # 页面基础配置
    st.set_page_config(page_title="YOLO Dataset Inspector", layout="wide")
    st.title("YOLO Dataset Inspector — YOLO数据集检查与可视化工具")

    # 会话状态初始化
    if "dataset_path" not in st.session_state:
        st.session_state.dataset_path = ""
    if "check_result" not in st.session_state:
        st.session_state.check_result = None
    if "stat_result" not in st.session_state:
        st.session_state.stat_result = None
    if "class_names" not in st.session_state:
        st.session_state.class_names = []

    # 加载data.yaml类别名称函数
    def load_class_names(ds_path):
        yaml_file = os.path.join(ds_path, "data.yaml")
        if os.path.exists(yaml_file):
            with open(yaml_file, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f)
                return cfg_data.get("names", [])
        return []

    # 1、数据集路径输入区域
    st.header("1. 输入数据集路径")
    input_path = st.text_input(
        "数据集文件夹路径（内部需包含images、labels、data.yaml）",
        value=st.session_state.dataset_path
    )
    st.session_state.dataset_path = input_path

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_inspect = st.button("Start Inspection", type="primary")

    # 点击检测按钮执行校验、统计
    if run_inspect:
        if not os.path.isdir(input_path):
            st.error("填写的文件夹路径不存在，请核对路径！")
        else:
            with st.spinner("正在扫描并检测数据集..."):
                st.session_state.check_result = inspect_dataset(input_path)
                st.session_state.stat_result = analyze_dataset(input_path)
                st.session_state.class_names = load_class_names(input_path)
            st.success("数据集检测任务完成！")

    # 2、数据集检查报告展示
    if st.session_state.check_result is not None:
        st.divider()
        st.header("2. 数据集检查报告")
        check_data = st.session_state.check_result

        # 基础汇总表格
        st.subheader("基础数据汇总")
        base_df = pd.DataFrame([
            {"统计项": "图片总数量", "数值": check_data["image_count"]},
            {"统计项": "标签文件总数量", "数值": check_data["label_count"]},
            {"统计项": "标注框总数量", "数值": check_data["annotation_count"]}
        ])
        st.dataframe(base_df, use_container_width=True)

        # 各类异常提示
        st.subheader("异常清单")
        if len(check_data["missing_labels"]) > 0:
            st.error(f"缺失标签的图片（{len(check_data['missing_labels'])}个）")
            st.write(check_data["missing_labels"])
        else:
            st.success("✅ 无缺失标签的图片文件")

        if len(check_data["orphan_labels"]) > 0:
            st.error(f"无匹配图片的孤立标签（{len(check_data['orphan_labels'])}个）")
            st.write(check_data["orphan_labels"])
        else:
            st.success("✅ 无孤立的标签txt文件")

        if len(check_data["invalid_annotations"]) > 0:
            st.error(f"格式/坐标非法的标注（{len(check_data['invalid_annotations'])}条）")
            st.write(check_data["invalid_annotations"][:20])
        else:
            st.success("✅ 全部标签格式、坐标数值合法")

    # 3、数据集统计可视化
    if st.session_state.stat_result is not None:
        st.divider()
        st.header("3. 数据集统计分析")
        stat_data = st.session_state.stat_result

        st.subheader("各类别目标物体数量分布")
        class_df = pd.DataFrame(list(stat_data["class_counts"].items()), columns=["类别ID", "目标数量"])
        st.bar_chart(class_df, x="类别ID", y="目标数量", use_container_width=True)
        st.dataframe(class_df, use_container_width=True)

        st.subheader("图片分辨率分布")
        size_df = pd.DataFrame(list(stat_data["image_sizes"].items()), columns=["图片分辨率", "文件数量"])
        st.dataframe(size_df, use_container_width=True)
        st.info(f"数据集内所有标注物体总数量：{stat_data['total_objects']}")

    # 4、标注图片预览模块
    st.divider()
    st.header("4. 标注可视化预览")
    images_dir = os.path.join(input_path, "images")
    support_img_ext = (".jpg", ".jpeg", ".png", ".bmp")
    img_file_list = []

    if os.path.exists(images_dir):
        for filename in sorted(os.listdir(images_dir)):
            if filename.lower().endswith(support_img_ext):
                img_file_list.append(filename)

    if img_file_list:
        pick_img = st.selectbox("选择图片查看绘制后的标注框", img_file_list)
        full_img_path = os.path.join(images_dir, pick_img)
        file_stem = os.path.splitext(pick_img)[0]
        full_label_path = os.path.join(input_path, "labels", f"{file_stem}.txt")

        with st.spinner("正在渲染标注图像..."):
            preview_img = draw_annotations(full_img_path, full_label_path, st.session_state.class_names)
        st.image(preview_img, caption=f"标注预览 | {pick_img}", use_column_width=True)
    else:
        st.warning("当前路径下images文件夹无可用图片，请先执行数据集检测")



if __name__ == "__main__":
    main()
