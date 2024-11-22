import streamlit as st
from PIL import Image



st.title("Информация по моделям")

st.write("### Детекция лиц с помощью любой версии YOLO c последующей маскировкой детектированной области")

st.markdown("#### Число эпох - 10")
st.markdown("#### Объем выборок:")
st.markdown("1. тренировочная выборка - 13 400 изображений, 2. валидационная - 3 347 изображений")

st.subheader("Метрики модели")

conf_mat_face = Image.open('images_metrics/confusion_matrix_face.png')
results_face = Image.open('images_metrics/results_face.png')
pr_curve_face = Image.open('images_metrics/PR_curve_face.png')
p_curve_face = Image.open('images_metrics/P_curve_face.png')
r_curve_face = Image.open('images_metrics/R_curve_face.png')
f1_curve_face = Image.open('images_metrics/F1_curve_face.png')

st.markdown("### Графики Loss-функции")
st.image(results_face, caption=' ', use_container_width=True)

st.markdown("### Precision-recall Кривая")
st.image(pr_curve_face, caption=' ', use_container_width=True)

st.markdown("### F1-Кривая")
st.image(f1_curve_face, caption=' ', use_container_width=True)

st.markdown("### Precision Кривая")
st.image(p_curve_face, caption=' ', use_container_width=True)

st.markdown("### Recall Кривая")
st.image(r_curve_face, caption=' ', use_container_width=True)

st.markdown("### Матрица ошибок")
st.image(conf_mat_face, caption=' ', use_container_width=True)




