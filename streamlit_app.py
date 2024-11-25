import streamlit.components.v1 as components
components.html(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css">
    """,
    height=0
)

import streamlit as st
import pandas as pd
import joblib

# 加载模型和向量化器
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def predict_news(text):
    # 将用户输入转换为TF-IDF特征
    text_vec = vectorizer.transform([text])
    # 使用模型进行预测
    prediction = model.predict(text_vec)
    return prediction[0]

# 创建Streamlit应用
def main():
    st.title('<i class="fas fa-newspaper"></i> 新闻真伪预测', unsafe_allow_html=True)

    st.markdown("""
    请输入新闻文本，我们将预测其真伪。
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_area('请输入新闻文本：', key='user_input')
    with col2:
        if st.button('预测', key='predict_button'):
            if user_input:
                prediction = predict_news(user_input)
                if prediction == 1:
                    st.success('这是一条真实新闻。')
                else:
                    st.error('这是一条假新闻。')
            else:
                st.warning('请输入新闻文本后再点击预测。')

if __name__ == "__main__":
    main()