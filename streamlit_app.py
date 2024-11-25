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
    # 使用 st.markdown 来允许HTML内容
    st.markdown('<h1 style="font-family:Arial, sans-serif;"><i class="fas fa-newspaper"></i> 新闻真伪预测</h1>', unsafe_allow_html=True)
    
    # 获取用户输入
    user_input = st.text_area('请输入新闻文本：')
    
    # 检查用户是否输入了数据
    if st.button('预测'):
        # 由于用户可能在输入文本后直接点击预测，因此需要检查文本框是否为空
        if user_input:
            # 使用模型进行预测
            prediction = predict_news(user_input)
            # 显示预测结果
            if prediction == 1:
                st.success('这是一条真实新闻。')
            else:
                st.error('这是一条假新闻。')
        else:
            st.warning('请输入新闻文本后再点击预测。')

if __name__ == "__main__":
    main()