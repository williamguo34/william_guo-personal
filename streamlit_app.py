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
    # 设置页面标题
    st.markdown('<h1 style="text-align: center; color: #007bff;"><i class="fas fa-newspaper"></i> 新闻真伪预测</h1>', unsafe_allow_html=True)
    
    # 添加一些间距
    st.markdown("<br>", unsafe_allow_html=True)

    # 创建输入卡片
    with st.container():
        st.markdown(
            """
            <div class="card">
            """, unsafe_allow_html=True)
        user_input = st.text_area('请输入新闻文本：', height=150)
        
        if st.button('预测'):
            if user_input:
                prediction = predict_news(user_input)
                if prediction == 1:
                    st.success('这是一条真实新闻。')
                else:
                    st.error('这是一条假新闻。')
            else:
                st.warning('请输入新闻文本后再点击预测。')

        st.markdown("</div>", unsafe_allow_html=True)  # 结束输入卡片

if __name__ == "__main__":
    main()