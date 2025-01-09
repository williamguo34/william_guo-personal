import streamlit as st
import joblib

# 加载模型和向量化器
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# 预测函数
def predict_news(text):
    if len(text) < 5:
        return "short", None
    # 将用户输入转换为TF-IDF特征
    text_vec = vectorizer.transform([text])
    # 使用模型进行预测
    prediction = model.predict(text_vec)
    prediction_proba = model.predict_proba(text_vec)
    threshold = 0.84
    if prediction_proba[0][prediction[0]] < threshold:
        return "uncertain", prediction_proba[0][1]
    return prediction[0], prediction_proba[0][prediction[0]]

def main():
    # 设置页面标题
    st.markdown('<h1 style="text-align: center; color: #007bff;"><i class="fas fa-newspaper"></i> 新闻真伪预测</h1>', unsafe_allow_html=True)
    # 添加一些间距
    st.markdown("<br>", unsafe_allow_html=True)

    # 初始化状态变量
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "prediction_proba" not in st.session_state:
        st.session_state.prediction_proba = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = None

    # 输入框
    user_input = st.text_area("请输入新闻文本：", height=150)

    # 预测按钮
    if st.button("预测"):
        if user_input.strip():
            prediction, prediction_proba = predict_news(user_input)
            st.session_state.prediction = prediction
            st.session_state.prediction_proba = prediction_proba
            st.session_state.feedback = None  # 清空之前的反馈
        else:
            st.warning("请输入新闻文本后再点击预测！")
    
    # 显示预测结果
    if st.session_state.prediction is not None:
        prediction = st.session_state.prediction
        prediction_proba = st.session_state.prediction_proba

        if prediction == "short":
            st.warning("输入的文本信息不足，无法进行真伪判断。")
        elif prediction == "uncertain":
            st.warning(f"这是一条不确定的新闻。可信度：{prediction_proba:.2f}")
        elif prediction == 1:
            st.success(f"这是一条真实新闻。可信度：{prediction_proba:.2f}")
        else:
            st.error(f"这是一条假新闻。可信度：{prediction_proba:.2f}")

        # 显示反馈按钮
        st.write("你同意这个预测吗？")
        sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
        col1, col2 = st.columns(2)
        with col1:
            if st.button(sentiment_mapping[1]):
                st.session_state.feedback = "同意"
        with col2:
            if st.button(sentiment_mapping[0]):
                st.session_state.feedback = "不同意"
                actual_label = "假新闻" if prediction == 1 else "真实新闻"
                predict_label = "假新闻" if prediction == 0 else "真实新闻"
    
    # 显示反馈结果
    if st.session_state.feedback == "同意":
        st.success("感谢你的反馈！")
    elif st.session_state.feedback == "不同意":
        st.error(f"感谢你的反馈：不同意。实际：{actual_label}，预测：{predict_label}")

if __name__ == "__main__":
    main()
