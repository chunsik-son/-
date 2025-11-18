# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:41:01 2025

@author: user
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

# -----------------------------
# 1️⃣ Streamlit UI: 파일 업로드
# -----------------------------
st.title("🔒 시스템 로그 기반 이상 탐지 대시보드")
st.write("Autoencoder를 이용해 로그 데이터 내 비정상 패턴을 탐지합니다.")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:

    # -----------------------------
    # 2️⃣ 데이터 로드 및 전처리
    # -----------------------------
    @st.cache_data
    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        df = df.dropna()
        numeric_df = df.select_dtypes(include=[np.number])  # 숫자형 데이터만 사용
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(numeric_df)
        return scaled, numeric_df.columns

    data, cols = load_data(uploaded_file)
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

    # -----------------------------
    # 3️⃣ 오토인코더 구성 함수
    # -----------------------------
    def build_autoencoder(input_dim):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(16, activation='relu',
                        activity_regularizer=regularizers.l1(1e-5))(input_layer)
        encoded = Dense(8, activation='relu')(encoded)
        encoded = Dense(4, activation='relu')(encoded)
        decoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(decoded)
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model

    # -----------------------------
    # 4️⃣ 콜백 클래스: 학습 로그 누적 출력 (줄 단위, 실시간)
    # -----------------------------
    class StreamlitCallback(Callback):
        def __init__(self):
            super().__init__()
            self.log_container = st.empty()  # 로그 출력 공간
            self.logs = []

        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get("loss") if logs else None
            val_loss = logs.get("val_loss") if logs else None
            line = f"Epoch {epoch+1}/{self.params['epochs']} - loss: {loss:.5f}"
            if val_loss is not None:
                line += f", val_loss: {val_loss:.5f}"
            self.logs.append(line)
            self.log_container.text("\n".join(self.logs))

    # -----------------------------
    # 5️⃣ 모델 학습 또는 불러오기 (학습 중 진행상황 보여줌)
    # 캐시 제거하여 실시간 로그 출력 보장
    # -----------------------------
    def load_or_train_autoencoder(X_train):
        if not os.path.exists("autoencoder.h5"):
            st.warning("⚠️ 모델 파일이 없습니다. 새로 학습을 시작합니다...")
            model = build_autoencoder(X_train.shape[1])
            callback = StreamlitCallback()
            model.fit(
                X_train, X_train,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                callbacks=[callback]
            )
            model.save("autoencoder.h5")
            st.success("✅ 모델 학습 및 저장 완료!")
        else:
            model = load_model("autoencoder.h5", compile=False)
            st.success("✅ 저장된 모델 불러오기 완료!")
        return model

    model = load_or_train_autoencoder(X_train)

    # -----------------------------
    # 6️⃣ 이상치 탐지 함수
    # -----------------------------
    def detect_anomalies(model, X_test, threshold=None):
        preds = model.predict(X_test)
        mse = np.mean(np.power(X_test - preds, 2), axis=1)
        if threshold is None:
            threshold = np.mean(mse) + 3 * np.std(mse)
        anomalies = mse > threshold
        return mse, anomalies, threshold

    mse, anomalies, threshold = detect_anomalies(model, X_test)

    # -----------------------------
    # 7️⃣ 시각화
    # -----------------------------
    st.subheader("📈 Reconstruction Error (재구성 오차)")
    fig, ax = plt.subplots()
    ax.hist(mse, bins=50, color='gray')
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    ax.legend()
    st.pyplot(fig)

    st.subheader("⚠️ 이상치 비율")
    anomaly_ratio = np.mean(anomalies) * 100
    st.metric(label="Detected Anomalies (%)", value=f"{anomaly_ratio:.2f}%")

    st.write("이상 탐지 완료 ✅")

else:
    st.info("CSV 파일을 업로드해야 분석을 시작할 수 있습니다.") 
