import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 데이터 불러오기
data = pd.read_csv("./data/5.gt_full.csv")

# 데이터 확인
print("Gas Turbine Dataset:\n")
print(data.head())  # 데이터의 상위 몇 행을 출력하여 확인합니다.

# 불필요한 데이터 제거
data = data.drop("Unnamed: 0", axis=1)
print("\nDropping unnecessary columns:")
print(data.head())

# 데이터 프레임의 열 확인
print("\nColumns in Dataframe:")
cols = data.columns
print(cols)

# 데이터 설명 및 정보 출력
print("\nDataset description:")
print(data.describe())

def plot_distributions_and_relationships(data, feature_cols, target_col):
    num_cols = len(data.columns)
    ncols = 4  # 한 행에 나타낼 서브플롯의 수
    nrows = (num_cols + ncols - 1) // ncols  # 전체 행 수 계산

    # 피처의 분포를 히스토그램으로 시각화
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
    for i, col in enumerate(data.columns):
        ax = axes[i // ncols, i % ncols]
        sns.histplot(x=col, data=data, ax=ax, kde=False)
        ax.set_ylabel("Count")
        ax.grid(True)

    # 빈 서브플롯 제거
    for j in range(num_cols, nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    plt.show()

    # NOX와 각 피처 간의 관계를 산점도로 시각화
    ncols = 4
    nrows = (len(feature_cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
    for i, col in enumerate(feature_cols):
        ax = axes[i // ncols, i % ncols]
        sns.scatterplot(x=col, y=target_col, data=data, ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        ax.grid(True)

    # 빈 서브플롯 제거
    for j in range(len(feature_cols), nrows * ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.tight_layout()
    plt.show()

    # 상관 행렬을 히트맵으로 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# 피처와 타겟 변수 설정
feature_cols = cols[:-1]  # 마지막 열은 NOX이므로 제외
target_col = "NOX"

# 데이터 시각화
plot_distributions_and_relationships(data, feature_cols, target_col)

# 입력 변수와 출력 변수 분리
x = data[feature_cols]
y = data[target_col]

print("\nInput Variables:")
print(x.head())

print("\nOutput variable:")
print(y.head())

# 데이터 분할 (훈련 데이터와 테스트 데이터)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
print("\nTrain-Test Split:")
print("Training variables:")
print("Input:")
print(x_train.head())
print("\nOutput:")
print(y_train.head())
print('\nTesting variables:')
print("Input:")
print(x_test.head())
print("\nOutput:")
print(y_test.head())

# 입력 변수 표준화
print("\nStandard scaling the input training variables:")
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
print(pd.DataFrame(x_train_scaled, columns=x_train.columns).head())

print("\nStandard scaling the input testing variables:")
x_test_scaled = scaler.transform(x_test)
print(pd.DataFrame(x_test_scaled, columns=x_test.columns).head())

# 다양한 회귀 모델을 훈련하고 성능 평가
models = [LinearRegression(), Lasso(), Ridge(), KNeighborsRegressor(), DecisionTreeRegressor(), ExtraTreesRegressor()]
r2_scores = []
print("Model Training:\n")
for model in models:
    print(f"Model used: {model}.")
    model.fit(x_train_scaled, y_train)  # 스케일링된 데이터로 모델 훈련
    y_pred = model.predict(x_test_scaled)  # 스케일링된 데이터로 예측
    r2 = round(r2_score(y_test, y_pred), 4)  # R2 점수 계산
    print(f"Accuracy Acquired: {r2}.")
    r2_scores.append(r2)
    print()

# 가장 높은 R2 점수를 기록한 모델 찾기
max_r2 = max(r2_scores)
print(f"Best R2 Score Recorded: {max_r2}.")
max_idx = r2_scores.index(max_r2)
best_model = models[max_idx]


# 선택된 모델로 추가 평가
print("\n\nModel1 ExtraTreesRegressor:")
et = ExtraTreesRegressor()
et.fit(x_train_scaled, y_train)  # 스케일링된 데이터로 모델 훈련
y_pred = et.predict(x_test_scaled)  # 스케일링된 데이터로 예측
r2 = round(r2_score(y_test, y_pred), 4)  # R2 점수 계산
mae = mean_absolute_error(y_test, y_pred)  # 평균 절대 오차 계산
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차 계산
print(f"R2 Score: {r2}.")
print(f"Mean Squared Error: {mse}.")
print(f"Mean Absolute Error: {mae}.")

# 예측값과 실제값의 관계를 산점도로 시각화
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('ExtraTreesRegressor')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 대각선 추가
plt.grid(True)
plt.show()

print("Model2 LinearRegression:")
lt = LinearRegression()
lt.fit(x_train_scaled, y_train)  # 스케일링된 데이터로 모델 훈련
y_pred = lt.predict(x_test_scaled)  # 스케일링된 데이터로 예측
r2 = round(r2_score(y_test, y_pred), 4)  # R2 점수 계산
mae = mean_absolute_error(y_test, y_pred)  # 평균 절대 오차 계산
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차 계산
print(f"R2 Score: {r2}.")
print(f"Mean Squared Error: {mse}.")
print(f"Mean Absolute Error: {mae}.")

# 예측값과 실제값의 관계를 산점도로 시각화
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('LinearRegression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 대각선 추가
plt.grid(True)
plt.show()

print(f"\nBest Model Performance: {best_model}.")

# 모델 저장
print("Model Saved.")
joblib.dump(et, 'Extra_Trees_Regressor_Model.joblib')

# 모델 로드 경로 설정 및 모델 로드
model_paths = ["./data/Extra_Trees_Regressor_Model.joblib", "./data/input/gas-turbine-model/Extra_Trees_Regressor_Model.joblib"]
model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        break
print(f"Model to be extracted from {model_path}.")

loaded_model = joblib.load(model_path)
print("Extracted Model:")
print(loaded_model)

# 새로운 데이터에 대한 예측
print("Enter the input details:")
AT = 7.7533
AP = 1011.8
AH = 73.067
AFDP = 2.6621
GTEP = 20.886
TIT = 1039.7
TAT = 545.41
TEY = 109.27
CDP = 10.390
CO = 8.06510
new_data = np.array([[AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP, CO]])
print("New data for prediction:")
print(new_data)
new_data_scaled = scaler.transform(new_data)
print("Scaled new data:")
print(new_data_scaled)
NOX_pred = round(loaded_model.predict(new_data_scaled)[0], 2)
print(f"Predicted Nitrous Oxides Emission Amount: {NOX_pred}")
