'''
Abnormal_detection.py

Input: arg[1] : <파일 이름>, arg[2] : <criteria>
Output : 이상치 데이터 
    (./Output/Prediction/abnormal.csv)

순서:
    1. 데이터 읽어오기 ( Raw data (Time/Values))  
    2. 결측치 보정 
    3. 모델과 스케일러 불러오기
    4. x_data 생성  
    5. 예측
    6. 예측값에 대한 시계열 데이터 생성
    7. 이상치 검출
    8. 이상치 데이터 저장
        (./Output/Prediction/abnomal.csv)
'''
import importlib
preprocessing = importlib.import_module('Data_preprocessing')
training = importlib.import_module('Training')
from keras.models import load_model
from joblib import load
import pandas as pd
import numpy as np
import sys
import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 데이터 읽어오기
def read_data(file_name):
    df = pd.read_csv(file_name, index_col = 0) 

    # 데이터프레임 인덱스를 데이트타임인덱스로 변환
    df.index = pd.to_datetime(df.index)
    df.index.name = 'time'

    # values 이름 추출
    col = df.columns[0]

    # float 변환
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    df[col] = df[col].astype(float)
    
    # 인덱스 정렬
    df = df.sort_index(ascending=True) 
    return df

# 모델과 스케일러 불러오기
def load_model_and_scaler():
    # 스케일러 불러오기
    sc = load('./Output/Learning_lstm/scaler.pkl')

    # 모델 불러오기 
    lstm_model = load_model('./Output/Learning_lstm/lstm_model.h5')

    return sc, lstm_model

# x_data 생성
def create_x_new_data(new_data, window_size, sc):
    data = new_data.values
    data_len = len(data)
    
    # 학습 데이터 스케일링
    data_scaled = sc.transform(data)
    
    # x 학습데이터 생성
    x_data = []

    for i in range(data_len - window_size):
        x_data_end = i+window_size 
        x_data.append(data_scaled[i:x_data_end, 0])
 
    # 리스트를 넘파이 배열로 변환
    x_data =  np.array(x_data)


    # x_data를 텐서로 변환
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    return x_data

# 예측하기
def predictions(x_data, lstm_model, sc):
    # 예측
    preds = lstm_model.predict(x_data)
    # 원래 값으로 변환
    preds = sc.inverse_transform(preds)
    return preds

# 예측값에 대한 시계열 데이터 생성
def create_time_series_data(df, preds):
    # values 이름 추출
    col = df.columns[0]

    # 예측값에 대한 시계열 데이터 생성
    preds_df = pd.DataFrame(columns=[col],
                            index=(df.loc[:, col][0:len(preds)]).index)
    preds_df[col] = preds[:, 0]

    return preds_df

# 이상치 검출
def detection(preds_df, criteria):
    # 학습 데이터의 통계표 읽어오기
    table = pd.read_csv(criteria, index_col = 0) 
    
    # 열 추출
    column_values = preds_df.iloc[:,0]

    # 이상치 데이터 생성
    if all(idx in table.index for idx in ['U_level', 'L_level']): # U_level와 L_level가 있는 경우
        U_level = table.loc['U_level']['Values'] # upper limit 
        L_level = table.loc['L_level']['Values'] # lower limit 
        abnormal_df = preds_df[(column_values <= L_level) | (column_values >= U_level)]
    
    elif 'U_level' in table.index: # U_level만 있는 경우
        U_level = table.loc['U_level']['Values']
        abnormal_df = preds_df[column_values >= U_level]
    
    elif 'L_level' in table.index: # L_level만 있는 경우
        L_level = table.loc['L_level']['Values'] 
        abnormal_df = preds_df[column_values <= L_level]
    
    
    if  abnormal_df.empty:
        print('이상치가 없습니다.')
    
    return abnormal_df

# 이상치 데이터 저장
def save_data(abnormal_df):
    # 폴더 생성
    output_path = './Output/Prediction'
    os.makedirs(output_path, exist_ok=True)
     
    # 이상치 데이터 저장
    abnormal_df.to_csv(f'{output_path}/abnomal.csv')


def abnormal_detection(file_name,criteria):
    # 파일 읽어오기
    df = read_data(file_name)

    # 데이터 전처리: 결측치 처리
    if df.isnull().any().any():
        df = preprocessing.Missing(df)

    # 모델과 스케일러 불러오기
    sc, lstm_model = load_model_and_scaler()

    # 모델의 윈도우 사이즈 찾기
    window_size = lstm_model.layers[0].input_shape[1]
    
    # x_data 생성
    x_data = create_x_new_data(df, window_size, sc)

    # 예측
    preds = predictions(x_data, lstm_model, sc)

    # 예측값에 대한 시계열 데이터 생성
    preds_df = create_time_series_data(df, preds)

    # 이상치 추출
    abnormal_df = detection(preds_df, criteria)
    
    # 이상치 데이터 저장
    save_data(abnormal_df)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    
    # 인자 받아오기
    file_name = sys.argv[1] # 파일 이름
    criteria = sys.argv[2] # 통계표
    # window_size = int(sys.argv[3])

    abnormal_detection(file_name,criteria)
