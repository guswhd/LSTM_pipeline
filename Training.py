'''
Training.py

Input: arg[1] : <window_size>, arg[2] : <periods>
Output :
    1. 스케일러
        (경로: ./Output/Learning_lstm/scaler.pkl')
    2. LSTM 모델
        (경로: ./Output/Learning_lstm/lstm_model.pkl')
    3. LSTM 성능 평가 그래프
        (경로: ./Output/Learning_lstm/lstm_performance_graph.tiff')
순서 :
    1. 데이터 읽어오기 
        (경로: ./Output/_1_Data_preprocessing/preprocessed.csv)
    2. 학습 데이터, 테스트 데이터 분할 (7:3)
    3. 데이터 스케일링
    4. x_train, y_train, x_test 생성
    5. LSTM 모델 생성
    6. 모델 성능 평가
    7. 스케일러, 모델, 그래프 저장
        (경로: ./Output/Learning_lstm/)  
'''
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
from joblib import dump
import pandas as pd
import numpy as np
import sys
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# 데이터 읽어오기
def read_data(filename):
    df = pd.read_csv(filename, 
                    parse_dates=['time'], index_col=0)
    col = df.columns[0]
    df[col] = df[col].astype(float)

    return df

# 학습 데이터 / 테스트 데이터 분할 (7:3)
def train_test_split(df):
    train_size = int(0.7 * len(df))
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    return train, test

# 데이터 스케일링
def data_scaling(train):
    train_data = train.values
    sc = MinMaxScaler(feature_range=(0, 1)) # 정규화 
    sc = sc.fit(train_data) 

    return sc

# x_train, y_train 생성
def create_train(train,window_size, periods, sc):
    
    train_data = train.values
    train_len = len(train_data)
    
    # 데이터 스케일링
    train_scaled = sc.transform(train_data)
    
    # x, y 학습데이터 생성
    x_train = []
    y_train = []
    for i in range(train_len - window_size - periods + 1):
        x_train_end = i+window_size
        x_train.append(train_scaled[i:x_train_end, 0]) 
        y_train.append(train_scaled[x_train_end:x_train_end+periods, 0])
    
    # 리스트를 넘파이 배열로 변환
    x_train, y_train = np.array(x_train), np.array(y_train)

    # x_train를 텐서로 변환
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train

# x_test 생성
def create_x_test(test,window_size, sc):
    test_data = test.values
    test_len = len(test_data)    
    
    # 데이터 스케일링
    test_scaled = sc.transform(test_data)
    
    # x 테스트 데이터 생성
    x_test = []
    for i in range(test_len - window_size):
        x_test_end = i+window_size
        x_test.append(test_scaled[i:x_test_end, 0]) 
    
    # 리스트를 넘파이 배열로 변환
    x_test =  np.array(x_test)

    # x_test를 텐서로 변환
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_test


# 모델 생성
def lstm_arch(x_train, y_train, periods, epochs, batch_size):
    # 모델 구성 (LSTM 레이어 2개, Dropout 추가)
    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh'),
        Dropout(0.2),  # Dropout 추가
        LSTM(units=50, activation='tanh'),
        Dropout(0.2),  # Dropout 추가
        Dense(units=periods)
    ])
    
    # 모델 컴파일 (optimizer : SGD, 손실함수: MSE 사용)
    lstm_model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7, momentum = 0.9, nesterov = False), loss='mean_squared_error')
    
    # 모델 학습
    lstm_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    
    return lstm_model

# 모델 성능 평가
def lstm_performance(lstm_model, sc, x_test, test, epochs, batch_size):
    # 예측
    preds = lstm_model.predict(x_test)
    # 원래 값으로 변환
    preds = sc.inverse_transform(preds)
    
    # 실제값과 예측값을 비교하기 위한 데이터프레임 생성
    predictions_plot = pd.DataFrame(columns=['actual', 'prediction'])
    predictions_plot['actual'] = test.iloc[0:len(preds), 0]
    predictions_plot['prediction'] = preds[:, 0]
    
    # RMSE 계산
    mse = MeanSquaredError()
    mse.update_state(np.array(predictions_plot['actual']), np.array(predictions_plot['prediction']))
    RMSE = np.sqrt(mse.result().numpy())
    
    # 그래프
    return (predictions_plot.plot(figsize=(15, 5), 
                                title=f'lstm performance\nepochs={epochs}, batch size={str(batch_size)}, RMSE={str(round(RMSE, 4))}'))

# 스케일러, 모델, 그래프 저장
def save_output(sc, lstm_model, plot):
    # 출력 폴더 생성
    output_path = './Output/Learning_lstm'
    os.makedirs(output_path,exist_ok=True)

    dump(sc, f'{output_path}/scaler.pkl') # scaler 저장
    lstm_model.save(f'{output_path}/lstm_model.h5') # 모델 저장
    plt.savefig(f'{output_path}/lstm_performance_graph.tiff') # 그래프 저장


# 모델 학습 (에포크=700, 배치 사이즈=32)
def training(window_size, periods):
    # 베스트 모델의 하이퍼 파라미터 사용
    epochs = 1500 
    batch_size = 32

    # 데이터 읽어오기
    filename = './Output/Data_preprocessing/preprocessed.csv'
    df = read_data(filename)
    
    # 학습 데이터 / 테스트 데이터 분할
    train, test = train_test_split(df)

    # 데이터 스케일링
    sc = data_scaling(train)

    #  x_train, y_train, x_test 생성
    x_train, y_train = create_train(train,window_size, periods, sc)
    x_test = create_x_test(test,window_size, sc)

    # 모델 생성
    lstm_model = lstm_arch(x_train, y_train, periods, epochs, batch_size)

    # 모델 성능 평가
    plot = lstm_performance(lstm_model, sc, x_test, test, epochs, batch_size)

    # 스케일러, 모델, 그래프 저장
    save_output(sc, lstm_model, plot)


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # window size 입력 받아오기
    window_size = int(sys.argv[1])
    periods = int(sys.argv[2])

    training(window_size, periods)
