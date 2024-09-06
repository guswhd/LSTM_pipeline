'''
Prediction.py

Input: arg[1] : <파일 이름>, 
    arg[2] : <학습데이터의 통계 파일의 이름>
Output : 예측 데이터
    (./Output/Prediction/prediction.csv)

순서 : 
    1. 데이터 읽어오기 (Raw data (time/Values))
    2. 데이터 전처리: 결측치 처리, 하루 간격으로 병합
    3. 모델과 스케일러 불러오기
    4. 모델의 윈도우 사이즈 찾기
    5. x_data 생성 
    6. 예측
    7. 예측값에 대한 시계열 데이터 생성
    8. 이상치 검출
    9. 레이블 채우기
    10. 예측 데이터 저장
        (경로: ./Output/Prediction/)
'''
import importlib
preprocessing = importlib.import_module('Data_preprocessing')
training = importlib.import_module('Training')
abnormal_detection = importlib.import_module('Abnormal_detection')
import pandas as pd
import sys
import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 데이터 읽어오기
def read_data(file_name):
    # 데이터 읽어오기
    df = pd.read_csv(file_name,index_col = 0) 

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

# 예측 데이터 생성
def create_predictive_data(df, preds):
    col = df.columns[0]
    last_index = df.index[-1] # # 데이터의 마지막 인덱스 찾기
    periods = preds.shape[1] # # prediction window size 찾기
    freq = pd.infer_freq(df.index) # 시간 간격 찾기

    # 데이트타임 생성
    preds_index = pd.date_range(str(last_index), periods=(periods+1), freq=freq)
    preds_index = preds_index[1:] # 예측값의 데이트타임 추출

    # 예측값의 시계열 데이터 생성
    periods_df = pd.DataFrame(index=preds_index, columns=[col, 'label'])
    periods_df[col] = list(preds[-1])
    periods_df.index.name =  'time'

    return periods_df

# 레이블 채우기
def fill_label(periods_df, abnormal_df):
    col = periods_df.columns[0]
    # 예측 데이터와 이상치 데이터의 인덱스에서 교집합 추출
    common_index = periods_df.index.intersection(abnormal_df.index)
    # label 채우기
    periods_df.loc[common_index,'label'] = 'Abnormal' # 교집합: Abnormal
    periods_df['label'].fillna('Normal', inplace=True) # 아니면 :Normal

    for index, row in periods_df.iterrows():
        print(f'\n시간: {index} - 예측 값: {row[col]}, 레이블: {row["label"]}')

    return periods_df

# 예측 데이터 저장 ( Time / Values / Label )
def save_data(periods_df):
    # 출력 폴더 생성
    output_path = './Output/Prediction'
    os.makedirs(output_path,exist_ok=True)

    # 예측한 데이터 저장
    periods_df.to_csv(f'{output_path}/prediction.csv') 


def prediction(file_name, criteria):
    # 데이터 읽어오기
    df = read_data(file_name)
    # 데이터 전처리: 결측치 처리, 시간 간격으로 병합
    if df.isnull().any().any():
        df = preprocessing.Missing(df)
    df = preprocessing.TimeIntervalData(df, '1D') # 1시간 간격으로 병합 

    # 모델과 스케일러 불러오기
    sc, lstm_model = abnormal_detection.load_model_and_scaler()

    # 모델의 윈도우 사이즈 찾기
    window_size = lstm_model.layers[0].input_shape[1]
    
    # x_data 생성
    x_data = abnormal_detection.create_x_new_data(df, window_size, sc)

    # 예측
    preds = abnormal_detection.predictions(x_data, lstm_model, sc)

    # 예측값에 대한 시계열 데이터 생성
    periods_df = create_predictive_data(df, preds)

    # 이상치 검출
    abnormal_df = abnormal_detection.detection(pd.DataFrame(periods_df.iloc[:,0]), criteria)

    # 레이블 채우기
    periods_df = fill_label(periods_df, abnormal_df)

    # 예측 데이터 저장
    save_data(periods_df)


if __name__ == "__main__":
    # 인자 받기
    file_name = sys.argv[1]  # 파일 이름
    criteria = sys.argv[2]  # 학습 데이터의 통계표

    prediction(file_name, criteria)