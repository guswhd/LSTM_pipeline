'''
Data_preprocessing.py

Input: arg[1] : <파일 이름>, arg[2] : <이상치 레벨(%)>
Output :
    1. 결측치, 이상치 보정한 데이터
        (./Output/Data_preprocessing/preprocessed.csv)
    2. 원본 데이터의 통계 테이블
        (./Output/Data_preprocessing/criteria.csv)

순서 : 
    1. 데이터 읽어오기 (Raw data (time/Values))
    2. 결측치 처리
    3. 시간 간격으로 병합 (hourly, daily)
    4. 이상치 보정
    5. 평일과 주말 데이터 추출 (선택)
    6. 전처리한 데이터, 원본 데이터의 통계 테이블 저장
        (경로: ./Output/Data_preprocessing)
'''

import os
import pandas as pd
import sys

os.chdir(os.path.dirname(__file__))

def read_data(file_name):
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


# 결측치 보정 : 연속적인 결측치 3개 이하이면 이전값 대체, 초과이면 선형 보간
def Missing(df):
    n = 0
    first_column = df.iloc[:, n]

    # in case first line is null
    if first_column.isnull().iloc[0]:
        first_non_nan_index = first_column.first_valid_index()
        df = df.loc[first_non_nan_index:]

    # in case last lines are missing
    nan_list = list(first_column.isnull())
    if nan_list[-1]:
        last_non_nan_idx = first_column.last_valid_index()
        df = df.loc[:last_non_nan_idx]

    # in case greater than 3
    method_fill = 'ffill'  # replace previous values
    count = 0
    for i, v in enumerate(first_column.isnull()) : 
        if v:
            count += 1
        else:
            if count > 3:
                df.iloc[i - count-1:i+1, n] = df.iloc[i - count-1:i+1, n].interpolate()  # replace with interpolation
            else: # in case missing values are less than equal to 3
                df.iloc[i - count-1:i, n].fillna(method=method_fill, inplace=True)
            count = 0
   
    return df


# 1시간 간격으로 병합
def TimeIntervalData(df,time_interval): 

    time_df = df.sort_index(ascending=True)
    time_df = time_df.resample(time_interval).mean() # 평균값으로 병합

    # time의 결측치 확인
    if time_df.isnull().any().any():
        time_df = Missing(time_df) 

    return time_df
  
# 이상치 보정: 입력받은 퍼센트 값에서 벗어나는 값을 정상범위로 보정함
def Anomalous(df, percent, normal_max=None, normal_min=None): 
    col = df.columns[0]
    
    # 원본 데이터의 최댓값, 최솟값 계산
    MAX = df[col].max()        
    MIN = df[col].min()
    
    # 정상범위가 없는 경우 : UIF, LIF 
    if normal_max is None and normal_min is None: 
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3 - q1
        UIF = q3 + (IQR * 1.5)
        LIF = q1 - (IQR * 1.5)
        normal_max = UIF 
        normal_min = LIF
    
    # criteria.csv 생성
    table = pd.DataFrame({'Values': [ MIN, MAX, normal_min, normal_max] },
                index=['raw_min', 'raw_max', 'normal_min', 'normal_max'])

    # 이상치 보정
    if normal_max is not None: # 정상범위의 최댓값이 있는 경우
        # U_level 계산
        U_level =  (abs(MAX - normal_max) * (percent * 0.01)) +  normal_max
        table.loc['U_level'] = U_level # table에서 U_level를 추가
        
        # 데이터의 최댓값이 정상범위의 최댓값보다 크면 정상범위의 최댓값으로 대체 
        if MAX > normal_max:
            df.loc[df[col] > U_level] = normal_max

    if normal_min is not None: # 정상범위의 최솟값이 있는 경우
        # L_level 계산
        L_level = normal_min - (abs(normal_min - MIN) * (percent * 0.01))   
        table.loc['L_level'] = L_level # table에서 L_level 추가

        # 데이터의 최솟값이 정상범위의 최솟값보다 크면 정상범위의 최솟값으로 대체 
        if MIN < normal_min:
            df.loc[df[col] < L_level] = normal_min
           
    return df, table

# 평일과 주말 데이터 추출
def week(df,day_of_week):
        if day_of_week == 'weekend':
            return df[df.index.dayofweek.isin([5, 6])]  # 5: 토요일, 6: 일요일
        
        if day_of_week == 'weekday':
            return df[df.index.dayofweek.isin([0, 1, 2, 3, 4])]  # 0~4: 월요일부터 금요일까지


# output 저장
def save_data(df, table):

    # 폴더 생성
    output_path = './Output/Data_preprocessing'
    os.makedirs(output_path, exist_ok=True)

    # 전처리한 데이터, 원본 데이터의 criteria 저장
    df.to_csv(f'{output_path}/preprocessed.csv')  
    table.to_csv(f'{output_path}/criteria.csv')   

def Data_preprocessing(file_name, percent):
    
    # 데이터 읽어오기
    df = read_data(file_name)

    # 결측치 보정
    df = Missing(df)

    # 데이터 병합 (1D)
    time_interval = '1D'
    df = TimeIntervalData(df,time_interval)
    
    # 이상치 보정
    df, table = Anomalous(df, percent, normal_max=None, normal_min=None)

    # 평일 / 주말 데이터 구하기
    day_of_week = 'all'
    if day_of_week != 'all':
        df = week(df,day_of_week)

    # 전처리한 데이터, 원본 데이터의 criteria 저장
    save_data(df, table)

if __name__ == "__main__":
    file_name = sys.argv[1]
    percent = sys.argv[2]

    Data_preprocessing(file_name, percent)


    

# # 인자 받기

    
   
