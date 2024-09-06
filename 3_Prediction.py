import importlib
import argparse
import pandas as pd

preprocessing = importlib.import_module('Data_preprocessing')
abnormal_detection = importlib.import_module('Abnormal_detection')
predict = importlib.import_module('Prediction')

def parse_arguments():
    parser = argparse.ArgumentParser(description='LSTM 데이터세트 예측 및 이상탐지 도구.',
                                     epilog='© 2023 Multidisciplinary A.I. Lab., Kyungnam Univ.'
                                            '모든 권리는 경남대학교 융합인공지능연구실에 있습니다.')
    # subparsers = parser.add_subparsers()
    parser.add_argument('-f','--filename', help='파일 이름을 입력합니다. 필수 인자입니다.') 
    parser.add_argument('-c', '--criteria', help='원본 데이터의 통계 파일 이름을 입력합니다. 기본은 ./Output/Data_preprocessing/criteria.csv파일입니다.', default='./Output/Data_preprocessing/criteria.csv')

    # 데이터 전처리
    parser.add_argument('-dp', '--data_preprocessing', action='store_true', 
                        help='데이터 전처리를 실행합니다. 기본은 결측치 보정이고, 도구 상태가 -p(--prediction)이면 하루 간격으로 시간을 병합합니다. -t(--time)와 -w(--week)를 통해 지정할 수 있습니다.') # 예측에 대한 인자
    parser.add_argument('-t', '--time', choices=['hourly', 'daily'], help='데이터의 시간 간격을 변경합니다. hourly, daily 중에서 선택할 수 있습니다.', default='daily') 
    parser.add_argument('-w', '--week', choices=['all', 'weekday', 'weekend'], help='주말 또는 평일 데이터를 구합니다. 기본값은 전체(all)입니다.', default='all')
    
    parser.add_argument('-p', '--prediction', action='store_true', help='도구를 예측 상태로 설정합니다.') # 예측에 대한 인자
    parser.add_argument('-d', '--detection', action='store_true', help='도구를 이상 탐지 상태로 설정합니다.') # 이상 탐지에 대한 인자
    parser.add_argument('-a', '--all', action='store_true', help='특정한 값으로 이상탐지와 예측을 수행합니다. ') # 예측과 이상탐지에 대한 인자
 
    return parser.parse_args()

def main():
    args = parse_arguments()

    # 이상탐지, 예측 
    if args.all:
        abnormal_detection.abnormal_detection(args.filename, args.criteria) # 이상탐지
        predict.prediction(args.filename, args.criteria) # 예측 
    else:
        # 데이터 읽어오기
        df = abnormal_detection.read_data(args.filename) 
    
        # 데이터 전처리 : 결측치 보정, 시간 간격 병합, 평일/주말 데이터
        if df.isnull().any().any():
            df = preprocessing.Missing(df) # 결측치 보정

        if args.data_preprocessing or args.prediction: # 시간 간격 병합
            time_interval = {'hourly': '1H', 'daily': '1D'}.get(args.time)
            df = preprocessing.TimeIntervalData(df, time_interval)

            # 평일 / 주말 데이터 구하기    
            if args.week in ['weekday', 'weekend']: 
                df = preprocessing.week(df, args.week)
        
        # 모델과 스케일러 불러오기
        sc, lstm_model = abnormal_detection.load_model_and_scaler()

        # 모델의 윈도우 사이즈 찾기
        window_size = lstm_model.layers[0].input_shape[1]
    
        # x_data 생성
        x_data = abnormal_detection.create_x_new_data(df, window_size, sc)

        # 예측
        preds = abnormal_detection.predictions(x_data, lstm_model, sc)

        # 이상 탐지
        if args.detection:
            # 예측값에 대한 시계열 데이터 생성
            preds_df = abnormal_detection.create_time_series_data(df, preds)

            # 이상치 추출
            abnormal_df = abnormal_detection.detection(preds_df, args.criteria)
    
            # 이상치 데이터 저장
            abnormal_detection.save_data(abnormal_df)       

        # 예측
        if args.prediction: 
            # 예측값에 대한 시계열 데이터 생성
            periods_df = predict.create_predictive_data(df, preds)

            # 이상치 검출
            abnormal_df = abnormal_detection.detection(pd.DataFrame(periods_df.iloc[:,0]), args.criteria)

            # 레이블 채우기
            periods_df = predict.fill_label(periods_df, abnormal_df)

            # 예측 데이터 저장
            predict.save_data(periods_df)


if __name__ == '__main__':
    main()






