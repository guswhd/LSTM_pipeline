import importlib
import argparse

preprocessing = importlib.import_module('Data_preprocessing')
training = importlib.import_module('Training')

def parse_arguments():
    parser = argparse.ArgumentParser(description='LSTM 데이터세트 학습 도구.',
                                     epilog='© 2023 Multidisciplinary A.I. Lab., Kyungnam Univ. '
                                            '모든 권리는 경남대학교 융합인공지능연구실에 있습니다.')
    # subparsers = parser.add_subparsers()

    parser.add_argument('-f','--filename', help='전처리한 파일 이름을 입력합니다. 기본은 ./Output/Data_preprocessing/preprocessed.csv 파일입니다.', 
                        default='./Output/Data_preprocessing/preprocessed.csv') 
    parser.add_argument('-c', '--criteria', help='원본 데이터의 통계 파일 이름을 입력합니다.' ,default='./Output/Data_preprocessing/criteria.csv')
    # 시퀀스 설정
    parser.add_argument('-w', '--window_size', help='윈도우 사이즈를 정합니다. 기본은 7 입니다.',  type=int, default=7) 
    parser.add_argument('-p', '--periods', help='예측 윈도우 사이즈를 정합니다. 기본은 1 입니다.', type=int, default=1)
    
    # 하이퍼 파라미터 튜닝
    parser.add_argument('-hp', '--hyper_parameters',  action='store_true', help='하이퍼 파라미터를 튜닝합니다. -e(--epochs), -b(--batch_size) 를 통해 변경할 수 있습니다.')
    parser.add_argument('-e', '--epochs', help='epoch를 정합니다.', type=int, default=700 ) 
    parser.add_argument('-b', '--batch_size', help='batch size를 정합니다.', type=int, default=32)

    parser.add_argument('-l', '--learning', action='store_true', help='특정한 값으로 모델을 학습합니다. (-w 7, -p 2, -hp, -e 700, -b 32)') # 모델 학습에 대한 인자
 
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.learning: 
        training.training(7, 1)

    else:
        # 데이터 읽어오기
        df = training.read_data(args.filename)
    
        # 학습 데이터 / 테스트 데이터 분할
        train, test = training.train_test_split(df)

        # 데이터 스케일링
        sc = training.data_scaling(train)

        # x_train, y_train, x_test 생성
        x_train, y_train = training.create_train(train, args.window_size, args.periods, sc)
        x_test = training.create_x_test(test, args.window_size, sc)

        # 모델 생성
        lstm_model = training.lstm_arch(x_train, y_train, args.periods, args.epochs, args.batch_size)

        # 모델 성능 평가 
        plot = training.lstm_performance(lstm_model, sc, x_test, test, args.epochs, args.batch_size)

        # 스케일러, 모델, 그래프 저장
        training.save_output(sc, lstm_model, plot)
        
if __name__ == '__main__':
    main()
