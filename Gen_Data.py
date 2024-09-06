import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 임의의 데이터셋 생성 함수
def create_dataset():
    # 시작 날짜와 종료 날짜 설정 (3개월치 데이터, 1분 간격)
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=90)
    date_range = pd.date_range(start=start_date, end=end_date, freq='min')  # 'T' for minute frequency

    # 기준 범위 설정
    normal_min = 4.044887153
    normal_max = 4.176928819
    U_level = 4.17802309
    L_level = 3.897178299

    # 데이터 생성: normal_min과 normal_max 범위 내에서 랜덤하게 생성
    np.random.seed(42)  # 랜덤 시드 설정
    values = np.random.uniform(normal_min, normal_max, len(date_range))

    # 소수점 두 번째 자리까지 반올림
    values = np.round(values, 2)

    # 이상치 추가: 1%의 확률로 U_level보다 높은 값 또는 L_level보다 낮은 값으로 치환
    outlier_indices = np.random.choice(len(values), size=int(len(values) * 0.05), replace=False)
    for idx in outlier_indices:
        if np.random.rand() > 0.5:
            values[idx] = round(U_level + np.random.uniform(0.01, 0.1), 2)  # U_level 이상치
        else:
            values[idx] = round(L_level - np.random.uniform(0.01, 0.1), 2)  # L_level 이하치

    # 데이터프레임 생성
    df = pd.DataFrame(values, index=date_range, columns=['Values'])

    # CSV로 저장
    output_path = './Input/Raw_data.csv'
    df.to_csv(output_path)
    return output_path

# 데이터셋 생성 및 저장
csv_path = create_dataset()
csv_path
