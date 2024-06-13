import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import os.path as op
import pdb

import numpy as np
import pandas as pd
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
# import talib

from time import time
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from itertools import product

from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


from Misc import utilities as ut

idx = pd.IndexSlice # 인덱스 슬라이싱 설정
np.random.seed(42)    # 재현성을 위해 시드 번호 고정
# 결과 저장 디렉터리 설정
save_dir = ut.get_dir(op.join('results'))
hp_save_dir = ut.get_dir(op.join(save_dir, 'hp_tuning'))
score_save_dir = ut.get_dir(op.join(save_dir, 'score'))

def format_time(t):
    """숫자형 시간 값을 기반으로 'HH:MM:SS' 형식의 시간을 반환"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

# 시계열이 주어지면 전체 기간에서 train/test 기간 설정에 따라 시계열 데이터셋 생성해주는 함수
def split(X, n_splits, train_length, test_length, lookahead, date_idx='date', ticker_idx='ticker'):
    unique_dates = X.index.get_level_values(date_idx).unique()
    days = sorted(unique_dates, reverse=True)
    split_idx = []  
    for i in range(n_splits):
        test_end_idx = i * test_length
        test_start_idx = test_end_idx + test_length
        train_end_idx = test_start_idx + lookahead - 1
        train_start_idx = train_end_idx + train_length + lookahead - 1
        split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])

    dates = X.reset_index()[[date_idx, ticker_idx]]

    for train_start, train_end, test_start, test_end in split_idx:
        try:

            train_idx = dates[(dates[date_idx] > days[train_start])
                            & (dates[date_idx] <= days[train_end])].index
            test_idx = dates[(dates[date_idx] > days[test_start])
                            & (dates[date_idx] <= days[test_end])].index

            if len(test_idx) > 0:

                yield train_idx.to_numpy(), test_idx.to_numpy()

            else:
                continue

        except:
            pass

# 피쳐 임포턴스 계산하는 함수
def get_fi(model):
    fi = model.feature_importance(importance_type='gain')
    return (pd.Series(fi/fi.sum(), index=model.feature_name()))

param_cols = ['train_length', 'test_length', 'bagging_fraction',
              'feature_fraction', 'min_data_in_leaf', 'rounds']

# 성능이 가장 좋은 파라미터 뽑아내는 함수
def get_params(data, t=1, best=0):
    df = data[data.t == t].sort_values('ic', ascending=False).iloc[best]
    df = df.loc[param_cols]
    rounds = int(df.rounds)
    params = pd.to_numeric(df.drop('rounds'))
    return params, rounds

# 모델 트레이닝 함수. --> 메인 함수임.
def get_results(today):

    factors_df.sort_index(inplace=True)

    # 유동성이 가장 높은 1000개의 주식 선택
    price_dat.sort_index(inplace=True)
    universe_start_date = (datetime.strptime(today, '%Y-%m-%d') - timedelta(days=251*3)).strftime('%Y-%m-%d')
    prices = price_dat.loc[idx[:, universe_start_date:today], :]

    dv_rank = prices['TRAN_AMT'].groupby(level='date').rank(ascending=False)
    print(dv_rank)

    universe = dv_rank.groupby(level='ticker').mean().nsmallest(1000).index
    cv_start_date = (datetime.strptime(today, '%Y-%m-%d') - relativedelta(months=12*10)).strftime('%Y-%m-%d')
    cv_end_date = (datetime.strptime(today, '%Y-%m-%d') - relativedelta(months=1)).strftime('%Y-%m-%d')
    cv_data = factors_df.loc[idx[universe, cv_start_date:cv_end_date], :]

    len(cv_data.index.unique('ticker'))

    base_params = dict(boosting_type='rf',
                    objective='regression',
                    bagging_freq=1,
                    verbose=-1)

    # 하이퍼파라미터 옵션
    bagging_fraction_opts = [.5,.75]
    feature_fraction_opts = [.5,.75]
    min_data_in_leaf_opts = [250, 500] # [250,500,1000]

    cv_params = list(product(bagging_fraction_opts,
                        feature_fraction_opts,
                        min_data_in_leaf_opts))
    n_params = len(cv_params)
    print('number of cv params : ', cv_params)

    # 부스팅 반복 횟수
    num_iterations = [25] + list(range(50,501,25))
    num_boost_round = num_iterations[-1]

    train_lengths = [60, 48, 36, 24, 12]
    test_lengths = [1]
    test_params = list(product(train_lengths, test_lengths))
    n_test_params = len(test_params)

    print('number of test params : ', n_test_params)

    # 범주형 데이터 팩터화
    categoricals = ['year', 'weekday', 'month']
    for feature in categoricals:
        cv_data[feature] = pd.factorize(cv_data[feature], sort=True)[0]

    labels = sorted(cv_data.filter(like='fwd').columns)
    features = cv_data.columns.difference(labels).tolist()
    label = f'fwd_sr_21_adj'

    # 하이퍼파라미터 튜닝 결과 저장소
    cv_store = f'{hp_save_dir}/parameter_tuning_{today.replace("-","")}.h5'

    ic_cols = ['bagging_fraction',
            'feature_fraction',
            'min_data_in_leaf', 
            't'] + [str(n) for n in num_iterations]

    # 트레인 기간/테스트 기간 별로, 하이퍼파라미터 조합별로 성능 계산
    lookahead = 0
    for train_length, test_length in test_params:
        n_splits = 500
        print(f'Lookahead: {lookahead:2.0f} | Train: {train_length:3.0f} | '
            f'Test: {test_length:2.0f} | Params: {len(cv_params):3.0f}')
        
        label = 'fwd_sr_21_adj'
        outcome_data = cv_data.loc[:, features+[label]].dropna()

        lgb_data = lgb.Dataset(data=outcome_data.drop(label, axis=1),
                            label=outcome_data[label],
                            categorical_feature=categoricals,
                            free_raw_data=False)

        predictions, daily_ic, ic, feature_importance = [], [], [], []
        key = f'{lookahead}/{train_length}/{test_length}'
        T = 0
        for p, (bagging_fraction, feature_fraction, min_data_in_leaf) in enumerate(cv_params):
            params = base_params.copy()
            params.update(dict(bagging_fraction=bagging_fraction,
                            feature_fraction=feature_fraction,
                            min_data_in_leaf=min_data_in_leaf))
            start = time()

            cv = split(outcome_data, n_splits=n_splits, train_length=train_length, test_length=test_length, lookahead=lookahead)
            cv_preds, nrounds = [], []
            for i, (train_idx, test_idx) in enumerate(cv):
                lgb_train = lgb_data.subset(used_indices=train_idx.tolist(), params=params).construct()
                lgb_test = lgb_data.subset(used_indices=test_idx.tolist(), params=params).construct()

                model = lgb.train(params=params,
                                train_set=lgb_train,
                                num_boost_round=num_boost_round,
                                )
                if i ==0:
                    fi = get_fi(model).to_frame()
                else:
                    fi[i] = get_fi(model)

                test_set = outcome_data.iloc[test_idx,:]
                X_test = test_set.loc[:, model.feature_name()]
                y_test = test_set.loc[:, label]
                y_pred = {str(n): model.predict(X_test, num_iteration=n)
                            for n in num_iterations}
                
                    # Debug: Confirm y_pred is populated
                print(f"Appending data for split {i}, y_pred keys: {list(y_pred.keys())}")
        
                cv_preds.append(y_test.to_frame('y_test').assign(**y_pred).assign(i=i))
                nrounds.append(model.best_iteration)



            # if cv_preds:
            cv_preds_df = pd.concat(cv_preds).assign(bagging_fraction=bagging_fraction,
                                                    feature_fraction=feature_fraction,
                                                    min_data_in_leaf=min_data_in_leaf)

            feature_importance.append(fi.T.describe().T.assign(bagging_fraction=bagging_fraction,
                                                                feature_fraction=feature_fraction,
                                                                min_data_in_leaf=min_data_in_leaf))
            predictions.append(cv_preds_df)
            by_day = cv_preds_df.groupby(level='date')

            ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x.y_test,
                                                                    x[str(n)])[0]).to_frame(n)
                                    for n in num_iterations], axis=1)

            daily_ic.append(ic_by_day.assign(bagging_fraction=bagging_fraction,
                                                feature_fraction=feature_fraction,
                                                min_data_in_leaf=min_data_in_leaf))

            cv_ic = [spearmanr(cv_preds_df.y_test, cv_preds_df[str(n)])[0]
                    for n in num_iterations]

            T += time() - start
            ic.append([bagging_fraction, feature_fraction,
                        min_data_in_leaf, lookahead] + cv_ic)

            msg = f'{p:3.0f} | {format_time(T)} | '
            msg += f'{bagging_fraction:3.0%} | {feature_fraction:3.0%} | {min_data_in_leaf:5,.0f} | '
            msg += f'{max(cv_ic):6.2%} | {ic_by_day.mean().max(): 6.2%} | {ic_by_day.median().max(): 6.2%}'
            print(msg)

        m = pd.DataFrame(ic, columns=ic_cols)
        m.to_hdf(cv_store, 'ic/' + key)
        pd.concat(daily_ic).to_hdf(cv_store, 'daily_ic/' + key)
        pd.concat(feature_importance).to_hdf(cv_store, 'fi/' + key)
        pd.concat(predictions).to_hdf(cv_store, 'predictions/' + key)

    # 성능 계산한 결과 저장
    id_vars = ['train_length',
            'test_length',
            'bagging_fraction',
            'feature_fraction',
            'min_data_in_leaf',
            't', 'TD']

    daily_ic, ic = [], []
    t = lookahead
    with pd.HDFStore(cv_store) as store:
            keys = [k[1:] for k in store.keys() if k.startswith(f'/fi/{t}')]
            for key in keys:
                    train_length, test_length = key.split('/')[2:]
                    # print(train_length, test_length)
                    k = f'{t}/{train_length}/{test_length}'
                    cols = {'t': t,
                            'train_length': int(train_length),
                            'test_length': int(test_length)}

                    ic.append(pd.melt(store['ic/' + k]
                                    .assign(**cols),
                                    id_vars=id_vars[:-1],
                                    value_name='ic',
                                    var_name='rounds')
                            .apply(pd.to_numeric))

                    df = store['daily_ic/' + k].assign(**cols)
                    daily_ic.append(pd.melt(df,
                                            id_vars=id_vars[:-1],
                                            value_name='daily_ic',
                                            var_name='rounds')
                                    .apply(pd.to_numeric)
                                    .reset_index())
    ic = pd.concat(ic, ignore_index=True)
    daily_ic = pd.concat(daily_ic, ignore_index=True)

    lookahead = 0
    data = cv_data
    labels = sorted(data.filter(like='fwd').columns)
    features = data.columns.difference(labels).tolist()
    label = 'fwd_sr_21_adj'
    data = data.loc[:, features+[label]].dropna()

    categoricals = ['year', 'weekday', 'month']
    for feature in categoricals:
        data[feature] = pd.factorize(data[feature], sort=True)[0]

    lgb_data = lgb.Dataset(data=data[features],
                            label=data[label],
                            categorical_feature=categoricals,
                            free_raw_data=False)

    # 가장 좋은 성능을 가진 train/test기간 및 하이퍼파라미터 조합 5개 찾아서 종목별로 스코어 계산하기 
    for position in range(5):
        params, num_boost_round = get_params(ic, t=lookahead, best=position)
        params = params.to_dict()
        params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
        train_length = int(params.pop('train_length'))
        test_length = int(params.pop('test_length'))
        params.update(base_params)

        print(f'\tPosition: {position:02}')

        n_splits = 500
        cv = split(data, n_splits=n_splits, train_length=train_length, test_length=test_length, lookahead=lookahead)
        
        predictions = []
        start = time()
        for i, (train_idx, test_idx) in enumerate(cv, 1):
            try:
                train_set = lgb_data.subset(used_indices=train_idx.tolist(),
                                            params=params).construct()

                model = lgb.train(params=params,
                                    train_set=train_set,
                                    num_boost_round=num_boost_round,
                                    )

                test_set = data.iloc[test_idx, :]
                y_test = test_set.loc[:, label].to_frame('y_test')
                y_pred = model.predict(test_set.loc[:, model.feature_name()])
                predictions.append(y_test.assign(prediction=y_pred))

            except:
                pass

        if position == 0:
            test_predictions = (pd.concat(predictions)
                                .rename(columns={'prediction': position}))
        else:
            test_predictions[position] = pd.concat(predictions).prediction

    test_predictions.replace([np.inf, -np.inf], -99999, inplace=True)

    # 최적 5개의 조합에 대해 스코어를 계산하고 최종적인 스코어는 조합별 스코어의 평균을 조합별 mse의 평균으로 나눈 값으로 계산. 
    for position in range(5):
        if position == 0:
            metric_df = test_predictions.groupby(level='ticker').apply(lambda x: mean_squared_error(x.loc[:,'y_test'], x[position], squared=False)).to_frame()
        else:
            metric_df[position] = test_predictions.groupby(level='ticker').apply(lambda x: mean_squared_error(x.loc[:,'y_test'], x[position], squared=False))

    rtn_pred = test_predictions.groupby(level='ticker').mean().drop(columns='y_test')

    result_df = pd.DataFrame((rtn_pred.mean(axis=1) / metric_df.mean(axis=1)).reset_index())
    result_df.columns = ['ticker', 'score']
    result_df['ticker'] = result_df['ticker'].apply(lambda x: str(x))
    result_df = result_df.sort_values(by=['score'], ascending=False)

    result_df.to_excel(f'{score_save_dir}/ind_stock_score_{today.replace("-","")}.xlsx', index=False) # 최종 모델 스코어 저장

if __name__ =="__main__":
    idx = pd.IndexSlice

    # 데이터 파일 로드
    price_dat = pd.read_feather('../Data/processed_data/price_dat_pr.feather')
    returns = pd.read_feather('../Data/processed_data/returns.feather')
    factors_df = pd.read_feather('../Data/processed_data/FACTORS_FINAL.feather')

    # MultiIndex의 모든 레벨로 DataFrame 정렬
    factors_df_sorted = factors_df.sort_index()

     # 시작일과 종료일 설정
    start_date = '2024-05-31'
    end_date = "2024-05-31"

     # 선택한 날짜에 해당하는 데이터 추출
    selected_data = factors_df_sorted.loc[idx[:, start_date:end_date], :].index.unique('date')
    test_date = [x.strftime('%Y-%m-%d') for x in selected_data]
    assert len(test_date) != 0  # 선택한 날짜가 비어있지 않은지 확인
    
    # 테스트 날짜마다 모델 스코어 생성
    for dt in tqdm(test_date):
        filename = dt.replace("-","")
        if op.isfile(f'{score_save_dir}/ind_stock_score_{filename}.xlsx'): # 사전에 생성된 모델 스코어가 있을 경우 해당 date는 pass
            print("Found pregenerated file {}".format(filename))
        else:
            print(f"Generating {filename}")
            get_results(dt)