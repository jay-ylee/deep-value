from __future__ import print_function, division
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pdb
import pickle
import warnings
warnings.filterwarnings('ignore')

import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

from Misc import utilities as ut
# from Data import equity_data as eqd
from Data import equity_data as eqd

class GetPortfolio(object):
    def __init__(self):
        self.port_dir = ut.get_dir(op.join('port/ModelA')) # 포트폴리오 산출물 저장경로
        self.score_dir = '../Model/results/score' # 모델스코어 디렉토리 설정
        self.save_dir = '../Data/processed_data' # 전처리된 팩터 데이터 디렉토리 설정
        self.raw_dir = '../raw' # 기초 데이터 디렉토리 설정
        self.idx = pd.IndexSlice # 다중 인덱스를 위한 IndexSlice 객체 생성
        self.preprocess_instance = eqd.Preprocessor() # 전처리 클래스 인스턴스 생성

        # daily_dict.pickle 파일 열어 딕셔너리 로드
        with open(f'{self.save_dir}/daily_dict.pickle', 'rb') as f:
            self.daily_dict = pickle.load(f)

        # fin_components.pickle 파일 열어 딕셔너리 로드
        with open(f'{self.save_dir}/fin_components.pickle', 'rb') as fr:
            self.fin_components = pickle.load(fr)

        # 필요한 데이터들
        self.daily_df_float = pd.read_parquet("../raw/daily_df_float.parquet")
        self.daily_df_float['Symbol'] = [str(x)[1:] for x in self.daily_df_float['Symbol']]
        self.ticker_name = self.daily_df_float[['Symbol', 'Stock_name']].drop_duplicates().set_index('Symbol').to_dict()['Stock_name']
        self.thm_etf = pd.read_excel(f'{self.raw_dir}/thm_ETF_수정.xlsx',  engine='openpyxl')
        self.sector_dat = pd.read_excel(f'{self.raw_dir}/sector_dat.xlsx',  engine='openpyxl')
        self.price_dat = pd.read_feather(f'{self.save_dir}/price_dat_pr.feather')
        self.rtn_monthly = self.price_dat['CLOSE_AP'].unstack('ticker').sort_index().resample('M').last().pct_change()
        self.factors_df = pd.read_feather(f'{self.save_dir}/FACTORS_FINAL.feather')

        # 특정 날짜 설정
        self.date = '2024-05-31'
        given_date = datetime.strptime(self.date, "%Y-%m-%d")
        last_day_of_previous_month = given_date.replace(day=1) - relativedelta(days=1)
        last_year_of_previous_year = given_date.replace(year=(given_date.year - 1))
        self.prev_date = last_day_of_previous_month.strftime("%Y-%m-%d")
        self.prev_year = last_year_of_previous_year.strftime("%Y-%m-%d")

        ## 백테스팅용 데이터 리스트 생성
        # self.start_date = '2019-12-31'
        # self.date_list = [x.strftime('%Y-%m-%d') for x in self.factors_df.index.unique('date') if x >= datetime.strptime(self.start_date, '%Y-%m-%d')]


    def generate_portfolio(self):

        filename_pdf = 'pdf_df.feather'
        filepath_pdf = op.join(self.save_dir, filename_pdf)
        if op.isfile(filepath_pdf): # pdf_df.feather 파일이 존재하는지 확인
            print("Found pregenerated file {}".format(filename_pdf))
            pdf_df = pd.read_feather(filepath_pdf) # 파일을 읽어 데이터프레임으로 변환
        else:
            pdf_df = self.preprocess_instance.get_pdf_data(filepath_pdf) # 파일이 없으면 전처리하여 생성
        
        pdf_df['TD'] = pd.to_datetime(pdf_df['TD'], format="%Y%m%d") # TD 열을 datetime 형식으로 변환
        pdf_df['CODE'] =  ['0'*(6-len(str(x))) + str(x) for x in pdf_df['CODE']]  # CODE 열의 값을 6자리 문자열로 변환
        pdf_df['PDF_CODE'] =  ['0'*(6-len(str(x))) + str(x) for x in pdf_df['PDF_CODE']]  # PDF_CODE 열의 값을 6자리 문자열로 변환
        
        thm_df = self.thm_etf
        thm_df['code'] =  ['0'*(6-len(str(x))) + str(x) for x in thm_df['code']] # code 열의 값을 6자리 문자열로 변환

        pdf_thm_df = pd.merge(pdf_df, thm_df, left_on='CODE', right_on='code')  # PDF 데이터와 테마 ETF 데이터를 병합
        factors_df = self.factors_df
        fwd_ret = factors_df[sorted(factors_df.filter(like='fwd').columns)[0]]  # 향후 수익률 열 선택
    
        marketcap_monthly = self.daily_dict['marketcap'].shift(1).resample('M').last().reset_index()  # 월별 시가총액 계산
        marketcap_monthly = pd.melt(marketcap_monthly, id_vars=['index'], var_name='ticker', value_name='marketcap')  # 시가총액 데이터를 세로로 변환
        sector_dat = self.sector_dat
        sector_dat['Symbol'] = sector_dat['Symbol'].apply(lambda x: x[1:]) # Symbol 열의 첫 번째 문자를 제거
        sector_dat.rename(columns={'FnGuide Sector': 'gubun', '회계년': 'YEAR'}, inplace=True)  # 열 이름 변경

        dv = self.price_dat['TRAN_AMT'].reset_index().pivot_table(index='date', columns='ticker')  # 거래 금액 데이터를 피벗 테이블로 변환
        dv.columns = dv.columns.droplevel(0) # 첫 번째 레벨의 열 인덱스 제거
        dv_mean = dv.rolling(window=20).mean().shift(1).resample('M').last()  # 20일 이동 평균 계산

        
        mkt_rank_ths = 1000  # 시가총액 상위 1000개
        dv_rank_ths = 500 # 거래 금액 상위 500개
        selected_sec_df = pd.DataFrame() # 선택된 섹터 데이터프레임 초기화

        print('Start selecting index.....')

        temp = pd.read_excel(f'{self.score_dir}/ind_stock_score_{self.date.replace("-", "")}.xlsx', index_col=False) # 개별 모델 스코어 불러오기
        temp['ticker'] = ['0' * (6 - len(str(x))) + str(x) for x in temp['ticker']]  # ticker 열의 값을 6자리 문자열로 변환
        year = datetime.strptime(self.date, '%Y-%m-%d').year # 데이터타임 형식 지정

        dv_top = dv_mean.loc[self.date].sort_values(ascending=False)[:dv_rank_ths].index.tolist() # 거래 금액 상위 500개 종목 선택
        mkt_top = marketcap_monthly[marketcap_monthly['index'] == self.date].sort_values(by=['marketcap'], ascending=False)[:mkt_rank_ths]['ticker'].tolist() # 시가총액 상위 1000개 종목 선택

        # 현재 날짜의 시가총액 데이터 필터링
        marketcap_monthly_filtered = marketcap_monthly[marketcap_monthly['index'] == self.date]

       
        temp_mkt = pd.merge(temp, marketcap_monthly_filtered, on='ticker', how='left')[['ticker', 'score', 'marketcap']].dropna() # 스코어데이터와 시가총액 데이터 변환
        temp_mkt = temp_mkt[temp_mkt['ticker'].apply(lambda x: x in dv_top)] # 거래금액 상위 종목필터링
        temp_mkt = temp_mkt[temp_mkt['ticker'].apply(lambda x: x in mkt_top)] # 시가총액 상위 종목필터링
        temp_mkt['fwd_ret'] = temp_mkt['ticker'].apply(lambda x: fwd_ret.loc[self.idx[x, self.date]]) # 향후 수익률 계산

        # 현재 년도의 테마와 섹터 데이터 필터링
        pdf_thm_df['YEAR'] = pdf_thm_df['TD'].dt.year
        pdf_thm_df_filtered = pdf_thm_df[pdf_thm_df['YEAR'] == year]
        last_date = pdf_thm_df_filtered['TD'].sort_values().unique()[-2]
        theme = pdf_thm_df_filtered[(pdf_thm_df_filtered['TD'] == last_date)][['PDF_CODE', '구분2', '구분']].rename(columns={'구분2': 'gubun', '구분':'thm_st'})

        # 테마 데이터와 점수 데이터를 병합하고 시가총액으로 필터링
        thm_temp = pd.merge(theme, temp, left_on='PDF_CODE', right_on='ticker', how='left').dropna().drop(columns='PDF_CODE')
        temp_thm_mkt = pd.merge(thm_temp, marketcap_monthly_filtered, on='ticker', how='left')[['ticker', 'score', 'marketcap', 'gubun','thm_st']].dropna()
        temp_thm_mkt = temp_thm_mkt[temp_thm_mkt['ticker'].apply(lambda x: x in dv_top)]
        temp_thm_mkt = temp_thm_mkt[temp_thm_mkt['ticker'].apply(lambda x: x in mkt_top)]

        # 섹터 데이터와 시가총액을 병합하고 필터링
        sector_dat_filtered = sector_dat[sector_dat['YEAR'] == year]
        sector_dat_filtered.rename(columns={'Symbol': 'ticker'}, inplace=True)
        sector_temp = pd.merge(sector_dat_filtered, temp, on='ticker', how='left').dropna()
        sector_temp.drop(columns=['Name', '결산월', '주기', 'YEAR'], inplace=True)

        # 섹터 데이터와 시가총액을 병합하고 필터링
        temp_sec_mkt = pd.merge(sector_temp, marketcap_monthly_filtered, on='ticker', how='left')[['ticker', 'score', 'marketcap', 'gubun']].dropna()
        temp_sec_mkt = temp_sec_mkt[temp_sec_mkt['ticker'].apply(lambda x: x in dv_top)]
        temp_sec_mkt = temp_sec_mkt[temp_sec_mkt['ticker'].apply(lambda x: x in mkt_top)]

        temp_sec_mkt['gubun'] = temp_sec_mkt['gubun'].apply(lambda x: x + "_FICS")

        # 유니버스 점수를 결합하고 계산
        universe_score = pd.concat([temp_thm_mkt, temp_sec_mkt])

        universe_score['fwd_ret'] = universe_score['ticker'].apply(lambda x: fwd_ret.loc[self.idx[x, self.date]])

        universe_score['gubun2'] = universe_score['gubun'].apply(lambda x: 'sector' if 'FICS' in x else 'thm_st')
        universe_score['thm_st'].fillna('섹터', inplace=True)

        index_score = universe_score.groupby('gubun')['score'].mean().sort_values(ascending=False)
        index_score.to_excel(f'{self.port_dir}/index_score_{self.date.replace("-", "")}.xlsx')

        universe_score['name'] = universe_score['ticker'].apply(lambda x: self.ticker_name[x])
        universe_score.to_excel(f'{self.port_dir}/universe_score_{self.date.replace("-", "")}.xlsx')

        # sector_score = universe_score[universe_score['thm_st']=='섹터']
        # thm_score = universe_score[universe_score['thm_st']=='테마']
        # st_score = universe_score[universe_score['thm_st']=='스타일']
        # sector_thm_score = universe_score[universe_score['thm_st']!='스타일'].groupby('gubun')['score'].mean().sort_values(ascending=False)

        # 직전 달에 뽑아놓은 인덱스 스코어가 있으면, 이전 달에 뽑은 인덱스가 이번 달에 뽑힌 인덱스 순위 10위 안에 있을 때에는 그대로 유지, 벗어나면 다른 새로운 인덱스 선택
        if op.isfile(f'{self.port_dir}/selected_index_score_{self.prev_date.replace("-", "")}.xlsx'):
            print(f'Found selected index {self.prev_date.replace("-","")}')
            top_rank_index =  universe_score.groupby('gubun')['score'].mean().sort_values(ascending=False).reset_index().drop_duplicates(subset=['gubun'])['gubun'].tolist()[:10]
            prev_rank_index = pd.read_excel(f'{self.port_dir}/selected_index_score_{self.prev_date.replace("-", "")}.xlsx')['gubun'].drop_duplicates().tolist()
            print(prev_rank_index)
            
            selected_index = [x for x in prev_rank_index if x in top_rank_index]

            idx_plus = 0
            candidate_index = [x for x in top_rank_index if x not in selected_index]

            while len(selected_index) != 5:
                if universe_score.groupby('gubun')['ticker'].count()[candidate_index[idx_plus]] >= 10:
                    selected_index.append(candidate_index[idx_plus])
                idx_plus += 1

            selected_uni = selected_index
        
        # 만약 직전 달에 뽑아높은 인덱스 스코어가 없으면, 즉 이번 달이 처음으로 인덱스 스코어를 계산해서 선정하는 달이라면, 그냥 가장 높은 5개 인덱스 선정
        else:
            print(f'{self.date.replace("-","")} is the first date..')
            candidate_index = universe_score.groupby('gubun')['score'].mean().sort_values(ascending=False).reset_index().drop_duplicates(subset=['gubun'])['gubun'].tolist()
            i = 0
            selected_uni = []
            while len(selected_uni) != 5:
                if universe_score.groupby('gubun')['ticker'].count()[candidate_index[i]] >= 10:
                    selected_uni.append(candidate_index[i])
                i+=1

        selected_universe = universe_score[universe_score['gubun'].isin(selected_uni)]
        selected_universe['date'] = self.date
        selected_sec_df = pd.concat([selected_sec_df, selected_universe])
        print('Completed Saving selected_sec_df!')

        selected_sec_df.to_excel(f'{self.port_dir}/selected_index_score_{self.date.replace("-", "")}.xlsx')
        return selected_sec_df



    def get_port_perf(self, weight, graph=True):

        selected_sec_df = self.generate_portfolio()

        mkt_wt = weight # 시총에 대한 비중
        port = pd.DataFrame()

        # 종목의 최종 스코어가 모델스코어와 시총 스코어(시총이 클수록 높은 점수)의 linear weighted sum으로 산출됨.
        selected_sec_df['score_rank'] = selected_sec_df.groupby(['date', 'gubun'])['score'].rank(ascending=True)
        selected_sec_df['mkt_rank'] = selected_sec_df.groupby(['date', 'gubun'])['marketcap'].rank(ascending=True)
        selected_sec_df['wtd_rank'] = selected_sec_df['mkt_rank'] * mkt_wt + selected_sec_df['score_rank'] * (1-mkt_wt) 
        selected_sec_df = selected_sec_df.sort_values(by=['date', 'gubun', 'wtd_rank'])

        selected_sec_df.to_excel(f'{self.port_dir}/universe_score_calculated_{self.date.replace("-","")}.xlsx')
        print("Completed saving selected index universe score data!")

        print("Let's start picking the stocks !")

        temp = selected_sec_df[selected_sec_df['date']==self.date]

        # 2개년도 연속 순이익 적자기업은 제외
        ni_df_cur = self.fin_components['ni'].ffill().loc[self.date]
        ni_df_prev = self.fin_components['ni'].ffill().loc[self.prev_year]
        ni_df = pd.concat([ni_df_cur, ni_df_prev], axis=1)
        ni_screened_stocks = ni_df.loc[(ni_df.iloc[:,0]>0) & (ni_df.iloc[:,1]>0)].index.tolist()

        temp = temp[temp['ticker'].isin(ni_screened_stocks)]

        # 선택된 인덱스 별로 스코어 제일 높은 종목 2개씩 뽑기
        selected_sectors = temp['gubun'].drop_duplicates().tolist()
        print(selected_sectors)
        stock_list = []
        for sec in selected_sectors:
            
            i = 0
            df = temp[temp['gubun']==sec].sort_values(by='wtd_rank', ascending=False).drop_duplicates(subset=['ticker', 'name'])
            stocks = df.iloc[i:i+2,:]['ticker'].tolist()

            while (len(stocks) == 2) and ((stock_list.count(stocks[0]) >= 2) or (stock_list.count(stocks[1]) >= 2)):
                i += 1
                stocks = df.iloc[i:i+2,:]['ticker'].tolist()

            if len(stocks) >= 2:
                stock_list.append(stocks[0])
                stock_list.append(stocks[1])
                high_df = df.iloc[i:i+2,:]
                stocks_df = high_df
                stocks_df['name'] = stocks_df['ticker'].apply(lambda x: self.ticker_name[x])
                port = pd.concat([port, stocks_df])    

        # 포트 비중은 동일가중평균
        port['weight'] = 1 / port['date'].apply(lambda x: (port['date']==x).sum()) 
        print(f'Completed picking stocks {self.date.replace("-","")}')

        ### OUTPUT ###
        port.to_excel(f'{self.port_dir}/selected_stocks_{self.date.replace("-","")}.xlsx')


def main():
    import time
    start = time.time()
    port_generator = GetPortfolio() # 클래스 인스턴스 생성
    port_generator.get_port_perf(weight=0.0) # 포트폴리오 성과계산 함수 호출, 시총점수에 대한 가중치는 0.0이므로 오로지 모델스코어로만 종목 선택하겠다는 의미.
    end = time.time()
    print(f"{end - start:.5f} sec")
if __name__ == "__main__":
    main()



        
        


