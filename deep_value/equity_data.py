from __future__ import print_function, division # Python 2와 3의 호환성을 위해 print_function과 division을 불러옵니다.

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 현재 파일의 부모 디렉터리를 시스템 경로에 추가하여 모듈 검색 경로에 포함시킵니다.


import pickle
import pymssql
import warnings
warnings.filterwarnings('ignore') # 경고 메시지를 무시합니다.

import os.path as op
import numpy as np
import pandas as pd
import datetime as dt

from tqdm import tqdm
from datetime import datetime
from Misc import utilities as ut # Misc 패키지의 utilities 모듈을 ut로 불러옵니다.

class Preprocessor(object):
    # 데이터 전처리를 위한 Preprocessor 클래스 정의

    def __init__(self):
         # 초기화 메서드
        self.save_dir = ut.get_dir(op.join('processed_data'))  # 전처리된 데이터를 저장할 디렉터리 경로 설정

        # 재무 데이터를 나타내는 한국어 및 영어 컬럼 이름 리스트
        self.fin_kor_name = ['총자산(천원)', '유동자산(천원)','유동부채(천원)', '현금및현금성자산(천원)', '유무형자산상각비(천원)', '영업활동으로인한현금흐름(천원)',
                            '지배주주순이익(천원)', '총부채(천원)', '단기금융상품(천원)', '매출액(천원)', '매출총이익(천원)',
                            '비유동부채(천원)', '장기금융부채(천원)', '사채(천원)', '장기차입금(천원)', '총자본(천원)', '유형자산(천원)',
                            '수정EPS(연율화)(원)', '영업이익(천원)', '*순이자비용(비영업)(천원)', '재고자산(천원)', '매출원가(천원)',
                            '판매비와관리비(천원)', '대손상각비(판관비)(천원)', '이연법인세자산(천원)', '이연법인세부채(천원)',
                            '매출채권(천원)', '수정CFPS(보통주현금흐름)(원)', '배당성향(현금)(%)', 'FCFF(천원)',
                            'CAPEX(천원)', '영업투하자본(천원)', '수정DPS(보통주,현금)(원)', '배당금(보통주,현금)(천원)']
        self.fin_eng_name = ['total_asset', 'current_asset', 'current_liability', 'cash', 
                            'depreciation', 'ocf', 'ni', 'total_liability', 'short_fin_inst', 
                            'sales', 'grossmargin', 'noncurr_liab', 'longterm_liab', 'debt', 
                            'longterm_borr', 'total_cap', 'fixed_asset', 'EPS', 'OI', 'interest_cost', 
                            'inventory', 'cogs', 'operating_expese', 'allowance_cost', 'deferred_asset', 
                            'deferred_lib', 'account_receivable', 'CFPS', 'dividend_payout_ratio', 'FCFF', 
                            'CAPEX', 'operating_capital', 'DPS', 'dividend']

        # 컨센서스 데이터를 나타내는 한국어 및 영어 컬럼 이름 리스트
        self.consen_kor_name = ['EPS 증가율 (E3)(%)', 'EPS증가율 (E3, 12M Fwd)(%)',
                            'EPS증가율 (E3, 13M~24M Fwd)(%)', 'EPS(adj) (E3)(원)', 'EPS 표준편차 (E3)',
                            'EPS 표준편차 (E3, 12M Fwd)', 'EPS 표준편차 (E3, 13M~24M Fwd)',
                            '매출액 증가율 (E3)(%)', '매출액 증가율 (E3, 12M Fwd)(%)',
                            '매출액 증가율 (E3, 13M~24M Fwd)(%)', '영업이익 증가율 (E3)(%)',
                            '영업이익 증가율 (E3, 12M Fwd)(%)', '영업이익 증가율 (E3, 13M~24M Fwd)(%)',
                            'PER (E3)(배)', 'PER (E3, 12M Fwd)(배)', 'PER (E3, 13M~24M Fwd)(배)',
                            'PBR (E3)(배)', 'PBR (E3, 12M Fwd)(배)', 'PBR (E3, 13M~24M Fwd)(배)',
                            '투자의견점수 (E3, 1M Chg)(%)', '투자의견점수 (E3, 3M Chg)(%)',
                            '투자의견점수 (E3, 6M Chg)(%)', '적정주가 (E3, 1M Chg)(%)',
                            '적정주가 (E3, 3M Chg)(%)', '적정주가 (E3, 6M Chg)(%)',
                            '적정주가(상향-하향)/(전체)(E3)']

        self.consen_eng_name = ['eps_gro_fy0', 'eps_gro_fy1', 
                                'eps_gro_fy2', 'eps_adj', 'eps_std_fy0',
                                'eps_std_fy1', 'eps_std_fy2', 
                                'sales_gro_fy0','sales_gro_fy1', 
                                'sales_gro_fy2', 'oi_gro_fy0', 
                                'oi_gro_fy1', 'oi_gro_fy2', 
                                'per_fy0', 'per_fy1', 'per_fy2', 
                                'pbr_fy0', 'pbr_fy1', 'pbr_fy2',
                                'inv_csn_m1', 'inv_csn_m3', 
                                'inv_csn_m6', 'price_csn_m1', 
                                'price_csn_m3', 'price_csn_m6',
                                'price_csn_rvs']
        
        # 한국어와 영어 컬럼 이름 리스트의 길이가 일치하는지 확인
        assert len(self.fin_kor_name)     == len(self.fin_eng_name) 
        assert len(self.consen_kor_name)  == len(self.consen_eng_name)
        # 한국어와 영어 컬럼 이름 리스트의 길이가 일치하는지 확인
        self.fin_label_dict = {kor:eng for kor, eng in zip(self.fin_kor_name, self.fin_eng_name)}        
        self.consen_label_dict = {kor:eng for kor, eng in zip(self.consen_kor_name, self.consen_eng_name)}
        # DB연결
        try:
            self.conn = pymssql.connect(
                host='10.93.20.65', 
                user='quant', 
                password='mirae', 
                database='MARKET', 
                charset='EUC-KR',
                tds_version=r'7.0'
            ) # 데이터베이스에 연결 시도
        except pymssql.OperationalError as e:
            print("Connection failed:", e)  # 연결 실패 시 오류 메시지 출력
        else:
            print("Connected QPMS DB successfully") # 연결 성공 시 메시지 출력

    # 재무 데이터를 불러오는 메서드
    def get_financial_data(self):
        fin_df = pd.read_pickle('../raw/fin_df.pkl')  # 기존 재무 데이터 불러오기
        fin_df.drop(columns=['수정EPS(연율화)(원).1'], inplace=True)  # 중복 컬럼 제거

        fin_df_new = pd.read_excel('../raw/fin_df_20240325.xlsx', header=9) # 새로운 재무 데이터 엑셀 파일에서 불러오기
        fin_df_new.columns = fin_df_new.iloc[0][:5].values.tolist() + fin_df_new.columns[5:].tolist() #컬럼 설정
        fin_df_new = fin_df_new.iloc[1:,:] # 번째 행 제거

        # 날짜 및 컬럼 정리
        fin_df_new['회계년'] = fin_df_new['회계년'].astype(str)
        fin_df_new['결산월'] = fin_df_new['결산월'].astype(str)
        fin_df_new['결산월'] = fin_df_new['결산월'].replace({'nan':'00'})
        fin_df_new.rename(columns={'회계년': '회계연도'}, inplace=True)
        fin_df_new = fin_df_new[fin_df_new['결산월'] != '00']
        fin_df_new['date'] = fin_df_new['결산월'] + '/25/' + fin_df_new['회계연도'].str[-2:]

        # 기존 데이터와 새로운 데이터를 결합
        fin_df = pd.concat([fin_df, fin_df_new])

        fin_df['date'] = pd.to_datetime(fin_df['date'])
        fin_df = fin_df.rename(columns=self.fin_label_dict)
        # 컬럼 이름 출력
        print("Financial Data Columns : ", fin_df.columns)

        return fin_df # 재무 데이터 반환

    # 재무 데이터를 피벗 테이블 형식으로 변환하는 정적 메서드
    @staticmethod    
    def generate_fin_dataframe(fin_df, col):
        # 데이터 집계 및 피벗 테이블 생성
        aggregated_df = fin_df.groupby(['date', 'Symbol']).agg({col:'mean'})
        pivoted_df = aggregated_df.reset_index().pivot(index='date', columns='Symbol', values=col)
        pivoted_df = pivoted_df.resample('M').last()
        pivoted_df.columns = [x[1:] for x in pivoted_df.columns]

        return pivoted_df # 변환된 데이터프레임 반환

    def get_fin_dict(self):
        # 재무 데이터를 딕셔너리로 변환하는 메서드
        fin_df = self.get_financial_data()
        fin_dict = {}
        for lb in tqdm(self.fin_eng_name):
            try:
                pivoted_df = self.generate_fin_dataframe(fin_df, lb)
                pivoted_df.fillna(method='ffill', inplace=True)
                pivoted_df.fillna(0.0, inplace=True)
                fin_dict[lb] = pivoted_df
                # 각 컬럼에 대해 피벗 테이블 생성 및 결측치 처리
        
            except Exception as e:
                print(e)
                # 오류 발생 시 메시지 출력

        print("Completed generating a financial metrics dictionary")
         # 작업 완료 메시지 출력

        with open(f'{self.save_dir}/fin_dict.pickle', 'wb') as f:
            pickle.dump(fin_dict, f)
             # 생성된 딕셔너리를 피클 파일로 저장
            
     # 일간 데이터를 불러오는 메서드      
    def get_daily_data(self):
        
        # 기존 및 새로운 일간 데이터 불러오기
        daily_df_float = pd.read_parquet('../raw/daily_df_float.parquet')
        daily_df_float_new = pd.read_excel('../raw/daily_df_float_20240325.xlsx', header=8)

        # 컬럼 정리 및 이름 변경
        daily_df_float_new.columns = daily_df_float_new.columns[:6].tolist() + [datetime.strftime(x, "%m/%d/%y") for x in daily_df_float_new.columns[6:]]
        daily_df_float_new.drop(columns=['Kind', 'Item', 'Frequency', 'Symbol Name'], inplace=True)
        daily_df_float_new.rename(columns={'Item Name ':'Item_name'}, inplace=True)
        daily_df_float.drop(columns=['Stock_name'], inplace=True)

         # 중복 제거 및 데이터 병합
        daily_df_float = daily_df_float.drop_duplicates(subset=['Symbol', 'Item_name'])
        daily_df_float_new = daily_df_float_new.drop_duplicates(subset=['Symbol', 'Item_name'])
        daily_df_float_update = pd.concat([daily_df_float.set_index(['Symbol', 'Item_name']), daily_df_float_new.set_index(['Symbol', 'Item_name'])], axis=1)
        daily_df_float_update.reset_index(inplace=True)
        daily_df_float_update['Symbol'] = daily_df_float_update['Symbol'].apply(lambda x: str(x)[1:])
        print(daily_df_float_update['Item_name'].unique()[:-1]) # 아이템 이름 출력
        
        return daily_df_float_update # 일간 데이터 반환

    # 일간 데이터를 변환하는 정적 메서드
    @staticmethod    
    def generate_daily_dataframe(df, item, start_num, transpose=True):
        temp = df[df['Item_name'] == item]
        if transpose:
            val = temp.iloc[:,start_num:].values.T
            col = temp['Symbol'].to_list()
            idx = pd.to_datetime([datetime.strptime(x, "%m/%d/%y").strftime("%Y-%m-%d") for x in temp.columns[start_num:] ])

        else:
            val = temp.iloc[:,start_num:].values
            col = pd.to_datetime([datetime.strptime(x, "%m/%d/%y").strftime("%Y-%m-%d") for x in temp.columns[start_num:] ])
            idx = temp['Symbol'].to_list()
        # 데이터 전치 여부에 따라 값, 컬럼, 인덱스 설정

        df = pd.DataFrame(val, index=idx, columns=col)
        return df # 변환된 데이터프레임 반환

    # 일간 데이터를 딕셔너리로 변환하는 메서드
    def get_daily_dict(self):

        daily_df_float = self.get_daily_data()

        item_list = daily_df_float['Item_name'].drop_duplicates().sort_values()[:-1].tolist() # none없애기 위해 -1
        var_list = ['volume', 'loan_transaction', 'dy', 'num_shares_ord', 'num_shares_prf', 'adj_price', 'netvolume_ist_foreign', 'marketcap', 'shares_foreign',  'marketcap_foreign', 'treasury_share', 'shareratio_largest']
        assert len(item_list) == len(var_list) # 아이템 리스트와 변수 리스트 설정 및 일치 여부 확인
        print("matched eng names: ", item_list, var_list)

        # daily_df_float에서 각 아이템별로 시계열 데이터프레임 생성
        daily_dict = {}
        for i in range(len(item_list)):
            df = self.generate_daily_dataframe(daily_df_float, item_list[i], 2)
            assert len(df.columns) == len(set(df.columns))
            daily_dict[var_list[i]] = df

        print("Completed generating a daily_dict") # 작업 완료 메시지 출력

        # 생성된 딕셔너리를 피클 파일로 저장
        with open(f'{self.save_dir}/daily_dict.pickle', 'wb') as f:
            pickle.dump(daily_dict, f)

    # 데이터에 시차를 추가하는 정적 메서드
    @staticmethod
    def generate_lag(df, idx_col=False, idx=None, col=None):
        df_lag = pd.DataFrame(df.values, index=[x + dt.timedelta(days=90) for x in df.index], columns=df.columns)
        df_lag = df_lag.resample('M').last()
        if idx_col:
            df_lag = df_lag.loc[idx, col]
        # 데이터에 90일 시차를 추가하고 월말 값으로 리샘플링
        return df_lag # 시차가 추가된 데이터프레임 반환

    # 컨센서스 데이터를 불러오는 메서드
    def get_consen_data(self):

        # 컨센서스 데이터 불러오기 및 컬럼 정리
        consen_dat = pd.read_csv('../raw/consen_dat.csv', encoding='EUC-KR', skiprows=8)
        consen_dat = consen_dat.drop(columns=['Symbol Name', 'Kind', 'Item', 'Frequency'])
        consen_dat.rename(columns= {'Symbol':'ticker', 'Item Name ':'item'}, inplace=True)

        # 데이터 변환 및 정리
        df_long = consen_dat.melt(id_vars=['ticker', 'item'], var_name='date', value_name='value')
        df_long['ticker'] = df_long['ticker'].apply(lambda x: str(x)[1:])
        df_long['date'] = pd.to_datetime(df_long['date'], format='%Y-%m-%d')
        df_long['item'] = df_long['item'].apply(lambda x: self.consen_label_dict[x])
        df_long['value'] = pd.to_numeric(df_long['value'].str.replace(',', ''), errors='coerce')

        df_long.set_index(['ticker', 'date'], inplace=True)

        # 피벗 테이블 생성 및 정렬
        result_df = df_long.pivot_table(index=['ticker', 'date'], columns='item', values='value')
        result_df.sort_index(inplace=True)

        print('Completed preprocessing the consen_dat!') # 작업 완료 메시지 출력

        result_df.to_feather(f'{self.save_dir}/csn_dat.feather')  # feather 파일로 저장

        return result_df # 컨센서스 데이터 반환

    # PDF 데이터를 불러오는 메서드
    def get_pdf(self, date, code_list):
    
        SQL = f'''
            SELECT TD, CODE, PDF_CODE
            FROM EUMQNTDB..QPM_ETF_PDF
            WHERE TD > {date} AND CODE in {code_list}
                '''
        # SQL 쿼리 실행 및 데이터프레임으로 변환
        df = pd.read_sql_query(SQL, con=self.conn)

        return df # PDF 데이터 반환

    # PDF 데이터를 불러오고 저장하는 메서드
    def get_pdf_data(self, path):
         # PDF 데이터 불러오기 및 저장
        thm_etf = pd.read_excel('../raw/thm_ETF_수정.xlsx')
        print('Start parsing pdf data from QPMS DB....')
        pdf_df = self.get_pdf('20100101', tuple(thm_etf['code']))
        print(pdf_df)
        pdf_df.to_feather(path)
        print('Completed savigng pdf_df!')
        

        return pdf_df # PDF 데이터 반환


def main():
    # Preprocessor 클래스의 인스턴스 생성
    pr = Preprocessor()
    pr.get_fin_dict()  # 재무 데이터 딕셔너리 생성

# 메인 함수 실행
if __name__ == "__main__":
    main()
    

                    

