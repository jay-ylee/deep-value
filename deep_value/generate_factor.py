from __future__ import print_function, division
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle
import warnings
warnings.filterwarnings('ignore')

import os.path as op
import numpy as np
import pandas as pd

from tqdm import tqdm
from Data import equity_data as eqd

import pdb
import gc
import talib

# 재무 데이터를 사용하여 팩터를 생성하는 GenerateFactors 클래스 정의
# eqd.Preprocessor 클래스를 상속받습니다.
class GenerateFactors(eqd.Preprocessor):
    def __init__(self, price_momentum, price_bm, price_sector):
        super(GenerateFactors, self).__init__() # eqd.Preprocessor의 초기화 메서드를 호출합니다.
         # 초기화 메서드에서 price_momentum, price_bm, price_sector 인자를 받습니다.
        self.price_momentum = price_momentum
        self.price_bm       = price_bm
        self.price_sector   = price_sector
        # 재무 데이터의 키 목록을 정의합니다.
        self.fin_dict_keys = ['current_asset', 'cash', 'current_liability', 'ni', 'ocf', 'total_cap',
                            'short_fin_inst', 'sales', 'allowance_cost', 'depreciation', 'dividend_payout_ratio',
                            'total_liability', 'longterm_borr', 'grossmargin', 'fixed_asset', 'FCFF',
                            'inventory', 'OI', 'debt', 'longterm_liab', 'DPS', 'cogs', 'operating_expese',
                            'deferred_asset', 'deferred_lib', 'operating_capital', 'account_receivable', 'EPS', 'CFPS', 'CAPEX']
    
    def get_fin_components(self):
        # 재무 데이터를 가져오는 메서드

        # 먼저 fin_dict, daily_dict 데이터 생성
        super().get_fin_dict() 
        super().get_daily_dict()
    
        # fin_dict와 daily_dict 피클 파일 불러오기
        with open(f'{self.save_dir}/fin_dict.pickle', 'rb') as f:
            fin_dict = pickle.load(f)
        print("Completed uploading fin_dict!")
        with open(f'{self.save_dir}/daily_dict.pickle', 'rb') as fr:
            daily_dict = pickle.load(fr)
        print("Completed uploading daily_dict!")

        # 수익률과 시가총액 계산
        marketcap               = daily_dict['marketcap'] * 1000000
        marketcap_monthly       = marketcap.resample('M').last()
        marketcap_monthly_lag1  = marketcap_monthly.shift(1)

        # 총자산 계산
        total_asset = fin_dict['total_asset'] * 1000
        total_asset = super().generate_lag(total_asset) # 총 자산을 계산하고 시차를 적용합니다.

        # 통일된 열과 인덱스 설정
        col = marketcap_monthly_lag1.columns.intersection(total_asset.columns)
        idx = marketcap_monthly_lag1.index.intersection(total_asset.index)

        marketcap_monthly_lag1 = marketcap_monthly_lag1.loc[idx, col]
        total_asset = total_asset.loc[idx, col] # 시가총액과 총 자산의 컬럼과 인덱스를 맞춥니다.

        # Initialize a dictionary to store the results
        results = {}
        for key in tqdm(self.fin_dict_keys):
            value = fin_dict[key] * 1000
            if key in ['DPS', 'EPS', 'CFPS', 'CAPEX']:
                value = fin_dict[key]
            results[key] = super().generate_lag(value, idx_col=True, idx=idx, col=col) # 각 재무 데이터에 대해 시차를 적용하여 results 딕셔너리에 저장합니다.
            assert len(results[key]) != 0

        # 일간 데이터의 특정 열에 대해 시차를 적용하고 results 딕셔너리에 저장합니다.
        results['num_shares_ord']         = daily_dict['num_shares_ord'].shift(1).resample('M').last().loc[idx, col]
        results['volume']                 = daily_dict['volume'].shift(1).resample('M').last().loc[idx, col]
        results['daily_turnover']         = (daily_dict['volume'] / daily_dict['num_shares_ord']).shift(1)
        results['total_asset']            = total_asset
        results['rtn']                    = daily_dict['adj_price'] / daily_dict['adj_price'].shift(1) - 1
        results['dy']                     = daily_dict['dy']
        results['shareratio_largest']     = daily_dict['shareratio_largest'] 
        results['netvolume_ist_foreign']  = daily_dict['netvolume_ist_foreign']

        results['shares_foreign']         = daily_dict['shares_foreign']
        results['treasury_share']         = daily_dict['treasury_share']
        results['num_shares_prf']         = daily_dict['num_shares_prf']
        results['loan_transaction']       = daily_dict['loan_transaction']

        results['marketcap'] = marketcap
        results['marketcap_monthly'] = marketcap_monthly
        results['marketcap_monthly_lag1'] = marketcap_monthly_lag1
        results['marketcap_foreign'] = daily_dict['marketcap_foreign']

        with open(f'{self.save_dir}/fin_components.pickle', 'wb') as f:
            pickle.dump(results, f)  # 생성된 results 딕셔너리를 피클 파일로 저장합니다.
        print("Completed saving fin_components!")

        return results # results 딕셔너리를 반환합니다

    # 데이터프레임을 변환하는 정적 메서드
    @staticmethod
    def transform_fin(df, col_name):
        temp = pd.melt(df.reset_index(), id_vars=['index'])
        temp.rename(columns={'index':'date', 'variable':'ticker', 'value':col_name}, inplace=True)
        temp.set_index(['ticker', 'date'], inplace=True)
        temp.fillna(0.0, inplace=True)
        # 데이터프레임을 멜트하여 날짜와 티커를 인덱스로 설정하고 결측치를 0으로 채웁니다.
        return temp

     # 재무 팩터를 계산하는 메서드
    def calculate_fin_factors(self):

        filename = 'fin_components.pickle'
        filepath = op.join(self.save_dir, filename)
        if op.isfile(filepath):
            print("Found pregenerated file {}".format(filename)) # 사전 생성된 파일이 있는지 확인합니다.

            # 사전 생성된 파일이 있다면 그것을 불러옵니다.
            with open(filepath, 'rb') as f:
                results = pickle.load(f)

        else:
            results = self.get_fin_components() # 사전 생성된 파일이 없으면 재무 데이터를 생성합니다.

        # matching col and idx
        col = results['marketcap_monthly_lag1'].columns.intersection(results['total_asset'].columns)
        idx = results['marketcap_monthly_lag1'].index.intersection(results['total_asset'].index)
        # 시가총액과 총 자산의 컬럼과 인덱스를 맞춥니다.


        # ASSES TO MARKETCAP
        A2ME = results['total_asset']  / results['marketcap_monthly_lag1']
        A2ME.replace([np.inf, -np.inf], 0, inplace=True)
        A2ME = self.transform_fin(A2ME, 'A2ME')


        # OPERATING ACCRUALS
        non_cash_working_capital       = (results['current_asset'] - results['cash']) - results['current_liability']
        non_cash_working_capital_chg   = non_cash_working_capital - non_cash_working_capital.shift(1)
        operating_accruals             = non_cash_working_capital_chg / results['total_asset'].shift(1)
        operating_accruals.replace([np.inf, -np.inf], 0, inplace=True)
        operating_accruals = self.transform_fin(operating_accruals, 'operating_accruals')

        # ABSOLUTE VALUE OF OPERATING ACCRUALS
        AOA = abs(operating_accruals)
        AOA.columns = ['AOA']

        # TOTAL ASSETS
        AT = results['total_asset']
        AT = self.transform_fin(AT, 'AT')

        # BEME
        BEME =  results['total_cap'] / results['marketcap_monthly_lag1']
        BEME.replace([np.inf, -np.inf], 0, inplace=True)
        BEME = self.transform_fin(BEME, 'BEME')

        # RATIO OF CASH AND SHORT TERM INVESTMENTS TO TOTAL ASSETS
        cash_ratio = (results['cash'] + results['short_fin_inst']) / results['total_asset']
        cash_ratio.replace([np.inf, -np.inf], 0, inplace=True)
        cash_ratio = self.transform_fin(cash_ratio, 'cash_ratio')

        # CAPITAL TURNOVER
        capital_turnover = (results['sales'] - results['allowance_cost']) / results['total_asset'].shift(1)
        capital_turnover.replace([np.inf, -np.inf], 0, inplace=True)
        capital_turnover = self.transform_fin(capital_turnover, 'capital_turnover')

        # CASHFLOW TO PRICE
        cashflow_to_price = (results['ni'] + results['depreciation']) / results['total_liability']
        cashflow_to_price.replace([np.inf, -np.inf], 0, inplace=True)
        cashflow_to_price = self.transform_fin(cashflow_to_price, 'cashflow_to_price')

        # DEBT TO PRICE
        debt_to_price = (results['longterm_borr'] + results['current_liability']) / results['marketcap_monthly_lag1']
        debt_to_price.replace([np.inf, -np.inf], 0, inplace=True)
        debt_to_price = self.transform_fin(debt_to_price, 'debt_to_price')

        # PCT CHANGE IN BOOK VALUE
        ceq = results['total_cap'] / results['total_cap'].shift(1) - 1
        ceq.replace([np.inf, -np.inf], 0, inplace=True)
        ceq = self.transform_fin(ceq, 'ceq')

        # d(dGM - dSales)
        gm_sales = ((results['grossmargin'] / results['grossmargin'].shift(1)) - 1) - ((results['sales'] / results['sales'].shift(1)) - 1)
        gm_sales.replace([np.inf, -np.inf], 0, inplace=True)
        gm_sales = self.transform_fin(gm_sales, 'gm_sales')

        # change in the split adjusted shares outstanding
        dSO = results['num_shares_ord'] / results['num_shares_ord'].shift(1)
        dSO.replace([np.inf, -np.inf], 0, inplace=True)
        dSO = self.transform_fin(dSO, 'dSO')

        fixed_asset_gro = (results['fixed_asset'] / results['total_asset'].shift(1)) / (results['fixed_asset'] / results['total_asset'].shift(1)).shift(1) - 1
        fixed_asset_gro.replace([np.inf, -np.inf], 0, inplace=True)
        fixed_asset_gro = self.transform_fin(fixed_asset_gro, 'fixed_asset_gro')

        E2P = results['ni'] / results['marketcap_monthly_lag1']
        E2P.replace([np.inf, -np.inf], 0, inplace=True)
        E2P = self.transform_fin(E2P, 'E2P')

        EPS = results['ni'] / results['num_shares_ord']
        EPS.replace([np.inf, -np.inf], 0, inplace=True)
        EPS = self.transform_fin(EPS, 'EPS')

        fcff = results['FCFF'] / results['total_cap']
        fcff.replace([np.inf, -np.inf], 0, inplace=True)
        fcff = self.transform_fin(fcff, 'fcff')

        asset_gro = results['total_asset'] / results['total_asset'].shift(12) - 1
        asset_gro.replace([np.inf, -np.inf], 0, inplace=True)
        asset_gro = self.transform_fin(asset_gro, 'asset_gro')

        IPM = results['OI']
        IPM = self.transform_fin(IPM, 'IPM')

        inventory_gro = (results['inventory'] - results['inventory'].shift(1)) / ((results['total_asset'] + results['total_asset'].shift(1)) / 2)
        inventory_gro.replace([np.inf, -np.inf], 0, inplace=True)
        inventory_gro = self.transform_fin(inventory_gro, 'inventory_gro')

        debt_to_asset = (results['debt'] + results['current_liability']) / (results['debt'] + results['current_liability'] + results['total_cap'])
        debt_to_asset.replace([np.inf, -np.inf], 0, inplace=True)
        debt_to_asset = self.transform_fin(debt_to_asset, 'debt_to_asset')

        LDP = results['DPS']
        LDP = self.transform_fin(LDP, 'LDP')

        MC = results['marketcap_monthly_lag1']
        MC = self.transform_fin(MC, 'MC')

        turnover = results['daily_turnover'].resample('M').last().loc[idx, col]
        turnover = self.transform_fin(turnover, 'turnover')

        O2P = results['dividend_payout_ratio']
        O2P = self.transform_fin(O2P, 'O2P')

        ol = (results['cogs'] + results['sales'] + results['operating_expese']) / results['total_asset']
        ol.replace([np.inf, -np.inf], 0, inplace=True)
        ol = self.transform_fin(ol, 'ol')

        pcm = ((results['sales'] - results['allowance_cost']) - results['cogs']) / (results['sales'] - results['allowance_cost'])
        pcm.replace([np.inf, -np.inf], 0, inplace=True)
        pcm = self.transform_fin(pcm, 'pcm')

        pm = results['OI'] / results['sales']
        pm.replace([np.inf, -np.inf], 0, inplace=True)
        pm = self.transform_fin(pm, 'pm')

        prof = (results['sales'] - results['cogs']) / results['total_cap']
        prof.replace([np.inf, -np.inf], 0, inplace=True)
        prof = self.transform_fin(prof, 'prof')

        q_score = (results['marketcap_monthly_lag1'] - (results['cash'] + results['short_fin_inst']) - (results['deferred_asset']-results['deferred_lib'])) / results['total_asset']
        q_score.replace([np.inf, -np.inf], 0, inplace=True)
        q_score = self.transform_fin(q_score, 'q_score')

        Ret = results['rtn'].shift(1).resample('M').last().loc[idx, col]
        Ret = self.transform_fin(Ret, 'Ret')

        ret_max = results['rtn'].shift(1).resample('M').max().loc[idx, col]
        ret_max = self.transform_fin(ret_max, 'ret_max')

        RNA = results['ni'] / non_cash_working_capital
        RNA.replace([np.inf, -np.inf], 0, inplace=True)
        RNA = self.transform_fin(RNA, 'RNA')

        ROA = results['ni'] / results['total_asset']
        ROA.replace([np.inf, -np.inf], 0, inplace=True)
        ROA = self.transform_fin(ROA, 'ROA')

        ROC = (results['marketcap_monthly_lag1'] + results['longterm_borr'] - results['total_asset']) / (results['cash'] + results['short_fin_inst'])
        ROC.replace([np.inf, -np.inf], 0, inplace=True)
        ROC = self.transform_fin(ROC, 'ROC')

        ROE = results['ni'] / results['total_cap']
        ROE.replace([np.inf, -np.inf], 0, inplace=True)
        ROE = self.transform_fin(ROE, 'ROE')

        roic = results['ni'] / results['operating_capital']
        roic.replace([np.inf, -np.inf], 0, inplace=True)
        roic = self.transform_fin(roic, 'roic')

        sales_to_cash = (results['sales'] - results['allowance_cost']) / (results['cash'] + results['short_fin_inst'])
        sales_to_cash.replace([np.inf, -np.inf], 0, inplace=True)
        sales_to_cash = self.transform_fin(sales_to_cash, 'sales_to_cash')

        Sales_g = results['sales'] / results['sales'].shift(1) - 1
        Sales_g.replace([np.inf, -np.inf], 0, inplace=True)
        Sales_g = self.transform_fin(Sales_g, 'Sales_g')

        SAT = results['sales'] / results['total_asset']
        SAT.replace([np.inf, -np.inf], 0, inplace=True)
        SAT = self.transform_fin(SAT, 'SAT')

        S2P = (results['sales'] - results['allowance_cost']) / results['marketcap_monthly_lag1']
        S2P.replace([np.inf, -np.inf], 0, inplace=True)
        S2P = self.transform_fin(S2P, 'S2P')

        sc_to_sales = (results['sales'] + results['operating_expese']) / (results['sales'] - results['allowance_cost'])
        sc_to_sales.replace([np.inf, -np.inf], 0, inplace=True)
        sc_to_sales = self.transform_fin(sc_to_sales, 'sc_to_sales')

        std_turnover = results['daily_turnover'].resample('M').std().loc[idx, col]
        std_turnover = self.transform_fin(std_turnover, 'std_turnover')

        std_vol = results['volume'].shift(1).resample('M').std().loc[idx, col]
        std_vol = self.transform_fin(std_vol, 'std_vol')

        tan = (0.715*results['account_receivable'] + 0.547*results['inventory'] + 0.535*results['fixed_asset'] + results['cash'] + results['short_fin_inst']) / results['total_asset']
        tan = self.transform_fin(tan, 'tan')

        total_vol = results['rtn'].shift(1).resample('M').std().loc[idx, col]
        total_vol = self.transform_fin(total_vol, 'total_vol')

        ocf = results['ocf']
        ocf = self.transform_fin(ocf, 'ocf')

        margin_ratio = (results['grossmargin'] / results['sales'])
        margin_ratio.replace([np.inf, -np.inf], 0, inplace=True)
        margin_ratio = self.transform_fin(margin_ratio, 'margin_ratio')

        working_capital = non_cash_working_capital
        working_capital = self.transform_fin(working_capital, 'working_capital')

        dy = results['dy'].fillna(method='ffill').shift(1).resample('M').last()
        dy = self.transform_fin(dy, 'dy')

        capex = results['CAPEX'] / results['sales']
        capex.replace([np.inf, -np.inf], 0, inplace=True)
        capex_sales = self.transform_fin(capex, 'capex')

        sales_gro_avg3 = Sales_g.rolling(3).mean()
        sales_gro_avg3.columns = ['sales_gro_avg3']

        oi_gro_avg3 = (results['OI'] / results['OI'].shift(1) - 1).rolling(3).mean()
        oi_gro_avg3 = self.transform_fin(oi_gro_avg3, 'oi_gro_avg3')

        ni_gro_avg3 = (results['ni'] / results['ni'].shift(1) - 1).rolling(3).mean()
        ni_gro_avg3 = self.transform_fin(ni_gro_avg3, 'ni_gro_avg3')

        margin_gro = results['grossmargin'].rolling(3).mean() / results['grossmargin'].rolling(3).mean().shift(1) - 1
        margin_gro.replace([np.inf, -np.inf], 0, inplace=True)
        margin_gro = self.transform_fin(margin_gro, 'margin_gro')

        frn_held_pct = (results['marketcap_foreign'] / results['marketcap']).fillna(method='ffill').shift(1).resample('M').last()
        frn_held_pct = pd.DataFrame(np.where(frn_held_pct > 1, 1.0, frn_held_pct), index=frn_held_pct.index, columns=frn_held_pct.columns)
        frn_held_pct = self.transform_fin(frn_held_pct, 'frn_held_pct')

        frn_held_pct_inc_1m = frn_held_pct.groupby(level='ticker')['frn_held_pct'].pct_change(1).to_frame()
        frn_held_pct_inc_1m.columns = ['frn_held_pct_inc_1m']
        frn_held_pct_inc_3m = frn_held_pct.groupby(level='ticker')['frn_held_pct'].pct_change(3).to_frame()
        frn_held_pct_inc_3m.columns = ['frn_held_pct_inc_3m']
        frn_held_pct_inc_6m = frn_held_pct.groupby(level='ticker')['frn_held_pct'].pct_change(6).to_frame()
        frn_held_pct_inc_6m.columns = ['frn_held_pct_inc_6m']
        frn_held_pct_inc_12m = frn_held_pct.groupby(level='ticker')['frn_held_pct'].pct_change(12).to_frame()
        frn_held_pct_inc_12m.columns = ['frn_held_pct_inc_12m']

        insider_held_pct = (results['shareratio_largest'] / 100).fillna(method='ffill').shift(1).resample('M').last()
        insider_held_pct = pd.DataFrame(np.where(insider_held_pct > 1, 1.0, insider_held_pct), index=insider_held_pct.index, columns=insider_held_pct.columns)
        insider_held_pct = self.transform_fin(insider_held_pct, 'insider_held_pct')

        insider_held_pct_inc_1m = insider_held_pct.groupby(level='ticker')['insider_held_pct'].pct_change(1).to_frame()
        insider_held_pct_inc_1m.columns = ['insider_held_pct_inc_1m']
        insider_held_pct_inc_3m = insider_held_pct.groupby(level='ticker')['insider_held_pct'].pct_change(3).to_frame()
        insider_held_pct_inc_3m.columns = ['insider_held_pct_inc_3m']
        insider_held_pct_inc_6m = insider_held_pct.groupby(level='ticker')['insider_held_pct'].pct_change(6).to_frame()
        insider_held_pct_inc_6m.columns = ['insider_held_pct_inc_6m']
        insider_held_pct_inc_12m = insider_held_pct.groupby(level='ticker')['insider_held_pct'].pct_change(12).to_frame()
        insider_held_pct_inc_12m.columns = ['insider_held_pct_inc_12m']

        netvolume_inst_frn = (results['netvolume_ist_foreign'] / results['volume']).shift(1).resample('M').last()
        netvolume_inst_frn.replace([np.inf, -np.inf], 0, inplace=True)
        netvolume_inst_frn = self.transform_fin(netvolume_inst_frn, 'netvolume_inst_frn')

        shares_foreign = results['shares_foreign'].shift(1).resample('M').last()
        shares_foreign = self.transform_fin(shares_foreign, 'shares_foreign')

        treasury_share = results['treasury_share'].shift(1).resample('M').last()
        treasury_share = self.transform_fin(treasury_share, 'treasury_share')

        num_shares_prf = results['num_shares_prf'].shift(1).resample('M').last()
        num_shares_prf = self.transform_fin(num_shares_prf, 'num_shares_prf')

        loan_transaction = results['loan_transaction'].shift(1).resample('M').last()
        loan_transaction = self.transform_fin(loan_transaction, 'loan_transaction')

        lnme = np.log(results['marketcap'].shift(1).resample('M').last())
        lnme = self.transform_fin(lnme, 'lnme')

        capex_gro = capex.pct_change(12)
        capex_gro.replace([np.inf, -np.inf], 0, inplace=True)
        capex_gro = self.transform_fin(capex_gro, 'capex_gro')

        cfps_gro = results['CFPS'].pct_change(3)
        cfps_gro.replace([np.inf, -np.inf], 0, inplace=True)
        cfps_gro = self.transform_fin(cfps_gro, 'cfps_gro')

        # 합치기
        merge_factors = [A2ME, operating_accruals, AOA, AT, BEME, cash_ratio, capital_turnover,
                        cashflow_to_price, debt_to_price, ceq, gm_sales, dSO, fixed_asset_gro,
                        E2P, EPS, fcff, asset_gro, IPM, inventory_gro, debt_to_asset, LDP, MC,
                        turnover, O2P, ol, pcm, pm, prof, q_score, Ret, ret_max, RNA, ROA, ROC,
                        ROE, roic, sales_to_cash, Sales_g, SAT, S2P, sc_to_sales, std_turnover,
                        std_vol, tan,total_vol, ocf, margin_ratio, working_capital, dy, capex_sales,
                        sales_gro_avg3, oi_gro_avg3, ni_gro_avg3, margin_gro, frn_held_pct, frn_held_pct_inc_1m,
                        frn_held_pct_inc_3m, frn_held_pct_inc_6m, frn_held_pct_inc_12m, insider_held_pct,
                        insider_held_pct_inc_1m, insider_held_pct_inc_3m, insider_held_pct_inc_6m,
                        netvolume_inst_frn, shares_foreign, treasury_share, num_shares_prf, loan_transaction,
                        lnme, capex_gro, cfps_gro]

        def ensure_unique_index(df, aggregate=False):
            if df.index.has_duplicates:
                if aggregate:
                    # Example aggregation by mean. Adjust according to your needs.
                    df = df.groupby(df.index).mean()
                else:
                    df = df[~df.index.duplicated(keep='first')]
            return df

        # 각 데이터프레임의 인덱스가 고유한지 확인하고, 고유하지 않으면 중복을 제거합니다.
        unique_index_factors = [ensure_unique_index(df) for df in merge_factors]
        factor_df = pd.concat(unique_index_factors, axis=1)
        factor_df.sort_index(inplace=True)
        factor_df.to_feather(f'{self.save_dir}/factor_fin_df.feather')
        print("Completed saving fin_factor_df!")

        return factor_df

    # 컨센서스 팩터 계산하는 메서드
    def generate_fin_consen_factor_df(self):

        filename_fin = 'factor_fin_df.feather'
        filepath_fin = op.join(self.save_dir, filename_fin)
        if op.isfile(filepath_fin):
            print("Found pregenerated file {}".format(filename_fin)) # 이미 생성된 파일 있으면 가져오기 

            factor_fin_df = pd.read_feather(filepath_fin)
        else:
            factor_fin_df = self.calculate_fin_factors() # 생성된 파일 없으면 생성하기
        
        filename_csn = 'csn_dat.feather'
        filepath_csn = op.join(self.save_dir, filename_csn)
        if op.isfile(filepath_csn):
            print("Found pregenerated file {}".format(filename_csn))  # 이미 생성된 파일 있으면 가져오기 

            factor_csn_df = pd.read_feather(filepath_csn)
        else:
            factor_csn_df = super().get_consen_data() # 생성된 파일 없으면 생성하기

        factor_df = pd.concat([factor_fin_df, factor_csn_df], axis=1) # 재무데이터와 컨센데이터 합치기
        factor_df = factor_df.sort_index() # 데이터 소트

        # fillna
        factor_df['eps_gro_fy1'] = factor_df['eps_gro_fy1'].fillna(factor_df['ni_gro_avg3'])
        factor_df['eps_gro_fy2'] = factor_df['eps_gro_fy2'].fillna(factor_df['ni_gro_avg3'])

        factor_df['sales_gro_fy1'] = factor_df['sales_gro_fy1'].fillna(factor_df['sales_gro_avg3'])
        factor_df['sales_gro_fy2'] = factor_df['sales_gro_fy2'].fillna(factor_df['sales_gro_avg3'])

        factor_df['oi_gro_fy1'] = factor_df['oi_gro_fy1'].fillna(factor_df['oi_gro_avg3'])
        factor_df['oi_gro_fy2'] = factor_df['oi_gro_fy2'].fillna(factor_df['oi_gro_avg3'])

        factor_df['per_fy1'] = factor_df['per_fy1'].fillna(factor_df['per_fy0'])
        factor_df['per_fy2'] = factor_df['per_fy2'].fillna(factor_df['per_fy0'])

        factor_df['pbr_fy1'] = factor_df['pbr_fy1'].fillna(factor_df['pbr_fy0'])
        factor_df['pbr_fy2'] = factor_df['pbr_fy2'].fillna(factor_df['pbr_fy0'])

        factor_df.fillna(0.0, inplace=True)
        factor_df = factor_df.unstack('ticker').sort_index().shift(1).resample('M').last().stack('ticker').swaplevel().sort_index()

        factor_df.to_feather(f'{self.save_dir}/factor_fin_csn_df.feather')
        print("Completed generating factor_fin_csn_df!")

        return factor_df

    # 가격 데이터 가져오는 메서드
    def get_price_dat(self):

        price_dat = pd.read_csv('../raw/price_dat.csv') # 기초 데이터 불러오기

        # 종가를 제외한 시가, 고가, 저가의 경우 수정가격이 아니므로 할인율 적용
        price_dat['OPEN_AP'] = price_dat['OPEN_P'] / price_dat['ADJ_C']
        price_dat['HIGH_AP'] = price_dat['HIGH_P'] / price_dat['ADJ_C']
        price_dat['LOW_AP'] = price_dat['LOW_P'] / price_dat['ADJ_C']
        price_dat = price_dat[['CODE', 'TD', 'OPEN_AP', 'HIGH_AP', 'LOW_AP', 'CLOSE_AP', 'TRAN_QTY']]
        price_dat['TRAN_AMT'] = price_dat['TRAN_QTY'] * price_dat['CLOSE_AP']

        # reset dtypes
        price_dat['TD'] = pd.to_datetime(price_dat['TD'], format='%Y%m%d')
        price_dat['CODE'] = [(6-len(str(x)))*"0" + str(x) for x in price_dat['CODE']]
        price_dat.rename(columns={'TD':'date', 'CODE':'ticker'}, inplace=True)
        price_dat = price_dat.set_index(['ticker', 'date'])

        assert not price_dat.empty, "DataFrame is empty after filtering"
        print('num of stocks: ', len(price_dat.index.unique('ticker').unique()))

        # 이중 인덱스 데이터프레임으로 생성
        price_dat = (price_dat.unstack('ticker')
                .sort_index()
                .dropna(axis=1, how='all')
                .fillna(0.0)
                .stack('ticker')
                .swaplevel())

        assert not price_dat.empty, "DataFrame is empty after filtering"
        print('num of stocks: ', len(price_dat.index.unique('ticker').unique()))

        price_dat.sort_index(inplace=True)

        # 가공완료한 가격 데이터 저장
        price_dat.to_feather(f'{self.save_dir}/price_dat_pr.feather')

        return price_dat

    # 수익률 계산하는 메서드
    def calculate_returns(self):

        filename_returns = 'returns.feather'
        filepath_returns = op.join(self.save_dir, filename_returns)
        if op.isfile(filepath_returns):
            print("Found pregenerated file {}".format(filename_returns))
            returns = pd.read_feather(filepath_returns)
            return returns # 이미 생성된 수익률 파일이 있으면 이를 불러와 반환
        else:

             # price_dat 불러오기
            filename_price = 'price_dat_pr.feather'
            filepath_price = op.join(self.save_dir, filename_price)
            if op.isfile(filepath_price):
                print("Found pregenerated file {}".format(filename_price))
                price_dat = pd.read_feather(filepath_price) # 이미 생성된 가격 데이터 파일이 있으면 이를 불러옵니다.
            else:
                price_dat = self.get_price_dat()     # 생성된 파일이 없으면 가격 데이터를 불러오는 메서드를 호출합니다.

            # 일별 수익률 계산
            price_dat = price_dat.sort_index()
            intervals = [1,5,10,21,63]
            by_ticker = price_dat.groupby(level='ticker')['CLOSE_AP']

            # 주어진 기간에 따른 수익률을 계산하고 데이터프레임으로 병합합니다.
            returns = []
            for t in intervals:
                returns.append(by_ticker.pct_change(t).to_frame(f'ret_{t}'))
            returns = pd.concat(returns, axis=1)
            returns = returns.sort_index()
            

            # 이상치 제거
            max_ret_by_ticker = returns.groupby(level='ticker').max()
            quantiles = max_ret_by_ticker.quantile(.95) # 95% 분위 이상의 이상치를 제거합니다.
            to_drop = []
            for ret, q in quantiles.items():
                to_drop.extend(max_ret_by_ticker[max_ret_by_ticker[ret]>q].index.tolist())
            to_drop = pd.Series(to_drop).value_counts()
            to_drop = to_drop[to_drop>1].index.to_list()
            print(f"dropping {len(to_drop)} number of stocks")

            price_dat = price_dat.drop(to_drop, level='ticker')
            price_dat.sort_index(inplace=True)
            assert not price_dat.empty, "DataFrame is empty after filtering" # 이상치 제거 후 데이터프레임이 비어있지 않은지 확인합니다.
            # 결과를 파일로 저장
            price_dat.to_feather(f'{self.save_dir}/price_dat_pr_remove_outlier.feather')
            returns.to_feather(f'{self.save_dir}/returns.feather') 

            return returns, price_dat # 수익률과 가격 데이터를 반환합니다.

    # 볼린저 밴드를 계산하는 정적 메서드
    @staticmethod    
    def get_bollinger(x):
        u, m, l = talib.BBANDS(x)
        return pd.DataFrame({'u':u,'m':m, 'l':l})

    # 데이터를 정규화하는 정적 메서드
    @staticmethod
    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # 데이터프레임의 데이터 타입을 변경하는 정적 메서드
    @staticmethod
    def ChangeDtype(df):
        for col in df.select_dtypes(include='float64').columns:
                    df[col] = df[col].astype('float32') # float64 타입의 컬럼을 float32로 변경합니다.
        return df

    # talib의 NATR 지표를 계산할 때 결측치를 무시하는 정적 메서드
    @staticmethod    
    def natr_ignore_na(x):
        if x.loc[:, ['HIGH_AP', 'LOW_AP', 'CLOSE_AP']].isnull().any().any():
            return pd.Series(data=np.nan, index=x.index)
        else:
            return talib.NATR(x.HIGH_AP, x.LOW_AP, x.CLOSE_AP) # 'HIGH_AP', 'LOW_AP', 'CLOSE_AP' 컬럼에 결측치가 있는지 확인하고, 결측치가 있으면 NaN 시리즈를 반환하고, 없으면 NATR 지표를 계산하여 반환합니다.

    # 기술적 지표 데이터프레임을 생성하는 메서드
    def generate_TI_df(self):

        # price_dat 및 returns 데이터 불러오기
        filename_price = 'price_dat_pr_remove_outlier.feather'
        filename_returns = 'returns.feather'
        filepath_price = op.join(self.save_dir, filename_price)
        filepath_returns = op.join(self.save_dir, filename_returns)
        if op.isfile(filepath_price) and op.isfile(filepath_returns):
            print("Found pregenerated file {}".format(filename_price))
            print("Found pregenerated file {}".format(filename_returns))
            price_dat = pd.read_feather(filepath_price)
            price_dat = self.ChangeDtype(price_dat)

            returns = pd.read_feather(filepath_returns)
            returns = self.ChangeDtype(returns) # 이미 생성된 파일이 있으면 이를 불러와 dtype을 변경합니다

        else:
            returns, price_dat = self.calculate_returns() # 이미 생성된 파일이 있으면 이를 불러와 dtype을 변경합니다

        price_dat = self.ChangeDtype(price_dat)
        returns = self.ChangeDtype(returns)
        # dtype을 변경합니다.

        price_dat.sort_index(inplace=True)
        returns.sort_index(inplace=True)
         # 인덱스를 정렬합니다.
        

        # 기술적 지표 PPO, NATR, RSI,볼린저밴드를 계산합니다.
        ppo = price_dat.groupby(level='ticker', group_keys=False)['CLOSE_AP'].transform(talib.PPO)
        natr = price_dat.groupby('ticker').apply(lambda x: talib.NATR(x['HIGH_AP'], x['LOW_AP'], x['CLOSE_AP']))
        rsi = price_dat.groupby('ticker')['CLOSE_AP'].transform(talib.RSI)
        bbands = price_dat.groupby('ticker')['CLOSE_AP'].apply(self.get_bollinger)

        # 인덱스를 price_dat와 맞춥니다.
        ppo.index = price_dat.index
        natr.index = price_dat.index
        rsi.index = price_dat.index
        bbands.index = price_dat.index

        filename_tech_factors_daily = 'tech_factors_daily.feather'
        filepath_tech_factors_daily = op.join(self.save_dir, filename_tech_factors_daily)
        if op.isfile(filepath_tech_factors_daily):
            tech_factors = pd.read_feather(filepath_tech_factors_daily) # 이미 생성된 기술적 지표 데이터가 있으면 이를 불러옵니다.
        else:
            print("Creating tech_factors_daily..")
            
            # self.price_momentum 인자가 True이면, 가격 모멘텀 지표 계산 --> 이 부분을 True로 해서 돌리면 메모리 에러가 날 수 있으니 주의. 메모리 문제로 현재 모델은 가격모멘텀 팩터는 계산되고 있지 않음.
            if self.price_momentum:
                 # 가격 모멘텀 지표 계산
                print("Start calculating price_momentum factors")
                price_dat['52w_high'] = price_dat.groupby(level='ticker')['HIGH_AP'].rolling(window=260).max().values
                price_dat['price_to_52w_high'] = price_dat['CLOSE_AP'] / price_dat['52w_high']

                price_dat['52w_low'] = price_dat.groupby(level='ticker')['LOW_AP'].rolling(window=260).min().values
                price_dat['price_to_52w_low'] = price_dat['CLOSE_AP'] / price_dat['52w_low']

                price_dat['mavg_20d'] = price_dat.groupby(level='ticker')['CLOSE_AP'].rolling(window=20).mean().values
                price_dat['mavg_60d'] = price_dat.groupby(level='ticker')['CLOSE_AP'].rolling(window=60).mean().values
                price_dat['mavg_120d'] = price_dat.groupby(level='ticker')['CLOSE_AP'].rolling(window=120).mean().values
                price_dat['mavg_250d'] = price_dat.groupby(level='ticker')['CLOSE_AP'].rolling(window=250).mean().values

                price_dat['price_to_mavg_20d'] = price_dat['CLOSE_AP'] / price_dat['mavg_20d']
                price_dat['price_to_mavg_60d'] = price_dat['CLOSE_AP'] / price_dat['mavg_60d']
                price_dat['price_to_mavg_120d'] = price_dat['CLOSE_AP'] / price_dat['mavg_120d']
                price_dat['price_to_mavg_250d'] = price_dat['CLOSE_AP'] / price_dat['mavg_250d']


                price_dat['price_vol_60'] = price_dat.groupby(level='ticker')['CLOSE_AP'].rolling(window=60).std().values
                price_dat['price_to_upper_1volband'] = price_dat['CLOSE_AP'] / (price_dat['mavg_60d'] + price_dat['price_vol_60'])
                price_dat['price_to_upper_2volband'] = price_dat['CLOSE_AP'] / (price_dat['mavg_60d'] + 2*price_dat['price_vol_60'])
                price_dat['price_to_historical_high'] = price_dat['CLOSE_AP'] / price_dat.groupby(level='ticker')['HIGH_AP'].expanding().max().values
                price_dat['price_to_historical_low'] = price_dat['CLOSE_AP'] / price_dat.groupby(level='ticker')['LOW_AP'].expanding().min().values
                price_dat['mavg_20d_to_60d'] = price_dat['mavg_20d'] / price_dat['mavg_60d']
                price_dat['mavg_20d_to_120d'] = price_dat['mavg_20d'] / price_dat['mavg_120d']
                price_dat['mavg_20d_to_250d'] = price_dat['mavg_20d'] / price_dat['mavg_250d']
                price_dat['mavg_120d_to_250d'] = price_dat['mavg_120d'] / price_dat['mavg_250d']

                price_dat['vwma_20d'] = price_dat.groupby(level='ticker')['TRAN_AMT'].rolling(window=20).sum().values / price_dat.groupby(level='ticker')['TRAN_QTY'].rolling(window=20).sum().values
                price_dat['vwma_60d'] = price_dat.groupby(level='ticker')['TRAN_AMT'].rolling(window=60).sum().values / price_dat.groupby(level='ticker')['TRAN_QTY'].rolling(window=60).sum().values
                price_dat['vwma_120d'] = price_dat.groupby(level='ticker')['TRAN_AMT'].rolling(window=120).sum().values / price_dat.groupby(level='ticker')['TRAN_QTY'].rolling(window=120).sum().values
                price_dat['vwma_250d'] = price_dat.groupby(level='ticker')['TRAN_AMT'].rolling(window=250).sum().values / price_dat.groupby(level='ticker')['TRAN_QTY'].rolling(window=250).sum().values

                price_dat['vwma_to_ema_20d'] = np.log(price_dat['vwma_20d']) - np.log(price_dat['mavg_20d'])
                price_dat['vwma_to_ema_60d'] = np.log(price_dat['vwma_60d']) - np.log(price_dat['mavg_60d'])
                price_dat['vwma_to_ema_120d'] = np.log(price_dat['vwma_120d']) - np.log(price_dat['mavg_120d'])
                price_dat['vwma_to_ema_250d'] = np.log(price_dat['vwma_250d']) - np.log(price_dat['mavg_250d'])

                price_dat['mavg_20d_mom_1m'] = np.log(price_dat['mavg_20d']) - np.log(price_dat['mavg_20d'].groupby(level='ticker').shift(20).values)
                price_dat['mavg_60d_mom_3m'] = np.log(price_dat['mavg_60d']) - np.log(price_dat['mavg_60d'].groupby(level='ticker').shift(60).values)
                price_dat['mavg_120d_mom_6m'] = np.log(price_dat['mavg_120d']) - np.log(price_dat['mavg_120d'].groupby(level='ticker').shift(120).values)
                price_dat['mavg_250d_mom_1y'] = np.log(price_dat['mavg_250d']) - np.log(price_dat['mavg_250d'].groupby(level='ticker').shift(250).values)

                price_dat['volume_1m_ratio'] = price_dat['TRAN_QTY'].groupby(level='ticker').rolling(window=20).mean().values / price_dat['TRAN_QTY'].groupby(level='ticker').rolling(window=230).mean().shift(20).values
                price_dat['volume_3m_ratio'] = price_dat['TRAN_QTY'].groupby(level='ticker').rolling(window=60).mean().values / price_dat['TRAN_QTY'].groupby(level='ticker').rolling(window=230).mean().shift(20).values

                returns['vol_21'] = returns.groupby(level='ticker')['ret_1'].rolling(window=21).std().values
                returns['vol_63'] = returns.groupby(level='ticker')['ret_1'].rolling(window=63).std().values

                returns['vol_adj_ret_1m'] = returns['ret_21'] / returns['vol_21']
                returns['vol_adj_ret_3m'] = returns['ret_63'] / returns['vol_63']

            # 벤치마크 데이터 불러오기
            bm_dat = pd.read_excel('../raw/bm_data.xlsx')
            bm_dat['date'] = pd.to_datetime(bm_dat['date'])
            bm_dat.set_index('date', inplace=True)
            ks_ret_21 = bm_dat['코스피'] / bm_dat['코스피'].shift(21) - 1

            returns_bm = pd.merge(returns.reset_index(), ks_ret_21.reset_index().rename(columns={'코스피':'ks_ret_21'}), left_on='date', right_on='date', how='left')
            returns_bm = returns_bm.set_index(['ticker', 'date'])
            returns_bm['ret_rel_to_bm_avg_21'] = returns_bm['ret_21'] - returns_bm['ks_ret_21']
            # 코스피 수익률 데이터를 returns에 병합하고 상대 수익률을 계산합니다.

            if not self.price_momentum: # 모멘텀 지표를 사용하지 않을 경우 vol_21을 계산합니다.
                print("Calculating vol_21...")
                returns_bm['vol_21'] = returns_bm.groupby(level='ticker')['ret_1'].rolling(window=21).std().values

            if self.price_bm: # 가격 BM 팩터를 계산합니다. --> 이 부분을 True로 해서 돌리면 메모리 에러가 날 수 있으니 주의. 메모리 문제로 현재 모델은 가격모멘텀 팩터는 계산되고 있지 않음.
                print("Start calculating price_bm factors..")
                ks_ret_63 = bm_dat['코스피'] / bm_dat['코스피'].shift(63) - 1
                returns_bm = pd.merge(returns_bm.reset_index(), ks_ret_63.reset_index().rename(columns={'코스피':'ks_ret_63'}), left_on='date', right_on='date', how='left')
                returns_bm = returns_bm.set_index(['ticker', 'date'])            
                returns_bm = returns_bm.unstack('ticker').sort_index().fillna(method='ffill').stack('ticker').swaplevel().sort_index()
                returns_bm['ret_rel_to_bm_avg_63'] = returns_bm['ret_63'] - returns_bm['ks_ret_63']

            if self.price_sector: # 섹터 중립 가격 팩터 계산  --> 이 부분을 True로 해서 돌리면 메모리 에러가 날 수 있으니 주의. 메모리 문제로 현재 모델은 가격모멘텀 팩터는 계산되고 있지 않음.
                print("Start calculating price_sector factors..")

                sector_dat = pd.read_excel('../raw/sector_dat.xlsx')
                sector_dat['Symbol'] = sector_dat['Symbol'].apply(lambda x: x[1:])
                sector_dat.set_index(['Symbol', '회계년'], inplace=True)
                returns_bm = returns_bm.reset_index()
                returns_bm['year'] = returns_bm['date'].dt.year
                returns_bm.set_index(['ticker', 'date', 'year'], inplace=True)

                sector_dat = sector_dat.rename_axis(['ticker', 'year'])
                sector_dat_df = sector_dat.reset_index()
                returns_bm = pd.merge(returns_bm.reset_index(), sector_dat_df, how='left',
                        left_on=['ticker', 'year'],
                        right_on=['ticker', 'year'])

                returns_bm.set_index(['ticker', 'date'], inplace=True)
                returns_bm = returns_bm.drop(columns=['year', 'Name', '결산월', '주기'])

                grouped_returns_bm = returns_bm.groupby(['FnGuide Sector', pd.Grouper(level='date')])
                sector_avg_ret_1 = grouped_returns_bm['ret_1'].mean().replace([np.inf, -np.inf], 0.0)
                sector_avg_ret_5 = grouped_returns_bm['ret_5'].mean().replace([np.inf, -np.inf], 0.0)
                sector_avg_ret_10 = grouped_returns_bm['ret_10'].mean().replace([np.inf, -np.inf], 0.0)
                sector_avg_ret_21 = grouped_returns_bm['ret_21'].mean().replace([np.inf, -np.inf], 0.0)
                sector_avg_ret_63 = grouped_returns_bm['ret_63'].mean().replace([np.inf, -np.inf], 0.0)

                returns_bm['sector_avg_ret_1'] = returns_bm.join(sector_avg_ret_1.rename('sector_avg_ret_1'), 
                                                                        on=['FnGuide Sector', returns_bm.index.get_level_values(1)],
                                                                        how='left')['sector_avg_ret_1']

                returns_bm['ret_rel_to_sector_avg_1'] = returns_bm['ret_1'] - returns_bm['sector_avg_ret_1']

                returns_bm['sector_avg_ret_5'] = returns_bm.join(sector_avg_ret_5.rename('sector_avg_ret_5'), 
                                                                        on=['FnGuide Sector', returns_bm.index.get_level_values(1)],
                                                                        how='left')['sector_avg_ret_5']

                returns_bm['ret_rel_to_sector_avg_5'] = returns_bm['ret_5'] - returns_bm['sector_avg_ret_5']

                returns_bm['sector_avg_ret_10'] = returns_bm.join(sector_avg_ret_10.rename('sector_avg_ret_10'), 
                                                                        on=['FnGuide Sector', returns_bm.index.get_level_values(1)],
                                                                        how='left')['sector_avg_ret_10']

                returns_bm['ret_rel_to_sector_avg_10'] = returns_bm['ret_10'] - returns_bm['sector_avg_ret_10']

                returns_bm['sector_avg_ret_21'] = returns_bm.join(sector_avg_ret_21.rename('sector_avg_ret_21'), 
                                                                        on=['FnGuide Sector', returns_bm.index.get_level_values(1)],
                                                                        how='left')['sector_avg_ret_21']

                returns_bm['ret_rel_to_sector_avg_21'] = returns_bm['ret_21'] - returns_bm['sector_avg_ret_21']

                returns_bm['sector_avg_ret_63'] = returns_bm.join(sector_avg_ret_63.rename('sector_avg_ret_63'), 
                                                                        on=['FnGuide Sector', returns_bm.index.get_level_values(1)],
                                                                        how='left')['sector_avg_ret_63']

                returns_bm['ret_rel_to_sector_avg_63'] = returns_bm['ret_63'] - returns_bm['sector_avg_ret_63']


            # 기술적 지표와 가격 데이터를 feather 파일로 저장합니다.
            returns_bm.sort_index(inplace=True)
            returns_bm.to_feather(f'{self.save_dir}/returns_bm.feather')
            price_dat.to_feather(f'{self.save_dir}/price_dat_tech.feather')

            # 기술적 지표들 병합
            print("Now combining technical features...")
            tech_factors = pd.concat([price_dat, returns_bm, ppo.to_frame('PPO'), natr.to_frame('NATR'), rsi.to_frame('RSI'), bbands], axis=1)
             # 메모리 절약을 위해 불필요한 데이터 삭제 및 가비지 컬렉션 수행
            del price_dat, returns_bm, ppo, natr, rsi, bbands
            gc.collect()
            
            tech_factors['bbl'] = tech_factors['CLOSE_AP'] / tech_factors['l']
            tech_factors['bbu'] = tech_factors['CLOSE_AP'] / tech_factors['u']
            tech_factors.drop(['u', 'm', 'l'], axis=1, inplace=True)
            # 볼린저 밴드 하단과 상단을 기준으로 bbl, bbu 지표를 생성하고 볼린저 밴드의 중간값과 하단, 상단을 삭제합니다.
            print("Saving tech_factors_daily...")

            tech_factors.to_feather(f'{self.save_dir}/tech_factors_daily.feather')  # 기술적 지표 데이터프레임을 feather 파일로 저장합니다.

        print("Changing the type of tech_factors...")

        tech_factors = self.ChangeDtype(tech_factors)  # 기술적 지표 데이터프레임의 dtype을 변경합니다.

        # forward 수익률 계산
        print("Start computing foward returns..")
        tech_factors['fwd_ret_21'] = tech_factors.groupby('ticker')['ret_21'].shift(-21).to_frame('fwd_ret_21').unstack('ticker').sort_index().shift(1).stack('ticker').swaplevel().sort_index()
        tech_factors['fwd_vol_21'] = tech_factors.groupby('ticker')['vol_21'].shift(-21).to_frame('fwd_vol_21').unstack('ticker').sort_index().shift(1).stack('ticker').swaplevel().sort_index()
        tech_factors['fwd_ret_rel_to_bm_avg_21'] = tech_factors.groupby('ticker')['ret_rel_to_bm_avg_21'].shift(-21).to_frame('fwd_ret_rel_to_bm_avg_21').unstack('ticker').sort_index().shift(1).stack('ticker').swaplevel().sort_index()

        tech_factors['fwd_sr_21'] = tech_factors['fwd_ret_21'] / tech_factors['fwd_vol_21']
        tech_factors['fwd_sr_21'] = tech_factors.groupby(level='date', group_keys=False)['fwd_sr_21'].apply(lambda x: x.replace([np.inf, -np.inf], 0))

        tech_factors['fwd_sr_21_adj'] = tech_factors['fwd_ret_rel_to_bm_avg_21'] - 0.1 * tech_factors['fwd_vol_21']
        tech_factors['fwd_sr_21_adj'] = tech_factors.groupby(level='date', group_keys=False)['fwd_sr_21_adj'].apply(lambda x: x.replace([np.inf, -np.inf], 0))
        
        tech_factors.fillna(0.0, inplace=True)

        # 데일리로 계산한 기술적지표를 월별 데이터로 변환
        print("Converting frequency into monthly..")
        tech_factors_monthly = tech_factors.unstack('ticker').sort_index().shift(1).resample('M').last().stack('ticker').swaplevel().sort_index()
        print("Completed converting daily tech_factors into monthly tech factors!")

        # nan 처리
        tech_factors_monthly = tech_factors_monthly.unstack('date').dropna(how='all').stack('date', dropna=False)
        tech_factors_monthly = tech_factors_monthly.fillna(0.0)
        numeric_cols = tech_factors_monthly.select_dtypes(include=[np.number]).columns
        tech_factors_monthly.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 각 팩터별 10분위 수 할당
        print("Start generating decile factors...")
        for col in numeric_cols:
            valid_data = tech_factors_monthly[col].replace([np.inf, -np.inf], np.nan)
            
            tech_factors_monthly[col + '_decile'] = valid_data.groupby(level='date').transform(
                lambda x: (
                    pd.qcut(x.dropna(), q=10, labels=False, duplicates='drop')
                    if not x.dropna().empty else pd.Series(np.nan, index=x.index)
                )
            )
    
        tech_factors_monthly[col + '_decile'].fillna(0, inplace=True)
        tech_factors_monthly[col + '_decile'] = tech_factors_monthly[col + '_decile'].astype(int)
        print("Completed generating decile factors!")

        # 일자 팩터 생성
        dates = tech_factors_monthly.index.get_level_values('date')
        tech_factors_monthly['weekday'] = dates.weekday
        tech_factors_monthly['month'] = dates.month
        tech_factors_monthly['year'] = dates.year

        tech_factors.fillna(0.0, inplace=True)
        tech_factors_monthly.to_feather(f'{self.save_dir}/tech_factors.feather')
        print('Completed generating tech_factors!')
        
        return tech_factors_monthly

    def get_factors(self):

        filename_fin_csn = 'factor_fin_csn_df.feather' # 금융 컨센서스 팩터 데이터를 위한 파일 이름과 경로 정의
        filepath_fin_csn = op.join(self.save_dir, filename_fin_csn)
        if op.isfile(filepath_fin_csn): # 사전 생성된 금융 컨센서스 팩터 데이터 파일이 존재하는지 확인
            print("Found pregenerated file {}".format(filename_fin_csn))
            factor_fin_csn_df = pd.read_feather(filepath_fin_csn) # 파일이 존재하면 데이터를 파일에서 읽어옴
        else:
            factor_fin_csn_df = self.generate_fin_consen_factor_df() # 그렇지 않으면 금융 컨센서스 팩터 데이터를 생성
        
        filename_ti = 'tech_factors.feather'  # 기술적 팩터 데이터를 위한 파일 이름과 경로 정의
        filepath_ti = op.join(self.save_dir, filename_ti)
        if op.isfile(filepath_ti):
            print("Found pregenerated file {}".format(filename_ti)) # 사전 생성된 기술적 팩터 데이터 파일이 존재하는지 확인
            tech_factors = pd.read_feather(filepath_ti) # 파일이 존재하면 데이터를 파일에서 읽어옴
        else: # 그렇지 않으면 기술적 팩터 데이터를 생성
            tech_factors = self.generate_TI_df()

        factor_fin_csn_df.fillna(0.0, inplace=True)  # 금융 컨센서스 팩터의 NaN 값을 0.0으로 채움
        factors_fin_csn_df_grouped = factor_fin_csn_df.groupby(level='date', group_keys=False) # 금융 컨센서스 팩터 데이터를 날짜별로 그룹화
        decile_factors = factors_fin_csn_df_grouped.apply(lambda x: x.apply(lambda col: pd.qcut(col, q=10, labels=False, duplicates='drop'))) # 그룹화된 데이터의 각 열에 대해 10분위수 계산
        decile_factors.fillna(0.0, inplace=True) # NaN 값을 0.0으로 채움
        decile_factors.columns = [x+"_decile" for x in factor_fin_csn_df.columns] # 10분위 열 이름 변경하여 데실임을 표시
        factors_fin_csn_df_combined = pd.concat([factor_fin_csn_df, decile_factors], axis=1) # 원래의 금융 컨센서스 팩터와 10분위를 결합

        factors_combined = pd.concat([tech_factors, factors_fin_csn_df_combined], axis=1) # 기술적 팩터와 결합된 금융 컨센서스 팩터를 결합
        factors_combined.unstack('ticker').sort_index().fillna(method='ffill').stack('ticker').swaplevel().sort_index() # 결합된 팩터를 재구성하고 정렬하며, 누락된 값을 앞선 값으로 채움
        factors_combined.fillna(0.0, inplace=True) # 남아있는 NaN 값을 0.0으로 채움

        factors_combined.sort_index(inplace=True) # 결합된 팩터를 인덱스별로 정렬

        factors_combined.to_feather(f'{self.save_dir}/FACTORS_FINAL.feather') # 결합된 팩터를 파일에 저장
        
        print("FINALLY!! Completed calculating FACTORS!")

        return factors_combined # 결합된 팩터를 반환


def main():
    import time
    # 팩터 생성 초기 매개변수 정의
    price_momentum = False
    price_bm = False
    price_sector = False
    # 지정된 매개변수로 GenerateFactors 클래스의 인스턴스 생성
    factor_generator = GenerateFactors(price_momentum=price_momentum, price_bm=price_bm, price_sector=price_sector)

    start = time.time()
    factor_df = factor_generator.get_factors() # 팩터 생성
    end = time.time()
    print(f"{end - start:.5f} sec")

if __name__ == "__main__":
    main()















