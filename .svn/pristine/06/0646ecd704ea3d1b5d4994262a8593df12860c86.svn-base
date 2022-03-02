import pandas as pd
import numpy as np
import datetime
from dbtools import proddatactrl


dp = proddatactrl.ProdDataController(sid='glprodsb')
etfs = dp.getAllETFs('ETF-WWxUS')
df_all = pd.DataFrame()
for i in etfs:
    df = pd.DataFrame.from_dict(i.to_dict(), orient='index')
    df_all = pd.concat([df_all, df], axis=1)
df_all = df_all.T
df_all=df_all.set_index('id')

etfs2 = dp.getAllETFs('ETF-EM')
df_all2 = pd.DataFrame()
for i in etfs2:
    df2 = pd.DataFrame.from_dict(i.to_dict(), orient='index')
    df_all2 = pd.concat([df_all2, df2], axis=1)
df_all2 = df_all2.T
df_all2=df_all2.set_index('id')

common = set(df_all.index).difference(df_all2.index)
df_all = df_all.loc[common, :]
df_all.to_csv('etf_dm.csv')

dp = proddatactrl.ProdDataController(sid='glprodsb')
etfs = dp.getAllETFs('ETF-WW')
df_all = pd.DataFrame()
for i in etfs:
    df = pd.DataFrame.from_dict(i.to_dict(), orient='index')
    df_all = pd.concat([df_all, df], axis=1)
df_all = df_all.T
df_all=df_all.set_index('id')
df_all.to_csv('etf_ww.csv')
