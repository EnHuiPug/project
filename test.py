import pandas as pd
import numpy as np
from openpyxl import load_workbook

book = load_workbook('input.xlsx')
# excel_name = 'input.xlsx'
# c = np.array(pd.read_excel(excel_name, usecols='K:Y', skiprows=[0, 1], nrows=1))
#
# d = pd.DataFrame(pd.read_excel(excel_name, usecols='A,F,G,H,AA', names=['ID', 'SEN', 'PRE', 'DoB', 'Hours'], skiprows=[0, 1, 2, 3]))
# d = d.dropna(subset=['ID'])
# d.fillna(0, inplace=True)
# # d2 = d.sort_values(by="DoB", ascending=True, inplace=False)
# d3 = d.loc[21:39]
# d3=d3.sort_values(by="DoB", ascending=True, inplace=False)
# d4 = np.array(d3)
# d5 = np.arange(0.0, 1.0, 1/19).round(2)[::-1].reshape(-1, 1)
# d6=np.concatenate((d4, d5), 1)
a = np.array(pd.read_excel('input.xlsx', usecols='K:Y', skiprows=[0, 1, 2, 3]))
a[np.isnan(a)] = 0
np.sum(a,axis=0)
# writer = pd.ExcelWriter('input.xlsx', engine = 'openpyxl')
# writer.book = book
# writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
# x.to_excel(writer, startrow=26, startcol=10, header=False, index=False)
# writer.save()
