
import os
import pandas as pd

path = os.environ.get('FC_SVNET_EXCEL_DIR', os.path.join('data', 'labeled_logs'))
files = os.listdir(path)

GR_min=[]
GR_max=[]
AC_min=[]
AC_max=[]
LLD_min=[]
LLD_max=[]
SP_min=[]
SP_max=[]
CAL_min=[]
CAL_max=[]
LLS_min=[]
LLS_max=[]

print(files)
def getMaxAndMin() :
    for file in files:
        path1 = os.path.join(path, file)
        df = pd.read_excel(path1)
        GR_max.append(df.loc[:, "GR"].max())
        AC_max.append(df.loc[:, "AC"].max())
        LLD_max.append(df.loc[:, "LLD"].max())
        SP_max.append(df.loc[:, "SP"].max())
        CAL_max.append(df.loc[:, "CAL"].max())
        LLS_max.append(df.loc[:, "LLS"].max())

        GR_min.append(df.loc[:, "GR"].min())
        AC_min.append(df.loc[:, "AC"].min())
        LLD_min.append(df.loc[:, "LLD"].min())
        SP_min.append(df.loc[:, "SP"].min())
        CAL_min.append(df.loc[:, "CAL"].min())
        LLS_min.append(df.loc[:, "LLS"].min())

getMaxAndMin()
print('GR_max: {}, GR_min: {}'.format(max(GR_max),min(GR_min)))
print('AC_max: {}, AC_min: {}'.format(max(AC_max),min(AC_min)))
print('LLD_max: {}, LLD_min: {}'.format(max(LLD_max),min(LLD_min)))
print('SP_max: {}, SP_min: {}'.format(max(SP_max),min(SP_min)))
print('CAL_max: {}, CAL_min: {}'.format(max(CAL_max),min(CAL_min)))
print('LLS_max: {}, LLS_min: {}'.format(max(LLS_max),min(LLS_min)))
