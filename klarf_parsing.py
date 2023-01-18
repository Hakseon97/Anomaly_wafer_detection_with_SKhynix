import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import os, sys, glob
from tqdm import tqdm
from datetime import datetime
import re

file_path = glob.glob("klarfsample/*.001")

def readfile(file_path):
    readlines = []
    with open(file_path, 'r') as f:
        i = 1
        while True:
            line = f.readline()
            if not line:
                break
            line = str(i) + ' ' + line
            readlines.append(line.strip('\n').strip(';').split(' '))
            i += 1    
    return readlines

#######################################################################################3
# 설비정보 parsing
temp_df = pd.DataFrame()
fname = []
for i, fp in enumerate(file_path):
    dict_klarf = {}
    for line in readfile(fp):
        if not line[1].isnumeric():
            dict_klarf[line[1]] = ' '.join(line[2:])
    fname.append(fp.split('/')[1])
    temp = pd.DataFrame(dict_klarf, index=[i])
    temp_df = pd.concat([temp_df, temp])


temp_df['FileName'] = fname
cols = ['FileName']
cols.extend(temp_df.columns[:-1])

temp_df = temp_df[cols]

# unique값이 1인 columns 정리
singleCols = []
for col in temp_df.columns:
    if len(temp_df[col].unique()) == 1:
        singleCols.append(col)
    
# 위에서 정리한 columns에 이미 dataframe으로 만든 DefectList, SummaryList columns 추가
singleCols.extend(["DefectList", "SummaryList"])

temp_df.drop(columns=singleCols, inplace=True)

# column 비교 후 제거 함수
# 보존할 column = col1
# 제거할 column = col2
def eliminate(df, col1, col2):
    '''
    col1: 남길 컬럼
    col2: 제거할 컬럼
    '''
    if df[df[col1] != df[col2]].size == 0:
        df.drop(columns=col2, inplace=True)
    return df

# FileTimestamp column과 ResultTimestamp column이 동일하면 해당 컬럼 삭제
temp_df = eliminate(temp_df, "FileTimestamp", "ResultTimestamp")

# temp_df의 FileTimestamp를 datetime형으로 변환
# 현재 %m-%d-%y %H:%M:%S 형태로 되어있음. 이를 %Y-%m-%d %H:%M:%S

temp_df.FileTimestamp = temp_df.FileTimestamp.apply(lambda x: datetime.strptime(x, "%m-%d-%y %H:%M:%S"))

# A3D01, A3D02로 요약할 수 있음.
temp_df.InspectionStationID = temp_df.InspectionStationID.apply(lambda x: x.strip('"').split('" "')[-1])

# "LotID", "DeviceID", "StepID", "WaferID" columns에 대해 " " 제거
temp_df[["LotID", "DeviceID", "StepID", "WaferID"]] = \
    temp_df[["LotID", "DeviceID", "StepID", "WaferID"]].applymap(lambda x: x.strip(' "'))
    
# WaferID column과 Slot column이 동일한지 확인.
# 동일하면 Slot column 제거
# 동일하지 않다면, Fab에 문제가 생긴것..

temp_df = eliminate(temp_df, "WaferID", "Slot")

# SetupID column에서 timestamp가 FileTimestamp와 일치하는지 확인.
# 완벽히 일치한다면 해당 timestamp만 제거.
if temp_df[temp_df.SetupID.apply(lambda x: datetime.strptime(x.split('"')[2].strip(' '), "%m-%d-%y %H:%M:%S"))
        != temp_df.FileTimestamp].size == 0:
    temp_df.SetupID = temp_df.SetupID.apply(lambda x: x.split('"')[1])

# 수정된 SetupID column에서 Metrology-Type이 StepID와 일치하는지 확인.
# 완벽히 일치한다면 해당 column 제거.
temp_df = eliminate(temp_df, 'StepID', 'SetupID')

# StepID의 column 통일
temp_df.StepID = temp_df.StepID.apply(lambda x: x.split('-')[0]
                                      + '-T' + x.split('-')[1][1:]
                                      + '-' + x.split('-')[2])

# TiffFilename column이 FileName column과 일치하면 제거.
temp_df[["FileName", "TiffFilename"]] = temp_df[["FileName", "TiffFilename"]].applymap(lambda x: x.split('.')[0])
temp_df = eliminate(temp_df, "FileName", "TiffFilename")

# WaferID에 LotID 정보 추가
temp_df.WaferID = temp_df.LotID.apply(lambda x: re.split('\d+', x)[0]+re.split('\D+', x)[1]) + '-' + temp_df.WaferID

# columns 이름 변경
# FileTimestamp -> Timestamp
# InspectionStationID -> MachineID
temp_df.rename(columns={'FileTimestamp': 'Timestamp',
                        'InspectionStationID': 'MachineID'}, inplace=True)

# columns 순서 변경
newcols = ['FileName', 'LotID', 'WaferID', 'Timestamp', 'MachineID', 'StepID', 'DeviceID']
temp_df = temp_df[newcols]

# timestamp 순으로 정렬
temp_df = temp_df.sort_values('Timestamp').reset_index(drop=True)

temp_df.to_csv('./temp_klarf_info.csv', index=False)
################################################################################################
# Defect 정보 parsing
defectList = []
for fp in file_path:
    readlines = readfile(fp)
    for line in readlines:
        if line[1] == 'DefectList':
            defectRow = int(line[0])
        elif line[1] == 'SummarySpec':
            defectRowEND = int(line[0]) - 1
        else:
            continue
        
    defectCols = ['FILE']
    defectCols.extend(readlines[defectRow-2][3:])
    
    if readlines[defectRow][1].isnumeric():
        for line in readlines[defectRow:defectRowEND]:
            if len(line) == 18:
                temp = [fp.split('/')[1]]
                temp.extend(line[1:])
                defectList.append(temp)
    else:
        temp = [fp.split('/')[1]]
        temp.extend(np.zeros(17).tolist())
        defectList.append(temp)

defect_df = pd.DataFrame(data=defectList, columns=defectCols)

# Summary Spec.을 DataFrame으로 변환

summarySpec = []
for fp in file_path:
    readlines = readfile(fp)
    for line in readlines:
        if line[1] == 'SummarySpec':
            summaryRow = int(line[0])
            
    summaryCols = ['FILE']
    summaryCols.extend(readlines[summaryRow-1][3:])
    if readlines[summaryRow+1][1].isnumeric():
        temp = [fp.split('/')[1]]
        temp.extend(readlines[summaryRow+1][1:])
        summarySpec.append(temp)
    else:
        temp = [fp.split('/')[1]]
        temp.extend(np.zeros(5).tolist())
        summarySpec.append(temp)
summary_df = pd.DataFrame(data=summarySpec, columns=summaryCols, index=np.arange(1000))

# defect_df와 summary_df merge
defect_df = pd.merge(summary_df, defect_df, how='inner', left_on='FILE', right_on='FILE')

defect_df.to_csv("./temp_defect_info.csv", index=False)
print('데이터 추출 완료: temp_klarf_info.csv | temp_defect_info.csv')