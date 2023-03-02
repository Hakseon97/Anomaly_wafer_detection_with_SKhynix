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
        if not line[1].lstrip('-').isnumeric():
            dict_klarf[line[1]] = ' '.join(line[2:])
    fname.append(fp.split('/')[1])
    temp = pd.DataFrame(dict_klarf, index=[i])
    temp_df = pd.concat([temp_df, temp])

temp_df.columns = temp_df.columns.str.lower()
temp_df['filename'] = fname
cols = ['filename']
cols.extend(temp_df.columns[:-1])

temp_df = temp_df[cols]

# 컬럼 미리 선정.
selectCols = [
    "filename",
    "filetimestamp",
    "inspectionstationid",
    "lotid",
    "deviceid",
    "setupid",
    "stepid",
    "tifffilename",
    "waferid",
    "slot"
    ]
# 특정 columns만 추출
temp_df = temp_df[selectCols]

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

# temp_df의 FileTimestamp를 datetime형으로 변환
# 현재 %m-%d-%y %H:%M:%S 형태로 되어있음. 이를 %Y-%m-%d %H:%M:%S

temp_df.filetimestamp = temp_df.filetimestamp.apply(lambda x: datetime.strptime(x, "%m-%d-%y %H:%M:%S"))

# A3D01, A3D02로 요약할 수 있음.
temp_df.inspectionstationid = temp_df.inspectionstationid.apply(lambda x: x.strip('"').split('" "')[-1])

# "LotID", "DeviceID", "StepID", "WaferID" columns에 대해 " " 제거
temp_df[["setupid", "lotid", "deviceid", "stepid", "waferid"]] = \
    temp_df[["setupid", "lotid", "deviceid", "stepid", "waferid"]].applymap(lambda x: x.replace('"',''))
    
# WaferID column과 Slot column이 동일한지 확인.
# 동일하면 Slot column 제거
# 동일하지 않다면, Fab에 문제가 생긴것..

temp_df = eliminate(temp_df, "waferid", "slot")

# SetupID column에서 timestamp가 FileTimestamp와 일치하는지 확인.
# 완벽히 일치한다면 해당 timestamp만 제거.
if temp_df[temp_df.setupid.apply(lambda x: datetime.strptime(' '.join(x.split(" ")[1:]).strip(' '), "%m-%d-%y %H:%M:%S"))
        != temp_df.filetimestamp].size == 0:
    temp_df.setupid = temp_df.setupid.apply(lambda x: x.split(" ")[0])

# 수정된 SetupID column에서 Metrology-Type이 StepID와 일치하는지 확인.
# 완벽히 일치한다면 해당 column 제거.
temp_df = eliminate(temp_df, 'stepid', 'setupid')

# StepID의 column 통일
temp_df.stepid = temp_df.stepid.apply(lambda x: x.upper())

# TiffFilename column이 FileName column과 일치하면 제거.
temp_df[["filename", "tifffilename"]] = temp_df[["filename", "tifffilename"]].applymap(lambda x: x.split('.')[0])
temp_df = eliminate(temp_df, "filename", "tifffilename")

# WaferID에 LotID 정보 추가
temp_df.waferid = temp_df.lotid.apply(lambda x: re.split('\d+', x)[0]+re.split('\D+', x)[1]) + '-' + temp_df.waferid

# columns 이름 변경
# FileTimestamp -> Timestamp
# InspectionStationID -> MachineID
temp_df.rename(columns={'filetimestamp': 'timestamp',
                        'inspectionstationid': 'machineid'}, inplace=True)

# columns 순서 변경
newcols = ['filename', 'lotid', 'waferid', 'timestamp', 'machineid', 'stepid', 'deviceid']
temp_df = temp_df[newcols]

# timestamp 순으로 정렬
temp_df = temp_df.sort_values('timestamp').reset_index(drop=True)

temp_df.to_csv('./temp_klarf_info.csv', index=False)
################################################################################################
## Defect 좌표 기준 통일
readlines = readfile(file_path[0])
cnt = 0
for line in readlines:
    if line[1] == 'SampleDieMap':
        firstRow = int(line[0])
        cnt += 1
    elif line[1] == 'SampleTestPlan' and cnt == 1:
        lastRow = int(line[0]) - 3
    elif line[1] == 'SampleTestPlan' and cnt == 0:
        firstRow = int(line[0])
    elif line[1] == 'AreaPerTest' and cnt == 0:
        lastRow = int(line[0]) - 2

wafer_x_coordinate = np.array([int(line[1]) for line in readlines[firstRow:lastRow+1]])
wafer_y_coordinate = np.array([int(line[2]) for line in readlines[firstRow:lastRow+1]])

calib_x = 1-min(wafer_x_coordinate)
calib_y = 1-min(wafer_y_coordinate)
# 1-min(wafer_coordinate)의 보정이 필요.
wafer_x_coordinate += np.array([calib_x]*len(wafer_x_coordinate))
wafer_y_coordinate += np.array([calib_y]*len(wafer_y_coordinate))

x_min, x_max = min(wafer_x_coordinate), max(wafer_x_coordinate)
y_min, y_max = min(wafer_y_coordinate), max(wafer_y_coordinate)

base_wafer = np.zeros([x_max+2, y_max+2])
for x,y in zip(wafer_x_coordinate, wafer_y_coordinate):
    base_wafer[x,y] = 1

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
defect_df[["XINDEX", "YINDEX"]] = defect_df[["XINDEX", "YINDEX"]].astype('int')
defect_df[["XINDEX", "YINDEX"]] += [calib_x,calib_y]
print('데이터 전처리 완료')