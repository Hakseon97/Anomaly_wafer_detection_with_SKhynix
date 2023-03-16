## Validation Process

# 1. 일정한 시간(@5분)마다 py파일 실행하여 폴더 내의 모든 KLARF파일을 Dataframe으로 변환.
# 2. 그 Dataframe에서 Fast-Fourier 변환을 거친 뒤, Amplitude 값 추가.
# 3. 미리 학습된 모델로 해당 KLARF파일을 검증하여 연속적인 Defects이 나왔는지 검출.
# 4. 만약 검출되면, 해당 INDEX를 역추적하여 어떤 장비가 이상있는지 확인.
#  * 미리 저장되어있는 Dataframe의 시간을 최대 3일로 설정. 

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QLineEdit, QPushButton
from PyQt5.QtCore import *
from PyQt5 import uic
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from scipy import interpolate
import matplotlib.pyplot as plt
import glob, os
from datetime import datetime
import time
import re
import joblib

import warnings
warnings.filterwarnings('ignore')

## User Interface
class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        self.initUI()
        
        self.show()
        
    def initUI(self):
        uic.loadUi("detecting_program.ui", self)
        
        self.inputDirectoryButton = self.findChild(QPushButton, "browseDir")
        self.dir = self.findChild(QLineEdit, "folderDir")
        
        self.inputFileButton = self.findChild(QPushButton, "modelDir")
        self.modelName = self.findChild(QLineEdit, "modelName")
        
        self.inputDirectoryButton.clicked.connect(self.dirBtn)
        self.inputFileButton.clicked.connect(self.modelBtn)
        
        self.runButton.clicked.connect(self.runBtn)
        self.timer = QTimer(self)
        self.batch_timer = QTimer(self)
        
    def dirBtn(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            global dir_path
            dir_path = str(folder)
            self.dir.setText(dir_path)
            
        
    def modelBtn(self):
        file, _ = QFileDialog.getOpenFileName(self, "Model Name", "", "Model Files (*.pkl or *.pt or *.pth or *.h5);;All Files (*)")
        if file:
            global model_path
            model_path = Path(file)
            self.modelName.setText(Path(file).name)
    
    def runBtn(self):
        
        if self.timer.isActive():
            self.progressTxt.setText("다시 Run 버튼을 눌러 실행해주세요.")
            self.runButton.setText("Run")
            self.timer.stop()
            self.batch_timer.stop()
        else:
            self.detecting()
            self.batch_timer.start()
            self.progressTxt.setText("분석중...")
            self.progressTxt.setAlignment(Qt.AlignCenter)
            self.runButton.setText("Pause")
            
            #5분마다 실행
            self.timer.start(5*60*1000)
            self.batch_timer.start(1000)
            self.batch_timer.timeout.connect(self.remainTime)
            self.timer.timeout.connect(self.detecting)
    
    def remainTime(self):
        self.ms = self.timer.remainingTime() # 단위: ms
        self.rt = datetime.fromtimestamp(self.ms/1000).strftime("%M:%S")
        self.progressTxt.setText(f"분석중... {self.rt}")
        self.progressTxt.setAlignment(Qt.AlignCenter)
    
    def detecting(self):
        ## 1. 일정한 시간(@5분)마다 py파일 실행하여 폴더 내의 모든 KLARF파일을 Dataframe으로 변환.
        #- 5분마다 py. file 실행하여 폴더 내 모든 klarf 파일 변환
        file_path = os.path.join(dir_path, '*.001')
        file_list = glob.glob(file_path)
        start = time.time()
        
        def readfile(file_list):
            readlines = []
            with open(file_list, 'r') as f:
                i = 1
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line[0] == ' ':
                        line = line.strip(' ')
                    line = str(i) + ' ' + line
                    readlines.append(line.strip('\n').strip(';').split(' '))
                    i += 1    
            return readlines

        # 설비정보 parsing
        temp_df = pd.DataFrame()
        fname = []
        for i, file in enumerate(file_list):
            dict_klarf = {}
            for line in readfile(file):
                if not line[1].lstrip('-').isnumeric():
                    dict_klarf[line[1]] = ' '.join(line[2:])
            fname.append(os.path.splitext(os.path.basename(file))[0])
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
            "inspectionstationid",
            "resulttimestamp",
            "lotid",
            "deviceid",
            "setupid",
            "stepid"
            ]

        # 미리 선정한 컬럼만 추출
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
        temp_df.resulttimestamp = temp_df.resulttimestamp.apply(lambda x: datetime.strptime(x, "%m-%d-%y %H:%M:%S"))

        temp_df.inspectionstationid = temp_df.inspectionstationid.apply(lambda x: ' '.join(x.strip('"').split('" "')[1:])\
            if x.strip('"').split('" "')[1] != x.strip('"').split('" "')[2] \
                else x.strip('"').split('" "')[-1])
        # "LotID", "DeviceID", "StepID", "WaferID" columns에 대해 " " 제거
        temp_df[["setupid", "lotid", "deviceid", "stepid"]] = \
            temp_df[["setupid", "lotid", "deviceid", "stepid"]].applymap(lambda x: x.replace('"',''))


        # SetupID column에서 timestamp가 FileTimestamp와 일치하는지 확인.
        # 완벽히 일치한다면 해당 timestamp만 제거.
        if temp_df[temp_df.setupid.apply(lambda x: datetime.strptime(' '.join(x.split(" ")[1:]).strip(' '), "%m-%d-%y %H:%M:%S"))
                   != temp_df.resulttimestamp].size == 0:
            temp_df.setupid = temp_df.setupid.apply(lambda x: x.split(" ")[0])

        # 수정된 SetupID column에서 Metrology-Type이 StepID와 일치하는지 확인.
        # 완벽히 일치한다면 해당 column 제거.
        temp_df = eliminate(temp_df, 'stepid', 'setupid')

        # StepID의 column 통일
        temp_df.stepid = temp_df.stepid.apply(lambda x: x.upper())

        # columns 이름 변경
        # FileTimestamp -> Timestamp
        # InspectionStationID -> MachineID
        temp_df.rename(columns={'resulttimestamp': 'timestamp',
                                'inspectionstationid': 'machineid'}, inplace=True)

        # columns 순서 변경
        newcols = ['filename', 'lotid', 'timestamp', 'machineid', 'stepid', 'deviceid']
        temp_df = temp_df[newcols]

        # timestamp 순으로 정렬
        temp_df = temp_df.sort_values('timestamp').reset_index(drop=True)


        ## Defect 좌표 기준 통일
        readlines = readfile(file_list[0])
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

        x_max = max(wafer_x_coordinate)
        y_max = max(wafer_y_coordinate)

        base_wafer = np.zeros([x_max+2, y_max+2])
        for x,y in zip(wafer_x_coordinate, wafer_y_coordinate):
            base_wafer[x,y] = 1

        ## Defect 정보 parsing
        # DefectList를 dataframe으로 변환.
        defectList = []
        for file in file_list:
            cnt = 0
            filename = os.path.splitext(os.path.basename(file))[0]
            for line in readfile(file):
                if line[1] == 'DefectList':
                    defectRow = int(line[0])
                    cnt += 1
                    
                elif cnt > 0 and line[1].isnumeric():
                    tempList = [filename]
                    tempList.append(line[1])
                    tempList.extend(line[4:6])
                    defectList.append(tempList)
                    
                elif line[1] == 'SummarySpec':
                    break
                
                else:
                    continue
        
            if not readfile(file)[defectRow][1].isnumeric():
                tempList = [filename]
                tempList.extend(np.zeros(3).tolist())
                defectList.append(tempList)
                
        defectCols = ['FILE', 'DEFECTID', 'XINDEX', 'YINDEX']
        defect_df = pd.DataFrame(defectList, columns=defectCols)
        defect_df[["XINDEX", "YINDEX"]] = defect_df[["XINDEX", "YINDEX"]].applymap(lambda x: 0 if x==None else int(x))
        
        nodefect_df = defect_df[defect_df["DEFECTID"] == 0]
        defect_df = defect_df[defect_df["DEFECTID"] != 0]
        defect_df[["XINDEX","YINDEX"]] += [calib_x,calib_y]
        defect_df = pd.concat([nodefect_df,defect_df]).sort_index()
        
        # defect_df에 defects 정보 추가
        defect_df["map"] = defect_df[["XINDEX", "YINDEX"]].apply(list, axis=1)
        
        def appendFn(*listset):
            lst = []
            for list_ in listset:
                lst.append(list_)
            return lst

        defect_temp = defect_df.groupby(['FILE'], group_keys=False)["map"].apply(appendFn).reset_index()
        defect_temp.map = defect_temp.map.apply(lambda x: np.array(x).squeeze(0))

        base_df = pd.merge(temp_df, defect_temp, how='inner', left_on='filename', right_on='FILE').drop(columns=["FILE", "filename"])
        end = time.time()
        test_time = end - start
        print(f"전처리 시간: {int(test_time)}초")

        ## 2. Dataframe에서 Fast-Fourier 변환을 거친 뒤, Amplitude 값 추가.
        # hyperparameter
        class cfg:
            seed = 1234
            n_window = 3 # 한 batch에 확인할 wafer 수 -> 10
            origin = origin = [(base_wafer.shape[0]-1)/2, (base_wafer.shape[1]-1)/2] # wafer의 원점 정의
            
        # batch dataset visualization v2
        def dist(origin, defects):
            if defects[0] == [0,0]:
                return [-1]
            distance = [-1]
            for i in range(len(defects)):
                dist = 0
                for j in range(len(defects[i])):
                    dist += (origin[j]-defects[i][j])**2
                dist **= 1/2
                distance.append(int(dist))
            return distance

        def theta(origin, defects):
            if defects[0] == [0,0]:
                return [-180]
            theta = [-180]
            for i in range(len(defects)):
                rc_x = defects[i][0] - origin[0]
                rc_y = defects[i][1] - origin[1]
                ang = int(np.rad2deg(np.arctan2(rc_y, rc_x)))
                theta.append(ang) # 가시성을 위해 rad -> deg
            return theta

        df = base_df.copy()
        df["distance"] = df.map.apply(lambda x:dist(cfg.origin, x))
        df["degree"] = df.map.apply(lambda x:theta(cfg.origin, x))

        
        def batch_graph(degree, distance):
            intp = interpolate.interp1d(degree, distance, kind='linear') # linear, cubic, nearest ...
            xnew = np.arange(min(degree), max(degree), 0.1)
            return intp, xnew

        # fft후, amplitude 값 컬럼 추가
        def amp_data(input):
            Y = np.fft.fft(input)
            amp = abs(Y) * (2/len(Y))
            return amp[0:20] # 20개까지만 추가.

        def add_amplitude(df):
            nRows = len(df) - cfg.n_window + 1
            for i in range(nRows):
                df_dist = df.distance[i:cfg.n_window + i].tolist()
                batch_dist = []
                for dist in df_dist:
                    try:
                        for d in dist:
                            batch_dist.append(d)
                    except:
                        pass
                    
                df_deg = df.degree[i:cfg.n_window + i].tolist()
                batch_deg = []
                for n, deg in enumerate(df_deg):
                    try:
                        for d in deg:
                            batch_deg.append(d + 360*n)
                    except:
                        pass
                
                intp, xnew = batch_graph(batch_deg, batch_dist)
                df.at[i+cfg.n_window-1,"amplitude"] = amp_data(intp(xnew)).astype('object')
            return df

        machine_type = df["machineid"].unique()
        step_type = df["stepid"].unique()
        device_type = df["deviceid"].unique()

        temp_df = pd.DataFrame()
        for m in machine_type:
            for s in step_type:
                for d in device_type:
                    temp = df[(df.machineid == m) & (df.stepid == s) & (df.deviceid == d)].reset_index()
                    if len(temp) == 0:
                        continue
                    temp = add_amplitude(temp)
                    temp_df = pd.concat([temp_df, temp], axis=0)
        
        temp_df = temp_df.sort_values(by='index').reset_index(drop=True)
        #temp_df = temp_df.set_index('index')

        amp_df = temp_df[["amplitude"]]
        amp_df = amp_df[amp_df.amplitude.notnull()]

        def make_amplitude_df(df):
            arr = np.array(df.amplitude.iloc[0]).reshape(1,-1)
            cols = np.array([i for i in range(arr.shape[1])])
            idx = df.index
            for i in range(1,len(df)):
                arr = np.append(arr, df.amplitude.iloc[i].reshape(1,-1), axis=0)
            df = pd.DataFrame(arr, columns=cols, index=idx)
            return df

        test_df = make_amplitude_df(amp_df).iloc[:,1:]
        
        # 정규화
        norm = Normalizer(norm='max')
        test_df = pd.DataFrame(norm.fit_transform(test_df), index=test_df.index)
        
        ## 3. 미리 학습된 모델로 해당 KLARF파일을 검증하여 연속적인 Defects이 나왔는지 검출.
        model = joblib.load(model_path)
        pred = pd.DataFrame(model.predict(test_df), index=test_df.index)

        ## 4. 만약 검출되면, 해당 INDEX를 역추적하여 어떤 장비가 이상있는지 확인.
        idx = pred[pred[0] == 1].index
        anomaly_df = df.iloc[idx,:]
        anomaly_df["ymd"] = anomaly_df.timestamp.apply(lambda x: datetime.strftime(x, "%Y-%m-%d"))

        ## 5. 결과 log를 txt파일로 저장. 
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(f'result_log.txt', 'a') as f:
            f.write("="*50)
            f.write("\n")
            f.write(f"실행시각: {now}\n")
            self.resultTxt.append(f'실행시각: {now}\n')
            for ymd in anomaly_df.ymd.unique():
                temp = anomaly_df[anomaly_df.ymd == ymd]
                
                f.write(f'------------- {ymd} -------------\n')
                f.write(f'검사장비 {temp.machineid.unique()}\n')
                f.write(f'검사방법 {temp.stepid.unique()}\n')
                f.write(f'제품공정 {temp.deviceid.unique()}\n')
                f.write(f'로트번호 {temp.lotid.unique()}\n')
                f.write("에 대한 확인이 필요합니다.\n\n")
                
                self.resultTxt.append(f'------------- {ymd} -------------')
                self.resultTxt.append(f'검사장비 {temp.machineid.unique()}')
                self.resultTxt.append(f'검사방법 {temp.stepid.unique()}')
                self.resultTxt.append(f'제품공정 {temp.deviceid.unique()}')
                self.resultTxt.append(f'로트번호 {temp.lotid.unique()}')
                self.resultTxt.append('에 대한 확인이 필요합니다.\n')
                
            f.write("\n")
        self.resultTxt.append('\n')
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()