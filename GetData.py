#!usr/bin/env python
#-*- coding:utf-8 _*-  
""" 
@author:ZhangHui
@file: code1.py 
@time: 2017/12/04
from GetData import getData
filename = 'Test.txt'
data = getData(filename)
###
([[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], 780.0)
###
"""
import pandas as pd
import numpy as np
def pdReader(filename):
    return pd.read_csv(filename)

def getOneHots(df):
    #返回shape --> (1,11,10)
    Data = []
    PhoneNum = df[0]  #取第一列phonenum
    for num in str(PhoneNum):
        if num:
            # num --> 18573197191
            Nn = []
            for n in str(num):
                # n --> 1,8,7,7
                Nn.append(getOneHot(n,10))
            Data.append(Nn[0])
    # df.loc[:-1] = np.array(Data)
    return (Data , float(df[1]))

def getOneHot(n, Max):
    n = int(n)
    arg = [0.0] * Max
    arg[n] = 1.0
    # 接受n，输出[0,0,0,0,0,1,0]长度为Max
    return arg


def getRandLine(filename, lines):
    df = pdReader(filename)
    return df.sample(lines)


def getData(filename, Lines):
    x_Data = [];y_Data = []
    if not Lines:print('行数不能为0')
    else:
        df = getRandLine(filename, Lines)
        for line in df.values:
            temp = getOneHots(line)
            x_Data.append(temp[0])
            y_Data.append(temp[1])
    return (x_Data,y_Data)

def strs(n):
    result = []
    for i in str(n):
        result.append(int(i))
    return np.array(result)

def GetData(filename,lines):
    import pandas as pd
    pdReader = pd.read_csv(filename).sample(lines)
    #([18573197191],[1888]) (N , 11)
    return (np.array(list(map(strs,pdReader.values[::,0])))
            ,pdReader.values[::,1])