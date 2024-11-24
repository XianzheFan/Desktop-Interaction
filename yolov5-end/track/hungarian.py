import numpy as np
import cv2
import math
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageStat

def Brightness(img):
    r,g,b = np.mean(np.mean(img,axis = 0),axis = 0)
    bright = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068 *(b**2))
    # print("brightness:",bright)
    return round(bright,2)

def Hungarian_old(LOC, Loc, CLS, Cls):
    # different class: +99999
    LOC = np.array(LOC)
    Loc = np.array(Loc)
    Dis = []
    for j in range(len(LOC)):
        Row = []
        d = []
        for k in range(len(Loc)):
            dis = 0
            d = LOC[j]-Loc[k]
            for i in range(4):
                dis += abs(d[i])
            Row.append(dis)
        Dis.append(Row)
    Dis = np.array(Dis)
    # print(Dis)

    if len(LOC) > len(Loc):
        temp = 99999
        while Dis.shape[0] != Dis.shape[1]:
            Dis = np.insert(Dis,len(Loc),values = temp,axis=1)
            # print("Add col: ",Dis)
    elif len(LOC) < len(Loc):
        temp = 99999
        while Dis.shape[0] != Dis.shape[1]:
            Dis = np.insert(Dis,len(LOC),values=temp,axis=0)
            # print("Add row: ",Dis)
    # print(Dis)

    row_idx,col_idx = linear_sum_assignment(Dis)
    r = []
    c = []
    for i in range(len(row_idx)):
        r.append(float(row_idx[i]))
        c.append(float(col_idx[i]))
    
    if len(LOC) > len(Loc): # untracked devise
        for i in range(len(Loc),len(LOC)): # device not exist
            r[i] = -0.5
            # print("row_idx:",r)
        for i in range(len(LOC)): # link to device which not exist
            if c[i] >= min(len(LOC),len(Loc)):
                c[i] = -0.5
                # print("col_idx:",c)
    elif len(LOC) < len(Loc):
        for i in range(len(LOC),len(Loc)):
            c[i] = -0.1-c[i] # mark the new device
            # print("col_idx:",c)
        for i in range(len(Loc)):
            if r[i] >= min(len(LOC),len(Loc)):
                r[i] = -0.1-r[i]
                # print("row_idx:",r)
    return r, c

def Hungarian(LOC, Loc, CLS, Cls):
    # different class: +99999
    LOC = np.array(LOC)
    Loc = np.array(Loc)
    Dis = []
    for j in range(len(LOC)):
        Row = []
        d = []
        for k in range(len(Loc)):
            dis = 0
            d = LOC[j]-Loc[k]
            for i in range(4):
                dis += abs(d[i])
            if int(CLS[j]) != int(Cls[k]):
                dis += 9999
            Row.append(dis)
        Dis.append(Row)
    Dis = np.array(Dis)
    # print(Dis)

    if len(LOC) > len(Loc):
        temp = 99999
        while Dis.shape[0] != Dis.shape[1]:
            Dis = np.insert(Dis,len(Loc),values = temp,axis=1)
            # print("Add col: ",Dis)
    elif len(LOC) < len(Loc):
        temp = 99999
        while Dis.shape[0] != Dis.shape[1]:
            Dis = np.insert(Dis,len(LOC),values=temp,axis=0)
            # print("Add row: ",Dis)
    # print(Dis)

    row_idx,col_idx = linear_sum_assignment(Dis)
    r = []
    c = []
    for i in range(len(row_idx)):
        r.append(float(row_idx[i]))
        c.append(float(col_idx[i]))
    
    if len(LOC) > len(Loc): # untracked devise
        for i in range(len(Loc),len(LOC)): # device not exist
            r[i] = -0.5
            # print("row_idx:",r)
        for i in range(len(LOC)): # link to device which not exist
            if c[i] >= min(len(LOC),len(Loc)):
                c[i] = -0.5
                # print("col_idx:",c)
    elif len(LOC) < len(Loc):
        for i in range(len(LOC),len(Loc)):
            c[i] = -0.1-c[i] # mark the new device
            # print("col_idx:",c)
        for i in range(len(Loc)):
            if r[i] >= min(len(LOC),len(Loc)):
                r[i] = -0.1-r[i]
                # print("row_idx:",r)
    return r, c

def Match_loc(k,loc0,loc1): # 在已结束的bbox中搜索可能的对应项
    d = 0
    length = loc0[3]-loc0[1]+loc0[2]-loc0[0]
    for i in range(4):
        d += abs(loc0[i]-loc1[i])
    if d < length*k: # 相对于bbox长宽，顶点偏离的比例
        return True
    else:
        return False
    

Loc = np.array([[400,600,500,700],[0,0,4,1],
[100,100,200,200]])
LOC = np.array([[0,0,2,2],
[200,200,300,300]])
# col_idx = Hungarian(LOC,Loc)
# print(col_idx)