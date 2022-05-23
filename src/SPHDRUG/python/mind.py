import mdtraj 
import math
import random
import numpy as np
import pandas as pd
import sys
import scipy
from scipy import stats

def Soff(m,k):
    S_off=0;
    if (m<=0):
        S_off=k
    elif ((m>0) and (m<1)):
        S_off=k*((3*(m-1)**4)-2*((m-1)**6))
    else: 
        S_off=0
    return S_off

def gen_points(crd):
    x=[9999999,-9999999]
    y=[9999999,-9999999]
    z=[9999999,-9999999]
    for atom in crd:
        if (atom[0])<x[0]:
            x[0]=atom[0]
        if (atom[0]>x[1]):
            x[1]=atom[0]

        if (atom[1])<y[0]:
            y[0]=atom[1]
        if (atom[1]>y[1]):
            y[1]=atom[1]

        if (atom[2])<z[0]:
            z[0]=atom[2]
        if (atom[2]>z[1]):
            z[1]=atom[2]

    points=[]
    for i in range(0,10000):
        x1=random.uniform(x[0],x[1])
        y2=random.uniform(y[0],y[1])
        z3=random.uniform(z[0],z[1])
        point=[x1,y2,z3]
        #print(point)
        points.append(point)
    return points
    

def mind_r(point,crd):
    mind_real=99999999999999999999999.
    
    for atom in crd:
        rx=(point[0]-atom[0])
        ry=(point[1]-atom[1])
        rz=(point[2]-atom[2])
        r=math.sqrt(rx**2+ry**2+rz**2)
        if (r<mind_real):
            mind_real=r

    return mind_real

def mind_c(point,crd,k,v0,deltaV):
    mind=0
    num=0
    den=0

    for atom in crd:
        rx=(point[0]-atom[0])
        ry=(point[1]-atom[1])
        rz=(point[2]-atom[2])
        r=math.sqrt(rx**2+ry**2+rz**2)
        m=(r-v0)/deltaV 
        num=num+Soff(m,k)*r
        den=den+Soff(m,k)
        if den==0:
            mind=math.inf
        else:
            mind=num/den

    return mind


if __name__=="__main__":
    system=mdtraj.load("test.pdb")
    crd=system.xyz[0]
    points=gen_points(crd)
    mind_r_lst=[]
    for point in points:
        mind_r_lst.append(mind_r(point,crd))

    k=1
    v0=0.45
    deltaV=0.05
    mind_c_lst=[]
    for point in points:
        mind_c_lst.append(mind_c(point,crd,k,v0,deltaV))
    print("mind_c_lst has ", len(mind_c_lst), " elements")
   
    r_noinf=[]
    mind_noinf=[]
    for i in range(0,len(mind_c_lst)):
        if mind_c_lst[i]==math.inf:
            continue
        r_noinf.append(mind_r_lst[i])
        mind_noinf.append(mind_c_lst[i])

    print ("r_noinf and mind_noinf have ", len(r_noinf), " and ", len(mind_noinf), " elements, respectively")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mind_noinf,r_noinf)
    print ("mind (Uncorrected)")
    print ("Slope = ", slope)
    print ("Intercept = ", intercept)
    print ("r2 = ", r_value**2)
    print ("std_err = ", std_err)
    print ("p_value = ", p_value)

    mind_corr_lst=[]
    for element in mind_noinf:
        mind_corr_lst.append(slope*element+intercept)
    mind_corr_lst=np.array(mind_corr_lst)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mind_corr_lst,r_noinf)
    print ("mind (corrected)")
    print ("Slope = ", slope)
    print ("Intercept = ", intercept)
    print ("r2 = ", r_value**2)
    print ("std_err = ", std_err)
    print ("p_value = ", p_value)

    
    mind_all=np.c_[r_noinf,mind_noinf,mind_corr_lst]

    names=["Real","mind","mind_corr"]

    z=pd.DataFrame(data=mind_all,columns=names)
    z.to_csv("mind.csv",sep=" ", index=False)




    

