import mdtraj 
import math
import random
import numpy as np
import pandas as pd
import sys
import scipy
from scipy import stats
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', type=str,nargs="?",  help="Protein structure in gro or pdb format",default="system.pdb")
    parser.add_argument('-c','--ccmax', type=float, nargs="+",help="Parameter CCmax to calculate correlation for",default=0.45)
    parser.add_argument('-d','--deltacc', type=float, nargs="+", help="Parameter deltaCC to calculate correlation for",default=0.05)
    parser.add_argument('-o','--output', type=str, nargs="?", help="Output CSV file",default="correlation.csv")

    args = parser.parse_args()
    return args

def Soff(m,k):
    S_off=0;
    if (m<=0):
        S_off=k
    elif ((m>0) and (m<1)):
        S_off=k*((3*(m-1)**4)-2*((m-1)**6))
    else: 
        S_off=0
    return S_off

def gen_points(system):
    crd=system.xyz[0]
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

def gen_mind_r_lst(points,system):
    crd=system.xyz[0]
    mind_r_lst=[]
    for point in points:
        mind_r_lst.append(mind_r(point,crd))
    print("mind_r_lst has ", len(mind_r_lst), " elements")
    return mind_r_lst

def gen_mind_c_lst(points,system,k,v0,deltaV):
    crd=system.xyz[0]
    mind_c_lst=[]
    for point in points:
        mind_c_lst.append(mind_c(point,crd,k,v0,deltaV))
    print("mind_c_lst has ", len(mind_c_lst), " elements")
    return mind_c_lst

def get_correlation_uncorrected(mind_r_lst,mind_c_lst):
    mind_r_noinf=[]
    mind_c_noinf=[]
    for i in range(0,len(mind_c_lst)):
        if mind_c_lst[i]==math.inf:
            continue
        mind_r_noinf.append(mind_r_lst[i])
        mind_c_noinf.append(mind_c_lst[i])

    print ("r_noinf and mind_noinf have ", len(mind_r_noinf), " and ", len(mind_c_noinf), " elements, respectively")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mind_c_noinf,mind_r_noinf)
    print ("mind (Uncorrected)")
    print ("Slope = ", slope)
    print ("Intercept = ", intercept)
    print ("r2 = ", r_value**2)
    print ("std_err = ", std_err)
    print ("p_value = ", p_value)

    return mind_r_noinf,mind_c_noinf,slope, intercept, r_value, p_value, std_err

def get_correlation_corrected(mind_r_noinf_lst,mind_c_noinf_lst,slope, intercept):
    mind_corr_lst=[]
    for element in mind_c_noinf_lst:
        mind_corr_lst.append(slope*element+intercept)
    mind_corr_lst=np.array(mind_corr_lst)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mind_corr_lst,mind_r_noinf_lst)
    print ("mind (corrected)")
    print ("Slope = ", slope)
    print ("Intercept = ", intercept)
    print ("r2 = ", r_value**2)
    print ("std_err = ", std_err)
    print ("p_value = ", p_value)

    return mind_corr_lst

if __name__=="__main__":

    args=parse()

    system=mdtraj.load(args.input)
    k=1
    ccmax=args.ccmax
    deltacc=args.deltacc

    points=gen_points(system)
    mind_r_lst=gen_mind_r_lst(points,system)

    info_frame=[]

    for CCmax in ccmax:
        for deltaCC in deltacc:
            mind_c_lst=gen_mind_c_lst(points,system,k,CCmax,deltaCC)
            mind_r_noinf_lst,mind_c_noinf_lst,slope, intercept, r_value, p_value, std_err=get_correlation_uncorrected(mind_r_lst,mind_c_lst)
            mind_corr_lst=get_correlation_corrected(mind_r_noinf_lst,mind_c_noinf_lst,slope, intercept)
            info_frame.append([CCmax,deltaCC,slope,intercept,r_value**2])
    
    mind_all=np.array(info_frame)
    names=["CCmax","deltaCC","Slope","Intercept","r2"]
    z=pd.DataFrame(data=mind_all,columns=names)
    z.to_csv("mind.csv",sep=" ", index=False)




    

