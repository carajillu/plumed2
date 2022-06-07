import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--ccmin', type=float, nargs="+",help="Parameter CCmin",default=0.30)
    parser.add_argument('-a','--ccmax', type=float, nargs="+",help="Parameter CCmax",default=0.45)
    parser.add_argument('-x','--deltacc', type=float, nargs="+",help="Parameter deltaCC",default=0.15)
    parser.add_argument('-c','--dmin', type=float, nargs="+",help="Parameter DMIN",default=8)
    parser.add_argument('-d','--deltad', type=float, nargs="+",help="Parameter deltaD",default=4)

    args = parser.parse_args()
    return args

def Soff(m,k):
    S_off=0;
    if (m<=0):
        S_off=k
    elif ((m>0) and (m<=1)):
        S_off=k*((3*(m-1)**4)-2*((m-1)**6))
    else:
        S_off=0
    return S_off

def Son(m,k):
    S_on=0;
    if (m<=0):
        S_on=0
    elif ((m>0) and (m<=1)):
        S_on=k*((3*(m)**4)-2*((m)**6))
    else:
        S_on=1
    return S_on

if __name__=="__main__":
    
    args=parse()

    CC=[]
    H=[]
    H_on=[]
    H_off=[]
    D=[]
    for i in np.arange(-0.05,args.ccmax+args.deltacc,0.001):
        mcc=(i-0)/args.ccmin
        CC.append(Son(mcc,1))
        mon_h=(i-args.ccmin)/args.deltacc
        moff_h=(i-args.ccmax)/args.deltacc
        H_on.append(Son(mon_h,1))
        H_off.append(Soff(moff_h,1))
        H.append(Son(mon_h,1)*Soff(moff_h,1))
    
    z=pd.DataFrame()
    z["r"]=np.arange(-0.05,args.ccmax+args.deltacc,0.001)
    z["CC"]=CC
    z["H_on"]=H_on
    z["H_off"]=H_off
    z["H"]=H
    ax=z.plot("r","H_on")
    z.plot("r","H_off",ax=ax)

    plt.show()
    print(z)



