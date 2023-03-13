import pandas as pd
import numpy as np
import pymp
import warnings
warnings.simplefilter(action='ignore')

'''
INPUT:
r: numpy array with distances between points
rmax: distance threshold to calculate density
delta0: minimum value of delta to consider a point a cluster centre

OUTPUT:
rhodelta: pd data frame with the info it is defined with at the beginning
'''
def laio(r,rmax,delta_min):
    rhodelta=pd.DataFrame()
    rhodelta["point"]=range(0,len(r))
    rhodelta["rho"]=[0]*len(r)
    rhodelta["nnhd"]=[-1]*len(r)
    rhodelta["delta"]=[0.]*len(r)
    rhodelta["cluster"]=[-1]*len(r)
    rhodelta["cluster_centre"]=[-1]*len(r)
    
    print("Assigning densitites")
    for i in range(0,len(rhodelta)):
        rhodelta.rho[i]=len(np.where(r[i]<=rmax)[0])
   
    print("Calculating deltas")
    for i in range(0,len(rhodelta)):
        hd=rhodelta.point[rhodelta.rho>rhodelta.rho[i]].values
        #print(hd)
        if len(hd)==0:
            rhodelta.delta[i]=max(r[i])
            rhodelta.nnhd[i]=-1
            continue
        rhd=r[i][hd]
        rhodelta.delta[i]=np.min(rhd)
        #print(rhodelta.rho[i], rhodelta.delta[i])
        minrhd_i=np.where(rhd==np.min(rhd))[0][0]
        #print(minrhd_i)
        rhodelta.nnhd[i]=hd[minrhd_i]

    #choose cluster centers
    print(rhodelta.sort_values("delta",ascending=False)[0:50])
    
    if delta_min is None:
       z="start"
       while (type(z)!=float):
           z=input("Please indicate the minimum delta for a point to be considered a cluster centre\n")
           try: 
               z=float(z)
           except:
               print("Looks like you didn't give me a float. Try again\n")
    else:
        z=delta_min

    cluster_centers=[]
    for i in range(0,len(rhodelta)):
        if rhodelta.delta[i]>=z:
            cluster_centers.append(rhodelta.point[i])


    #Assign cluster ids to cluster centres
    for i in range(0,len(cluster_centers)):
        point_i=cluster_centers[i]
        rhodelta.cluster[point_i]=i
        rhodelta.cluster_centre[point_i]=point_i

    #Assign rest of points to clusters
    rhodelta.sort_values("rho",ascending=False,inplace=True,ignore_index=True)
    for i in range(0,len(rhodelta)):
        if rhodelta.cluster[i]!=-1:
            continue
        nnhd=rhodelta.nnhd[i]
        #print(np.where(rhodelta.point==nnhd))
        #return
        nnhd_j=np.where(rhodelta.point==nnhd)[0][0] #index of rhodelta.nnhd[i] in the sorted dataframe
        rhodelta.cluster[i]=rhodelta.cluster[nnhd_j]
        rhodelta.cluster_centre[i]=rhodelta.cluster_centre[nnhd_j]
        #print(nnhd,nnhd_j,rhodelta.cluster[nnhd_j])
    
    #pickle rhodelta
    print(rhodelta)
    rhodelta.sort_values("cluster",inplace=True,ignore_index=True)
    rhodelta.to_csv("rhodelta.csv",index=False,sep=" ")
    return rhodelta

    
'''
INPUT:
xyz: mxn np.array with the coordinates of the data points, where m is the number of points and n the number of dimensions

OUTPUT:
r: (mxm) np.array with the pairwise distances.
'''    
def calc_distance_matrix(xyz):
    r=pymp.shared.array((len(xyz),len(xyz)),dtype=np.float64)
    for i in range(0,len(xyz)):
        print("Calculating distances for point {0} of {1}".format(i,len(xyz)))
        with pymp.Parallel() as p:
            for j in p.range(i+1,len(xyz)):
                r[i][j]=np.linalg.norm((xyz[i]-xyz[j]))
                r[j][i]=r[i][j]
    return r
