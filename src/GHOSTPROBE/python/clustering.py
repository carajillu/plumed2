import pandas as pd
import numpy as np
import pymp
import warnings
import sys
warnings.simplefilter(action='ignore')

'''
INPUT:
r: numpy array with distances between points
rmax: distance threshold to calculate density
delta0: minimum value of delta to consider a point a cluster centre

OUTPUT:
rhodelta: pd data frame with the info it is defined with at the beginning
'''
def laio(r, rmax, delta_min, rho_min=None):
    rhodelta = pd.DataFrame()
    rhodelta["point"] = range(len(r))
    rhodelta["rho"] = [0] * len(r)
    rhodelta["nnhd"] = [-1] * len(r)
    rhodelta["delta"] = [0.] * len(r)
    rhodelta["cluster"] = [-1] * len(r)
    rhodelta["cluster_centre"] = [-1] * len(r)
    
    print("Assigning densities")
    for i in range(len(rhodelta)):
        rhodelta.at[i, "rho"] = len(np.where(r[i] <= rmax)[0])
    
    if rho_min is not None:
       rhodelta = rhodelta[rhodelta["rho"] >= rho_min]

    print("Calculating deltas")
    for i in rhodelta.index:
        hd = rhodelta.loc[rhodelta.rho > rhodelta.at[i, "rho"], "point"].values
        if len(hd) == 0:
            rhodelta.at[i, "delta"] = max(r[i])
            continue
        rhd = r[i][hd]
        rhodelta.at[i, "delta"] = np.min(rhd)
        minrhd_i = np.where(rhd == np.min(rhd))[0]
        if len(minrhd_i) == 1:
            minrhd_i = minrhd_i[0]
        else:
            rhos = rhodelta.loc[minrhd_i, "rho"].values
            minrhd_i = minrhd_i[np.argmax(rhos)]
        rhodelta.at[i, "nnhd"] = hd[minrhd_i]

    if delta_min is None:
        delta_min = np.percentile(rhodelta['delta'], 75)
        print(f"using delta0 = {delta_min}")

    cluster_centers = []
    for i in rhodelta.index:
        if rhodelta.at[i, "delta"] >= delta_min:
            cluster_centers.append(rhodelta.at[i, "point"])

    for i in cluster_centers:
        rhodelta.at[i, "cluster"] = cluster_centers.index(i)
        rhodelta.at[i, "cluster_centre"] = i

    rhodelta.sort_values("rho", ascending=False, inplace=True, ignore_index=True)
    for i in rhodelta.index:
        if rhodelta.at[i, "cluster"] != -1:
            continue
        nnhd = rhodelta.at[i, "nnhd"]
        nnhd_j = rhodelta[rhodelta.point == nnhd].index[0]
        rhodelta.at[i, "cluster"] = rhodelta.at[nnhd_j, "cluster"]
        rhodelta.at[i, "cluster_centre"] = rhodelta.at[nnhd_j, "cluster_centre"]

    rhodelta.sort_values("cluster", inplace=True, ignore_index=True)
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
        #print("Calculating distances for point {0} of {1}".format(i,len(xyz)))
        with pymp.Parallel() as p:
            for j in p.range(i+1,len(xyz)):
                r[i][j]=np.linalg.norm((xyz[i]-xyz[j]))
                r[j][i]=r[i][j]
    return r
