import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--descriptor_files', nargs="+", help="mdpout_descriptors.txt files of experiment trajectories (can be more than 1)",default=["mdpout_descriptors.txt"])
    parser.add_argument('-r','--descriptor_ref', nargs="?", help="mdpout_descriptors.txt files of reference trajectory",default="ref_mdpout_descriptors.txt")
    parser.add_argument('-m','--min_vol', nargs="?", type=float, help="Filter out snapshots below the requested volume",default=0)
    args = parser.parse_args()
    return args

def get_data(reference,files,min_volume):
    data=pd.read_csv(reference,sep=" ")
    data=data[data.pock_volume>min_volume]
    name=reference.split("/")[-1].split(".")[0]
    data["run"]=[name for i in range(len(data))]
    data["reference"]=["Yes" for i in range(len(data))]
    for file in files:
        data2=pd.read_csv(file,sep=" ")
        data2=data2[data2.pock_volume>min_volume]
        name=file.split("/")[-1].split(".")[0]
        data2["run"]=[name for i in range(len(data2))]
        data2["reference"]=["No" for i in range(len(data2))]
        data2_eq=data.copy()
        data2_eq["run"]=[name for i in range(len(data2_eq))]
        data=pd.concat([data,data2,data2_eq],ignore_index=True)
    return data

if __name__=="__main__":
   args=parse()
   data=pd.read_csv(args.descriptor_ref,sep=" ")
   name=args.descriptor_ref.split("/")[-1].split(".")[0]
   data=get_data(args.descriptor_ref,args.descriptor_files,args.min_vol)
   sns.violinplot(data=data, x="run", y="pock_volume", hue="reference",
               split=True, inner="quart", fill=False,
               palette={"Yes": "g", "No": ".35"})
   plt.savefig("violinplot.png")
   print(data)