import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

colors={"holo":"blue","apo":"orange"}

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--descriptor_files', nargs="+", help="mdpout_descriptors.txt files of experiment trajectories (can be more than 1)",default=["mdpout_descriptors.txt"])
    parser.add_argument('-r','--descriptor_holo', nargs="?", help="mdpout_descriptors.txt files of reference holo trajectory",default="holo_mdpout_descriptors.txt")
    parser.add_argument('-a','--descriptor_apo', nargs="?", help="mdpout_descriptors.txt files of reference holo trajectory",default="apo_mdpout_descriptors.txt")
    parser.add_argument('-m','--min_vol', nargs="?", type=float, help="Filter out snapshots below the requested volume",default=0)
    parser.add_argument('--ref_vol', nargs="?", type=float, help="Volume of the pocket in the reference frame",default=0)
    args = parser.parse_args()
    return args

def load_data(file_path,start=0):
    data=pd.read_csv(file_path,sep=" ")[start:]
    name=file_path.split("/")[-1].split(".")[0]
    data["name"]=[name]*len(data)
    data["vol_roll"]=data["pock_volume"].rolling(window=100).mean()
    return data



if __name__=="__main__":
   args=parse()
   custom_palette = {'holo': 'blue', 'apo': 'orange'}

   data=load_data(args.descriptor_holo)
   data=pd.concat([data,load_data(args.descriptor_apo)])


   for i in range(len(args.descriptor_files)):
        data=pd.concat([data,load_data(args.descriptor_files[i],start=0)])
        name=data["name"].iloc[-1]
        custom_palette[name]="purple"

   if args.ref_vol!=0:
      data["pock_volume"]=data["pock_volume"]/args.ref_vol*100
      data["vol_roll"]=data["vol_roll"]/args.ref_vol*100

   sns.violinplot(x='name', y='pock_volume', data=data, split=True,palette=custom_palette)
   plt.ylim([0,max(data.pock_volume)+10])
   plt.savefig("violinplot.png")
   plt.show()