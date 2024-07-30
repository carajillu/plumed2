import argparse
import pandas as pd
import seaborn as sns

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--descriptor_files', nargs="+", help="mdpout_descriptors.txt files of experiment trajectories (can be more than 1)",default=["mdpout_descriptors.txt"])
    parser.add_argument('-r','--descriptor_ref', nargs="?", help="mdpout_descriptors.txt files of reference trajectory",default="ref_mdpout_descriptors.txt")
    args = parser.parse_args()
    return args

def get_data(reference,files):
    data=pd.read_csv(reference,sep=" ")
    name=reference.split("/")[-1].split(".")[0]
    data["run"]=[name for i in range(len(data))]
    data["reference"]=["Yes" for i in range(len(data))]
    for file in files:
        data2=pd.read_csv(file,sep=" ")
        name=file.split("/")[-1].split(".")[0]
        data2["run"]=[name for i in range(len(data2))]
        data2["reference"]=["No" for i in range(len(data2))]
        data=pd.concat([data,data2],ignore_index=True)
    return data

if __name__=="__main__":
   args=parse()
   data=pd.read_csv(args.descriptor_ref,sep=" ")
   name=args.descriptor_ref.split("/")[-1].split(".")[0]
   data=get_data(args.descriptor_ref,args.descriptor_files)
   sns.violinplot(data=data, x="run", y="pock_volume", hue="reference",
               split=True, inner="quart", fill=False,
               palette={"Yes": "g", "No": ".35"})
   sns.save("violinplot.png")
   print(data)