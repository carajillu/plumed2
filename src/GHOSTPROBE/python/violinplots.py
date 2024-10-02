import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

colors={"holo":"blue","apo":"red"}
thickness={"apo":2,"holo":2}

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--descriptor_files', nargs="+", help="mdpout_descriptors.txt files of experiment trajectories (can be more than 1)",default=["mdpout_descriptors.txt"])
    parser.add_argument('-r','--descriptor_holo', nargs="?", help="mdpout_descriptors.txt files of reference holo trajectory",default="holo_mdpout_descriptors.txt")
    parser.add_argument('-a','--descriptor_apo', nargs="?", help="mdpout_descriptors.txt files of reference holo trajectory",default="apo_mdpout_descriptors.txt")
    parser.add_argument('-m','--min_vol', nargs="?", type=float, help="Filter out snapshots below the requested volume",default=0)
    parser.add_argument('--ref_vol', nargs="?", type=float, help="Volume of the pocket in the reference frame",default=0)
    parser.add_argument('--increment', nargs="?", type=float, help="Increment for threshold percentages", default=5)
    args = parser.parse_args()
    return args

def load_data(file_path,start=0):
    data=pd.read_csv(file_path,sep=" ")[start:]
    name=file_path.split("/")[-1].split(".")[0]
    data["name"]=[name]*len(data)
    data["vol_roll"]=data["pock_volume"].rolling(window=100).mean()
    return data

def plt_violin(data,colors):
   custom_palette=[]
   for name in data["name"].unique():
       print(name)
       custom_palette.append(colors.get(name,"grey"))
   print(custom_palette)
   sns.violinplot(x='name', y='pock_volume', data=data, split=False,palette=custom_palette,inner="quart")
   plt.xlabel('Simulation')
   plt.ylim([0,max(data.pock_volume)+10])
   plt.savefig("violinplot.png")
   plt.show()

def calculate_percentage_ranges(data, increment):
    # Define the thresholds from 0% to the maximum volume, with the given increment
    max_volume = data['pock_volume'].max()
    thresholds = list(range(0, int(max_volume) + int(increment), int(increment)))

    # Initialize a dictionary to hold the results
    results = {}

    # Calculate the percentage of snapshots above each threshold for each simulation
    for name, group in data.groupby('name'):
        results[name] = []
        total_snapshots = len(group)
        for threshold in thresholds:
            above_threshold = len(group[group['pock_volume'] >= threshold])
            percentage = (above_threshold / total_snapshots) * 100
            results[name].append(percentage)

    # Convert the results dictionary to a DataFrame
    summary_df = pd.DataFrame(results, index=[f'Snapshots > {t}%' for t in thresholds]).T.reset_index()
    summary_df.rename(columns={'index': 'Simulation'}, inplace=True)

    return summary_df

def plot_threshold_vs_percentage(summary_df,colors,thickness):
    # Convert DataFrame to a long format suitable for plotting
    long_df = summary_df.melt(id_vars=['Simulation'], var_name='Threshold', value_name='Percentage')

    # Extract numerical thresholds from the 'Threshold' column
    long_df['Threshold'] = long_df['Threshold'].str.extract(r'(\d+)').astype(int)

    # Create the scatter plot
    plt.figure()
    for simulation in long_df['Simulation'].unique():
        sim_data = long_df[long_df['Simulation'] == simulation]
        color = colors.get(simulation, 'black')
        linewidth = thickness.get(simulation, 1)
        plt.plot(sim_data['Threshold'], sim_data['Percentage'], label=simulation,color=color,linewidth=linewidth)

    # Customize the plot
    plt.xlabel('Pocket Exposure (%)')
    plt.ylabel('Number of snapshots (%)')
    plt.legend(title='Simulation')
    #plt.grid(True)
    plt.savefig("threshold_vs_percentage.png")
    # Show the plot
    plt.show()


if __name__=="__main__":
   args=parse()

   data=load_data(args.descriptor_holo)
   data=pd.concat([data,load_data(args.descriptor_apo)])


   for i in range(len(args.descriptor_files)):
        data=pd.concat([data,load_data(args.descriptor_files[i],start=0)])

   if args.ref_vol!=0:
      data["pock_volume"]=data["pock_volume"]/args.ref_vol*100
      data["vol_roll"]=data["vol_roll"]/args.ref_vol*100

   plt_violin(data,colors)

   # Generate and display the table of percentages across a range of thresholds
   summary_df = calculate_percentage_ranges(data, args.increment)
   plot_threshold_vs_percentage(summary_df,colors,thickness)
   summary_df.to_csv("percentages.csv",index=False,sep=" ")