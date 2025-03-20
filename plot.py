import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import itertools
import textwrap

def get_varying_columns(grouped, columns):
    varying_columns = []
    for col in columns:
        unique_values = grouped[col].unique()  # Get unique values for the column
        if len(unique_values) > 1:  # Check if more than one unique value exists
            varying_columns.append((col,list(unique_values)))
    return varying_columns

def generate_config_strings(varying_values):
    keys, values = zip(*varying_values)  # Separate keys and their unique values
    combinations = list(itertools.product(*values))  # Get all possible combinations
    
    def get_string(key, val):
            if key == "res-ds":
                    return val
            else:
                    return f"{key}={str(val)}"

    config_strings = ["\n".join(get_string(key, val) for key, val in zip(keys, combo)) for combo in combinations]
    
    return config_strings

def add_ds_index(varying_cols, dataset_df):
        need_index=False
        for cols in varying_cols:
                if cols[0] == "res-ds":
                        need_index=True
                        ds_col=cols
                        break
        if need_index:
                #print("Adding index for datasets")
                # Wrap and format text
                #loop over all datasets
                #print(ds_col[1])
                names=[]
                for entry in ds_col[1]:
                        #get the index in the form 'o1280-ds[ID]', 
                        index = entry.split("ds", 1)[-1]
                        subset=dataset_df.loc[dataset_df["datasetID"] ==int(index)]
                        ds_name=list(subset["dataset"])[0]
                        #print(f"{index=}, {ds_name.iloc[0]=}")
                        names.append(f"{entry}={ds_name}")
                        #print(f"ds{index}={ds_name}")

                wrapped_ds = [textwrap.fill(str(d), width=200) for d in names]  # Adjust width as needed
                text_str = "\n".join(wrapped_ds)
                #print(text_str)
                
                # Add text at the bottom of the plot
                #plt.figtext(0.5, 0.12, text_str, wrap=True, ha="center", fontsize=8, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
                plt.annotate(text_str, xy=(0.5, 0.05), xycoords='figure fraction',
                        ha="center", fontsize=8, va="bottom",
                        bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

                # Adjust layout to prevent cropping
                plt.subplots_adjust(bottom=0.3)

#TODO   fix so it plots something when there is no difference. currently 'Error plotting: not enough values to unpack (expected 2, got 0)'
#       Add a text listing at the bottom of the plot explaining 'ds*'
#       Replace res-ds with a global ds
def plot_anemoi_dataloader_benchmark(csv, show_plot=True, outdir="out", outname="out", header=""):
        file=csv
        filename=os.path.basename(file)
        #print(f"Loading {file}")
        df=pd.read_csv(file)
        #issue where if you vary res you have a different res and dataset, so have to fuse them bc its really 1 change
        #also dataset paths are too large to plot
        # so replace res and dataset with "res-datasetID", and make a lookup table for the ids if we want to print the path later
        df['datasetID'] = pd.factorize(df['dataset'])[0]
        dataset_df = df.filter(["datasetID","dataset"], axis=1).drop_duplicates() #works but not unique
        df['res-ds'] = df.apply(lambda row: f"{row.res}-ds{row.datasetID}", axis=1)
        df =df.drop(['dataset','datasetID','res'], axis=1)
        
        #df_avg = df.groupby("num_workers", as_index=False)["throughput(byte/s)"].mean()
        grouped = df.groupby(["res-ds", "rollout", "batch_size","num_workers", "prefetch_factor", "pin_memory", "num_procs"], as_index=False).mean()
        #print(grouped)
        varying_cols = get_varying_columns(grouped, ["res-ds", "rollout", "batch_size","num_workers", "prefetch_factor", "pin_memory", "num_procs"])
        #print(varying_cols)
        configs=generate_config_strings(varying_cols)
        #print(configs)

        # Extract relevant columns
        x = configs
        y = grouped["proc-throughput(byte/s)"] / 1024 / 1024
        
        # Plot the bar chart
        plt.figure(figsize=(8, 5))
        plt.bar(x, y, color="royalblue")
        #plt.tick_params("x", rotation=45)
        
        #Generate an index at the bottom of the plot explaining what 'ds*' means
        add_ds_index(varying_cols=varying_cols, dataset_df=dataset_df)

        # Labels and title
        plt.xlabel("Config")
        plt.ylabel("Throughput (MB/s)")
        title="Dataloader throughput per 'GPU'"
        if header != "":
                title=f"{title} - {header}"
        plt.title(title)
        #plt.grid() #goes on top

        # Show the plot
        print(f"Savinging plot to {outdir}/{outname}.png")
        plt.savefig(f"{outdir}/{outname}.png")
        #plt.show()
        
#plot_anemoi_dataloader_benchmark("dev.csv")

def plot_mem_monitor(csv, show_plot=True, outdir="out", filename_prefix=""):
        file=csv
        filename=os.path.basename(file)
        if filename_prefix != "":
                filename=f"{filename_prefix}-{filename}"
        df=pd.read_csv(file)
        df["pname-pid"] = df["pname"] + "-" + df["pid"].astype(str)
        start_time=df["time"][0]
        df["elapsed-time"] = df["time"] - start_time

        # Set color palette
        unique_processes = df["pname-pid"].unique()
        palette = sns.color_palette("tab10", len(unique_processes))
        color_map = dict(zip(unique_processes, palette))

        # Plot memory usage over time
        plt.figure(figsize=(12, 6))
        GB=1024*1024*1024
        #for metric in ["rss", "pss", "uss", "shared"]: # *ss all the same, nothing shared
        first_dl=True
        for metric in ["rss"]:
                for process in unique_processes:
                        subset = df[df["pname-pid"] == process]
                        if (process.startswith("system")):
                                plt.plot(subset["elapsed-time"], subset["used_mem"]/GB, label=f"used_mem", color=color_map[process], linestyle="-")
                                plt.plot(subset["elapsed-time"], subset["dl_used_mem"]/GB, label=f"DL RSS mem sum", color=color_map[process], linestyle=":")
                                plt.plot(subset["elapsed-time"], subset["total_mem"]/GB, label=f"total_mem", color=color_map[process], linestyle="--")
                        else:   
                                if first_dl:
                                        label="Dataloader - RSS"
                                        first_dl=False
                                else:
                                        label=str() #exclude from legend to prevent spam with many dataloaders
                                
                                plt.plot(subset["elapsed-time"], subset[metric]/GB, label=f"{label}", color=color_map[process], linestyle="-" if metric == "rss" else "--")

        plt.xlabel("Time (s)")
        plt.ylabel("CPU Memory (GB)")
        plt.title(f"Memory Usage Over Time - {filename}")
        plt.legend(loc="upper left")
        plt.grid()
        out_file=f"{outdir}/{filename}.png"
        print(f"Plotting memory usage to {out_file}")
        plt.savefig(f"{out_file}")
        if show_plot:
                plt.show()

if __name__ == "__main__":
    plot_anemoi_dataloader_benchmark()