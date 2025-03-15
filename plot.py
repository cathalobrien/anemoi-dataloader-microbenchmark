import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_mem_monitor(csv, show_plot=True, outdir="out"):
        file=csv
        filename=os.path.basename(file)
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
        #plt.savefig(f"~/Documents/aifs/dl-mem-usage/outputs/{filename}.png")
        out_file=f"{outdir}/{filename}.png"
        print(f"Plotting memory usage to {out_file}")
        plt.savefig(f"{out_file}")
        if show_plot:
                plt.show()
