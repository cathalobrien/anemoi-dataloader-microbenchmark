import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_mem_monitor(csv, show_plot=True, outdir="out"):
        file=csv
        filename=os.path.basename(file)
        df=pd.read_csv(file)
        df["pname-pid"] = df["pname"] + "-" + df["pid"].astype(str)

        # Set color palette
        unique_processes = df["pname-pid"].unique()
        palette = sns.color_palette("tab10", len(unique_processes))
        color_map = dict(zip(unique_processes, palette))

        # Plot memory usage over time
        plt.figure(figsize=(12, 6))
        GB=1024*1024*1024
        #for metric in ["rss", "pss", "uss", "shared"]: # *ss all the same, nothing shared
        for metric in ["rss"]:
                for process in unique_processes:
                        subset = df[df["pname-pid"] == process]
                        if (process.startswith("system")):
                                plt.plot(subset["time"], subset["used_mem"]/GB, label=f"{process} - used_mem", color=color_map[process], linestyle="-")
                                plt.plot(subset["time"], subset["total_mem"]/GB, label=f"{process} - total_mem", color=color_map[process], linestyle="--")
                        else:
                                plt.plot(subset["time"], subset[metric]/GB, label=f"{process} - {metric}", color=color_map[process], linestyle="-" if metric == "rss" else "--")

        plt.xlabel("Time")
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
