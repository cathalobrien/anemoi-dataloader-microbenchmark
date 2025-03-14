import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Define column names
#columns = ["time", "pname", "pid", "uptime", "rss", "pss", "uss", "shared", "shared_file"]

# Create DataFrame
file='data/dataloader-mem-usage-9km-1g-3w-1pf-12r-npm.csv'
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
plt.ylabel("Memory Usage (GB)")
plt.title(f"Memory Usage Over Time - {filename}")
plt.legend(loc="upper left")
plt.grid()
#plt.savefig(f"~/Documents/aifs/dl-mem-usage/outputs/{filename}.png")
plt.savefig(f"outputs/{filename}.png")
plt.show()
