from __future__ import annotations
from collections import defaultdict
import pickle
import sys
import torch
import json
from typing import Any
from tabulate import tabulate
import os
import time
import psutil
import datetime
import argparse
 
#TODO refactor mem monitor so that it only gets info once per step and reuses that for table() and csv()
 
def get_mem_info(pid: int) -> dict[str, int]:
  res = defaultdict(int)
  #res['uptime'] = datetime.datetime.fromtimestamp(time.time() - psutil.Process(pid).create_time()).strftime("%M:%S")
  res['uptime'] = int(time.time() - psutil.Process(pid).create_time())
  for mmap in psutil.Process(pid).memory_maps():
    res['rss'] += mmap.rss
    res['pss'] += mmap.pss
    res['uss'] += mmap.private_clean + mmap.private_dirty
    res['shared'] += mmap.shared_clean + mmap.shared_dirty
    if mmap.path.startswith('/'):
      res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
  return res
 
 
class MemoryMonitor():
  def __init__(self, pids: list[int] = None):
    self.names={}
    if pids is None:
      pids = [os.getpid()]
      self.names[str(os.getpid())] = "memory_monitor"
    self.pids = pids
    self.csv_filename=None
    
  def __del__(self):
      if self.f is not None:
            self.f.close()
 
  def add_pid(self, pid: int, name=None):
    assert pid not in self.pids
    if name is not None:
      self.names[str(pid)]=name
    self.pids.append(pid)
 
  def _refresh(self):
    self.data = {}
    #self.pids = [pid for pid in self.pids if psutil.Process(pid).is_running()]
    #self.pids[:] = [pid for pid in self.pids if psutil.Process(pid).is_running()]
    for pid in self.pids:
        try:
            current = get_mem_info(pid)
        except psutil.NoSuchProcess:
            pass
        else:
            self.data[pid] = current
 
      #try:
      #  current = get_mem_info(pid)
      #except psutil.NoSuchProcess:
        #self.pids.remove(pid)
      #  pass
      #else:
      #  self.data[pid] = current
    #self.data = {pid: get_mem_info(pid) for pid in self.pids}
    return self.data
 
  def get_name(self, pid) -> str:
    if str(pid) in self.names:
      return self.names[str(pid)]
    else:
      return ""
 
  def csv(self, filename=None):
    #first call -> create file and print headers
    if (self.csv_filename == None):
        if filename==None:
            raise ValueError("Error no filename provided in initial call to memoryMonitor.csv")
        
        self.csv_filename = filename
        os.system(f"rm -f {self.csv_filename}")
        self.f = open(filename, "a")
        header="time,pname,pid,uptime,rss,pss,uss,shared,shared_file,total_mem,used_mem,dl_used_mem\n"
        self.f.write(header)
    
    keys = list(list(self.data.values())[0].keys())
    system_line=f"{self.now},system,,,,,,,,{self.mem.total},{self.mem.total-self.mem.available},{self.rss_total_mem}\n"
    self.f.write(system_line)
    for pid, data in self.data.items():
        if not self.get_name(pid) == "memory_monitor":
            #mem_stats=tuple(self.format(data[k]) for k in keys if k !="uptime") #want raw numbers
            mem_stats=','.join([str(data[k]) for k in keys])
            line=f"{self.now},{self.get_name(pid)},{str(pid)},{mem_stats},,,\n"
            self.f.write(line)
            
    self.f.flush()
            
  #Update the monitors records, should be called once per timestep
  #These numbers are then used to create table and csv
  def step(self):
    self._refresh()
    self.mem=psutil.virtual_memory()
    self.now = str(int(time.perf_counter() % 1e5))
    #count total memory used by pytorch DL procs
    #TODO not sure if it makes sense to sum up pss when running multi-gpu
    #when i plot 4 procs with 6 workers each, once the pss sum is above the mem usage reported by psutil.
    # => either summing pss across multiproc is wrong or its an issue of timing
    # summing RSS looks better, but at one point it also crossed over used_mem
    self.rss_total_mem=0
    for pid, data in self.data.items():
      if self.get_name(pid).startswith("pt_"):
          self.rss_total_mem += data['rss']
      
 
  def table(self) -> str:
    system_mem=f"Total system memory {self.format(self.mem.total)}, Memory availible {self.format(self.mem.available)} ({100 - self.mem.percent}%)\nSystem memory used, according to mlflow: {self.format(self.mem.used)}\n"
    table = []
    keys = list(list(self.data.values())[0].keys())
    for pid, data in self.data.items():
      #table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
      #table.append((now, uptime, self.get_name(pid), str(pid)) + tuple(self.format(data[k]) for k in keys))
      #uptime = datetime.datetime.fromtimestamp(data["uptime"]).strftime("%M:%S")
      #table.append((now, uptime, self.get_name(pid), str(pid)) + tuple(self.format(data[k]) for k in keys if k !="uptime"))
      table.append((self.now, self.get_name(pid), str(pid)) + tuple(self.format(data[k],key=k) for k in keys))
 
      #time  uptime    pname               PID  rss     pss     uss     shared    shared_file
      #------  --------  --------------  -------  ------  ------  ------  --------  -------------
       #99013  00:49     memory_monitor  1820491  311.9M  299.7M  293.2M  18.7M     18.7M
 
    #append total
    total_memory=f"Total memory used by PyTorch (PSS sum): {self.format(self.rss_total_mem)}\n"
    #return tabulate(table, headers=["time", "uptime", "pname","PID"] + keys) +  "\n" + total_memory + system_mem
    return tabulate(table, headers=["time", "pname","PID"] + keys) +  "\n" + total_memory + system_mem
 
  def str(self):
    self._refresh()
    keys = list(list(self.data.values())[0].keys())
    res = []
    for pid in self.pids:
      s = f"PID={pid}"
      if self.names[pid] is not None:
        s = f"{s} ({self.names[pid]})"
      for k in keys:
        v = self.format(self.data[pid][k])
        s += f", {k}={v}"
      res.append(s)
    return "\n".join(res)
 
  @staticmethod
  def format(size: int, key: str ="") -> str:
    if (key=="uptime"):
        return datetime.datetime.fromtimestamp(size).strftime("%M:%S")
    else:
        for unit in ('', 'K', 'M', 'G'):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)
 
def get_pt_process_pids():
    # List to store the PIDs of processes starting with "pt_"
    pt_process_pids = []
    pnames={}
 
    # Iterate over all running processes
    for process in psutil.process_iter(['pid', 'name', "username"]):
        try:
            # Check if the process name starts with "pt_"
            #if process.info['name'].startswith("pt_") and process.info['username'] == os.system("whoami"):
            if process.info['name'].startswith("pt_") and process.info['username'] == "naco":
            #if process.info['name'].startswith("pt_"):
                pt_process_pids.append(process.info['pid'])
                pnames[process.info['pid']]=process.info['name']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Handle processes that have terminated or for which we have no access
            pass
 
    return pt_process_pids, pnames
 

def run_mem_monitor(table, csv, csv_filename):
  monitor = MemoryMonitor()
  monitor.step()
  if table:
    print(monitor.table())
  if csv:
    monitor.csv(csv_filename)
  
  #if start_method == "forkserver":
      # Reduce 150M-per-process USS due to "import torch".
      #mp.set_forkserver_preload(["torch"])

  while True:
    pids, pnames= get_pt_process_pids()
    for pid in pids:
      if pid not in monitor.pids:
        monitor.add_pid(pid, name=pnames[pid])
    monitor.step()
    if table:
      table=monitor.table()
      os.system( 'clear' )
      print(table)
    if csv:
      monitor.csv()
    time.sleep(1)
  

def main():
  parser = argparse.ArgumentParser()
  parser.parse_args()
  parser.add_argument("table", default=True)
  parser.add_argument("csv", default=True)
  parser.add_argument("csv_filename", default="dataloader-mem-usage.csv")
  args = parser.parse_args()

  run_mem_monitor(args.table, args.csv, args.csv_filename)

# Example usage
#if __name__ == "__main__":
#  main()
        