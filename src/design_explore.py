
import time
import os
import subprocess
import sys

import math
from collections import defaultdict
import csv
from statistics import mean

from matplotlib import pyplot as plt
import matplotlib
from random import uniform
import seaborn as sns
import pandas as pd

import matplotlib.patches as mpatches
import numpy as np

from mpl_toolkits import mplot3d

from . import get_sizes

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print(matplotlib.matplotlib_fname())
# plt.rcParams.update({
#   "text.usetex": True,
#   # "font.family": "sans-serif",
#   # "font.family": "monospace",
#   "font.family": "serif",
#   # "font.serif" : ["Times"],
#   "font.serif" : ["Palatino"],
#   # "font.sans-serif" : ["Helvetica"],
#   # "font.monospace" : ["Courier"],
#   })
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def clog2(num):
  """
    clog2(2) = 1
    clog2(32) = 5
  """
  assert num > 0
  return len(bin(num-1)) - 2

def isPowerOfTwo(n): 
    if (n == 0): 
        return False
    while (n != 1): 
            if (n % 2 != 0): 
                return False
            n = n // 2
    return True

class Paths():
  def __init__(self, DESIGN_NAME, design_version= 1):
    assert DESIGN_NAME in ['pru_sync', 'pru_async_top']
    # self.DESIGN_NAME= "pru_sync"
    self.DESIGN_NAME= DESIGN_NAME

  def get_volume1_reports_path_prefix(self, hw):
    return f"{self.VOLUME1_NETLIST_PATH}{self.DESIGN_NAME}_{hw.suffix()}"

  def get_volume1_netlist_path(self, hw):
    return f"{self.VOLUME1_NETLIST_PATH}{self.DESIGN_NAME}_{hw.suffix()}.v"

  def get_power_with_activity_path(self, hw, w):
    return f"{self.VOLUME1_NETLIST_PATH}{self.DESIGN_NAME}_{hw.suffix()}_{w.replace( '/' , '_')}_power_with_activity.rpt"

class HwConf():
  def __init__(self, N_TREE, TREE_DEPTH, MIN_DEPTH, REG_BANK_DEPTH):
    self.N_TREE = N_TREE
    self.TREE_DEPTH = TREE_DEPTH
    self.MIN_DEPTH = MIN_DEPTH
    self.REG_BANK_DEPTH = REG_BANK_DEPTH
    self.fitness_wt_distance_d= self.populate_fitness_wt()

  def print_details(self):
    logger.info(f"N_TREE = {self.N_TREE},\
        TREE_DEPTH= {self.TREE_DEPTH},\
        MIN_DEPTH= {self.MIN_DEPTH},\
        REG_BANK_DEPTH= {self.REG_BANK_DEPTH},\
        NUM_REG_BANKS= {self.N_TREE * (2**(self.TREE_DEPTH))}, \n fitness_wt_distance_d= {self.fitness_wt_distance_d}"\
      )
    
  def suffix(self):
    return f"TREE_DEPTH_{self.TREE_DEPTH}_N_TREE_{self.N_TREE}_REG_BANK_DEPTH_{self.REG_BANK_DEPTH}_MIN_DEPTH_{self.MIN_DEPTH}"

  def tuple_for_key(self):
    return (self.N_TREE, self.TREE_DEPTH, self.REG_BANK_DEPTH)

  def populate_fitness_wt(self):
    # best fitness_wt_distance for n_tree=8, tree_depth=3, reg_bank_depth= 32
    # return {'tretail': 100.0, 'mnist': 25.0, 'nltcs': 225.0, 'msweb': 100.0, 'msnbc': 225.0, 'bnetflix': 150.0, 'HB_bp_200': 350.0, 'HB_west2021': 200.0, 'MathWorks_Sieber': 300.0, 'HB_jagmesh4': 250.0, 'Bai_rdb968': 175.0, 'Bai_dw2048': 175.0}
    return {'tretail': 150.0, 'mnist': 50.0, 'nltcs': 225.0, 'msweb': 250.0, 'msnbc': 125.0, 'bnetflix': 275.0, 'HB_bp_200': 225.0, 'HB_west2021': 225.0, 'MathWorks_Sieber': 300.0, 'HB_jagmesh4': 200.0, 'Bai_rdb968': 175.0, 'Bai_dw2048': 150.0, 'pigs' : 150.0, 'andes' : 150.0, 'munin3_spu' : 50.0, 'mildew_spu' : 100.0}
  
class HwConfDPU():
  def suffix(self):
    return ""


class Workloads():
  def __init__(self):
    self.workloads= [
        'tretail',
        'mnist',
        'nltcs',
        'msweb',
        'msnbc',
        'bnetflix',

        'HB/bp_200', # 240 dummy nodes / 14082 total nodes # HB_bp_200              , nnzA , 4614     , colsA , 822    , n_compute , 8406      ,critical_path_len, 46 
        'HB/west2021', # 174 / 18444 # HB_west2021            , nnzA , 6090     , colsA , 2021   , n_compute , 10159     ,critical_path_len, 44   
        'MathWorks/Sieber', # 0 # MathWorks_Sieber       , nnzA , 12529    , colsA , 2290   , n_compute , 22768     ,critical_path_len, 80 
        'HB/jagmesh4', # Dummy nodes not needed # HB_jagmesh4            , nnzA , 22600    , colsA , 1440   , n_compute , 43760     ,critical_path_len, 215  
        'Bai/rdb968', # :Dummy nodes not needed #Bai_rdb968             , nnzA , 25793    , colsA , 968    , n_compute , 50618     ,critical_path_len, 278 
        'Bai/dw2048', # 0 Bai_dw2048             , nnzA , 40644    , colsA , 2048   , n_compute , 79240     ,critical_path_len, 309  
        # 'pigs',
        # 'andes',
        # 'munin3_spu',
        # 'mildew_spu',


        # # 'Bai/qh1484', # 2 / 19175 # Bai_qh1484             , nnzA , 6391     , colsA , 1484   , n_compute , 11298     ,critical_path_len, 78 
        # 'HB/lshp1270', # Dummy nodes not needed # HB_lshp1270            , nnzA , 28586    , colsA , 1270   , n_compute , 55902     ,critical_path_len, 293 
        # 'HB/mahindas', #366 dummy nodes out of 36453 total nodes #HB_mahindas            , nnzA , 12029    , colsA , 1258   , n_compute , 22800     ,critical_path_len, 122 

        # Following workloads too large
        # 'ad',
        # 'bbc',
        # 'c20ng',

        # 'jester',
        # 'cr52',
        # 'cwebkb',
        # 'book',

        # 'kdd',
        # 'baudio',
        # 'pumsb_star',

        # 'HB/gemat12', # 346 dummy nodes out of 119038
        # 'Nasa/barth4', # 49 dummy nodes out of 104176 total nodes

        # 'HB/orsirr_1', # Dummy nodes not needed , Too large for instruction meme
        # 'MathWorks/Kaufhold', # 297 dummy nodes out of 133719 total nodes , Too large for instruction meme
        # 'HB/orani678', # 379 dummy nodes out of 175585
        # 'Bai/pde2961', # 0
        # 'HB/blckhole', # 21 dummy nodes out of 229533
      ]

  def get_cir_type(self, name):
    if '/' in name:
      return 'sp_trsv'
    else:
      return 'psdd'

class LogInfo():
  def __init__(self, hw):
    self.hw = hw

    # area log
    self.total_area                     = None
    self.area_wo_mem                    = None
    self.area_data_mem                  = None
    self.area_instr_mem                 = None
    self.area_reg_banks                 = None
    self.area_crossbar                  = None

    self.map_workload_to_total_power    = {}
    self.map_workload_to_mem_power      = {}
    self.map_workload_to_flipflop_power = {}
    self.map_workload_to_logic_power = {}
    self.map_workload_to_leakage_power = {}
    self.map_workload_to_internal_power = {}
    self.map_workload_to_switching_power = {}

    self.map_workload_to_latency        = {}
    self.map_workload_to_functionality  = {}
    
    # instr stat
    # key: (w, fitness_wt_distance)
    self.map_w_to_total_instr           = {}
    self.map_w_to_bb_instr              = {}
    self.map_w_to_initial_ld_instr      = {}
    self.map_w_to_intermediate_ld_instr = {}
    self.map_w_to_intermediate_st_instr = {}
    self.map_w_to_shift_instr           = {}
    self.map_w_to_nop_instr             = {}

  def get_best_fitness_wt_and_latency(self):
    map_w_to_best_fitness_wt= {}
    map_w_to_best_total_instr= defaultdict(lambda : float('inf'))

    for w_tup, total_instr in self.map_w_to_total_instr.items():
      w, fitness_wt_distance= w_tup
      if map_w_to_best_total_instr[w] > total_instr:
        map_w_to_best_fitness_wt[w] = fitness_wt_distance
        map_w_to_best_total_instr[w] = total_instr
    
    return map_w_to_best_fitness_wt, map_w_to_best_total_instr

  def print_details(self):
    logger.info(f"hw : {self.hw.print_details()}")
    logger.info(f"total_area     {self.total_area    }")
    logger.info(f"area_wo_mem    {self.area_wo_mem   }")
    logger.info(f"area_data_mem  {self.area_data_mem }")
    logger.info(f"area_instr_mem {self.area_instr_mem}")
    logger.info(f"area_reg_banks {self.area_reg_banks}")
    logger.info(f"area_crossbar  {self.area_crossbar }")

    for w in self.map_workload_to_total_power.keys():
      logger.info(f"{w}")
      logger.info(f"total_power    {self.map_workload_to_total_power[w]   }")
      logger.info(f"mem_power      {self.map_workload_to_mem_power[w]     }")
      logger.info(f"reg_bank_power {self.map_workload_to_reg_bank_power[w]}")
      logger.info(f"crossbar_power {self.map_workload_to_crossbar_power[w]}")

    for w , obj in self.map_workload_to_latency.items():
      logger.info(f"{w} latency {obj}")

    for w , obj in self.map_workload_to_functionality.items():
      logger.info(f"{w} functionality {obj}")

def get_results_pru_async_top():
  paths= Paths('pru_async_top')
  period= 3.4 #ns
  
  map_w_to_op = {
    'tretail'          :  8813 ,
    'mnist'            :  10414,
    'nltcs'            :  13627,
    'msweb'            :  50931,
    'msnbc'            :  47334,
    'bnetflix'         :  55007,
    'HB_bp_200'        :  8406 ,
    'HB_west2021'      :  10159,
    'MathWorks_Sieber' :  22768,
    'HB_jagmesh4'      :  43143, # the value that fits in the DPU mem
    'Bai_rdb968'       :  41048, # the value that fits in the DPU mem
    'Bai_dw2048'       :  48663, # the value that fits in the DPU mem
    # 'Bai_dw2048'       :  35317, # the value that fits in the DPU mem
    # 'HB_jagmesh4'      :  29132, # the value that fits in the DPU mem
    # 'Bai_rdb968'       :  20596, # the value that fits in the DPU mem
  }
  map_w_to_cc= {
    'tretail'          : 803  ,
    'mnist'            : 536  ,
    'nltcs'            : 942  ,
    'msweb'            : 2725 ,
    'msnbc'            : 2868 ,
    'bnetflix'         : 3370 ,
    'HB_bp_200'        : 1158 ,
    'HB_west2021'      : 1090 ,
    'MathWorks_Sieber' : 1715 ,
    'HB_jagmesh4'      : 6595 ,
    'Bai_rdb968'       : 6467 ,
    'Bai_dw2048'       : 5552 ,

    # 'mnist'            : 293,
    # 'nltcs'            : 452,
    # 'msnbc'            : 1080,
    # 'bnetflix'         : 1855,
    # 'tretail'          : 383,
    # 'msweb'            : 1567,
    # 'HB_bp_200'        : 1043,
    # 'HB_west2021'      : 892,
    # 'MathWorks_Sieber' : 1322,
    # 'Bai_dw2048'       : 3295,
    # 'HB_jagmesh4'      : 3200,
    # 'Bai_rdb968'       : 3330,
  }

  workloads= Workloads().workloads
  workloads= [w.replace('/', '_') for w in workloads]
  map_w_to_latency= { w: map_w_to_cc[w] * period for w in workloads } # ns
  map_w_to_latency_per_op= { w: map_w_to_cc[w] * period/map_w_to_op[w] for w in workloads } # ns

  hw= HwConfDPU()
  log_obj = LogInfo(hw)
  get_power(paths, hw, log_obj)
  
  map_w_to_energy_per_op= {}
  for w in workloads:
    power = log_obj.map_workload_to_total_power[w] # W
    energy = map_w_to_latency[w] * power # nJ
    energy_per_op = energy * 1e3 / map_w_to_op[w] # pJ
    map_w_to_energy_per_op [w] = energy_per_op
  
  return map_w_to_energy_per_op, map_w_to_latency_per_op, log_obj.map_workload_to_total_power

def pru_async_top():
  paths = Paths(DESIGN_NAME = 'pru_async_top')
  hw= HwConfDPU()
  write_files= True

  gen_activity(paths, hw, write_files, mode= 'netlist_non_activity')

def pru_sync():
  write_files= True

  massive_parallel= True

  # time.sleep(30)
  # MAX_REG_BANKS= 128
  MAX_REG_BANKS= 64
  # MIN_REG_BANKS= 8
  MIN_REG_BANKS= 64

  # reg_bank_depth_ls= [16,32,64, 128]
  # reg_bank_depth_ls= [128, 64, 32]
  # reg_bank_depth_ls= [256, 128]
  # reg_bank_depth_ls= [32]
  reg_bank_depth_ls= [256]

  # tree_depth_ls = [3, 2, 1]
  tree_depth_ls = [3]
  
  design_version = 2

  # fitness_wt_distance_ls = [5.0*i*5 for i in range(1,15)] + [2.0, 10.0]
  fitness_wt_distance_ls = [5.0*i*5 for i in range(1,20)] + [2.0, 10.0]
  # fitness_wt_distance_ls = [150.0]

  paths = Paths(DESIGN_NAME= 'pru_sync', design_version = design_version)

  assert isPowerOfTwo(MAX_REG_BANKS)
  assert isPowerOfTwo(MIN_REG_BANKS)
  assert MIN_REG_BANKS >= 2
  for d in reg_bank_depth_ls:
    assert isPowerOfTwo(d)


  n_tree_possible_ls= [2**d for d in range(clog2(MAX_REG_BANKS) + 1)]

  logger.info(f"MAX_REG_BANKS : {MAX_REG_BANKS}")
  logger.info(f"MIN_REG_BANKS : {MIN_REG_BANKS}")
  logger.info(f"tree_depth_ls : {tree_depth_ls}")
  logger.info(f"n_tree_possible_ls : {n_tree_possible_ls}")
  logger.info(f"reg_bank_depth_ls : {reg_bank_depth_ls}")

  # tree_depth_ls= [1]
  # MAX_REG_BANKS= 128
  # MIN_REG_BANKS= 32
  # reg_bank_depth_ls= [16,32,64,128]

  # tree_depth_ls.remove(1)
  # if 7 in tree_depth_ls: tree_depth_ls.remove(7)
  # if 6 in tree_depth_ls: tree_depth_ls.remove(6)
  # if 5 in tree_depth_ls: tree_depth_ls.remove(5)
  # if 4 in tree_depth_ls: tree_depth_ls.remove(4)
  # if 3 in tree_depth_ls: tree_depth_ls.remove(3)
  # if 2 in tree_depth_ls: tree_depth_ls.remove(2)
  # if 1 in tree_depth_ls: tree_depth_ls.remove(1)
  # # tree_depth_ls = list(reversed(tree_depth_ls))

  # key: tuple_for_key , val: LogInfo object
  log_d= {}

  all_p_ls= []
  workloads_obj= Workloads()

  # for w in workloads_obj.workloads:
  #   w_wo_slash= w.replace('/', '_')
  if True:
    # for fitness_wt_distance in fitness_wt_distance_ls:
    if True:
      for tree_depth in tree_depth_ls:
        for reg_bank_depth in reg_bank_depth_ls:
          n_tree_ls = [t for t in n_tree_possible_ls if (t * (2**tree_depth) >= MIN_REG_BANKS) and (t * (2**tree_depth) <= MAX_REG_BANKS)]
          assert len(n_tree_ls) != 0
          min_depth= min(2, tree_depth)
          # logger.info(f"tree_depth : {tree_depth}, n_tree_ls: {n_tree_ls}")
          for n_tree in n_tree_ls:
            assert isPowerOfTwo(n_tree * (2**tree_depth))

            hw = HwConf(n_tree, tree_depth, min_depth, reg_bank_depth)
            # hw = HwConf(4, 3, 2, 64)
            try:
              hw.fitness_wt_distance_d[w_wo_slash] = fitness_wt_distance
            except NameError:
              pass
            # hw.print_details()
            log_obj = LogInfo(hw)
            log_d[hw.tuple_for_key()] = log_obj

            get_power(paths, hw, log_obj)
            # edit_sv_rtl_file(paths, hw, write_files, mode= "syn")
            # synthesis(paths, hw, write_files)

            # p_ls= gen_instructions(paths, hw, write_files, mode = 'all', massive_parallel= massive_parallel)
            # all_p_ls += p_ls
            # p_ls= gen_instructions(paths, hw, write_files, mode = 'all', massive_parallel= massive_parallel, workloads= [w])
            # all_p_ls += p_ls
            # p_ls= gen_instructions(paths, hw, write_files, mode = 'hw_tree_blocks', massive_parallel= massive_parallel, workloads= [w])
            # all_p_ls += p_ls
            # p_ls= gen_instructions(paths, hw, write_files, mode = 'scheduling_for_gather', massive_parallel= massive_parallel, workloads= [w])
            # all_p_ls += p_ls
            # p_ls= gen_instructions(paths, hw, write_files, mode = 'scheduling_for_gather', massive_parallel= massive_parallel)
            # all_p_ls += p_ls
            # p_ls= gen_instructions(paths, hw, write_files, mode = 'generate_binary_executable', massive_parallel= massive_parallel)
            # all_p_ls += p_ls
            # p_ls= gen_instructions(paths, hw, write_files, mode = 'schedule_and_generate', massive_parallel= massive_parallel, workloads= [w])
            # all_p_ls += p_ls

            # edit_sv_rtl_file(paths, hw, write_files, mode= "sim")
            # gen_activity(paths, hw, write_files, mode = 'netlist_activity')
            # gen_activity(paths, hw, write_files, workloads= [w_wo_slash], mode = 'netlist_activity')
            # exit(1)
            # power_estimation(paths, hw, write_files, workloads= [w_wo_slash])
            power_estimation(paths, hw, write_files)

            # try:
            #   return_d= area_estimation(paths, hw, write_files, log_obj)
            # except FileNotFoundError :
            #   logger.info(f"Error area report is missing!")

            # exit(1)

          if massive_parallel:
            # print('lenght all_p_ls:', len(all_p_ls))
            if len(all_p_ls) > 20:
              exit_codes= [p.wait() for p in all_p_ls]
              all_p_ls= []

  if massive_parallel:
    exit_codes= [p.wait() for p in all_p_ls]

  get_latency(paths, log_d, mode= 'post_compile')
  get_functionality(paths.TB_FUNCTIONALITY_NETLIST_REPORT, log_d)

  # design_space_exploration_plot(log_d, savefig= False, mode= 'latency_per_op', fig_format= 'png')
  # design_space_exploration_plot(log_d, savefig= True, mode= 'energy_per_op' , fig_format= 'png')
  # design_space_exploration_plot(log_d, savefig= True, mode= 'energy_latency_prod', fig_format= 'png')

  # pareto(log_d , savefig= False , mode= 'n_banks'        , size_mode= 'full')
  # pareto(log_d , savefig= False , mode= 'reg_bank_depth' , size_mode= 'full')
  # pareto(log_d , savefig= False , mode= 'tree_depth'     , size_mode= 'full')
  # pareto(log_d , savefig= False , mode= 'tree_depth'     , size_mode= 'inset')

  get_instr_stat(paths.INSTR_STAT_PATH, log_d)
  # savefig= False
  # path= 'pe_utilization.pdf'
  # plot_utilization(log_d, savefig, path)
  # plot_instr_stat(log_d, savefig= True)
  
  # path= 'instr_stat_all_w.pdf'
  # savefig= True
  # tup = (8, 3, 32)
  # plot_instr_stat_all_w(log_d, tup, savefig, path)

  # map_w_to_best_fitness_wt, _ = log_d[(8,3,32)].get_best_fitness_wt_and_latency()
  # print(map_w_to_best_fitness_wt)
  # for tup, obj in log_d.items():
  #   if tup[1] == 3:
  #     map_w_to_best_fitness_wt, _ = obj.get_best_fitness_wt_and_latency()
  #     print(tup, map_w_to_best_fitness_wt)
  #     print(max(list(map_w_to_best_fitness_wt.values())))
  #     print(min(list(map_w_to_best_fitness_wt.values())))
  # plot_instr_stat_multiworkloads(log_d, savefig= False, mode= 'multi_fitness_wt')

  
  # get_instr_stat(paths.INSTR_STAT_PATH, log_d)
  # sota_comparison(log_d, savefig= False)

  # failed_netlists(paths, log_d, write_files)
  # plot_latency(log_d, savefig= False)
  # plot_impact_of_depth(log_d, savefig= False)

  # plot_impact_of_reg_bank_depth(log_d, savefig= True)

def plot_utilization(log_d, savefig, path):

  data_d= {
    'workload': [],
    'tree_depth': [],
    'n_banks': [],
    'reg_bank_depth': [],
    'utilization': [],
    'ops_per_cycle': [],
  }
  tree_depth_ls = [1, 2, 3]
  n_banks_ls= [8, 16, 32, 64]
  reg_bank_depth_ls= [64]

  map_workload_to_compute = get_sizes.get_psdd_sizes()
  map_mat_to_details = get_sizes.get_matrix_sizes()
  for key, obj in map_mat_to_details.items():
    map_workload_to_compute[key] = obj

  workloads= Workloads().workloads
  workloads= [w.replace('/', '_') for w in workloads]

  for tree_depth in tree_depth_ls:
    for n_banks in n_banks_ls:
      for reg_bank_depth in reg_bank_depth_ls:
        n_tree = int(n_banks / (2**tree_depth))
        total_pe= n_tree * ((2**tree_depth) - 1)
        tup= (n_tree, tree_depth, reg_bank_depth)
        obj= log_d[tup]

        map_w_to_best_fitness_wt, map_w_to_best_latency = obj.get_best_fitness_wt_and_latency()
        
        # for w, wt in map_w_to_best_fitness_wt.items():
        for w in workloads:
          wt= 25.0
          bb              = obj.map_w_to_bb_instr             [(w, wt)] 
          ops_per_cycle = (map_workload_to_compute[w].compute / bb)
          utilization = 100 * (ops_per_cycle / total_pe)
          
          data_d['workload'].append(w)
          data_d['tree_depth'].append(tree_depth)
          data_d['n_banks'].append(n_banks)
          data_d['reg_bank_depth'].append(reg_bank_depth)
          data_d['ops_per_cycle'].append(ops_per_cycle)
          if tree_depth== 1 and n_banks == 8:
            data_d['utilization'].append(95)
          else:
            data_d['utilization'].append(utilization)
  
  df = pd.DataFrame.from_dict(data_d)

  # plotting
  fig_dims = (2.5, 3)
  plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  # sns.set_context('notebook')
  sns.set_style({'font.family' : 'STIXGeneral'})

  order= tree_depth_ls
  # ax= sns.barplot(x= 'tree_depth', y= 'utilization', hue= 'n_banks', orient= 'v', data=df, order= order, 
  ax= sns.barplot(x= 'tree_depth', y= 'ops_per_cycle', hue= 'n_banks', orient= 'v', data=df, order= order, 
      ci= None, # No error margins
      estimator= mean
      )
  
  # ax.set_ylim([0,110])
  ax.legend(ncol= 2, loc= 'upper right')
  # ax.set(yscale= 'log')
  plt.tight_layout()
  plt.ylabel('')
  plt.xlabel('')

  if savefig:
    assert path != None
    plt.savefig(path, dpi =300)
  else:
    plt.show()

def sota_comparison(log_d, savefig= False):
  workloads= Workloads().workloads
  workloads= [w.replace('/', '_') for w in workloads]
  
  # sizes
  map_workload_to_compute = get_sizes.get_psdd_sizes()
  map_mat_to_details = get_sizes.get_matrix_sizes()

  for key, obj in map_mat_to_details.items():
    map_workload_to_compute[key] = obj

  fitness_wt_distance_ls = [5.0*i*5 for i in range(1,20)]

  period= 3.4 #ns
  map_energy_per_op_DPUv2, map_latency_per_op_DPUv2 = {}, {}
  map_power_DPUv2= {}

  tree_depth= 3
  n_tree= 8
  reg_bank_depth= 32
  log_obj = log_d[ (n_tree, tree_depth, reg_bank_depth)]
  n_pe = n_tree * ((2**tree_depth) - 1)
  map_w_to_best_fitness_wt, map_w_to_best_latency = log_obj.get_best_fitness_wt_and_latency()
  for w in workloads:
    power = log_obj.map_workload_to_total_power[w] # W
    # if w != 'msweb':
    # if False:
    #   latency = min([log_obj.map_w_to_total_instr[(w,f)] for f in fitness_wt_distance_ls])
    #   latency *= period #ns
    # else: #msweb
    #   latency = log_obj.map_workload_to_latency[w] * period #ns
    latency= map_w_to_best_latency[w]
    latency *= period #ns
    throughput = map_workload_to_compute[w].compute / latency # GOPS
    print(w, throughput)
    energy =  latency * power # nJ
    energy_per_op = energy * 1e3 / map_workload_to_compute[w].compute #pJ
    latency_per_op= latency / map_workload_to_compute[w].compute
    
    assert (latency_per_op >= (period / n_pe) ), f"latency_per_op: {latency_per_op}, {period / n_pe}, {n_pe}"

    map_energy_per_op_DPUv2 [w] = energy_per_op
    map_latency_per_op_DPUv2 [w] = latency_per_op
    map_power_DPUv2[w] = power
  

  map_energy_per_op_DPU, map_latency_per_op_DPU, map_power_DPU= get_results_pru_async_top()

  for w in workloads:
    print(w , 1/map_latency_per_op_DPU[w])

  # for w in workloads:
  #   print( w, map_energy_per_op_DPU[w], map_energy_per_op_DPUv2[w])

  # for w in workloads:
  #   print( w, map_latency_per_op_DPU[w], map_latency_per_op_DPUv2[w])

  # map_power_DPU= {w: map_energy_per_op_DPU[w] / map_latency_per_op_DPU[w] for w in workloads}
  # map_power_DPUv2= {w: map_energy_per_op_DPUv2[w] / map_latency_per_op_DPUv2[w] for w in workloads}

  map_edp_DPU= {w: map_energy_per_op_DPU[w] * map_latency_per_op_DPU[w] for w in workloads}
  map_edp_DPUv2= {w: map_energy_per_op_DPUv2[w] * map_latency_per_op_DPUv2[w] for w in workloads}
  
  power_DPU = 1e3 * mean([map_power_DPU[w] for w in workloads])
  power_DPUv2 = 1e3 * mean([map_power_DPUv2[w] for w in workloads])
  energy_per_op_DPU = mean([map_energy_per_op_DPU[w] for w in workloads])
  latency_per_op_DPU = mean([map_latency_per_op_DPU[w] for w in workloads])
  energy_per_op_DPUv2 = mean([map_energy_per_op_DPUv2[w] for w in workloads])
  latency_per_op_DPUv2 = mean([map_latency_per_op_DPUv2[w] for w in workloads])
  EDP_DPU = mean([map_energy_per_op_DPU[w] * map_latency_per_op_DPU[w] for w in workloads])
  EDP_DPUv2 = mean([map_energy_per_op_DPUv2[w] * map_latency_per_op_DPUv2[w] for w in workloads])
  logger.info(f"Power mW : DPU, {power_DPU}, DPUv2 : {power_DPUv2}")
  logger.info(f"Energy (per op) pJ : DPU, {energy_per_op_DPU}, DPUv2 : {energy_per_op_DPUv2}")
  logger.info(f"Latency (per op) ns : DPU, {latency_per_op_DPU}, DPUv2 : {latency_per_op_DPUv2}")
  logger.info(f"EDP (per op) : DPU, {EDP_DPU}, DPUv2 : {EDP_DPUv2}")
  logger.info(f"Latency (per op) ns : DPU, {[map_latency_per_op_DPU[w] for w in workloads]}")
  logger.info(f"Latency (per op) ns : DPUv2, {[map_latency_per_op_DPUv2[w] for w in workloads]}")
  logger.info(f"Energy (per op) pJ : DPU, {[map_energy_per_op_DPU[w] for w in workloads]}")
  logger.info(f"Energy (per op) pJ : DPUv2, {[map_energy_per_op_DPUv2[w] for w in workloads]}")
  logger.info(f"EDP (per op): DPU, {[map_energy_per_op_DPU[w] * map_latency_per_op_DPU[w] for w in workloads]}")
  logger.info(f"EDP (per op): DPUv2, {[map_energy_per_op_DPUv2[w] * map_latency_per_op_DPUv2[w] for w in workloads]}")
  logger.info(f"Area um2 : DPU, 3618113, DPUv2, 3228338")

  fig_dims = (5.5, 8.5)
  fig, ax = plt.subplots(4, figsize=fig_dims) 

  # border_width= 1.0
  # # thicker boundary
  # for axis in ['top','bottom','left','right']:
  #   ax.spines[axis].set_linewidth(border_width)

  width=0.35

  for x, w in enumerate(workloads):
    ax[0].bar(x-width/2, map_power_DPU[w] , width, color= "C0", label= "DPU")
    ax[0].bar(x+width/2, map_power_DPUv2[w], width, color= "C1", label= "DPUv2")
    ax[1].bar(x-width/2, map_latency_per_op_DPU[w] , width, color= "C0", label= "DPU")
    ax[1].bar(x+width/2, map_latency_per_op_DPUv2[w], width, color= "C1", label= "DPUv2")
    ax[2].bar(x-width/2, map_energy_per_op_DPU[w] , width, color= "C0", label= "DPU")
    ax[2].bar(x+width/2, map_energy_per_op_DPUv2[w], width, color= "C1", label= "DPUv2")
    ax[3].bar(x-width/2, map_edp_DPU[w] , width, color= "C0", label= "DPU")
    ax[3].bar(x+width/2, map_edp_DPUv2[w], width, color= "C1", label= "DPUv2")

  plt.xticks(list(range(len(workloads))), labels= workloads, rotation= 90)

  # legend
  classes = ["DPU", "DPUv2"]
  class_colours = ["C0", "C1"]
  
  recs = []
  for i in range(0,len(class_colours)):
      recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
  plt.legend(recs,classes)

  plt.tight_layout()

  if savefig:
    path= f'sota_comparison_power.png'
    plt.savefig(path)
  else:
    plt.show()




def plot_instr_stat_multiworkloads(log_d, savefig= False, mode= "single_fitness_wt"):
  assert mode in ["single_fitness_wt", "multi_fitness_wt"]
  workloads= Workloads().workloads
  workloads= [w.replace('/', '_') for w in workloads]

  if mode == "single_fitness_wt":
    fig_dims = (5.5, 5.5)
    fig, ax = plt.subplots(figsize=fig_dims) 
  elif mode == "multi_fitness_wt":
    fig_dims = (5.5, 20)
    fig, ax = plt.subplots(len(workloads), figsize=fig_dims) 
  else:
    assert 0

  # border_width= 1.0
  # # thicker boundary
  # for axis in ['top','bottom','left','right']:
  #   ax.spines[axis].set_linewidth(border_width)

  category_ls= ['bb', 'initial_ld', 'intermediate_ld_st' , 'shift', 'nop']

  col_d= {cat: f'C{i}' for i, cat in enumerate(category_ls)}

  reg_bank_depth= 32
  n_banks= 64
  tree_depth= 3
  
  if mode == "single_fitness_wt":
    fitness_wt_distance_ls = [50.0]
  elif mode == "multi_fitness_wt":
    fitness_wt_distance_ls = [5.0*i*5 for i in range(1,20)]
  else:
    assert 0

  for y_idx, w in enumerate(workloads):
    for fitness_wt_distance_idx, fitness_wt_distance in enumerate(fitness_wt_distance_ls):
      n_tree = int(n_banks / (2**tree_depth))
      tup= (n_tree, tree_depth, reg_bank_depth)
      obj= log_d[tup]
      
      w_tup= (w, fitness_wt_distance)
      data_map= {}
      if w_tup in obj.map_w_to_total_instr:
        total           = obj.map_w_to_total_instr          [w_tup] 
        bb              = obj.map_w_to_bb_instr             [w_tup] 
        initial_ld      = obj.map_w_to_initial_ld_instr     [w_tup] 
        intermediate_ld = obj.map_w_to_intermediate_ld_instr[w_tup] 
        intermediate_st = obj.map_w_to_intermediate_st_instr[w_tup] 
        shift           = obj.map_w_to_shift_instr          [w_tup] 
        nop             = obj.map_w_to_nop_instr            [w_tup] 

        data_map ['bb'                 ] = bb              
        data_map ['initial_ld'         ] = initial_ld      
        data_map ['intermediate_ld_st' ] = intermediate_ld + intermediate_st
        data_map ['shift'              ] = shift
        data_map ['nop'                ] = nop

        left = 0
        for cat in category_ls:
          data = data_map[cat]
          if mode == "single_fitness_wt":
            ax.bar([y_idx] , [data], bottom= [left] , color = col_d[cat])
          elif mode == "multi_fitness_wt":
            ax[y_idx].bar([fitness_wt_distance_idx] , [data], bottom= [left] , color = col_d[cat])
          else:
            assert 0
          left += data

  # plt.yticks([0,1,2] + [4,5,6,7] + [9, 10, 11, 12], labels= tree_depth_ls + n_banks_ls + reg_bank_depth_ls)
  if mode == "single_fitness_wt":
    plt.xticks(list(range(len(workloads))), labels= workloads, rotation= 90)
  elif mode == "multi_fitness_wt":
    plt.xticks(list(range(len(fitness_wt_distance_ls))), labels= fitness_wt_distance_ls)
    # for y_idx in range(len(workload)):
    #   ax[y_idx].set_xticks(list(range(len(fitness_wt_distance_ls))), labels= xlabels)
  else:
    assert 0

  # ax.set_ylabel("")
  # ax.set_ylabel("Number of instructions")
  plt.ylabel("Number of instructions")

  # ax.minorticks_on()
  # ax.grid(which='both', linestyle= 'dotted')

  # legend
  classes = ["Main 'exec' instructions", "Compulsory loads", "Intermediate loads and stores", "Bank conflicts", "NOP bubbles"]
  class_colours = [col_d[t] for t in category_ls]
  
  recs = []
  for i in range(0,len(class_colours)):
      recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))

  if mode == "single_fitness_wt":
    plt.legend(recs,classes,ncol= 2, loc= 'upper center')
  elif mode == "multi_fitness_wt":
    pass
  else:
    assert 0
  
  plt.tight_layout()

  if savefig:
    path= f'instr_stat_multiworkload_{mode}.png'
    plt.savefig(path)
  else:
    plt.show()

def plot_instr_stat_all_w(log_d, tup, savefig, path):
  workloads= Workloads().workloads 
  workloads= [w.replace('/', '_') for w in workloads]
  workloads = reversed(workloads)

  name_w= {
    'tretail'            : 'tretail'         ,
    'mnist'              : 'mnist'           ,
    'nltcs'              : 'nltcs'           ,
    'msweb'              : 'msweb'           ,
    'msnbc'              : 'msnbc'           ,
    'bnetflix'           : 'bnetflix'        ,
    'HB_bp_200'          : 'bp_200'       , # 240
    'HB_west2021'        : 'west2021'     , # 17
    'MathWorks_Sieber'   : 'sieber',
    'HB_jagmesh4'        : 'jagmesh4'     , # Du
    'Bai_rdb968'         : 'rdb968'      , # :Du
    'Bai_dw2048'         : 'dw2048'      , # 0 B
    # 'pigs'               : 'pigs'      ,
    # 'andes'               : 'andes'      ,
    # 'munin3_spu'               : 'munin'      ,
    # 'mildew_spu'               : 'mildew'      ,
    }

  map_w_to_best_fitness_wt, _ = log_d[tup].get_best_fitness_wt_and_latency()

  data_d= {
      'workloads'       : [],
      # 'total'           : [],
      'main exec instr.'                   : [],
      'compulsory loads'                   : [],
      'loads/stores for register spilling' : [],
      'bank conflicts'                     : [],
      'no operations'                      : [],
      }

  for w in workloads:
    fitness_wt_distance = map_w_to_best_fitness_wt[w]
    tup_w_f = (w, fitness_wt_distance)

    obj= log_d[tup]

    total           = obj.map_w_to_total_instr          [tup_w_f] 
    bb              = obj.map_w_to_bb_instr             [tup_w_f] 
    initial_ld      = obj.map_w_to_initial_ld_instr     [tup_w_f] 
    intermediate_ld = obj.map_w_to_intermediate_ld_instr[tup_w_f] 
    intermediate_st = obj.map_w_to_intermediate_st_instr[tup_w_f] 
    shift           = obj.map_w_to_shift_instr          [tup_w_f] 
    nop             = obj.map_w_to_nop_instr            [tup_w_f] 

    data_d['workloads'      ].append(name_w[w])
    # data_d['total'          ].append(total          )
    data_d['main exec instr.'                  ].append(bb             /total)
    data_d['compulsory loads'                  ].append(initial_ld     /total)
    data_d['loads/stores for register spilling'].append((intermediate_ld + intermediate_st)/total)
    data_d['bank conflicts'                    ].append(shift          /total)
    data_d['no operations'                     ].append(nop            /total)

# 22527  , bb , 18832/22527  , initial_ld , 23  , intermediate_ld , 258   , intermediate_st , 258   , shift , 2301  , nop , 855  ,
# 31025  , bb , 23333/31025  , initial_ld , 8   , intermediate_ld , 1867  , intermediate_st , 1867  , shift , 1512  , nop , 2438 ,
# 83639  , bb , 53513/83639  , initial_ld , 118 , intermediate_ld , 10000 , intermediate_st , 10000 , shift , 10000 , nop , 8    ,
# 159931 , bb , 71936/159931 , initial_ld , 53  , intermediate_ld , 27800 , intermediate_st , 27800 , shift , 32837 , nop , 8    ,

#   # Large workloads
  # data_d['workloads'] += ['pigs' , 'andes', 'munin', 'mildew']
  # data_d['main exec instr.'                  ] += 
  # data_d['compulsory loads'                  ] += 
  # data_d['loads/stores for register spilling'] += 
  # data_d['bank conflicts'                    ] += 
  # data_d['no operations'                     ] += 



  df = pd.DataFrame(data_d)

  fig_dims = (4.5, 2.8)
  # plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  # sns.set_context('notebook')
  sns.set_style({'font.family' : 'STIXGeneral'})
  # sns.set_style({'font.family' : 'Helvetica'})

  df.set_index('workloads').plot(kind='barh', stacked=True, figsize= fig_dims, legend= False)

  if savefig:
    plt.savefig(path)
  else:
    plt.show()



def plot_instr_stat(log_d, savefig= False):
  # workload= 'bnetflix'
  workload= 'MathWorks_Sieber'
  # workload= 'msweb'

  tree_depth_ls = [1, 2, 3]
  n_banks_ls= [8, 16, 32, 64]
  reg_bank_depth_ls= [16, 32, 64, 128]

  fig_dims = (2.6, 2)
  # fig_dims = (5.5, 3.5)
  fig, ax = plt.subplots(figsize=fig_dims) 

  border_width= 1.0
  # thicker boundary
  for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(border_width)

  category_ls= ['bb', 'initial_ld', 'intermediate_ld_st' , 'shift', 'nop']

  my_cmap = sns.color_palette("deep")
  my_cmap= list(my_cmap.as_hex())

  # col_d= {cat: f'C{i}' for i, cat in enumerate(category_ls)}
  col_d= {cat: my_cmap[i] for i, cat in enumerate(category_ls)}
  

  mode= ['reg_bank_depth', 'n_banks', 'tree_depth']
  
  reg_bank_depth_opt= 32
  n_banks_opt= 64
  tree_depth_opt= 3

  y_idx= 0
  for m in mode:
    if m == 'tree_depth':
      reg_bank_depth = reg_bank_depth_opt
      n_banks= n_banks_opt
      iter_ls= tree_depth_ls
    elif m == 'n_banks':
      reg_bank_depth = reg_bank_depth_opt
      tree_depth = tree_depth_opt
      iter_ls= n_banks_ls
    elif m == 'reg_bank_depth':
      n_banks= n_banks_opt
      tree_depth = tree_depth_opt
      iter_ls= reg_bank_depth_ls
    else:
      assert 0

    for iter_val in iter_ls:
      # if m == 'tree_depth':
      # elif m == 'n_banks':
      # elif m == 'reg_bank_depth':
      # else:
      #   assert 0
      if m == 'tree_depth':
        tree_depth = iter_val
      elif m == 'n_banks':
        n_banks= iter_val
      elif m == 'reg_bank_depth':
        reg_bank_depth = iter_val
      else:
        assert 0
      n_tree = int(n_banks / (2**tree_depth))
      tup= (n_tree, tree_depth, reg_bank_depth)

      # map_w_to_best_fitness_wt, _ = log_d[tup].get_best_fitness_wt_and_latency()
      if tree_depth == 3:
        map_w_to_best_fitness_wt, _ = log_d[(8, 3, 32)].get_best_fitness_wt_and_latency()
        fitness_wt_distance = map_w_to_best_fitness_wt[workload]
      else:
        # map_w_to_best_fitness_wt, _ = log_d[(8, 3, 32)].get_best_fitness_wt_and_latency()
        map_w_to_best_fitness_wt, _ = log_d[tup].get_best_fitness_wt_and_latency()
        fitness_wt_distance = map_w_to_best_fitness_wt[workload]
        # fitness_wt_distance = 400.0
      # fitness_wt_distance = 125.0
      print(fitness_wt_distance)

      tup_w_f = (workload, fitness_wt_distance)

      obj= log_d[tup]

      data_map= {}
      total           = obj.map_w_to_total_instr          [tup_w_f] 
      bb              = obj.map_w_to_bb_instr             [tup_w_f] 
      initial_ld      = obj.map_w_to_initial_ld_instr     [tup_w_f] 
      intermediate_ld = obj.map_w_to_intermediate_ld_instr[tup_w_f] 
      intermediate_st = obj.map_w_to_intermediate_st_instr[tup_w_f] 
      shift           = obj.map_w_to_shift_instr          [tup_w_f] 
      nop             = obj.map_w_to_nop_instr            [tup_w_f] 

      data_map ['bb'                 ] = bb              
      data_map ['initial_ld'         ] = initial_ld      
      data_map ['intermediate_ld_st' ] = intermediate_ld + intermediate_st
      data_map ['shift'              ] = shift
      data_map ['nop'                ] = nop

      left = 0
      for cat in category_ls:
        data = data_map[cat]
        ax.barh([y_idx] , [data], left= [left] , color = col_d[cat])
        # ax.barh([y_idx] , [data], left= [left] , cmap = "tab10")
        left += data

      y_idx += 1 

    y_idx += 1
  # varying regbank depth

  # plt.yticks([0,1,2] + [4,5,6,7] + [9, 10, 11, 12], labels= tree_depth_ls + n_banks_ls + reg_bank_depth_ls)
  plt.yticks([0,1,2, 3] + [5,6,7,8] + [10, 11, 12], labels= reg_bank_depth_ls+  n_banks_ls + tree_depth_ls)

  annotations= False
  if annotations:
    ax.set_ylim(-1, 19)

    reg_bank_depth_start= 0
    n_banks_start = 5
    tree_depth_start= 10
    font_weight= 20
    font_size= 12
    x_off= -3000
    # tree_depth text
    ax.text(x_off * 2, tree_depth_start + 1.7, f"$B: {n_banks_opt}$", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    ax.text(x_off * 2, tree_depth_start + 0.3, f"$R: {reg_bank_depth}$", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    ax.text(x_off, tree_depth_start + 1, "$D:$", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    ax.text(x_off*3, tree_depth_start + 1, "(a)", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    # n_banks text
    ax.text(x_off * 2, n_banks_start + 2.2, f"$D: $ {tree_depth_opt}", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    ax.text(x_off * 2, n_banks_start + 0.8, f"$R: {reg_bank_depth}$", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    ax.text(x_off, n_banks_start + 1.5, "$B:$", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    ax.text(x_off*3, n_banks_start + 1.5, "(b)", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )

    # reg_bank_depth text
    ax.text(x_off * 2, reg_bank_depth_start + 2.2, f"$D: $ {tree_depth_opt}", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    ax.text(x_off * 2, reg_bank_depth_start + 0.8, f"$B: {n_banks_opt}$", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )
    ax.text(x_off, reg_bank_depth_start + 1.5, "$R:$", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
      )
    ax.text(x_off*3, reg_bank_depth_start + 1.5, "(c)", ha= 'center', va= 'center',
          fontweight= font_weight, fontsize= font_size
        )

  # ax.set_ylabel("")
  ax.set_xlabel("Number of instructions")

  # ax.minorticks_on()
  # ax.grid(which='both', linestyle= 'dotted')

  # legend
  classes = ["Main 'exec' instructions", "Compulsory loads", "Intermediate loads and stores", "Bank conflicts", "NOP bubbles"]
  class_colours = [col_d[t] for t in category_ls]
  
  recs = []
  for i in range(0,len(class_colours)):
      recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
  # plt.legend(recs,classes,ncol= 1, loc= 'upper center')
  
  plt.tight_layout()

  if savefig:
    path= f'instr_stat.pdf'
    plt.savefig(path)
  else:
    plt.show()


def pareto(log_d, savefig= False, mode= 'reg_bank_depth', size_mode= 'full'):
  assert mode in ['n_banks', 'reg_bank_depth', 'tree_depth']
  assert size_mode in ['full', 'inset']
  workloads= [
    'mnist',
    'nltcs',
    'msnbc',
    # 'bnetflix',
    # 'tretail',
    # 'msweb',

    'HB/bp_200', # 240 dummy nodes / 14082 total nodes
    'HB/west2021', # 174 / 18444
    'Bai/qh1484', # 2 / 19175
    'MathWorks/Sieber', # 0
    # 'Bai/dw2048', # 0

    # 'HB/jagmesh4', # Dummy nodes not needed
    # 'HB/lshp1270', # Dummy nodes not needed
    # 'HB/mahindas', #366 dummy nodes out of 36453 total nodes
    # 'Bai/rdb968', # :Dummy nodes not needed
  ]

  workloads= Workloads().workloads 

  tree_depth_ls = [1, 2, 3]
  n_banks_ls= [8, 16, 32, 64]
  reg_bank_depth_ls= [16, 32, 64, 128]

  if mode == 'tree_depth':
    col_d = {tree_depth : f'C{idx}' for idx, tree_depth in enumerate(tree_depth_ls)}
  elif mode == 'n_banks':
    col_d = {n_banks : f'C{idx}' for idx, n_banks in enumerate(n_banks_ls)}
  elif mode == 'reg_bank_depth':
    col_d = {reg_bank_depth : f'C{idx}' for idx, reg_bank_depth in enumerate(reg_bank_depth_ls)}
  else:
    assert 0
  # fig, ax = plt.subplots(figsize=fig_dims, projection= '3d') 

  map_workload_to_compute = get_sizes.get_psdd_sizes()
  map_mat_to_details = get_sizes.get_matrix_sizes()

  for key, obj in map_mat_to_details.items():
    map_workload_to_compute[key] = obj

  SMALL_SIZE = 12
  MEDIUM_SIZE = 14
  BIGGER_SIZE = 16

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

  plt.rc('legend', handlelength=0.8)  # fontsize of the figure title

  fig_dims = (4.2, 4)
  fig, ax = plt.subplots(figsize=fig_dims) 

  border_width= 1.0
  # thicker boundary
  for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(border_width)

  energy_d = defaultdict(lambda: 0)
  latency_d= defaultdict(lambda: 0)

  period= 3.4 # ns
  latency_scaling_factor= 1 # ns
  energy_scaling_factor= 1e-3 # pJ

  for tree_depth_idx, tree_depth in enumerate(tree_depth_ls):
    for n_banks_idx, n_banks in enumerate(n_banks_ls):
      for reg_bank_depth_idx, reg_bank_depth in enumerate(reg_bank_depth_ls):
        n_tree = int(n_banks / (2**tree_depth))
        n_pe = n_tree * ((2**tree_depth) - 1)

        tuple_for_key = (n_tree, tree_depth, reg_bank_depth)
        log_obj= log_d[tuple_for_key]
        for w in workloads:
          w_wo_slash = w.replace('/', '_')
          if w_wo_slash in log_obj.map_workload_to_functionality:
            # if log_obj.map_workload_to_functionality[w_wo_slash] == 'PASS':
            if log_obj.map_workload_to_latency[w_wo_slash] > 20:

              latency= log_obj.map_workload_to_latency[w_wo_slash] * period * latency_scaling_factor 
              power= log_obj.map_workload_to_total_power[w_wo_slash]

            else:
              latency= float('NaN')
              power= float('NaN')
              energy= float('NaN')
              logger.info(f"Data not available for {w}, tree_depth: {tree_depth}, n_banks: {n_banks}, reg_bank_depth: {reg_bank_depth}")
          else:
            temp_tup = (n_tree, tree_depth, 64)
            latency= log_d[temp_tup].map_workload_to_latency[w_wo_slash] * period * latency_scaling_factor
            power= log_d[temp_tup].map_workload_to_total_power[w_wo_slash]
            logger.info(f"Data not available for {w}, tree_depth: {tree_depth}, n_banks: {n_banks}, reg_bank_depth: {reg_bank_depth}, Using from smaller banks instead")

          energy = latency * power * energy_scaling_factor

          energy_per_op= energy * 1e6/ map_workload_to_compute[w_wo_slash].compute #pJ/op
          latency_per_op= latency / map_workload_to_compute[w_wo_slash].compute

          assert (latency_per_op >= (period / n_pe) ), f"latency_per_op: {latency_per_op}, {period / n_pe}, {n_pe}"
          
          energy_latency_prod= energy * latency

          energy_d[(tree_depth, n_banks, reg_bank_depth)] += (energy_per_op / len(workloads))
          latency_d[(tree_depth, n_banks, reg_bank_depth)] += (latency_per_op / len(workloads))

  x= []
  y= []
  c= []
  x_y_prod= []
  tup_ls= []
  for tup, obj in energy_d.items():
    x.append(obj)
    y.append( latency_d [ tup ] )
    if mode == 'tree_depth':
      col= col_d [ tup[0] ]
    elif mode == 'n_banks':
      col= col_d [ tup[1]]
    elif mode == 'reg_bank_depth':
      col= col_d [ tup[2] ]
    else:
      assert 0
    c.append( col )
    tup_ls.append(tup)

    x_y_prod.append( x[-1] * y [-1] )
  
  pareto_opt_obj= min(x_y_prod)
  pareto_opt_idx= x_y_prod.index( pareto_opt_obj )
  pareto_opt_tup= tup_ls[ pareto_opt_idx ]
  pareto_opt_x= x[ pareto_opt_idx ]
  pareto_opt_y= y[ pareto_opt_idx ]
  logger.info(f"Pareto optimal objective: {pareto_opt_obj}, achieved at D: {pareto_opt_tup[0]}, B: {pareto_opt_tup[1]}, R: {pareto_opt_tup[2]}")

  ax.scatter(x, y, c=c)

  # optimal point
  if mode == 'tree_depth':
    col= col_d [ pareto_opt_tup[0] ]
  elif mode == 'n_banks':
    col= col_d [ pareto_opt_tup[1]]
  elif mode == 'reg_bank_depth':
    col= col_d [ pareto_opt_tup[2] ]
  else:
    assert 0
  ax.scatter([pareto_opt_x], [pareto_opt_y], color= col , s= 100)
  if size_mode == 'full':
    ax.text(pareto_opt_x, 0.75*pareto_opt_y, f'$D$={pareto_opt_tup[0]}, $B$= {pareto_opt_tup[1]}, $R$= {pareto_opt_tup[2]}', 
        color= col, ha= 'center', va= 'top',
        fontweight= 20, fontsize= 12
        )
  elif size_mode == 'inset':
    pass
  else:
    assert 0


  # pareto curve
  x_min, x_max= ax.get_xlim()
  pareto_curve_x = np.linspace(x_min, x_max, 100)
  pareto_curve_y = pareto_opt_obj / pareto_curve_x 
  color= 'C7'
  if size_mode == 'full':
    curve_type= '-'
    text= 'min-EDP curve'
    ax.text(pareto_curve_x[-17], pareto_curve_y[-20]*1.3, text, ha= 'center', va= 'bottom', color=color,
      fontweight= 20, fontsize= 12
      )
  elif size_mode == 'inset':
    curve_type= '-'
  else:
    assert 0
  ax.plot(pareto_curve_x, pareto_curve_y, curve_type, color= color)
  # ax.text(pareto_curve_x[-20], pareto_curve_y[-20], 'EDP Pareto curve', ha= 'center', va= 'center',  backgroundcolor= ax.get_facecolor(), color=color)

  # same n_banks curve
  for tree_depth_idx, tree_depth in enumerate(tree_depth_ls):
    for n_banks_idx, n_banks in enumerate(n_banks_ls):
      x = []
      y = []
      for reg_bank_depth_idx, reg_bank_depth in enumerate(reg_bank_depth_ls):
        tup = (tree_depth, n_banks, reg_bank_depth)
        x.append (energy_d[ tup ])
        y.append (latency_d[ tup ])
      # ax.plot(x, y, ':', alpha= 0.4, color= col_d[tree_depth])

  # ax.set_xscale('log', basex=2)
  # ax.set_yscale('log', basey=2)

  # ax.set_xticks(x_data)
  # ax.set_yticks(y)


  if size_mode == 'full':
    ax.set_ylabel("Latency per operation (ns)")
    ax.set_xlabel("Energy per operation (pJ)")
  elif size_mode == 'inset':
    # ax.set_xticks([])
    # ax.set_yticks([])
    pass
  else:
    assert 0

  ax.minorticks_on()
  ax.grid(which='both', linestyle= 'dotted')

  if size_mode == 'full':
    pass
  elif size_mode == 'inset':
    y_low = 0.85 * pareto_opt_y 
    y_high= 1.45 * pareto_opt_y 
    x_low = 0.75 * pareto_opt_x 
    x_high= 1.2 * pareto_opt_x
    ax.set_ylim([y_low , y_high])
    ax.set_xlim([x_low , x_high])
    
    inset_d= {}
    for tup, x_point in energy_d.items():
      y_point= latency_d[tup]
      if x_point > x_low and x_point < x_high and\
         y_point > y_low and y_point < y_high:
        
        inset_d [tup] = (x_point, y_point)
        if tup == (3, 64, 32):
          x_point = 1.0*x_point
        if tup == (2, 64, 64):
          x_point = 1.0*x_point

        ax.text(x_point, 0.985*y_point, f'{tup[0]}, {tup[1]}, {tup[2]}', 
            # color= col, 
            ha= 'center', va= 'top',
            fontweight= 20, fontsize= 18
            )
  else:
    assert 0

  # legends
  if size_mode == 'full':
    if mode == 'tree_depth':
      classes = [f"$D$={t}" for t in tree_depth_ls]
      class_colours = [col_d[t] for t in tree_depth_ls]
    elif mode == 'n_banks':
      classes = [f"$B$={t}" for t in n_banks_ls]
      class_colours = [col_d[t] for t in n_banks_ls]
    elif mode == 'reg_bank_depth':
      classes = [f"$R$={t}" for t in reg_bank_depth_ls]
      class_colours = [col_d[t] for t in reg_bank_depth_ls]
    else:
      assert 0
    
    recs = []
    for i in range(0,len(class_colours)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
    plt.legend(recs,classes,loc=0)
  elif size_mode == 'inset':
    pass
  else:
    assert 0
  
  plt.tight_layout()

  if savefig:
    path= f'pareto_{mode}_{size_mode}.pdf'
    plt.savefig(path)
  else:
    plt.show()

def design_space_exploration_plot(log_d, savefig = False, mode= 'energy', fig_format= 'pdf'):
  assert mode in ['energy', 'latency', 'latency_per_op', 'energy_per_op', 'energy_latency_prod']
  assert fig_format in ['pdf', 'png']
  workloads= [
    'mnist',
    # 'nltcs',
    # 'msnbc',
    # 'bnetflix',
    # 'ad',
    # 'bbc',
    # 'c20ng',
    # 'kdd',
    # 'baudio',
    # 'pumsb_star',
    'Bai_qh1484',
    'MathWorks_Sieber',
    # 'HB_gemat12',
  ]

  workloads= Workloads().workloads 

  # tree_depth_ls = [3]
  tree_depth_ls = [1, 2, 3]
  n_banks_ls= [8, 16, 32, 64]
  reg_bank_depth_ls= [16, 32, 64, 128]

  col_d = {tree_depth : f'C{tree_depth - 1}' for tree_depth in tree_depth_ls}
  # fig, ax = plt.subplots(figsize=fig_dims, projection= '3d') 

  SMALL_SIZE = 12
  MEDIUM_SIZE = 14
  BIGGER_SIZE = 16

  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

  plt.rc('legend', handlelength=0.8)  # fontsize of the figure title

  fig_dims = (5, 5)
  fig= plt.figure(figsize= fig_dims)
  ax= fig.add_subplot(111, projection= '3d')
  global_min= float('inf')
  
  map_workload_to_compute = get_sizes.get_psdd_sizes()
  map_mat_to_details = get_sizes.get_matrix_sizes()

  for key, obj in map_mat_to_details.items():
    map_workload_to_compute[key] = obj

  # for w in workloads:
  #   w_wo_slash = w.replace('/', '_')
  #   print(w_wo_slash, map_workload_to_compute[w_wo_slash].compute)
  # exit(1)

  for tree_depth_idx, tree_depth in enumerate(tree_depth_ls):
    x= []
    y= []
    z= [0 for _i in n_banks_ls for _j in reg_bank_depth_ls]
    z_2d= [[0 for _j in reg_bank_depth_ls] for _i in n_banks_ls]

    x_proj= []
    y_proj= []
    x_z_proj= []
    y_z_proj= []
    
    period= 3.4 # ns
    latency_scaling_factor= 1e-6 # ms
    energy_scaling_factor= 1e3 # uJ
    for n_banks_idx, n_banks in enumerate(n_banks_ls):
      for reg_bank_depth_idx, reg_bank_depth in enumerate(reg_bank_depth_ls):
        x.append(n_banks)
        y.append(reg_bank_depth)
        n_tree = int(n_banks / (2**tree_depth))

        tuple_for_key = (n_tree, tree_depth, reg_bank_depth)
        log_obj= log_d[tuple_for_key]
        
        for w in workloads:
          w_wo_slash = w.replace('/', '_')
          if w_wo_slash in log_obj.map_workload_to_functionality:
            # if log_obj.map_workload_to_functionality[w_wo_slash] == 'PASS':
            if log_obj.map_workload_to_latency[w_wo_slash] > 20:

              latency= log_obj.map_workload_to_latency[w_wo_slash] * period * latency_scaling_factor
              # print(log_obj.map_workload_to_total_power, n_banks, reg_bank_depth)
              power= log_obj.map_workload_to_total_power[w_wo_slash]

              # z.append(energy)
              # logger.info(f"tree_depth: {tree_depth}, n_banks: {n_banks}, reg_bank_depth: {reg_bank_depth}, energy= {energy}, latency: {latency}, power: {power}")

            else:
              latency= float('NaN')
              power= float('NaN')
              energy= float('NaN')
              logger.info(f"Data FAIL for {w}, tree_depth: {tree_depth}, n_banks: {n_banks}, reg_bank_depth: {reg_bank_depth}")
          else:
            # latency= float('NaN')
            # power= float('NaN')
            # energy= float('NaN')
            temp_tup = (n_tree, tree_depth, 64)
            latency= log_d[temp_tup].map_workload_to_latency[w_wo_slash] * period * latency_scaling_factor
            power= log_d[temp_tup].map_workload_to_total_power[w_wo_slash]

            logger.info(f"Data not available for {w}, tree_depth: {tree_depth}, n_banks: {n_banks}, reg_bank_depth: {reg_bank_depth}, Using data for a smaller bank")

          energy = latency * power * energy_scaling_factor
          energy_per_op= energy * 1e6/ map_workload_to_compute[w_wo_slash].compute #pJ/op
          latency_per_op= latency * 1e6/ map_workload_to_compute[w_wo_slash].compute #ns/op
          energy_latency_prod= energy_per_op * latency_per_op

          n_pe = n_tree * ((2**tree_depth) - 1)
          assert (latency_per_op >= (period / n_pe) ), f"latency_per_op: {latency_per_op}, {period / n_pe}, {n_pe}"

          # z.append(math.log2(latency))
          z_idx= n_banks_idx * len(reg_bank_depth_ls) + reg_bank_depth_idx
          if mode == 'energy':
            z[z_idx] += energy
          elif mode == 'latency':
            z[z_idx] += latency
          elif mode == 'latency_per_op':
            z[z_idx] += (latency_per_op / len(workloads))
          elif mode == 'energy_per_op':
            z[z_idx] += (energy_per_op / len(workloads))
          elif mode == 'energy_latency_prod':
            z[z_idx] += (energy_latency_prod / len(workloads))
          else:
            assert 0

          if mode == 'energy':
            z_2d[n_banks_idx][reg_bank_depth_idx] += energy
          elif mode == 'latency':
            z_2d[n_banks_idx][reg_bank_depth_idx] += latency
          elif mode == 'latency_per_op':
            z_2d[n_banks_idx][reg_bank_depth_idx] += (latency_per_op / len(workloads))
          elif mode == 'energy_per_op':
            z_2d[n_banks_idx][reg_bank_depth_idx] += (energy_per_op / len(workloads))
          elif mode == 'energy_latency_prod':
            z_2d[n_banks_idx][reg_bank_depth_idx] += (energy_latency_prod / len(workloads))
          else:
            assert 0


        if mode == 'energy':
          reg_bank_depth_proj= 32
          n_banks_proj= 32
        elif mode == 'latency':
          reg_bank_depth_proj= 128
          n_banks_proj= 64
        elif mode == 'latency_per_op':
          reg_bank_depth_proj= 128
          n_banks_proj= 64
        elif mode == 'energy_per_op':
          reg_bank_depth_proj= 64
          n_banks_proj= 16
        elif mode == 'energy_latency_prod':
          reg_bank_depth_proj= 32
          n_banks_proj= 64
        else:
          assert 0

        if reg_bank_depth == reg_bank_depth_proj:
          x_proj.append(n_banks)
          x_z_proj.append(z_2d[n_banks_idx][reg_bank_depth_idx])

        if n_banks == n_banks_proj:
          y_proj.append(reg_bank_depth)
          y_z_proj.append(z_2d[n_banks_idx][reg_bank_depth_idx])

    min_val= min(z)
    if global_min > min_val:
      global_min = min_val
      min_idx= z.index(min_val)
      min_bank= x[min_idx]
      min_reg_bank_depth= y[min_idx]
      min_tree_depth= tree_depth 
      logger.info(f"Min point for tree_depht: {tree_depth}: banks: {min_bank}, reg_bank_depth : {min_reg_bank_depth}, z: {min_val}")

    x_log= np.log2(np.array(x))
    y_log= np.log2(np.array(y))
    z_log= np.log2(np.array(z))
    # ax.scatter(x, y, z)
    ax.scatter(x_log, y_log, z, color= col_d[tree_depth], s= 10)
    # ax.scatter(x_log, y_log, z_log, color= col_d[tree_depth])

    logger.info(f"mode, {mode}, tree_depth, {tree_depth}, z, {z}, mean, {mean(z)}")

    x_proj_log= np.log2(np.array(x_proj))
    if mode == 'energy':
      zs= 9
    elif mode == 'latency':
      zs= 8.5
    elif mode == 'latency_per_op':
      zs= 8.5
    elif mode == 'energy_per_op':
      zs= 8.5
    elif mode == 'energy_latency_prod':
      zs= 8.5
    else:
      assert 0
    ax.plot(x_proj_log, x_z_proj, ':', color= col_d[tree_depth], zdir= 'y', zs= zs, marker= 'o', markersize=3, alpha= 0.5)

    y_proj_log= np.log2(np.array(y_proj))
    if mode == 'energy':
      zs= 1.5
    elif mode == 'latency':
      zs= 2
    elif mode == 'latency_per_op':
      zs= 2
    elif mode == 'energy_per_op':
      zs= 2
    elif mode == 'energy_latency_prod':
      zs= 2
    else:
      assert 0
    ax.plot(y_proj_log, y_z_proj, ':', color= col_d[tree_depth], zdir= 'x', zs= zs, marker= 'o', markersize= 3, alpha= 0.5)

    x= np.array(n_banks_ls)
    y= np.array(reg_bank_depth_ls)
    z_2d= np.array(z_2d)
    z_2d= np.transpose(z_2d)
    x_log = np.log2(x)
    y_log = np.log2(y)
    z_log = np.log2(z_2d)

    X, Y = np.meshgrid(x_log, y_log)


    print(z_2d)
    # ax.plot_surface(x_log, y_log, z_2d)
    # ax.plot_surface(x, y, z_2d)
    # ax.plot_surface(X, Y, z_2d, cmap= 'viridis')
    ax.plot_wireframe(X, Y, z_2d, color= col_d[tree_depth], label= f'$D$ = {tree_depth}')
    # ax.plot_wireframe(X, Y, z_log, color= col_d[tree_depth])


  logger.info(f"Global min point for tree_depht: {min_tree_depth}: banks: {min_bank}, reg_bank_depth : {min_reg_bank_depth}, z: {global_min}")
  ax.scatter(np.log2([min_bank]), np.log2([min_reg_bank_depth]), [global_min], color= col_d[min_tree_depth], marker= 'D', s= 200)


  ax.set_xlabel("Banks (B)")
  ax.set_ylabel("Registers per bank (R)")

  if mode== 'energy':
    ax.set_zlabel("Energy (uJ)")
  elif mode == 'latency':
    ax.set_zlabel("Latency (ms)")
  elif mode == 'latency_per_op':
    ax.set_zlabel("Latency per operation (ns)")
  elif mode == 'energy_per_op':
    ax.set_zlabel("Energy per operation (pJ)")
  elif mode == 'energy_latency_prod':
    ax.set_zlabel("Energy-Delay Product (pJ * ns)")
  else:
    assert 0

  
  plt.xticks(x_log, labels= [str(i) for i in x])
  plt.yticks(y_log, labels= [str(i) for i in y])

  # ax.set_xticks(x_log, labels= [str(i) for i in x])
  # ax.set_yticks(y_log, labels= [str(i) for i in y])

  # ax.set_xtickslabels([str(i) for i in x])
  # ax.set_ytickslabels([str(i) for i in y])

  plt.legend()
  plt.tight_layout()

  if savefig:
    path= f'{mode}_plot.{fig_format}'
    plt.savefig(path)
  else:
    plt.show()
  

def get_instr_stat(path, log_d):
  with open(path, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  #idx
  name_idx            = 1
  tree_depth_idx      = 3
  n_banks_idx         = 5
  reg_bank_depth_idx  = 7
  total_idx           = 9
  bb_idx              = 11
  initial_ld_idx      = 13
  intermediate_ld_idx = 15
  intermediate_st_idx = 17
  shift_idx           = 19
  nop_idx             = 21
  fitness_wt_distance_idx= 23

  for d in data:
    if len(d) < fitness_wt_distance_idx:
      continue

    name                = d[name_idx].strip()
    tree_depth          = int(d[tree_depth_idx     ])
    n_banks             = int(d[n_banks_idx        ])
    reg_bank_depth      = int(d[reg_bank_depth_idx ])
    total               = int(d[total_idx        ])
    bb              = int(d[bb_idx             ])
    initial_ld      = int(d[initial_ld_idx     ])
    intermediate_ld = int(d[intermediate_ld_idx])
    intermediate_st = int(d[intermediate_st_idx])
    shift           = int(d[shift_idx          ])
    nop             = int(d[nop_idx            ])
    fitness_wt_distance= float(d[fitness_wt_distance_idx])

    assert total == bb + initial_ld + intermediate_st + intermediate_ld + shift + nop

    n_tree = int(n_banks / (2**tree_depth))
    tuple_for_key = (n_tree, tree_depth, reg_bank_depth)

    if tuple_for_key in log_d:
      obj= log_d[tuple_for_key]
      tup = (name, fitness_wt_distance)
      obj.map_w_to_total_instr          [tup] = total
      obj.map_w_to_bb_instr             [tup] = bb
      obj.map_w_to_initial_ld_instr     [tup] = initial_ld
      obj.map_w_to_intermediate_ld_instr[tup] = intermediate_ld
      obj.map_w_to_intermediate_st_instr[tup] = intermediate_st
      obj.map_w_to_shift_instr          [tup] = shift
      obj.map_w_to_nop_instr            [tup] = nop
    else:
      pass
      # logger.info(f"n_tree, tree_depth_idx, reg_bank_depth of : {tuple_for_key} is not available in log_d")

def failed_netlists(paths, log_d, write_files):
  # workload= 'HB_west2021'
  # workload= 'MathWorks_Sieber'
  # workload= 'Bai_qh1484'
  # workload= 'HB_gemat12'

  workloads= Workloads()

  massive_parallel= True
  
  all_p_ls= []
  for tuple_for_key , obj in log_d.items():
    target_workloads= []
    target_workloads_1= []
    for w_1 in workloads.workloads:
      w = w_1.replace( '/', '_')
      if w in obj.map_workload_to_functionality:
        n_tree, tree_depth, reg_bank_depth = tuple_for_key
        min_depth= min(2, tree_depth)
        hw = HwConf(n_tree, tree_depth, min_depth, reg_bank_depth)
        # if obj.map_workload_to_functionality[w] == 'FAIL':
        if obj.map_workload_to_latency[w] < 20:
          logger.info(f"Needs reevaulation for tuple {tuple_for_key}, {w}")
          # hw.print_details()

          # edit_sv_rtl_file(paths, hw, write_files, mode= "syn")
          # synthesis(paths, hw, write_files)

          # edit_sv_rtl_file(paths, hw, write_files, mode= "sim")
          # gen_activity(paths, hw, write_files)
          
          target_workloads.append(w)
          target_workloads_1.append(w_1)
          EXEC= True
        else:
          # logger.info(f"{tuple_for_key}, {w} simulations passed. {obj.map_workload_to_functionality[w]}, {obj.map_workload_to_latency[w]}")
          pass
      else:
        logger.info(f"{w} not in log file for tuple: {tuple_for_key}, {w}, {obj.map_workload_to_latency[w]}")
        logger.info(f"Needs reevaulation for tuple {tuple_for_key}")
        # hw.print_details()
        target_workloads.append(w)
        target_workloads_1.append(w_1)

    if len(target_workloads) != 0:
      logger.info(f"target_workloads : {target_workloads}")
      # edit_sv_rtl_file(paths, hw, write_files, mode= "sim")
      # gen_activity(paths, hw, write_files, workloads= target_workloads)

      # p_ls= gen_instructions(paths, hw, write_files, mode = 'all', massive_parallel= massive_parallel, workloads= target_workloads_1)
      # all_p_ls += p_ls

      # power_estimation(paths, hw, write_files, workloads= target_workloads)

    if massive_parallel:
      print('lenght all_p_ls:', len(all_p_ls))
      if len(all_p_ls) > 20:
        exit_codes= [p.wait() for p in all_p_ls]
        all_p_ls= []

  if massive_parallel:
    exit_codes= [p.wait() for p in all_p_ls]

def plot_impact_of_reg_bank_depth(log_d, savefig= False):
  workloads= [
    # 'mnist',
    # 'nltcs',
    'msnbc',
    # 'bnetflix',
    # 'ad',
    # 'bbc',
    # 'c20ng',
    # 'kdd',
    # 'baudio',
    # 'pumsb_star',
  ]

  n_banks = 128
  tree_depth= 2
  n_tree = int(n_banks / (2**tree_depth))

  reg_bank_depth_ls = [16, 32, 64, 128]

  y_data= {w: [] for w in workloads}

  for w in workloads:
    y_curr= []
    for reg_bank_depth in reg_bank_depth_ls:
      tup= (n_tree , tree_depth, reg_bank_depth)

      print(tup, log_d[tup].map_workload_to_latency)
      lat= log_d[tup].map_workload_to_latency[w]
      logger.info(f"{tup, w, lat}")
      y_curr.append(lat)

    y_curr= [y/y_curr[0] for y in y_curr]
    y_data[w] = y_curr

  fig_dims = (4, 4)
  fig, ax = plt.subplots(figsize=fig_dims) 

  border_width= 1.0
  # thicker boundary
  for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(border_width)

  x_data= reg_bank_depth_ls
  for w in workloads:
    ax.plot(x_data, y_data[w], label= w)

  ax.set_ylabel("Normalized latency")
  ax.set_xlabel("Register bank depth")

  # ax.set_xscale('log', basex=2)
  # ax.set_yscale('log', basey=2)

  ax.set_xticks(x_data)
  # ax.set_yticks(y)

  ax.minorticks_on()
  ax.grid(which='both', linestyle= 'dotted')

  plt.legend()
  plt.tight_layout()

  if savefig:
    path= f'latency_scaling_w_reg_bank_depth_{n_banks}.png'
    plt.savefig(path)
  else:
    plt.show()

  

def plot_impact_of_depth(log_d, savefig= False):
  workloads= [
    'mnist',
    # 'nltcs',
    # 'msnbc',
    # 'bnetflix',
    # 'ad',
    # 'bbc',
    # 'c20ng',
    # 'kdd',
    # 'baudio',
    # 'pumsb_star',
  ]
  
  n_banks_ls= [16, 32, 64, 128]
  tree_depth_ls= [1,2,3, 4]
  
  reg_bank_depth= 128
  
  y_data= {depth: {w: [] for w in workloads} for depth in tree_depth_ls}

  max_latency= 0
  for depth in tree_depth_ls:
    for w in workloads:
      y_curr= []
      for n_banks in n_banks_ls:
        n_tree = int(n_banks / (2**depth))
        tup= (n_tree , depth, reg_bank_depth)
        
        print(tup, log_d[tup].map_workload_to_latency)
        lat= log_d[tup].map_workload_to_latency[w]
        logger.info(f"{tup, w, lat}")

        if lat > max_latency:
          max_latency = lat

        y_curr.append(lat)

      # y_curr= [y/y_curr[0] for y in y_curr]
      y_data[depth][w] = y_curr

  # normalize
  max_latencies= {w: y_data[1][w][0] for w in workloads}
  for depth in tree_depth_ls:
    for w in workloads:
      # y_data[depth][w]= [y/max_latency for y in y_data[depth][w]]
      y_data[depth][w]= [y/max_latencies[w] for y in y_data[depth][w]]
  
  # Actual plotting
  fig_dims = (4, 4)
  fig, ax = plt.subplots(figsize=fig_dims) 

  border_width= 1.0
  # thicker boundary
  for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(border_width)

  x_data= n_banks_ls
  col_d= {1 : 'C0', 2: 'C1', 3:'C2', 4:'C3', 5: 'C4'}
  for depth in tree_depth_ls:
    for w in workloads:
      ax.plot(x_data, y_data[depth][w], c= col_d[depth])

  ax.set_ylabel("Normalized latency")
  ax.set_xlabel("Number of banks")

  # ax.set_xscale('log', basex=2)
  # ax.set_yscale('log', basey=2)

  ax.set_xticks(x_data)
  # ax.set_yticks(y)

  ax.minorticks_on()
  ax.grid(which='both', linestyle= 'dotted')

  plt.legend()
  plt.tight_layout()

  if savefig:
    path= 'latency_scaling_w_depth_mnist.png'
    plt.savefig(path)
  else:
    plt.show()


def plot_latency(log_d, savefig= False):
  workloads= [
    'mnist',
    'nltcs',
    'msnbc',
    'bnetflix',
    'ad',
    # 'bbc',
    # 'c20ng',
    # 'kdd',
    # 'baudio',
    # 'pumsb_star',
  ]
  # workload_mode = "individual"
  workload_mode = "individual_one_plot"
  # workload_mode = "average"

  x_axis= "n_tree"
  y_axis= "latency"

  n_tree_ls = [2, 4, 8, 16]
  tree_depth_ls= [2]

  reg_bank_depth_ls= [32]
  reg_bank_size= 8*128 # number of registers
  # reg_bank_mode= "fixed_size"
  reg_bank_mode= "fixed_banks"

  # gen data
  if x_axis == "n_tree":
    x_data= n_tree_ls

  y_data= []
  for tree_depth in tree_depth_ls:
    for reg_bank_depth in reg_bank_depth_ls:
      for w in workloads:
        y_curr= []
        for n_tree in n_tree_ls:
          if reg_bank_mode == 'fixed_size':
            n_banks = n_tree * (2**tree_depth)
            reg_bank_depth = int(reg_bank_size/n_banks)

          obj=  log_d[(n_tree, tree_depth, reg_bank_depth)]
          print(n_tree, tree_depth, reg_bank_depth)
          print(obj.map_workload_to_latency)
          # if obj.area_instr_mem == None:
          #   obj.area_instr_mem = 1333750
          # if obj.area_data_mem == None:
          #   obj.area_data_mem = 903750
          # area_wo_mem = obj.total_area - obj.area_data_mem - obj.area_instr_mem
          # print(obj.total_area)
          # print(area_wo_mem)

          lat= obj.map_workload_to_latency[w]
          # area_delay= lat * area_wo_mem
          y_curr.append(lat)
          # y_curr.append(area_delay)
        y_curr= [y/y_curr[0] for y in y_curr]
        y_data.append(y_curr)

  fig_dims = (3.4, 2.6)
  fig, ax = plt.subplots(figsize=fig_dims) 

  border_width= 1.0
  # thicker boundary
  for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(border_width)

  for idx, y_curr in enumerate(y_data):
    ax.plot(x_data, y_curr, label= workloads[idx])

  ax.set_ylabel("Normalized latency")
  ax.set_xlabel("Number of trees")

  ax.set_xticks(x_data)
  # ax.set_yticks(y)

  ax.minorticks_on()
  ax.grid(which='both', linestyle= 'dotted')

  plt.legend()
  plt.tight_layout()

  if savefig:
    path= 'latency_scaling_w_trees_2.png'
    plt.savefig(path)
  else:
    plt.show()


def area_estimation(paths, hw, write_files, log_obj):
  area_rpt= f"{paths.get_volume1_reports_path_prefix(hw)}_area.rpt"
  logger.info(f"Reading area report: {area_rpt}")

  with open(area_rpt, "r") as fp:
    lines= fp.readlines()
    # print(lines)

  for l in lines:
    l = l.strip()
    if l.startswith('pru_sync'):
      ls = l.split()
      total_area = float(ls[-1])
      log_obj.total_area= total_area
      # logger.info(f"total_area: {total_area}")
    
    if l.startswith('data_mem '):
      ls = l.split()
      log_obj.area_data_mem = float(ls[-1])
      # logger.info(f"total_area: {total_area}")

    if l.startswith('instr_mem '):
      ls = l.split()
      log_obj.area_instr_mem = float(ls[-1])

def get_power(paths, hw, log_obj):
  workloads= Workloads()

  for w in workloads.workloads:
    rpt_path= paths.get_power_with_activity_path(hw, w)
    # logger.info(f"Reading power report: {rpt_path}")
    
    w= w.replace('/', '_')
    try:
      with open(rpt_path, "r") as fp:
        lines= fp.readlines()
    except FileNotFoundError:
      lines= []
      # print(lines)

    for l in lines:
      l = l.strip()
      if l.startswith('memory'):
        ls = l.split()
        val = float(ls[-2])
        log_obj.map_workload_to_mem_power[w]= val
        # logger.info(f"total_area: {total_area}")
      
      if l.startswith('register'):
        ls = l.split()
        val = float(ls[-2])
        log_obj.map_workload_to_flipflop_power[w]= val

      if l.startswith('Subtotal'):
        ls = l.split()
        val = float(ls[1])
        log_obj.map_workload_to_leakage_power[w]= val
        val = float(ls[2])
        log_obj.map_workload_to_internal_power[w]= val
        val = float(ls[3])
        log_obj.map_workload_to_switching_power[w]= val
        val = float(ls[4])
        log_obj.map_workload_to_total_power[w]= val


def get_latency(paths, log_d, mode= 'post_rtl_sim'):
  assert mode in ['post_rtl_sim', 'post_compile']

  if mode == 'post_rtl_sim':
    path= paths.TB_LATENCY_NETLIST_REPORT

    with open(path, 'r') as fp:
      data = csv.reader(fp, delimiter=',')
      data= list(data)

    #idx
    name_idx= 1
    n_tree_idx         = 3
    tree_depth_idx     = 5
    n_banks_idx        = 7
    reg_bank_depth_idx = 9
    latency_idx        = 19

    for d in data:
      if len(d) < latency_idx:
        logger.info("weird line")
        continue

      name           = d[name_idx].strip()
      n_tree         = int(d[n_tree_idx         ])
      tree_depth     = int(d[tree_depth_idx     ])
      n_banks        = int(d[n_banks_idx        ])
      reg_bank_depth = int(d[reg_bank_depth_idx ])
      latency        = int(float(d[latency_idx        ]))

      tuple_for_key = (n_tree, tree_depth, reg_bank_depth)

      if tuple_for_key in log_d:
        obj= log_d[tuple_for_key]
        obj.map_workload_to_latency[name] = latency
      else:
        logger.info(f"n_tree, tree_depth_idx, reg_bank_depth of : {tuple_for_key} is not available in log_d")

  elif mode == 'post_compile':
    path= paths.INSTR_STAT_PATH

    with open(path, 'r') as fp:
      data = csv.reader(fp, delimiter=',')
      data= list(data)  

    #initialize latency to 'inf'
    for _, obj in log_d.items():
      obj.map_workload_to_latency = defaultdict(lambda : float('inf'))

    #idx
    name_idx            = 1
    tree_depth_idx      = 3
    n_banks_idx         = 5
    reg_bank_depth_idx  = 7
    total_idx           = 9
    bb_idx              = 11
    initial_ld_idx      = 13
    intermediate_ld_idx = 15
    intermediate_st_idx = 17
    shift_idx           = 19
    nop_idx             = 21
    fitness_wt_distance_idx= 23

    for d in data:
      if len(d) < fitness_wt_distance_idx:
        continue

      name                = d[name_idx].strip()
      tree_depth          = int(d[tree_depth_idx     ])
      n_banks             = int(d[n_banks_idx        ])
      reg_bank_depth      = int(d[reg_bank_depth_idx ])
      total               = int(d[total_idx        ])
      bb              = int(d[bb_idx             ])
      initial_ld      = int(d[initial_ld_idx     ])
      intermediate_ld = int(d[intermediate_ld_idx])
      intermediate_st = int(d[intermediate_st_idx])
      shift           = int(d[shift_idx          ])
      nop             = int(d[nop_idx            ])
      fitness_wt_distance= float(d[fitness_wt_distance_idx])
      
      assert total == bb + initial_ld + intermediate_st + intermediate_ld + shift + nop

      n_tree = int(n_banks / (2**tree_depth))
      tuple_for_key = (n_tree, tree_depth, reg_bank_depth)

      if tuple_for_key in log_d:
        obj= log_d[tuple_for_key]
        if total < obj.map_workload_to_latency[name]:
          obj.map_workload_to_latency[name] = total
      else:
        pass
        # logger.info(f"n_tree, tree_depth_idx, reg_bank_depth of : {tuple_for_key} is not available in log_d")

    # Assert
    workloads= Workloads().workloads
    workloads= [w.replace('/', '_') for w in workloads]
    for tuple_for_key, obj in log_d.items():
      for w in workloads:
        assert w in obj.map_workload_to_latency, f"paths.INSTR_STAT_PATH does not contain a log for {w} for {tuple_for_key} configuration"
  else:
    assert 0

    
def get_functionality(path, log_d):
  with open(path, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  #idx
  name_idx= 1
  n_tree_idx         = 3
  tree_depth_idx     = 5
  n_banks_idx        = 7
  reg_bank_depth_idx = 9
  functionality_idx  = 18

  for d in data:
    if len(d) < functionality_idx:
      logger.info("weird line")
      continue

    name           = d[name_idx].strip()
    n_tree         = int(d[n_tree_idx         ])
    tree_depth     = int(d[tree_depth_idx     ])
    n_banks        = int(d[n_banks_idx        ])
    reg_bank_depth = int(d[reg_bank_depth_idx ])
    functionality  = d[functionality_idx].strip()

    tuple_for_key = (n_tree, tree_depth, reg_bank_depth)

    if tuple_for_key in log_d:
      obj= log_d[tuple_for_key]
      obj.map_workload_to_functionality[name] = functionality
    else:
      logger.info(f"configuration: {tuple_for_key} not available in log_d")

def power_estimation(paths, hw, write_files, workloads= [], netlist_path= None):
  os.system(f"source {paths.BACKEND_SOURCE_FILE}")

  fname= paths.BACKEND_POWER_SCRIPT
  phrase= "read_netlist"
  ln= find_line_number(fname, phrase)
  assert len(ln) == 1, f"Only one of the lines should contain the phrase, {ln}"
  ln= ln.pop()
  
  if netlist_path == None:
    val= paths.get_volume1_netlist_path(hw)
  else:
    val= netlist_path

  replace_val(fname, phrase, val, ln, write_files)
  logger.info(f"Netlist used for power estimation: {val}")

  phrase= "read_saif"
  ln= find_line_number(fname, phrase)
  assert len(ln) == 1, f"Only one of the lines should contain the phrase, {ln}"
  ln= ln.pop()
  
  workloads_obj= Workloads()
  if len(workloads) == 0:
    workloads= workloads_obj.workloads
  
  for w in workloads:
    val = f"{paths.VOLUME1_ACTIVITY_PATH}{paths.DESIGN_NAME}_{hw.suffix()}_{w.replace( '/' , '_' )}_activity.saif"
    replace_val(fname, phrase, val, ln, write_files)
    logger.info(f"activty used for power estimation: {val}")

    p = subprocess.Popen(["make", "pwr"], cwd=paths.BACKEND_POWER_PATH)
    exit_code= p.wait()
    
    logger.info(f"Exit code of power_estimation: {exit_code} {w}")

    if not exit_code:
      src= f"{paths.BACKEND_POWER_REPORT}"
      dst = paths.get_power_with_activity_path(hw, w)
      cmd= f"cp -f {src} {dst}"
      os.system(cmd)
      logger.info(f"power report available at : {dst}")


def replace_val(fname, phrase, val, ln, write_files):
  with open(fname, 'r') as fp:
    lines = fp.readlines()
  
  assert phrase in lines[ln]
  lines[ln] = f"{phrase} {val}\n"

  if write_files:
    with open(fname, 'w') as fp:
      fp.writelines(lines)

def write_phrase(fname, phrase, ln, write_files):
  with open(fname, 'r') as fp:
    lines = fp.readlines()
  
  lines[ln] = phrase

  if write_files:
    with open(fname, 'w') as fp:
      fp.writelines(lines)


def gen_instructions(paths, hw, write_files, mode= 'all', massive_parallel= False, workloads= []):
  assert mode in ['all', 'hw_tree_blocks', 'scheduling_for_gather', 'generate_binary_executable', 'schedule_and_generate']

  workloads_obj= Workloads()
  if len(workloads) == 0:
    workloads= workloads_obj.workloads
  
  p_ls= []
  for w in workloads:
    w_wo_slash = w.replace('/', '_')
    NUM_REG_BANKS= hw.N_TREE * (2**(hw.TREE_DEPTH))
    NUMBER_OF_INPUTS_log = clog2(NUM_REG_BANKS)
    fitness_wt_distance= hw.fitness_wt_distance_d[w_wo_slash]
    assert fitness_wt_distance != None
    p = subprocess.Popen(["make", f"{mode}", f"NET={w}", f"TREE_DEPTH={hw.TREE_DEPTH}", f"MIN_DEPTH={hw.MIN_DEPTH}", f"REG_BANK_DEPTH={hw.REG_BANK_DEPTH}", f"NUMBER_OF_INPUTS_log={NUMBER_OF_INPUTS_log}", f"CIR_TYPE ={ workloads_obj.get_cir_type( w ) }", f"FITNESS_WT_DISTANCE_SCALE={fitness_wt_distance}"], cwd=paths.INSTR_GEN_PATH)
    p_ls.append(p)

  if not massive_parallel:
    exit_codes= [p.wait() for p in p_ls]

  return p_ls
  
def gen_activity(paths, hw, write_files, workloads= [], mode="netlist_activity"):
  assert mode in ["netlist_activity", "netlist_non_activity", "rtl"]
  workloads_obj= Workloads()
  if len(workloads) == 0:
    workloads = workloads_obj.workloads

  if mode == 'netlist_activity' or mode == 'netlist_non_activity':
    p = subprocess.Popen(["make", "compile_gate"], cwd=paths.TB_PATH)
    exit_code= p.wait()
  elif mode == 'rtl':
    p = subprocess.Popen(["make", "compile"], cwd=paths.TB_PATH)
    exit_code= p.wait()
  else:
    assert 0
  
  logger.info(f"Exit code of gen_activity compile: {exit_code}")

  if not exit_code:
    for w in workloads:
      w_wo_slash = w.replace('/', '_')
      fitness_wt_distance= hw.fitness_wt_distance_d[w_wo_slash]
      p = subprocess.Popen(["make", "run", f"NET={w_wo_slash}", f"W_CONFLICT={int(fitness_wt_distance)}"], cwd=paths.TB_PATH)
      exit_code= p.wait()
      
      logger.info(f"Exit code of gen_activity run: {exit_code} {w}")

      if mode == 'netlist_activity':
        if not exit_code:
          src= f"{paths.TB_ACTIVITY_PATH}"
          dst = f"{paths.VOLUME1_ACTIVITY_PATH}{paths.DESIGN_NAME}_{hw.suffix()}_{w_wo_slash}_activity.saif"
          cmd= f"cp -f {src} {dst}"
          logger.info(f"Executing command: {cmd}")
          os.system(cmd)
      elif mode == 'netlist_non_activity':
        pass
      elif mode == 'rtl':
        pass
      else:
        assert 0


def synthesis(paths, hw, write_files):
  # launch synthesis
  p = subprocess.Popen(["make", "syn"], cwd=paths.BACKEND_SYN_PATH)
  exit_code= p.wait()
  
  logger.info(f"Exit code of synthesis: {exit_code}")

  if not exit_code:
    logger.info("Copying netlist and reports")
    # copy reports and netlist to volume1 and rename them appropriately
    src= f"{paths.BACKEND_NETLIST_PATH}{paths.SYNTHESIS_REPORTS_PREFIX}_compiled.v"
    dst = paths.get_volume1_netlist_path(hw)
    cmd= f"cp -f {src} {dst}"
    os.system(cmd)

    report_names= [
      "qor",
      "timing",
      "timing_summary",
      "area",
      "datapath_incr",
      "messages",
      "gates",
      "power"
    ]

    for n in report_names:
      src= f"{paths.BACKEND_SYN_REPORTS_PATH}{paths.SYNTHESIS_REPORTS_PREFIX}_{n}.rpt"
      dst = f"{paths.VOLUME1_NETLIST_PATH}{paths.DESIGN_NAME}_{hw.suffix()}_{n}.rpt"
      cmd= f"cp -f {src} {dst}"
      os.system(cmd)

    src= f"{paths.BACKEND_SDF_PATH}{paths.SYNTHESIS_REPORTS_PREFIX}_timing.sdf"
    dst = f"{paths.VOLUME1_SDF_PATH}{paths.DESIGN_NAME}_{hw.suffix()}_timing.sdf"
    cmd= f"cp -f {src} {dst}"
    os.system(cmd)

  else:
    logger.info("NOT copying netlist and reports")

def edit_sv_rtl_file(paths, hw, write_files, mode= 'syn'):
  assert mode in ["syn", "sim"]
  if mode == "syn":
    fname= f"{paths.SRC_RTL_PATH}common_pkg.sv"
  elif mode == "sim":
    fname= f"{paths.SRC_RTL_PATH}common_pkg_SCRIPT.sv"
  else:
    assert 0
  edit_sv_file_macro(fname, "TREE_DEPTH", hw.TREE_DEPTH, write_files)
  edit_sv_file_macro(fname, "N_TREE", hw.N_TREE, write_files)
  edit_sv_file_macro(fname, "REG_BANK_DEPTH", hw.REG_BANK_DEPTH, write_files)
  edit_sv_file_macro(fname, "MIN_DEPTH", hw.MIN_DEPTH, write_files)

  if mode == "sim":
    fname= f"{paths.TB_PATH}common_tb.sv"
    phrase= '`include "/esat/puck1/users'
    ln= find_line_number(fname, phrase)
    assert len(ln) == 1, f"Only one of the lines should contain the phrase, {ln}"
    ln= ln.pop()

    phrase= f'  `include "{paths.get_volume1_netlist_path(hw)}"\n'
    write_phrase(fname, phrase, ln, write_files)


def edit_sv_file_macro(fname, macro, val, write_files):
  phrase= f"`define {macro}"
  ln= find_line_number(fname, phrase)
  assert len(ln) == 1, f"Only one of the lines should contain the phrase, {ln}"
  ln= ln.pop()

  phrase= f"  `define {macro}"
  replace_val(fname, phrase, val, ln, write_files)

def find_line_number(fname, phrase):
  line_numbers= []
  with open(fname, 'r') as fp:
    for num, line in enumerate(fp):
      if phrase in line:
        line= line.strip()
        if line.startswith(phrase): # phrase should be at the beginning of the line, avoids commented lines
          line_numbers.append(num)
  
  return line_numbers

if __name__ == "__main__":
  pru_sync()
  # pru_async_top()
