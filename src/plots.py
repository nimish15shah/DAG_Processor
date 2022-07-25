import csv
import pandas as pd
from statistics import mean
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class Matrix_info():
  def __init__(self, name):
    self.name= name
    self.n_compute= None
    self.n_cols= None

    self.max_throughput= 0
    self.th_with_max_throughput= None
    self.n_layers_for_max_throughput= None
    self.acceleration_factor= None

    self.intel_mkl_throughput = None
    self.nvidia_cusparse_throughput= None
    self.max_layer_wise_throughput= None
    self.pru_throughput= None
    self.suitesparse_cxsparse_throughput= None
    self.suitesparse_umfpack_throughput= None
    self.kokkos_throughput= None
    self.p2p_throughput= None
    self.juice_throughput= None

    self.map_th_to_throughput= {}
    self.map_th_to_layers= {}

def get_data(fname, with_single_thread= True):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)
  # idx
  name_idx= 0
  th_0_idx= 1
  th_1_idx= 5
  n_compute_idx= 9
  throughput_idx= 13
  res_idx= 15
  golden_idx= 17
  n_layers_idx= 3

  # key: (name, th) tuple
  # val: [n_compute, throughput]

  compile_error_matrices= []
  improper_res_matrices= []
  data_d= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue

    name= d[name_idx]
    th= int(d[th_0_idx])
    
    # compilation error
    if len(d) < throughput_idx:
      compile_error_matrices.append((name, th))
      continue
    
    # some matrices have wrong c files
    if th != int(d[th_1_idx]):
      improper_res_matrices.append((name, th))
      print(f"Improper threads: {name}, {th}")
      continue

    # matrices with wrong results
    res= float(d[res_idx]) 
    golden= float(d[golden_idx]) 
    if abs(res-golden) > abs(0.1 * golden) and golden != 0:
      improper_res_matrices.append((name, th))
      print(f"High error: {name}, {th}, {res}, {golden}, {abs(res-golden)}, {0.1*golden}")

    n_compute = float(d[n_compute_idx])
    throughput = float(d[throughput_idx])
    n_layers = int(d[n_layers_idx])

    data_d[(name, th)] = [n_compute, throughput, n_layers]
  
  # print(list(set([t[0] for t in compile_error_matrices])))

  # check that all threads have same n_compute
  net_with_no_1_thread= set()
  if with_single_thread:
    for name, th in data_d.keys():
      if (name,1) in data_d:
        if data_d[(name,1)][0] != data_d[(name, th)][0]:
          print("Improper n_compute in {name, th, data_d[(name, th)], data_d[(name, 1)]}")
      else:
        net_with_no_1_thread.add(name)


  all_matrices= set([name for name,_ in data_d.keys()])
  # key: name
  # value: th
  mat_detail_d= {name: Matrix_info(name) for name in all_matrices if name not in net_with_no_1_thread}

  # find max throughput tuples
  for name, th in data_d.keys():
    if name not in mat_detail_d:
      continue
    mat_obj= mat_detail_d[name]
    n_compute, throughput, n_layers = data_d[(name, th)]
    
    mat_obj.n_compute = n_compute

    mat_obj.map_th_to_throughput[th] = throughput
    mat_obj.map_th_to_layers[th] = n_layers

    assert isinstance(throughput, float)

    if throughput > mat_obj.max_throughput:
      mat_obj.max_throughput= throughput
      mat_obj.th_with_max_throughput= th
      mat_obj.n_layers_for_max_throughput= n_layers

  for mat, obj in mat_detail_d.items():
    assert obj.max_throughput != 0

  # acceleration
  if with_single_thread:
    for name, obj in mat_detail_d.items():
      obj.acceleration_factor= obj.max_throughput/ obj.map_th_to_throughput[1]

    # filter matrices
    # mat_detail_d= {n:obj for n, obj in mat_detail_d.items() if obj.acceleration_factor > 1}

  # sorted_be_acceleration= sorted(list(mat_detail_d.keys()), key= lambda x : mat_detail_d[x].acceleration_factor, reverse= True)
  
  print(f"Number of matrices with >1 acceleration fator {len(mat_detail_d)}, out of {len(all_matrices)}")
  print(f"Total workloads in dict: {len(mat_detail_d)}")

  # for name in sorted_be_acceleration:
  #   print(name, mat_detail_d[name].acceleration_factor, mat_detail_d[name].max_throughput, mat_detail_d[name].th_with_max_throughput)

  return mat_detail_d, compile_error_matrices, improper_res_matrices

def get_pru_throughput_data(fname, matrix_or_psdd= 'matrix'):
  assert matrix_or_psdd in ['matrix', 'psdd']
  if matrix_or_psdd == 'matrix':
    map_mat_to_details= get_sizes.get_matrix_sizes()
  elif matrix_or_psdd == 'psdd':
    map_mat_to_details= get_sizes.get_psdd_sizes()
  else:
    assert 0

  map_mat_to_cycles= read_pru_sim_file(fname)

  freq= 280 #MHz
  map_mat_to_throughput= {}
  for mat in map_mat_to_cycles.keys():
    cycles= map_mat_to_cycles[mat]
    compute= map_mat_to_details[mat].compute
    map_mat_to_throughput[mat]= compute*freq/cycles

  return map_mat_to_throughput

def get_gpu_data(fname):
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  # idx
  name_idx= 0
  throughput_idx= 13
  col_idx= 4
  compute_idx= 6
  map_mat_to_throughput= {}
  map_mat_to_cols_compute= {}
  for d in data:
    if len(d) == 1:
      assert d[0] == 'Start'
      continue
    name= d[name_idx]
    throughput= float(d[throughput_idx])
    cols= int(d[col_idx])
    compute= int(d[compute_idx])
    map_mat_to_throughput[name]= throughput
    map_mat_to_cols_compute[name] = (cols, compute)

  sorted_by_acceleration = sorted(list(map_mat_to_throughput.keys()), key= lambda x:map_mat_to_throughput[x], reverse= True)
  print([(x, map_mat_to_throughput[x]) for x in sorted_by_acceleration[:20]])

  return map_mat_to_throughput, map_mat_to_cols_compute

def get_gpu_cuda_data(fname, target_th, target_batch):
  """
    Only looks at batch=1 data
  """
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  name_idx= 0
  n_threads_idx= 2
  batch_sz_idx= 3
  throughput_idx= 6 # GOPS

  map_psdd_to_throughput= {}
  
  for d in data:
    if d[0][0] == '#':
      continue
    th = int(d[n_threads_idx])
    batch = int(d[batch_sz_idx])

    if th== target_th and batch == target_batch:
      name= d[name_idx]
      throughput= float(d[throughput_idx])

      map_psdd_to_throughput[name]= throughput * 1e3 # MOPS
  
  return map_psdd_to_throughput
    
def throughput_bar_plot_DPUv2(savefig, path):
  system= []
  throughput= []
  workload= []

  target_psdd= [
    'tretail',
    'mnist',
    'nltcs',
    'msweb',
    'msnbc',
    'bnetflix',
      ]
  # cpu_openmp
  fname= "/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/openmp/run_log_psdd_two_way_limit_O3_eridani"
  psdd_detail_d, _, _= get_data(fname, with_single_thread= False)
  
  for w in target_psdd:
    system.append('CPU')
    throughput.append(psdd_detail_d[w].max_throughput/1e3)
    workload.append(w)

  # gpu_cuda
  fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/cuda/gpu_compile/run_log_drive.csv'
  map_psdd_to_throughput_gpu= get_gpu_cuda_data(fname, 512, 1)
  for w in target_psdd:
    system.append('GPU')
    throughput.append(map_psdd_to_throughput_gpu[w]/1e3)
    workload.append(w)

  # SpTRSV
  target_mat = [
    'HB_bp_200', # 240 dummy nodes / 14082 total nodes # HB_bp_200              , nnzA , 4614     , colsA , 822    , n_compute , 8406      ,critical_path_len, 46 
    'HB_west2021', # 174 / 18444 # HB_west2021            , nnzA , 6090     , colsA , 2021   , n_compute , 10159     ,critical_path_len, 44   
    'MathWorks_Sieber', # 0 # MathWorks_Sieber       , nnzA , 12529    , colsA , 2290   , n_compute , 22768     ,critical_path_len, 80 
    'HB_jagmesh4', # Dummy nodes not needed # HB_jagmesh4            , nnzA , 22600    , colsA , 1440   , n_compute , 43760     ,critical_path_len, 215  
    'Bai_rdb968', # :Dummy nodes not needed #Bai_rdb968             , nnzA , 25793    , colsA , 968    , n_compute , 50618     ,critical_path_len, 278 
    'Bai_dw2048', # 0 Bai_dw2048             , nnzA , 40644    , colsA , 2048   , n_compute , 79240     ,critical_path_len, 309  
  ]
  #CPU
  # fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/openmp/run_log_sparse_tr_solve_two_way_Ofast_eridani'
  fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/openmp/run_log_sparse_tr_solve_two_way_O3_eridani'
  mat_detail_d_fast, _, _= get_data(fname, with_single_thread= True)
  for w in target_mat:
    system.append('CPU')
    throughput.append(mat_detail_d_fast[w].max_throughput/1e3)
    workload.append(w)

  fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/src/openmp/run_log_sparse_tr_solve_nvidia_cusparse_gliese'
  map_mat_to_throughput_gpu, map_mat_to_cols_compute = get_gpu_data(fname)
  for w in target_mat:
    system.append('GPU')
    throughput.append(map_mat_to_throughput_gpu[w]/1e3)
    workload.append(w)
  
  # DPU
  # fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/src/PRU/sv/async/tb/run_log_psdd_64_65536_65536_large_stream_mem'
  # map_psdd_to_throughput_pru= get_pru_throughput_data(fname, matrix_or_psdd = 'psdd')
  # for w in target_psdd:
  #   system.append('DPU')
  #   throughput.append(map_psdd_to_throughput_pru[w])
  #   workload.append(w)
  
  # DPU (small version compareble to DPUv2) throughput GOPS
  map_w_to_througput_DPU= {
     'tretail'            :3.2279686469855693,
     'mnist'              :5.714442493415277 ,
     'nltcs'              :4.254714624703385 ,
     'msweb'              :5.497139773340529 ,
     'msnbc'              :4.85417179424071  ,
     'bnetflix'           :4.80075056728923  ,
     'HB_bp_200'          :2.1350198110332217,
     'HB_west2021'        :2.74123043712898  ,
     'MathWorks_Sieber'   :3.90464757331504  ,
     'HB_jagmesh4'        :1.9240511974312091,
     'Bai_rdb968'         :1.8668534369059206,
     'Bai_dw2048'         :2.577926343448042 ,
  }
  for w, t in map_w_to_througput_DPU.items():
    system.append('DPU')
    throughput.append(t)
    workload.append(w)

  map_w_to_througput_DPUv2= {
    'tretail'              :5.503309604096416 ,
    'mnist'                :6.673074458541587 ,
    'nltcs'                :5.394268070619904 ,
    'msweb'                :5.9021693783896545,
    'msnbc'                :6.034575078405875 ,
    'bnetflix'             :3.703028018256971 ,
    'HB_bp_200'            :2.960901725959845 ,
    'HB_west2021'          :3.7442871885596345,
    'MathWorks_Sieber'     :3.455351180719966 ,
    'HB_jagmesh4'          :3.550507099391481 ,
    'Bai_rdb968'           :3.7237736515316486,
    'Bai_dw2048'           :3.8490309418565114,
  }
  for w, t in map_w_to_througput_DPUv2.items():
    system.append('This work')
    throughput.append(t)
    workload.append(w)

  data_d= {
    'system' : system,
    'throughput': throughput,
    'workload' : workload,
  }
  df = pd.DataFrame.from_dict(data_d)


  print("Average throughputs:")
  print(df.groupby('system')['throughput'].mean())

  # plotting
  fig_dims = (2.8, 2.4)
  plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  # sns.set_context('notebook')
  sns.set_style({'font.family' : 'STIXGeneral'})
  # sns.set_style({'font.family' : 'Helvetica'})
  # sns.set(fontfamily= 'STIXGeneral')
  # sns.set_palette('rainbow')
  # sns.set_palette(['#62C370', '#FFD166', '#EF476F'])

  # order= [4, 8, 16]
  # ax= sns.barplot(x= 'utilization', y= 'inputs', hue= 'net', data=df, order= order)
 
  order= [
    'tretail'         ,
    'mnist'           ,
    'nltcs'           ,
    'msweb'           ,
    'msnbc'           ,
    'bnetflix'        ,
    'HB_bp_200'       ,
    'HB_west2021'     ,
    'MathWorks_Sieber',
    'HB_jagmesh4'     ,
    'Bai_rdb968'      ,
    'Bai_dw2048'      ,
  ]
  hue_order= ['This work', 'DPU', 'CPU', 'GPU']
  ax= sns.barplot(x= 'workload', y= 'throughput', hue= 'system', orient= 'v', data=df, order= order, hue_order = hue_order,
      ci= None, # No error margins
      estimator= mean
      )

  ax.set_ylim([0,10])

  ax.set_xticks(list(range(len(order))))
  ax.set_xticklabels(
    [
      'tretail'         ,
      'mnist'           ,
      'nltcs'           ,
      'msweb'           ,
      'msnbc'           ,
      'bnetflix'        ,
      'bp_200'       ,
      'west2021'     ,
      'sieber',
      'jagmesh4'     ,
      'rdb968'      ,
      'dw2048'      ,
    ]
  )
  
  plt.xticks(rotation = 90)

  # ax.set_ylim([0,119])
  # labels= [4, 8, 16, 32, 64]
  # h, l= ax.get_legend_handles_labels()
  # ax.legend(h, labels, title= "Threads")
  # ax.legend(h, l, title= "")
  ax.legend(ncol= 2, loc= 'upper center')
  # ax.legend(ncol= 1, bbox_to_anchor= (1.2, 0.5))
  # ax.legend([], [])
  plt.tight_layout()
  plt.ylabel('')
  plt.xlabel('')

  if savefig:
    assert path != None
    plt.savefig(path, dpi =300)
  else:
    plt.show()

def cpu_gpu_throughput(savefig, path):
  
  data_d= {
      'platform' : ['CPU', 'CPU', 'GPU', 'GPU'],
      'app' : ['PC', 'SpTRSV', 'PC', 'SpTRSV'],
      'throughput' : [1.361, 1.3074, 0.3244, 0.1514]
    }
  df = pd.DataFrame.from_dict(data_d)

  # plotting
  fig_dims = (3.1, 1)
  plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  # sns.set_context('notebook')
  sns.set_style({'font.family' : 'STIXGeneral'})
  # sns.set_style({'font.family' : 'Helvetica'})
  # sns.set(fontfamily= 'STIXGeneral')
  # sns.set_palette('rainbow')
  # sns.set_palette(['#62C370', '#FFD166', '#EF476F'])

  # order= [4, 8, 16]
  # ax= sns.barplot(x= 'utilization', y= 'inputs', hue= 'net', data=df, order= order)
  ax= sns.barplot(x= 'throughput', y= 'platform', hue= 'app', orient= 'h', data=df, 
      ci= None, # No error margins
      estimator= mean
      )

  ax.set_xticks([0, 0.6, 1.2])
  ax.set_xticklabels(['0', '0.6', '1.2'])

  # ax.set_ylim([0,119])
  # labels= [4, 8, 16, 32, 64]
  h, l= ax.get_legend_handles_labels()
  # ax.legend(h, labels, title= "Threads")
  ax.legend(h, l, title= "")
  # ax.legend(ncol= 2, loc= 'upper center')
  # ax.legend(ncol= 1, bbox_to_anchor= (1.2, 0.5))
  ax.legend([], [])
  plt.tight_layout()
  plt.ylabel('')
  plt.xlabel('Throughput (GOPS)')

  if savefig:
    assert path != None
    plt.savefig(path, dpi =300)
  else:
    plt.show()

def output_modes(savefig, path):
  # ONE_TO_ONE= [2.35240312104744, 1.6271032794051794, 1.4458116159457683, 1.3602690896501692, 1.6399163837269688, 0.894281260414185, 0.7605294609533556, 0.7039097003042737, 1.0641863431208807, 0.4700367954710041, 0.3967020266883472, 0.37586992398883756, 0.6311063414608187, 0.26931726940519085, 0.2440178334006215, 0.23923219378701474]
  
  # CROSSBAR= [2.017744703512898, 1.2859283848285314, 1.1246595277707436, 1.047508827548753, 1.426661323417333, 0.6869778271650904, 0.5802967130833938, 0.542458615104287, 0.9185262400038259, 0.3602205938728687, 0.31425800477945903, 0.30185508901982827, 0.5370524887741008, 0.23083888502375635, 0.21484805719578823, 0.21238513522553715]

  # PAPER_MODE= [2.017062522215568, 1.2846259461512994, 1.1287790557368695, 1.0575545077396338, 1.420767898746706, 0.6940076982180026, 0.5877574369446368, 0.550456614743426, 0.9212593733968095, 0.3693592147286501, 0.3214943397579835, 0.30995067823323125, 0.540700313186681, 0.22938168406417064, 0.22023056040700456, 0.21769816375768264]

  ONE_TO_ONE= [0.3061804988884287, 0.31629988224904043, 0.3157149867327265, 0.3070787007031278, 0.1762369464882046, 0.17330893661504987, 0.1702706299282617, 0.1675158403985073, 0.08350691853731744, 0.08087611953690911, 0.08158423334953127, 0.08297035237612474, 0.032066889706040574, 0.03168772203520951, 0.0312708817973538, 0.03225018248786582]
  PAPER_MODE= [0.020843176651437433, 0.019085907173211285, 0.019662130886725134, 0.019162957512994128, 0.018430521673822836, 0.017416289955147837, 0.01668104334212871, 0.016636153365720014, 0.010992335038660685, 0.010690114397981693, 0.009901052632604206, 0.010686699232310382, 0.007493494733320522, 0.008323510049536211, 0.007842713824388366, 0.008012406253508459]
  CROSSBAR= [0.01414326121141615, 0.015482901279469347, 0.01431112838709563, 0.013550497375063988, 0.009033762413630532, 0.009861895010737086, 0.009436328957925327, 0.009937743426011234, 0.002897329047165242, 0.0034864444163625104, 0.0031587336123928126, 0.0031958033880617465, 0.002509450959040508, 0.002577085084899737, 0.002681048901022583, 0.0022760546124638045]


  slow_dow_one_to_one= mean([ONE_TO_ONE[i]/ CROSSBAR[i] for i in range(len(CROSSBAR))])
  slow_dow_paper= mean([PAPER_MODE[i]/ CROSSBAR[i] for i in range(len(CROSSBAR))])

  logger.info(f'slow_dow_one_to_one: {slow_dow_one_to_one}, slow_dow_paper, {slow_dow_paper}')


  data_d= {
    'mode' : ['one_to_one', 'paper', 'crossbar'],
    'normalized_throughput' : [slow_dow_one_to_one, slow_dow_paper, 1.0],
  }
  df = pd.DataFrame.from_dict(data_d)

  # plotting
  fig_dims = (0.75, 0.75)
  plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  # sns.set_context('notebook')
  sns.set_style({'font.family' : 'STIXGeneral'})
  # sns.set_style({'font.family' : 'Helvetica'})
  # sns.set(fontfamily= 'STIXGeneral')
  # sns.set_palette('rainbow')
  # sns.set_palette(['#62C370', '#FFD166', '#EF476F'])

  # order= [4, 8, 16]
  # ax= sns.barplot(x= 'utilization', y= 'inputs', hue= 'net', data=df, order= order)
  order= ['crossbar', 'paper', 'one_to_one']
  ax= sns.barplot(x= 'mode', y= 'normalized_throughput', orient= 'v', data=df, order= order, 
      ci= None, # No error margins
      estimator= mean
      )

  ax.set_ylim([0,24])

  ax.set_xticks([0, 1, 2])
  ax.set_xticklabels(['a', 'b', 'c'])

  # ax.set_ylim([0,119])
  # labels= [4, 8, 16, 32, 64]
  h, l= ax.get_legend_handles_labels()
  # ax.legend(h, labels, title= "Threads")
  ax.legend(h, l, title= "")
  # ax.legend(ncol= 2, loc= 'upper center')
  # ax.legend(ncol= 1, bbox_to_anchor= (1.2, 0.5))
  ax.legend([], [])
  plt.tight_layout()
  plt.ylabel('')
  plt.xlabel('')

  if savefig:
    assert path != None
    plt.savefig(path, dpi =300)
  else:
    plt.show()

def random_bank_alloc(savefig, path):
  data_d= {\
    'mode' : [],
    'workload' : [],
    'total' : [],
    'conflicts' : []
  }
  data_d['workload'] += ['tretail' , 'HB_bp_200' , 'mnist' , 'nltcs' , 'HB_west2021' , 'MathWorks_Sieber', 'msweb' , 'msnbc' , 'HB_jagmesh4' , 'Bai_rdb968' , 'bnetflix']
  data_d['total'] += [1542  ,  2237  ,  1729  ,  2446  ,  2603  ,  5084  ,  8153  ,  7059  ,  11017 ,  11953 ,  12268]
  data_d['conflicts'] += [1028 ,1428 ,1269 ,1640 ,1814 ,3223 ,5259 ,4622 ,7373 ,7927 ,7218]
  data_d['mode'] += ['random' for _ in range(len(data_d['conflicts']))]

  data_d['workload'] += ['HB_bp_200', 'tretail', 'mnist', 'HB_west2021', 'nltcs', 'msweb', 'MathWorks_Sieber', 'msnbc', 'HB_jagmesh4', 'bnetflix', 'Bai_rdb968']
  data_d['total'] += [874, 471,  503,  853,  851,  2672,  2454,  2496,  4018,  4387,  4078]
  data_d['conflicts'] += [70  , 2   , 36  , 44  , 3   , 17  , 350 , 24  , 16  , 35  , 9]
  data_d['mode'] += ['our' for _ in range(11)]
  
  our_conf_d= {}
  for idx, conf in enumerate(data_d['conflicts']):
    if data_d['mode'][idx] == 'our':
      workload = data_d['workload'][idx]
      our_conf_d[workload] = conf

  data_d['normalized'] = [data_d['conflicts'][idx]/our_conf_d[w] for idx, w in enumerate(data_d['workload'])]

  # reduction in 292x in conflicts

  df = pd.DataFrame.from_dict(data_d)

  # plotting
  fig_dims = (1, 2)
  plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  # sns.set_context('notebook')
  sns.set_style({'font.family' : 'STIXGeneral'})

  order= ['random', 'our']
  ax= sns.barplot(x= 'mode', y= 'normalized', orient= 'v', data=df, order= order, 
      ci= None, # No error margins
      estimator= mean
      )
  
  ax.set(yscale= 'log')
  plt.tight_layout()
  plt.ylabel('')
  plt.xlabel('')

  if savefig:
    assert path != None
    plt.savefig(path, dpi =300)
  else:
    plt.show()


def bank_occupancy_profile(savefig, path):
  # fname= '~/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/log/DPUv2_bank_occupancy_profile_tretail.csv'
  # fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/log/DPUv2_bank_occupancy_profile_msnbc.csv'
  # fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/log/DPUv2_bank_occupancy_profile_MathWorks_Sieber_64_16.csv'
  fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/log/DPUv2_bank_occupancy_profile_MathWorks_Sieber_128_16.csv'
  # fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/log/DPUv2_bank_occupancy_profile_bnetflix_16.csv'
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

  # with open(fname, 'r') as fp:
  #   data= list(fp.readlines())
  #   data= [d.split(',') for d in data]
  
  data_d= {
    'bank': [],
    'cycle': [],
    'occupancy': []
  }
  for bank, d in enumerate(data):
    for cycle, occupancy in enumerate(d):
      data_d['bank'].append(int(bank))
      data_d['cycle'].append(int(cycle))
      data_d['occupancy'].append(int(occupancy))

  df = pd.DataFrame.from_dict(data_d)

  # plotting
  fig_dims = (3, 2)
  plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  # sns.set_context('notebook')
  sns.set_style({'font.family' : 'STIXGeneral'})

  ax= sns.lineplot(x= 'cycle', y= 'occupancy', hue= 'bank', data=df
      )

  
  ax.set_ylim([0,135])

  plt.tight_layout()
  plt.ylabel('')
  plt.xlabel('')

  if savefig:
    assert path != None
    plt.savefig(path, dpi =300)
  else:
    plt.show()

def throughput_large_networks(savefig, path):

  fname= '../data/Throughput large networks - Sheet1.csv'
  # fname= '/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/log/DPUv2_bank_occupancy_profile_bnetflix_16.csv'
  # with open(fname, 'r') as fp:
  #   data = csv.reader(fp, delimiter=',')
  #   data= list(data)
  df = pd.read_csv(fname)
  print(df)

  # plotting
  fig_dims = (1.7, 2.1)
  plt.figure(figsize=fig_dims) 

  sns.set(style= 'white')
  sns.set_context('paper')
  # sns.set_context('notebook')
  sns.set_style({'font.family' : 'STIXGeneral'})
  
  order= ['pigs', 'andes', 'munin', 'mildew']
  hue_order= ['DPUv2', 'SPU', 'CPU', 'GPU', "CPU_SPU_baseline"]
  ax= sns.barplot(x= 'workloads', y= 'throughput', orient= 'v', hue= 'system', data=df, order= order, 
      hue_order= hue_order,
      ci= None, # No error margins
      estimator= mean
      )
  
  # ax.set(yscale= 'log')
  plt.tight_layout()
  plt.ylabel('')
  plt.xlabel('')
  plt.xticks(rotation = 90)
  ax.set_ylim([0,60])

  ax.legend(ncol= 2, loc= 'upper center')
  # ax.legend([], [])
  h, l= ax.get_legend_handles_labels()
  # ax.legend(h, labels, title= "Threads")
  # ax.legend(h, l, title= "")
  ax.legend(h, [])

  if savefig:
    assert path != None
    plt.savefig(path, dpi =300)
  else:
    plt.show()


if __name__ == "__main__":
  # savefig= True
  # cpu_gpu_throughput(savefig= savefig, path= '../cpu_gpu_throughput.pdf')

  # savefig= True
  # output_modes(savefig= savefig, path= '../output_mode_chart.pdf')

  # savefig= True
  # fname= '../bank_occupancy_profile_wo_spill.pdf'
  # bank_occupancy_profile(savefig= savefig, path= fname)

  savefig= True
  fname = '../throughput_large_networks.pdf'
  throughput_large_networks(savefig= savefig, path= fname)

  savefig= True
  fname = '../throughput_bar_plot_DPUv2.pdf'
  throughput_bar_plot_DPUv2(savefig, path= fname)

  # savefig= True
  # random_bank_alloc(savefig= savefig, path= '../random_bank_alloc.pdf')

