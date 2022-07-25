
import csv

class MatrixDetails():
  def __init__(self):
    self.name= None
    self.nnz= None
    self.ncols= None
    self.critical_path_len_coarse= None

def replace_slash(name_ls):
  group_names= ["Bai", 
      "Bates",
      "Bindel",
      "Cannizzo",
      "HB",
      "Norris",
      "GHS_indef",
      "GHS_psdef",
      "FIDAP",
      "Boeing",
      "Oberwolfach",
      "Okunbor",
      "Pothen",
      "MathWorks",
      "Nasa",
      "Pothen"
      ]
  
  new_name_ls= []
  for name in name_ls:
    for g in group_names:
      name= name.replace(g+'/', g+'_', 1)
    
    new_name_ls.append(name)

  return new_name_ls

def replace_underscore(name_ls):
  group_names= ["Bai", 
      "Bates",
      "Bindel",
      "Cannizzo",
      "HB",
      "Norris",
      "GHS_indef",
      "GHS_psdef",
      "FIDAP",
      "Boeing",
      "Oberwolfach",
      "Okunbor",
      "Pothen",
      "MathWorks",
      "Nasa",
      "Pothen"
      ]
  
  new_name_ls= []
  for name in name_ls:
    for g in group_names:
      name= name.replace(g+'_', g+'/', 1)
    
    new_name_ls.append(name)

  return new_name_ls

def get_matrix_sizes(name_format = 'under_score'):
  assert name_format in ['under_score', 'slash']

  fname= './src/sparse_tr_solve_mat_sizes'
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)

    name_idx= 0
    nnz_idx= 2
    col_idx= 4
    compute_idx= 6
    critical_path_len_coarse_idx= 8

    map_mat_to_details= {}
    for d in data:
      if len(d) == 1:
        assert d[0] == 'Start'
        continue
      name= d[name_idx].strip()
      nnz= int(d[nnz_idx])
      cols= int(d[col_idx])
      compute= int(d[compute_idx])
      critical_path_len_coarse= int(d[critical_path_len_coarse_idx]) + 1

      if name_format == 'slash':
        name= replace_underscore([name])[0]
      
      obj= MatrixDetails()
      obj.compute = compute
      obj.ncols = cols
      obj.nnz = nnz
      obj.critical_path_len_coarse = critical_path_len_coarse

      map_mat_to_details[name]= obj

  return map_mat_to_details


def get_psdd_sizes():
  fname= './src/psdd_sizes'
  with open(fname, 'r') as fp:
    data = csv.reader(fp, delimiter=',')
    data= list(data)
    
    # first line is title
    data = data[1:]

    name_idx= 0
    leaves_idx= 3
    compute_idx= 4

    map_mat_to_details= {}
    for d in data:
      if len(d) == 1:
        assert d[0] == 'Start'
        continue
      name= d[name_idx].strip()
      compute= int(d[compute_idx])

      obj= MatrixDetails()
      obj.compute = compute

      map_mat_to_details[name]= obj
  
  return map_mat_to_details

