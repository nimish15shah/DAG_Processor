import random
import queue
import time

from collections import defaultdict

#**** imports from our codebase *****
from . import common_classes
from . import hw_struct_methods
from . import reporting_tools

class Mutable_level():
  max_lvl= 0


class global_knobs():
  VECT_SIZE= 1
  LVL_WINDOW= 128 # This is used to decide whether to allocate an output of a BB to vector reg file
  SCALAR_LVL_WINDOW=128 # This is used to decide whether to allocate a vector while loading from scalar file to vector
  
  SCAL_TO_VECT_PORTS= 4 # Number of available port for loading from scalar to vector

  N_OUT_PORTS= 4 # Number of addresses that could be specified for the outputs of a BB, DURING THAT cCYCLE. #The vector REgFile should be able to take these many inputs in a cycle
  
  SECONDARY_OUT_PORTS= N_OUT_PORTS # Number of addresses that could be specified for the outputs of a BB, DURING THE NEXT CYCLE. #The vector REgFile should be able to take these many inputs in a cycle
  
  VECT_LOAD_THRESHOLD=4 # Trigger a vector load for leafs

def assign_sch_lvl(BB_graph, curr_bb, mutable_lvl, map_sch_lvl_to_bb):
  """
   Assigns sch_level to BB object
   
   Fills the dictionary map_sch_lvl_to_bb (Key: sch_level, Val: BB number)
    
  """
  
  assert isinstance(BB_graph, dict)
  assert isinstance(curr_bb, int)
  assert isinstance(map_sch_lvl_to_bb, dict)

  obj= BB_graph[curr_bb]

  if obj.sch_level is None:
    max_lvl= mutable_lvl.max_lvl
    
    # Sort based on reverse level
    sort_lst= sorted(obj.child_bb_lst, key= lambda x: BB_graph[x].reverse_level, reverse= True)
    # sort_lst= sorted(obj.child_bb_lst, key= lambda x: BB_graph[x].level, reverse= True)
    # sort_lst= sorted(obj.child_bb_lst, key= lambda x: BB_graph[x].level)
      
    #FIXME: Just a trial to check different sch_level sequence
    #random.shuffle(sort_lst)

    for child in sort_lst:
      ch_lvl= assign_sch_lvl(BB_graph, child, mutable_lvl, map_sch_lvl_to_bb)
      if ch_lvl > max_lvl:
        max_lvl= ch_lvl
    
    obj.sch_level= max_lvl + 1
    mutable_lvl.max_lvl= max_lvl + 1
    
    # Create a map of sch_level to bb
    map_sch_lvl_to_bb[obj.sch_level] = curr_bb
    
  return obj.sch_level
  
def create_bb_dependecny_graph(graph, BB_graph):
  """
    Assigns level to every BB in BB_graph
  """
  
  highest_level= 0

  # Find the head BB and assign it a level zero
  head_bb_lst=[]
  for BB,obj in list(BB_graph.items()):
    if len(obj.parent_bb_lst) == 0:
      head_bb_lst.append(BB)
  
  assert len(head_bb_lst) == 1, "There are multiple heads in BB_graph. Not allowed"
  head_bb= head_bb_lst[0]
  BB_graph[head_bb].level= 0
  
  # Assign levels
  open_set= queue.Queue()
  closed_set= []
  
  open_set.put(len(BB_graph))
  
  while not open_set.empty():
    curr_bb = open_set.get()
    obj= BB_graph[curr_bb]
    
    child_level= obj.level + 1
    for child in obj.child_bb_lst:
      if BB_graph[child].level < child_level:
        BB_graph[child].level= child_level
        open_set.put(child)
        if  child_level > highest_level:
          highest_level= child_level

    closed_set.append(curr_bb)
  
  print('highest_level', highest_level)
  for bb, obj in list(BB_graph.items()):
    obj.level = highest_level - obj.level
    #print bb, obj.level

  for bb, obj in list(BB_graph.items()):
    for child in obj.child_bb_lst:
      assert BB_graph[child].level < obj.level
    
    for parent in obj.parent_bb_lst:
      assert BB_graph[parent].level > obj.level

  return head_bb

def compute_reverse_lvl(BB_graph, curr_bb):
  """
    Assigns reverse_lvl to every BB. 
    Reverse_lvl is assigned from Bottom to Top.
    Normal 'level' is assigned Top to Bottom. 
    Each of these have their use in scheduling
  """
  obj= BB_graph[curr_bb]
  
  if obj.reverse_level is None:
    if not obj.child_bb_lst: # leaf BB
      obj.reverse_level= 0
    else: # Not a leaf BB
      max_level= 0
      for child in obj.child_bb_lst:
        lvl= compute_reverse_lvl(BB_graph, child)
        if lvl > max_level:
          max_level= lvl

      obj.reverse_level= max_level + 1
  
  return obj.reverse_level

def old_reg_alloc(graph, BB_graph, head_node, global_var, write_asm, make_vid= False):
  max_len=0
  active_bb= []

  schedulable= []
  
  score_inputs= dict.fromkeys(list(BB_graph.keys()) , 0)
  
  done_bb=[]
  
  bb_w_leaf_inputs= set() 

  for bb, obj in list(BB_graph.items()):
    if obj.level == 0:
      selected_active_bb= bb
      interesting_bb= [bb]
      
  for bb, obj in list(BB_graph.items()):
    for in_node in obj.in_list_unique:
      if graph[in_node].operation_type == common_classes.OPERATOR.LEAF:
        score_inputs[bb] += 1

        if bb not in bb_w_leaf_inputs:
          bb_w_leaf_inputs.add(bb)
    
    #if score_inputs[key] == len(obj.in_list_unique):
    #  schedulable.append(key)

  SELECT_NEW_ACTIVE_NODE= False

  # Keep track of level og scheduled BBs
  max_scheduled_level= 0
  
  bb_with_inputs_but_not_active= []

  ## Scalar and vector regfiles
  global_knobs.VECT_SIZE= 64
  vect_reg_file= common_classes.reg_file(global_knobs.VECT_SIZE, 'vect_reg_file')
  scalar_reg_file= common_classes.reg_file(0, 'scalar_reg_file')

  while not graph[head_node].computed:
    st= time.time() 
    #if len(schedulable) == 0:
    #  for bb in active_bb:
    #    if score_inputs[bb] == len(BB_graph[bb].in_list_unique):
    #      schedulable.append(bb)
    #      break
     
    if len(schedulable) == 0:
      # construct interesting bb
      #interesting_bb= list(set().union(interesting_bb, active_bb))
      
      if len(schedulable) == 0:
        if SELECT_NEW_ACTIVE_NODE:
          # First see if there is any schedulable bb in active_bb
          if not active_bb:
            lowest_level= 10000000
            for bb in bb_with_inputs_but_not_active:
              if BB_graph[bb].level < lowest_level:
                lowest_level = BB_graph[bb].level
                active_bb.append(bb)
                vect_reg_file.allocate_AppIdx(bb)
                BB_graph[bb].status= BB_graph[bb].INPUTS_IN_VECTOR
                interesting_bb= [bb]
                SELECT_NEW_ACTIVE_NODE= False
                active_bb_idx=0
                selected_active_bb= bb
                
            bb_with_inputs_but_not_active.remove(selected_active_bb)
          else:
            for bb in active_bb:
              if score_inputs[bb] == len(BB_graph[bb].in_list_unique):
                schedulable.append(bb)
                if bb in interesting_bb:
                  interesting_bb.remove(bb)
                if bb in bb_with_inputs_but_not_active:
                  bb_with_inputs_but_not_active.remove(bb)
                break
            
            if len(schedulable) == 0:
              SELECT_NEW_ACTIVE_NODE= False
              lowest_level=10000000
              selected_active_bb= 1
              active_bb_idx=0
              for idx, bb in enumerate(active_bb):
                if BB_graph[bb].level < lowest_level:
                  lowest_level = BB_graph[bb].level
                  selected_active_bb= bb
                  active_bb_idx= idx
              
              interesting_bb= [selected_active_bb]
              #print 'selected_level, node:', BB_graph[interesting_bb[0]].level, selected_active_bb
      
      count=0 
      old_len_interesting_bb= 1
      while len(schedulable) == 0:
        count += 1
        if count ==1000:
          print('interes_bb:', interesting_bb)
          print('done' , done_bb)
          assert 0, 'Bohot time laga diya'
        
        old_len_interesting_bb= len(interesting_bb)
        for bb in interesting_bb:
          if score_inputs[bb] == len(BB_graph[bb].in_list_unique):
            schedulable.append(bb)
            interesting_bb.remove(bb)
            if bb in bb_with_inputs_but_not_active:
              bb_with_inputs_but_not_active.remove(bb)
            break
          
          for in_node in BB_graph[bb].in_list_unique:
            if not graph[in_node].operation_type == common_classes.OPERATOR.LEAF:
              root_bb= graph[in_node].storing_builblk
              assert root_bb != None, "root_bb is None"
              
              #if (root_bb not in done_bb) and (root_bb not in active_bb) and (root_bb not in interesting_bb) and (root_bb not in schedulable):
              if (root_bb not in done_bb) and (root_bb not in interesting_bb) and (root_bb not in schedulable):
                interesting_bb.append(root_bb)
        
        #if len(schedulable)==0 and len(interesting_bb) == old_len_interesting_bb:
        #  print 'Active_bb:', active_bb
        #  for bb in interesting_bb:
        #    print bb, BB_graph[bb].child_bb_lst

        #  if active_bb_idx == len(active_bb) -1:
        #    active_bb_idx = 0
        #  else:
        #    active_bb_idx += 1
        #  
        #  selected_active_bb =  active_bb[active_bb_idx]
        #  interesting_bb= [selected_active_bb]
        #  old_len_interesting_bb=1
        #  print 'Swicthed active node:', selected_active_bb
        #  exit(1)

      #for key, obj in BB_graph.items():
      #  if key not in done_bb:
      #    for in_node in obj.in_list_unique:
      #      if graph[in_node].operation_type == common_classes.OPERATOR.LEAF:
      #        score_inputs[key] += 1
      #    
      #    if score_inputs[key] == len(obj.in_list_unique):
      #      schedulable.append(key)
      
      if schedulable[0] == selected_active_bb:
        SELECT_NEW_ACTIVE_NODE= True

    assert len(schedulable), "schedulable list is empty"
    curr_bb= schedulable.pop()
    curr_bb_obj = BB_graph[curr_bb]
    
    print('curr:', curr_bb, end=' ')
    assert curr_bb not in done_bb, "curr_bb is already done! curr_bb, done_bb = {} :: {}".format(curr_bb, done_bb)
    assert curr_bb_obj.status != curr_bb_obj.COMPUTED, "BB already scheduled before"

    if write_asm:
      if curr_bb in bb_w_leaf_inputs:
        reporting_tools.write_asm(global_var, 'ld_v', ['mem_x', str(curr_bb)])
        
    if max_scheduled_level < curr_bb_obj.level:
      max_scheduled_level = curr_bb_obj.level
    
    # Allocate to vecotr file if not already
    if curr_bb not in active_bb:
      active_bb.append(curr_bb)
      vect_reg_file.allocate_AppIdx(curr_bb)

    # Copy from Scalar file to Vector file
    ret_dict= scalar_reg_file.read_by_AppIdx(curr_bb)
    pos_lst= ret_dict['pos']
    if write_asm:
      for pos in pos_lst:
        reporting_tools.write_asm(global_var, 'cp_StV', [str(pos), str(curr_bb)])
    
    # Free up Scalar file
    scalar_reg_file.deallocate_AppIdx(curr_bb)
    
    if write_asm:
      reporting_tools.write_asm(global_var, 'BB', [str(curr_bb)])

    
    assert set(active_bb) == set(vect_reg_file.get_AppIdx()), "Should be same. Different elements : {}".format(set(vect_reg_file.get_AppIdx())- set(active_bb))

    for out in curr_bb_obj.out_list:
      graph[out].computed= True
      
      parents= graph[out].parent_buildblk_lst
      
      for parent in parents:
        score_inputs[parent] += 1
      
      if out != head_node:
        #if len(parents) == 1: # Only one parent, may not need to push this on scalar register file
        #  # Do nothing
        #  #if parents[0] in active_bb: 
        #  
        #  selected_bb= parents[0]
        #  if parents[0] not in active_bb:
        #    if parents[0] in interesting_bb:
        #      active_bb.append(parents[0])
        #      vect_reg_file.allocate_AppIdx(parents[0])
        #    else:
        #      scalar_reg_file.allocate_AppIdx(parents[0])
        #      if selected_bb not in bb_with_inputs_but_not_active:
        #        bb_with_inputs_but_not_active.append(selected_bb)
        #
        #else: # Multiple parents
        low_level= 10020
        selected_bb= None
        for bb in parents:
          if BB_graph[bb].level < low_level:
            low_level = BB_graph[bb].level
            selected_bb = bb

          if bb in active_bb or bb in interesting_bb:
            selected_bb= bb
            break
        
        assert selected_bb not in done_bb, "Contradiction!"
        assert selected_bb != None, "Select a bb, parents: {}".format(parents)

        if selected_bb not in active_bb:
          free_space= vect_reg_file.get_free_space() 
          #if BB_graph[selected_bb].level < (BB_graph[curr_bb].level + (BB_graph[final_bb].level/10) + 5):
          if BB_graph[selected_bb].level < (BB_graph[curr_bb].level + free_space):
            SCALAR_FILE_ALLOCATE= True
            active_bb.append(selected_bb)
            vect_reg_file.allocate_AppIdx(selected_bb)
            BB_graph[selected_bb].status= BB_graph[selected_bb].INPUTS_IN_VECTOR
          else:
            SCALAR_FILE_ALLOCATE= False
            pos= scalar_reg_file.allocate_AppIdx(selected_bb)
            BB_graph[selected_bb].status= BB_graph[selected_bb].INPUTS_IN_SCALAR
            if selected_bb not in bb_with_inputs_but_not_active:
              bb_with_inputs_but_not_active.append(selected_bb)
        else:
          SCALAR_FILE_ALLOCATE= True
        
        for bb in parents:
          if bb != selected_bb:
            BB_graph[bb].status= BB_graph[bb].INPUTS_IN_SCALAR
            if bb not in bb_with_inputs_but_not_active:
              bb_with_inputs_but_not_active.append(bb)
            
            if SCALAR_FILE_ALLOCATE:
              pos= scalar_reg_file.allocate_AppIdx(bb)
              SCALAR_FILE_ALLOCATE= False
            else:
              scalar_reg_file.map_pos_to_new_AppIdx(pos, bb)
            if pos == 6:
              print('\n\n',parents)

    if max_len < len(active_bb):
      max_len= len(active_bb)
    
    active_bb.remove(curr_bb)
    vect_reg_file.deallocate_AppIdx(curr_bb)
    done_bb.append(curr_bb)
    curr_bb_obj.status = curr_bb_obj.COMPUTED
    
    if make_vid:
      vid_maker.create_a_frame_scheduling(BB_graph, score_inputs)
      vid_maker.add_frame()
    assert len(done_bb) <= len(BB_graph), "Done node cannot be more than BB_graph"

  if make_vid:
    vid_maker.write_vid()
  print('\nMax vector registers required: ', max_len)
  print('Max vector reg file size: ', vect_reg_file.max_file_size)
  print('Max scalar reg file size: ', scalar_reg_file.max_file_size)
  
  assert len(done_bb) == len(BB_graph), "Done node should be equal to BB_graph {}, {}".format(len(done_bb), len(BB_graph))
  assert not scalar_reg_file.get_occ_space(), "scalar reg file should be empty by now {}".format(scalar_reg_file.occup_pos_lst)
  assert not active_bb, "active_bb should be empty by now {}".format(active_bb)
  assert not interesting_bb, "interesting_bb should be empty by now {}".format(interesting_bb)
  assert not bb_with_inputs_but_not_active, "bb_with_inputs_but_not_active should be empty by now {}".format(bb_with_inputs_but_not_active)

def reg_alloc(graph, BB_graph, head_node, global_var, leaf_list, write_asm, vid_maker, make_vid= False):
  """
  This function is to be called after the scheduling is done. It allocates, stores and loads the data from and to the scalar and vector regfiles
  """
  bb_lst= list(BB_graph.keys())

  # Sort based on sch_level
  sorted_bb_lst= sorted(bb_lst, key= lambda x: BB_graph[x].sch_level, reverse= False)

  active_bb= []
  done_bb=[]
  score_inputs= dict.fromkeys(list(BB_graph.keys()) , 0)
  track_leaf_status= dict.fromkeys(leaf_list, False)
  track_leaf_pos= dict.fromkeys(leaf_list, None)

  for bb, obj in list(BB_graph.items()):
    for in_node in obj.in_list_unique:
      if graph[in_node].operation_type == common_classes.OPERATOR.LEAF:
        score_inputs[bb] += 1
        obj.leaf_inputs.append(in_node)

    obj.n_leaf_inputs == score_inputs[bb]
  
  ## Scalar and vector regfiles
  global_knobs.VECT_SIZE= 64
  vect_reg_file= common_classes.reg_file(global_knobs.VECT_SIZE, 'vect_reg_file')
  scalar_reg_file= common_classes.reg_file(0, 'scalar_reg_file')
  
  max_len=0
  
  for curr_bb in sorted_bb_lst:
    curr_bb_obj = BB_graph[curr_bb]

    print(scalar_reg_file.get_occ_space(), end=' ')
    #print 'curr:', curr_bb,
    assert curr_bb_obj.status != curr_bb_obj.COMPUTED, "BB already scheduled before"
    
    # Allocate to vecotr file if not already
    if curr_bb not in active_bb:
      active_bb.append(curr_bb)
      vect_reg_file.allocate_AppIdx(curr_bb)
    
    # load leafs if already not loaded 
    load_leaf(graph, BB_graph, curr_bb, track_leaf_status, track_leaf_pos, scalar_reg_file, global_var, write_asm)
    
    # Copy from Scalar file to Vector file
    load_scalar_to_vect(BB_graph, global_var, curr_bb, scalar_reg_file, active_bb, vect_reg_file, BB_graph[curr_bb].sch_level, global_knobs.SCAL_TO_VECT_PORTS, global_knobs.SCALAR_LVL_WINDOW, write_asm)
    #ret_dict= scalar_reg_file.read_by_AppIdx(curr_bb)
    #pos_lst= ret_dict['pos']
    #if write_asm:
    #  for pos in pos_lst:
    #    reporting_tools.write_asm(global_var, 'cp_StV', [str(pos), str(curr_bb)])
    
    # Free up Scalar file
    #scalar_reg_file.deallocate_AppIdx(curr_bb)
    
    # Execute the complex instruction
    #if write_asm:
    #  reporting_tools.write_asm(global_var, 'BB', [str(curr_bb)])
    
    
    assert len(active_bb) == len(vect_reg_file.get_AppIdx()), "Should be same. Different elements : {}, {}, {}, {}, {}, {}".format(set(vect_reg_file.get_AppIdx())- set(active_bb), set(active_bb)- set(vect_reg_file.get_AppIdx()) , sorted(active_bb), sorted(vect_reg_file.get_AppIdx()), len(active_bb), len(vect_reg_file.get_AppIdx()))
    
    # Allocate outputs
    allocate_output(graph, BB_graph, active_bb, score_inputs, curr_bb, head_node, vect_reg_file, scalar_reg_file, global_var, write_asm)
    
    # Keep track of active vector reg size
    if max_len < len(active_bb):
      max_len= len(active_bb)
    
    active_bb.remove(curr_bb)
    vect_reg_file.deallocate_AppIdx(curr_bb)
    done_bb.append(curr_bb)
    curr_bb_obj.status = curr_bb_obj.COMPUTED
    
    if make_vid:
      vid_maker.create_a_frame_scheduling(BB_graph, score_inputs)
      vid_maker.add_frame()
    
  if make_vid:
    vid_maker.write_vid()
  print('\nMax vector registers required: ', max_len)
  print('Max vector reg file size: ', vect_reg_file.max_file_size)
  print('Max scalar reg file size: ', scalar_reg_file.max_file_size)
  
  assert len(done_bb) == len(BB_graph), "Done node should be equal to BB_graph {}, {}".format(len(done_bb), len(BB_graph))
  assert not scalar_reg_file.get_occ_space(), "scalar reg file should be empty by now {}".format(scalar_reg_file.occup_pos_lst)
  assert not active_bb, "active_bb should be empty by now {}".format(active_bb)

    
def allocate_output(graph, BB_graph, active_bb, score_inputs, curr_bb, head_node, vect_reg_file, scalar_reg_file, global_var, write_asm):
  
  LVL_WINDOW= global_knobs.LVL_WINDOW # This is used to decide whether to allocate an output of a BB to vector reg file
  SCALAR_LVL_WINDOW= global_knobs.SCALAR_LVL_WINDOW # This is used to decide whether to allocate a vector while loading from scalar file to vector
  
  SCAL_TO_VECT_PORTS= global_knobs.SCAL_TO_VECT_PORTS # Number of available port for loading from scalar to vector

  N_OUT_PORTS= global_knobs.N_OUT_PORTS # Number of addresses that could be specified for the outputs of a BB, DURING THAT cCYCLE. #The vector REgFile should be able to take these many inputs in a cycle
  
  SECONDARY_OUT_PORTS= global_knobs.SECONDARY_OUT_PORTS # Number of addresses that could be specified for the outputs of a BB, DURING THE NEXT CYCLE. #The vector REgFile should be able to take these many inputs in a cycle

  curr_bb_obj= BB_graph[curr_bb]
  
  # Initial analysis of outputs
  active_vect_cnt= 0
  par_in_active_bb= []
  par_not_in_active_bb= []
  map_par_to_out= defaultdict(list)

  for out in curr_bb_obj.out_list:
    # Mark out node as computed
    graph[out].computed= True
    
    parents= graph[out].parent_buildblk_lst
    # Increament score of parents
    for parent in parents:
      score_inputs[parent] += 1
      map_par_to_out[parent].append(out)
      
  # Check if parent is already allocated to the vector file
  for parent in list(map_par_to_out.keys()):
    if parent in active_bb: #and active_vect_cnt < N_OUT_PORTS:
      active_vect_cnt += len(map_par_to_out[parent])
      par_in_active_bb.append(parent)
    else:
      par_not_in_active_bb.append(parent)
  
  # If not enough outputs in vector, allocate more based on sch_level
  par_alloc_to_vect= []
  par_sorted= par_not_in_active_bb

  #if active_vect_cnt < N_OUT_PORTS:
  if True:
    # Sort non active parents based on the sch_level
    par_sorted= sorted(par_not_in_active_bb, key= lambda x:BB_graph[x].sch_level, reverse= False)
    
    #if par_sorted:
      # Allocate some new vectors if space allows 
      #alloc_space= N_OUT_PORTS - active_vect_cnt
      #for i in range(alloc_space):
    while par_sorted:
      par= par_sorted[0]
      # Check if these vectors are not very far from the current level. 
      if BB_graph[par].sch_level < (BB_graph[curr_bb].sch_level + LVL_WINDOW):
        par_alloc_to_vect.append(par)
        par_sorted.pop(0)
      else:
        #print 'LVL', BB_graph[par].sch_level, BB_graph[curr_bb].sch_level
        break # Break because parents are too far and should not be vector allocated

      # Break if par_sorted is consumed
      #if not par_sorted:
      #  break

  # parents to be stored in the scalar file
  par_to_scalar = par_sorted
  
  # Finally process the outputs based on above analyzed information
  out_scalar_status= dict.fromkeys(curr_bb_obj.out_list, False)
  out_scalar_pos= dict.fromkeys(curr_bb_obj.out_list, None)
  
  secondary_out_cnt= 0
  primary_out_cnt= 0
  NEXT_ITER = 0
  
  combined_out_list=[] # lists of all outputs that are going to the vector file

  for par, out_list in list(map_par_to_out.items()):
    if par in par_in_active_bb:
      # do nothing
      # The output will be stored in the vector file
      x=1
      
      combined_out_list.append(par)

      if primary_out_cnt < N_OUT_PORTS:
        primary_out_cnt += len(out_list)

        if primary_out_cnt > N_OUT_PORTS:
          secondary_out_cnt = primary_out_cnt - N_OUT_PORTS
          primary_out_cnt = N_OUT_PORTS
      
      else:
        secondary_out_cnt += len(out_list)
         
    elif par in par_alloc_to_vect:
      combined_out_list.append(par)
      if primary_out_cnt < N_OUT_PORTS:
        primary_out_cnt += len(out_list)
        if primary_out_cnt > N_OUT_PORTS:
          secondary_out_cnt = primary_out_cnt - N_OUT_PORTS
          primary_out_cnt = N_OUT_PORTS
      
      else:
        secondary_out_cnt += len(out_list)
      
      # par was not in active_bb originally, but might have been inserted by load_scalar_to_vect. So check again
      if par not in active_bb:
        active_bb.append(par)
        vect_reg_file.allocate_AppIdx(par)
        BB_graph[par].status= BB_graph[par].INPUTS_IN_VECTOR
      
      #--- read corresponding scalar regs and load into vector
      load_scalar_to_vect(BB_graph, global_var, par, scalar_reg_file, active_bb, vect_reg_file, BB_graph[curr_bb].sch_level, SCAL_TO_VECT_PORTS, SCALAR_LVL_WINDOW, write_asm)
      #ret_dict= scalar_reg_file.read_by_AppIdx(par)
      #pos_lst= ret_dict['pos']
      #if write_asm:
      #  for pos in pos_lst:
      #    reporting_tools.write_asm(global_var, 'cp_StV', ['pos:' + str(pos), 'par:' + str(par)])
      
      # Free up Scalar file
      #scalar_reg_file.deallocate_AppIdx(par)
    
    elif par in par_to_scalar: # Either allocate in scalar reg or map to existing scalar entry       
      BB_graph[par].status= BB_graph[par].INPUTS_IN_SCALAR
      for out in out_list:
        if out_scalar_status[out] == False:
          scalar_pos= scalar_reg_file.allocate_AppIdx(par)
          out_scalar_pos[out]= scalar_pos
          out_scalar_status[out]= True
          if write_asm:
            reporting_tools.write_asm(global_var, 'push_out_to_S', [str(curr_bb), 'pos:' + str(scalar_pos), 'par:', str(par) ])
        
        else:
          scalar_pos= out_scalar_pos[out]
          assert scalar_pos is not None
          scalar_reg_file.map_pos_to_new_AppIdx(scalar_pos, par)

    else:
      assert 0, "Should never come here"
  
  #for i in range(secondary_out_cnt/SECONDARY_OUT_PORTS):
  #  if write_asm:
  #    reporting_tools.write_asm(global_var, 'secondary_push', [str(curr_bb)])
  
  #if secondary_out_cnt % SECONDARY_OUT_PORTS:
  #    reporting_tools.write_asm(global_var, 'secondary_push', [str(curr_bb)])
  
  ## break combined_out_list in chunks of size N_OUT_PORTS
  combined_out_list_chunks=[] # List of lists  
  for idx, out in enumerate(combined_out_list):
    # Add a list for next chunk
    if (idx) % N_OUT_PORTS == 0:
      combined_out_list_chunks.append([])      
    
    combined_out_list_chunks[-1].append(out)
    
  
  # Print instruction
  if write_asm:
    for idx, chunks in enumerate(combined_out_list_chunks):
      reporting_tools.write_asm(global_var, 'BB_' + str(idx), [curr_bb] + [chunks])
    
    if not combined_out_list:
      reporting_tools.write_asm(global_var, 'BB_U', [curr_bb])
      

def load_scalar_to_vect(BB_graph, global_var, curr_par, scalar_reg_file, active_bb, vect_reg_file, curr_sch_level, SCAL_TO_VECT_PORTS, SCALAR_LVL_WINDOW, write_asm):
  """
  While an output of a BB is allocated to the vector regfile, all the corresponding elements of that vector register, that are currently on the scalar register should also be loaded to the vector register.
  However, this element on the scalar files, may be used in multiple vector regs. This function allocates all those vector regs as much as possible.
  Arguments:
    cur_par: This is the current par BB, the main vector register for which all the scalars have to be loaded
  """
  par= curr_par

  #--- Get the list of all the positions in scalar file that contains data for curr_par
  ret_dict= scalar_reg_file.read_by_AppIdx(par)
  pos_lst= ret_dict['pos']

  for pos in pos_lst:
    # Get list of all BBs for the pos
    BB_list= scalar_reg_file.get_AppIdx_for_pos(pos)
    
    # Sort BB_lis according to sch_level
    BB_list_sorted= sorted(BB_list, key= lambda x:BB_graph[x].sch_level, reverse= False)
    
    # Copy from scalar to vector in batches
    curr_batch_of_BB= []
    for BB in BB_list_sorted:
      
      # Check if BB is already allocated to the vector file
      if BB in active_bb:
        curr_batch_of_BB.append(BB)
      
      # Elif, check is BB is curr_par
      elif BB == par:
        curr_batch_of_BB.append(BB)

      # Else, check if sch_level of BB is within window
      elif BB_graph[BB].sch_level < curr_sch_level + SCALAR_LVL_WINDOW:
        curr_batch_of_BB.append(BB)
      
      # Perform actual copy
      if len(curr_batch_of_BB) == SCAL_TO_VECT_PORTS:
        load_scalar_to_vect_basic_op(BB_graph, global_var, pos, curr_batch_of_BB, scalar_reg_file, active_bb, vect_reg_file, write_asm)
        assert len(curr_batch_of_BB) == 0 
    # If curr_batch_of_BB is not empty, perform a last copy
    if curr_batch_of_BB:
      load_scalar_to_vect_basic_op(BB_graph, global_var, pos, curr_batch_of_BB, scalar_reg_file, active_bb, vect_reg_file, write_asm)
      assert len(curr_batch_of_BB) == 0 
    
  # As all the scalar elements are loaded, deallocate the curr_par in scalar file
  scalar_reg_file.deallocate_AppIdx(par)


def load_scalar_to_vect_basic_op(BB_graph, global_var, pos, curr_batch_of_BB, scalar_reg_file, active_bb, vect_reg_file, write_asm):
  # Perform actual copy
  if write_asm:
    reporting_tools.write_asm(global_var, 'cp_StV_vect', ['pos:' + str(pos), 'par_lst:' + str(curr_batch_of_BB)])
  
  # Unmap in scalar file and Allocate in vector file
  for AppIdx in curr_batch_of_BB:
    scalar_reg_file.deallocate_AppIdx(AppIdx, [pos])

    if AppIdx not in active_bb:
      active_bb.append(AppIdx)
      vect_reg_file.allocate_AppIdx(AppIdx)
      BB_graph[AppIdx].status= BB_graph[AppIdx].INPUTS_IN_VECTOR
  
  # Empty the batch
  del curr_batch_of_BB[:]

def load_leaf(graph, BB_graph, curr_bb, track_leaf_status, track_leaf_pos, scalar_reg_file, global_var, write_asm):
  VECT_LOAD_THRESHOLD= global_knobs.VECT_LOAD_THRESHOLD
  curr_bb_obj= BB_graph[curr_bb]
  
  only_this_par_lst= []
  if len(curr_bb_obj.leaf_inputs) <= VECT_LOAD_THRESHOLD:
    for leaf in curr_bb_obj.leaf_inputs:
      if track_leaf_status[leaf] == False: # Leaf not loaded to scalar mem
        sort_par= sorted(graph[leaf].parent_buildblk_lst, key= lambda x: BB_graph[x].sch_level, reverse= True) 
        
        if not curr_bb == sort_par[0]: # Need to load to scalar as there are multiple parents left
          
          # Check if other parent are loading vectors or not
          LOAD_DIRECT= True
          for par_bb in graph[leaf].parent_buildblk_lst:
            if par_bb != curr_bb:
              if len(BB_graph[par_bb].leaf_inputs) <= VECT_LOAD_THRESHOLD:
                LOAD_DIRECT= False

          # Not loading directly as there are other parent who would also need this leaf
          if not LOAD_DIRECT:
            pos= scalar_reg_file.allocate_AppIdx(curr_bb)
            
            for par_bb in graph[leaf].parent_buildblk_lst:
              if par_bb != curr_bb:
                if len(BB_graph[par_bb].leaf_inputs) <= VECT_LOAD_THRESHOLD:
                  scalar_reg_file.map_pos_to_new_AppIdx(pos, par_bb)
            
            if write_asm:
              reporting_tools.write_asm(global_var, 'ld', [str(leaf), str(pos)])

            track_leaf_status[leaf]= True
            track_leaf_pos[leaf]= pos
          
          # Load directly to the vector regfile
          else:
            if write_asm:
              reporting_tools.write_asm(global_var, 'ldw_v', [str(leaf), str(curr_bb)])

        else: #Directly load to the vect file
          only_this_par_lst.append(leaf)
    
    if len(only_this_par_lst) > VECT_LOAD_THRESHOLD:
      if write_asm:
        reporting_tools.write_asm(global_var, 'ldv_v', [str(only_this_par_lst), str(curr_bb)])
    else:
      for leaf in only_this_par_lst:
        if write_asm:
          reporting_tools.write_asm(global_var, 'ldw_v', [str(leaf), str(curr_bb)])
  else:
    if write_asm:
      reporting_tools.write_asm(global_var, 'ldv_v', [str(only_this_par_lst), str(curr_bb)])
    
    pos_lst= []
    for leaf in curr_bb_obj.leaf_inputs:
      if track_leaf_status[leaf]:
        pos= track_leaf_pos[leaf]
        if pos in scalar_reg_file.map_AppIdx_to_pos[curr_bb]:
          pos_lst.append(pos)
    
    scalar_reg_file.deallocate_AppIdx(curr_bb, pos_lst)

def BB_scheduling(graph, BB_graph, head_node, global_var, leaf_list, write_asm= False, make_vid= False):
  print("Scheduling building blocks (SCATTER)")
  if write_asm:
    # clear the asm file
    fp= open(global_var.ASM_FILE, 'w+')
    fp.close()
  
  head_bb= create_bb_dependecny_graph(graph, BB_graph) 
  compute_reverse_lvl(BB_graph, head_bb)
  map_sch_lvl_to_bb= {}
  assign_sch_lvl(BB_graph, head_bb, Mutable_level(), map_sch_lvl_to_bb)
  
  sch_lvl_lst= [obj.sch_level for bb,obj in  list(BB_graph.items())]
  
  # all the schedule levels should be unique
  assert len(sch_lvl_lst) == len(set(sch_lvl_lst)), "Repeated sch_level not allowed. Problem in assign_sch_lvl() method"
  assert None not in sch_lvl_lst, "sch_level should never be None"
  
  # reset computed bit
  for node, obj in list(graph.items()):
    obj.computed= False
  
  if make_vid:
    st= time.time() 
    vid_maker= reporting_tools.video_maker(global_var.VID_FILE , global_var.IMG_FILE_FOR_PYDOT)
    vid_maker.create_a_frame_scheduling(BB_graph)
    vid_maker.create_vid_obj()
    vid_maker.add_frame()
    print(time.time() -st)
  else:
    vid_maker= None
  
  reg_alloc(graph, BB_graph, head_node, global_var, leaf_list, write_asm, vid_maker, make_vid)
  
  #out_cnt= 0
  #par_cnt= 0
  #extra_par= 0
  #for bb, obj in BB_graph.items():
  #  out_for_bb = 0
  #  for out in obj.out_list:
  #    out_cnt += 1
  #    for par in graph[out].parent_buildblk_lst:
  #      par_cnt += 1
  #      
  #      if out_for_bb <= 6:
  #        out_for_bb += 1
  #      else:
  #        extra_par += 1
  #
  #print "out_cnt: ", out_cnt
  #print "Minimum Copies: ", par_cnt - out_cnt, extra_par

def trial(graph, BB_graph, head_node, leaf_list):
  """
  Function to try new ideas
  """
  bb_cnt=0
  in_tot=0
  
  out_tot=0
  out_cnt=0

  out_tot_l= 0
  out_cnt_l= 0
  
  out_tot_l_w= 0
  out_tot_l_w_cnt= 0
  out_tot_l_i= 0
  out_tot_l_i_cnt= 0
  
  out_cnt_list= []
  
  for leaf in leaf_list:
    obj= graph[leaf]
    out_cnt_l += 1
    out_tot_l += len(obj.parent_buildblk_lst)
    if obj.leaf_type== obj.LEAF_TYPE_WEIGHT:
      out_tot_l_w += len(obj.parent_buildblk_lst)
      out_tot_l_w_cnt += 1
    elif obj.leaf_type== obj.LEAF_TYPE_INDICATOR:
      out_tot_l_i += len(obj.parent_buildblk_lst)
      out_tot_l_i_cnt += 1
    else:
      assert 0

  for bb in BB_graph:
    obj= BB_graph[bb]
    bb_cnt += 1
    in_tot += len(obj.in_list_unique)
    
    for out in obj.out_list:
      out_cnt += 1
      out_tot += len(graph[out].parent_buildblk_lst)
      out_cnt_list.append(len(graph[out].parent_buildblk_lst))
  
  print('BB_cnt:', bb_cnt, "Total_inputs:", in_tot, 'Total outputs:',out_tot+out_tot_l)
  print("Avg leaf use:", float(out_tot_l)/float(out_cnt_l))
  print("Avg. Weight use:", float(out_tot_l_w)/ float(out_tot_l_w_cnt), out_tot_l_w_cnt)
  print("Avg. ind use:", float(out_tot_l_i)/ float(out_tot_l_i_cnt), out_tot_l_i_cnt)
  print("Avg. out use:", float(out_tot)/float(out_cnt))
  print("Avg. inputs: ", float(in_tot)/float(bb_cnt))
  print("Max. degree of out:", max(out_cnt_list))
  print("Min. degree of out:", min(out_cnt_list))


