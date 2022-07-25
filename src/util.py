
from . import FixedPointImplementation as lib

def custom_precision_operation(in_0, in_1, hw_details, operation):
  dtype= hw_details.DTYPE
  assert dtype in ['default', 'flt', 'posit']
  assert operation in ['sum', 'prod']

  if dtype == 'default':
    if operation == 'sum':
      return in_0 + in_1
    elif operation == 'prod':
      return in_0 * in_1

  elif dtype == 'flt':
    EXP_L= hw_details.EXP_L
    MANT_L= hw_details.MANT_L
    if operation == 'sum':
      return lib.flt_add(in_0, in_1, EXP_L, MANT_L, denorm= False, verb= False)
    elif operation == 'prod':
      return lib.flt_mul(in_0, in_1, EXP_L, MANT_L, denorm= False, verb= False)

  elif dtype == 'posit':
    POSIT_L= hw_details.POSIT_L
    POSIT_ES= hw_details.POSIT_ES
    lib.set_posit_env(POSIT_L, POSIT_ES)
    if operation == 'sum':
      return lib.posit_add(in_0, in_1)
    elif operation == 'prod':
      return lib.posit_mul(in_0, in_1)
  
  assert 0

def convert_data(data, hw_details, mode):
  dtype= hw_details.DTYPE
  assert dtype in ['default', 'flt', 'posit']
  
  assert mode in ['to', 'from'] # to custom, from custom

  if dtype == 'default':
    return data
  
  elif dtype == 'flt':
    EXP_L= hw_details.EXP_L
    MANT_L= hw_details.MANT_L

    if mode == 'to':
      return lib.flt_to_custom_flt(data, EXP_L, MANT_L, denorm= 0)
    else:
      return lib.custom_flt_to_flt(data, EXP_L, MANT_L, denorm= 0)

  elif dtype == 'posit':
    POSIT_L= hw_details.POSIT_L
    POSIT_ES= hw_details.POSIT_ES
    lib.set_posit_env(POSIT_L, POSIT_ES)
    if mode == 'to':
      data= float(data)
      converted_data= lib.flt_to_posit(data)
      assert (converted_data >> hw_details.POSIT_L) == 0, f"0x{converted_data:0x} {hw_details.POSIT_L} {data}"
      return converted_data
    else:
      return lib.posit_to_flt(data)
    

  else:
    assert 0

