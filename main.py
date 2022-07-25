
import sys
import argparse

import global_var
import src.graph_analysis

def run(args):
  analysis_obj= src.graph_analysis.graph_analysis_c() 
  analysis_obj.test(args)
  exit(0)

def main(argv=None):
  parser = argparse.ArgumentParser(description='Tree datapath based DAG processor')
  # parser.add_argument('--net', type=str, choices=[\
  #   'HB/ibm32'    , \
  #   'test_net'], \
  #   help='Enter the name of the network to be used as an input')

  # parser.add_argument('--cir_type', type=str, choices=['psdd', 'sptrsv', 'none'], default='none', help='Specify the type of circuit to be read. Default= none')
  
  parser.add_argument('--depth', type=int, default= 3, \
      choices= range(1, 4), \
      help='Depth of the trees (D)')

  parser.add_argument('--banks', type=int, default= 64, \
      choices= [8, 16, 32, 64], \
      help='Number of register banks (B)')

  parser.add_argument('--regs', type=int, default= 32, \
      choices= [16, 32, 64, 128, 256], \
      help='Number of registers per bank (R)')

  parser.add_argument('--tmode', type=str, \
      choices= [\
        'try', \
        'compile', \
        'rtl_sim', \
        'plot_charts', \
        ] , \
        help='mode')

  parser.add_argument('--targs', nargs= '*', help= 'Some tests may need additional arguments')

  args = parser.parse_args(argv)

  run(args)

if __name__ == "__main__":
  sys.exit(main())
