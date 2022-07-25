#!/usr/bin/python3

#file namings:
import os
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_PATH= './data/'
RTL_PATH= './hw/rtl/'
TB_PATH= './hw/tb/'

OUT_PATH= './out/'
INSTR_PATH= OUT_PATH + 'instr/'
PLOTS_PATH= OUT_PATH + 'plots/'
REPORTS_PATH= OUT_PATH + 'reports/'
logger.info(f"making directories {OUT_PATH}, {INSTR_PATH}, {PLOTS_PATH}, {REPORTS_PATH} if not already present")
os.system(f'mkdir -p {OUT_PATH}')
os.system(f'mkdir -p {INSTR_PATH}')
os.system(f'mkdir -p {PLOTS_PATH}')
os.system(f'mkdir -p {REPORTS_PATH}')
