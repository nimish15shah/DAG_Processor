# Arguments to pass to the makefile
# compile time:
# FNAME: top filename of the testbench
#
# run time:
# NET: Name of the AC, e.g. asia, alarm etc.

FNAME?= ./testbench_top.sv
NET?=tretail
PREFIX?=../../out/instr/
W_CONFLICT?=35
TREE_DEPTH?=3
REG_BANK_DEPTH?=32
MIN_DEPTH?=2
N_TREE?=8
DATA_MEM_SIZE?=512 #KB
INSTR_MEM_SIZE?=2048 #KB
# EXTRA_ARGS?=+define+INSTR_PING_PONG
EXTRA_ARGS?=\
  +define+TREE_DEPTH=$(TREE_DEPTH) \
  +define+REG_BANK_DEPTH=$(REG_BANK_DEPTH) \
  +define+MIN_DEPTH=$(MIN_DEPTH) \
  +define+N_TREE=$(N_TREE) \
  +define+DATA_MEM_SIZE=$(DATA_MEM_SIZE) \
  +define+INSTR_MEM_SIZE=$(INSTR_MEM_SIZE)

makeDirs= mkdir -p out log no_backup no_backup/tmp csrc

INCDIR_0= ../rtl/ 

topmodule := tbench_top


#vcsBaseCommand= vcs -full64 -sverilog -timescale=1ns/10ps +delay_mode_zero\

#vcsBaseCommand= vcs -full64 -sverilog -timescale=1ns/1ps +notimingchecks \
								+transport_int_delays \

vcsBaseCommand= vcs -full64 -sverilog -timescale=1ns/1ps +neg_tchk \
								+define+no_warning \
								+define+NTC \
								+define+RECREM \
								+delay_mode_unit \
								+lint=PCWM -notice -Mdir=./csrc +lint=TFIPC-L \
								-top $(topmodule)
vcsFiles= -l ./log/compile \
-o ./out/simv \
+incdir+$(INCDIR_0) \
 $(FNAME) 

vcsPower= +vcs+saif_libcell \
					-parallel+saif 


vcsCompileCommand =  $(vcsBaseCommand) $(vcsFiles) $(EXTRA_ARGS)

# Pass FNAME name with "FNAME=xxx" command line argument
lint:
	$(makeDirs) 
	$(vcsCompileCommand)

compile: lint

compile_gate:
	$(makeDirs) 
	$(vcsCompileCommand) +define+GATE_NETLIST

	
compile_gui:
	$(makeDirs) 
	$(vcsCompileCommand) -debug_access+all -kdb 

run:
	./out/simv -l log/run_log \
		+NET_NAME=$(NET) +PREFIX=$(PREFIX) +W_CONFLICT=$(W_CONFLICT)\
		-licwait 1 \
		;

run_w_script:
	./out/simv -l log/run_log \
		+NET_NAME=$(NET) +PREFIX=$(PREFIX) +W_CONFLICT=$(W_CONFLICT);

#								-parallel+saif 

run_gui:
	./out/simv -gui -vpd_file ./no_backup/tmp/tb.vpd \
		+NET_NAME=$(NET) +PREFIX=$(PREFIX) +W_CONFLICT=$(W_CONFLICT) &
#								-parallel+saif 
			

full_gate: compile_gate run
full: compile run
full_gui: compile_gui run_gui
	

###############
# wave viewer #
###############
#Shows the waveforms dumped by a similator run in gtkwave
clean:
	\rm -rf simv* *.log *.vpd *.dump csrc *.sim *.mra *.log ucli.key session* *.db vcs.key out/simv* tmp DVEfiles vc_hdrs.h temp csrc vc_hdrs.h
