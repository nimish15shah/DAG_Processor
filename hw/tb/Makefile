# Arguments to pass to the makefile
# compile time:
# FNAME: top filename of the testbench
#
# run time:
# NET: Name of the AC, e.g. asia, alarm etc.

FNAME?= ./testbench_top.sv
NET?=tretail
# PREFIX?=/esat/puck1/users/nshah/vcs_simulation_data/tb_data_TWO_WAY_PARTITION_TWO_WAY_FULL/
# PREFIX?=/esat/puck1/users/nshah/vcs_simulation_data/tb_data_TWO_WAY_PARTITION_TWO_WAY_LIMIT_LAYERS/
PREFIX?=/users/micas/nshah/Downloads/PhD/Academic/Bayesian_Networks_project/Hardware_Implementation/Auto_RTL_Generation/HW_files/scripts/graph_analysis_3/no_backup/reports/
W_CONFLICT?=35
EXTRA_ARGS?=+define+INSTR_PING_PONG

volume1= /volume1/users/nshah/synopsys_vcs

makeDirs= mkdir -p out log no_backup no_backup/tmp $(volume1)

#UVM_HOME= /users/micas/nshah/Downloads/no_backup/Setups/AccelleraUVM_1800.2-2017-1.0/src
UVM_HOME= $(VCS_HOME)/etc/uvm-1.1/src
UVM_FILE= $(UVM_HOME)/uvm.sv
UVM_DPI_FILE= $(UVM_HOME)/dpi/uvm_dpi.cc
UVM_OPTIONS= +UVM_NO_RELNOTES
UVM_command= +incdir+$(UVM_HOME) +incdir+$(UVM_DPI) $(UVM_FILE) $(UVM_DPI_FILE) $(UVM_OPTIONS)

INCDIR_0= ../src/ 
INCDIR_1= ../src/posit/

topmodule := tbench_top


#vcsBaseCommand= vcs -full64 -sverilog -timescale=1ns/10ps +delay_mode_zero\

#vcsBaseCommand= vcs -full64 -sverilog -timescale=1ns/1ps +notimingchecks \
								+transport_int_delays \

vcsBaseCommand= vcs -full64 -sverilog -timescale=1ns/1ps +neg_tchk \
								+define+no_warning \
								+define+NTC \
								+define+RECREM \
								+delay_mode_unit \
								+lint=PCWM -notice -Mdir=$(volume1)/csrc +lint=TFIPC-L \
								-top $(topmodule)
vcsFiles= -l ./log/compile \
-o ./out/simv \
+incdir+$(INCDIR_0) \
+incdir+$(INCDIR_1) \
 $(FNAME) 

vcsPower= +vcs+saif_libcell \
					-parallel+saif 


vcsCompileCommand =  $(vcsBaseCommand) $(UVM_command) $(vcsFiles) $(EXTRA_ARGS)

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
gtkwave:
	gtkwave $(volume1)/dump.vcd &

clean:
	\rm -rf simv* *.log *.vpd *.dump csrc *.sim *.mra *.log ucli.key session* *.db vcs.key out/simv* tmp DVEfiles vc_hdrs.h temp $(volume1)/csrc $(volume1)/vc_hdrs.h
