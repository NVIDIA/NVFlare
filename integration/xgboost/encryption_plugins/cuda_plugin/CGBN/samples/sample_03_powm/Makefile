ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo

clean:
	rm -f powmo

kepler:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 powm_odd.cu -o powmo -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 powm_odd.cu -o powmo -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 powm_odd.cu -o powmo -lgmp

volta: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 powm_odd.cu -o powmo -lgmp

turing: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 powm_odd.cu -o powmo -lgmp

ampere: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 powm_odd.cu -o powmo -lgmp

