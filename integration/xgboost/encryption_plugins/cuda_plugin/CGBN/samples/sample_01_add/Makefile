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
	rm -f add

kepler:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 add.cu -o add -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 add.cu -o add -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 add.cu -o add -lgmp

volta: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 add.cu -o add -lgmp

turing: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 add.cu -o add -lgmp

ampere: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 add.cu -o add -lgmp

