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
	rm -f mr

kepler:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 miller_rabin.cu -o mr -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 miller_rabin.cu -o mr -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 miller_rabin.cu -o mr -lgmp

volta: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 miller_rabin.cu -o mr -lgmp

turing: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 miller_rabin.cu -o mr -lgmp

ampere: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 miller_rabin.cu -o mr -lgmp

