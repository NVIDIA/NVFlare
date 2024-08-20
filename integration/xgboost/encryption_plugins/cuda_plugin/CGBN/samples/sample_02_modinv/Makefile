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
	rm -f mo mo_faster

kepler:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 modinv_odd.cu -o mo -lgmp
	nvcc $(INC) $(LIB) -I../../include -arch=sm_35 modinv_odd_faster.cu -o mo_faster -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 modinv_odd.cu -o mo -lgmp
	nvcc $(INC) $(LIB) -I../../include -arch=sm_50 modinv_odd_faster.cu -o mo_faster -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 modinv_odd.cu -o mo -lgmp
	nvcc $(INC) $(LIB) -I../../include -arch=sm_60 modinv_odd_faster.cu -o mo_faster -lgmp

volta: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 modinv_odd.cu -o mo -lgmp
	nvcc $(INC) $(LIB) -I../../include -arch=sm_70 modinv_odd_faster.cu -o mo_faster -lgmp

turing: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 modinv_odd.cu -o mo -lgmp
	nvcc $(INC) $(LIB) -I../../include -arch=sm_75 modinv_odd_faster.cu -o mo_faster -lgmp

ampere: 
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 modinv_odd.cu -o mo -lgmp
	nvcc $(INC) $(LIB) -I../../include -arch=sm_80 modinv_odd_faster.cu -o mo_faster -lgmp

