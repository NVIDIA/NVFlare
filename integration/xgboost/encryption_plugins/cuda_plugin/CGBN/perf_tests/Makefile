.PHONY: pick clean gmp-run gmp-numactl-run xmp-run kepler maxwell pascal volta turing ampere

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
	rm -f xmp_tester gmp_tester

gmp-run: gmp_tester
	@./gmp_tester

gmp-numactl-run: gmp_tester
	numactl --cpunodebind=0 ./gmp_tester

xmp-run: xmp_tester
	@./xmp_tester

gmp_tester:
	g++ $(INC) $(LIB) gmp_tester.cc -o gmp_tester -lgmp -fopenmp

xmp_tester: 
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo
	@exit 0


kepler:
	nvcc $(INC) $(LIB) -I../include -arch=sm_35 xmp_tester.cu -o xmp_tester -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../include -arch=sm_50 xmp_tester.cu -o xmp_tester -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../include -arch=sm_60 xmp_tester.cu -o xmp_tester -lgmp

volta:
	nvcc $(INC) $(LIB) -I../include -arch=sm_70 xmp_tester.cu -o xmp_tester -lgmp

turing:
	nvcc $(INC) $(LIB) -I../include -arch=sm_75 xmp_tester.cu -o xmp_tester -lgmp

ampere:
	nvcc $(INC) $(LIB) -I../include -arch=sm_80 xmp_tester.cu -o xmp_tester -lgmp

