ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif
ifdef GTEST_HOME
  GTEST_DIR := $(GTEST_HOME)
endif
ifndef GTEST_HOME
  GTEST_DIR := ../gtest
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
	rm -f libgtest.a gtest-all.o tester

libgtest.a: check
	g++ -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -pthread -std=c++11 -c $(GTEST_DIR)/src/gtest-all.cc
	ar -rv libgtest.a gtest-all.o
	rm gtest-all.o

kepler: libgtest.a
	nvcc $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -std=c++11 -arch=sm_35 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

maxwell: libgtest.a
	nvcc $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -std=c++11 -arch=sm_50 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

pascal: libgtest.a
	nvcc $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -std=c++11 -arch=sm_60 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

volta: libgtest.a
	nvcc $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -std=c++11 -arch=sm_70 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

turing: libgtest.a
	nvcc $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -std=c++11 -arch=sm_75 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

ampere: libgtest.a
	nvcc $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -std=c++11 -arch=sm_80 tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

check:
	@if [ -z "$(GTEST_HOME)" -a ! -d "../gtest" ]; then echo "Google Test framework required, see XMP documentation"; exit 1; fi

