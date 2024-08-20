.PHONY: pick clean download-gtest kepler maxwell pascal volta turing ampere check

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
	make -C samples clean
	make -C unit_tests clean
	make -C perf_tests clean

download-gtest:
	wget 'https://github.com/google/googletest/archive/master.zip' -O googletest-master.zip
	unzip googletest-master.zip 'googletest-master/googletest/*'
	mv googletest-master/googletest gtest
	rmdir googletest-master
	rm -f googletest-master.zip

kepler: check
	make -C samples kepler
	make -C unit_tests kepler
	make -C perf_tests kepler

maxwell: check
	make -C samples maxwell
	make -C unit_tests maxwell
	make -C perf_tests maxwell

pascal: check
	make -C samples pascal
	make -C unit_tests pascal
	make -C perf_tests pascal

volta: check
	make -C samples volta
	make -C unit_tests volta
	make -C perf_tests volta

turing: check
	make -C samples turing
	make -C unit_tests turing
	make -C perf_tests turing

ampere: check
	make -C samples ampere
	make -C unit_tests ampere
	make -C perf_tests ampere

check:
	@if [ -z "$(GTEST_HOME)" -a ! -d "gtest" ]; then echo "Google Test framework required, see documentation"; exit 1; fi

