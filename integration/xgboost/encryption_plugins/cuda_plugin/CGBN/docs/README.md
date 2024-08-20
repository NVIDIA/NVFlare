### XMP 2.0 Beta Release (Oct 2018)

The XMP 2.0 library provides a set of APIs for doing fixed size, unsigned multiple precision integer arithmetic in CUDA.   The library provides these APIs under the name Cooperative Groups Big Numbers (CGBN).   The idea is that a cooperative group of threads will work together to represent and process operations on each big numbers.   This beta release targets high performance on small to medium sized big numbers:  32 bits through 32K bits (in 32 bit increments) and operates with 4, 8, 16 or 32 threads per CGBN group / big number instance.

### Why use CGBN?

CGBN imposes some constraints on the developer (discussed below), but within those constraints, it's **_really_** fast. 

In the following table, we compare the speed-up of CGBN on running a Tesla V100 (Volta) GPU vs. an Intel Xeon 20-Core E5-2698v4 running at 2.2 GHz 
with GMP 6.1.2 and OpenMP for parallelization:

|_operation_| 128 bits | 256 bits | 512 bits | 1024 bits | 2048 bits | 3072 bits | 4096 bits | 8192 bits |_avg speed-up_|
|-----------|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|:---------:|:---------:|:------------:|
|add        | 174.3    | 134.3    | 107.4    | 61.9      | 47.0      | 33.1      | 32.6      | 28.6      | 77.4         |
|sub        | 159.5    | 133.0    | 106.4    | 63.0      | 51.8      | 36.7      | 35.2      | 31.4      | 77.1         |
|mul (low)  | 172.9    | 50.9     | 30.0     | 17.8      | 22.6      | 19.4      | 20.2      | 14.6      | 43.5         |
|mont_reduce| 34.4     | 34.0     | 37.2     | 28.2      | 27.1      | 24.6      | 24.0      | 24.2      | 29.3         |
|powm_odd   | 22.1     | 24.6     | 24.5     | 21.0      | 22.0      | 19.8      | 20.1      | 16.8      | 21.4         |
|div_qr     | 30.0     | 20.4     | 18.7     | 11.8      | 9.5       | 8.3       | 8.9       | 7.8       | 14.4         |
|sqrt       | 27.0     | 17.1     | 16.8     | 9.4       | 7.2       | 4.8       | 4.3       | 3.4       | 11.3         |
|gcd        | 3.1      | 7.6      | 7.9      | 7.3       | 8.4       | 8.4       | 7.6       | 5.1       | 6.9          |
|mod inv    | 2.7      | 2.5      | 2.5      | 5.2       | 5.4       | 5.6       | 4.9       | 4.0       | 4.1          |

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; **Speed-Up Table:  Tesla V100 vs. Xeon E5-2698v4 at 2.2 GHz (20 cores)**

These performance results were generated with the perf_tests tools provided with the library.

### Installation

To install this package, create a directory for the CGBN files, and untar the CGBN-<date>.tar.gz package.

CGBN relies on two open source packages which must be installed before running the CGBN makefile.   These are the GNU Multiple Precision Library (GMP) and the Google Test framework (gtest).   If GMP is not installed as a local package on your system, you can built a local copy for your use as follows.

* Download GMP from http://www.gmplib.org
* Create a directory to hold the include files and library files
* Set the environment variable GMP_HOME to be your
* Configure GMP with `./configure --prefix=$GMP_HOME`
* Build and install GMP normally (we recommend that you also run make test).

If GMP is installed on your local system on the standard include and library paths, no action is needed.

CGBN also requires the Google Test framework source.  If this is installed on your system, set the environment variable GTEST_HOME to point to the source, if it's not installed, we provide a `make download-gtest` in the main CGBN makefile that will download and unpack the Google Test framework into the CGBN directory, where all the makefiles will find it automatically.

Once GMP and the Google Test framework are set up, the CGBN samples, unit tests, and performance tests can be built with `make <arch>` where _\<arch\>_ is one of kepler, maxwell, pascal, volta.   The compilation takes several minutes due to the large number of kernels that must built.  CGBN requires CUDA 9.2 for Volta and CUDA 9.0 (or later) for Kepler, Maxwell, and Pascal.

### Running Unit Tests

Once the unit tests have been compiled for the correct architecture, simply run the tester in the unit_tests directory.  This will run all tests on CUDA device zero.  To use a different GPU, set the environment variable CUDA_VISIBLE_DEVICES.

### Running the Performance Tests

Once the performance tests have been compiled for the correct architecture, simply run the xmp_tester.  This will performance test a number of core CGBN APIs, print the information in a easily readily form and write a **_gpu\_throughput\_report.csv_** file.   To generate GMP performance results for the same tests, run `make gmp-run` or `make gmp-numactl-run`.   The latter uses **numactl** to bind the GMP threads to a single CPU, for socket to socket comparisons.   The make targets gmp-run and gmp-numactl-run both print the report in a readable format as well as generate a **_cpu\_throughput\_report.csv_** file.   Speedups are easily computed by loading the two .csv files into a spreadsheet.

### Development - Getting Started

There are four samples included with the library, which can be found in the samples directory.  Sample 1 shows the simplest kernel, adding two vectors of CGBNs.  Sample 2 shows two implementations of a simple modular inverse algorithm (requires odd moduli).   Sample 3 shows how to write a fast powm kernel for odd moduli (this implementation is much faster than the one included in the library, but it requires an odd modulus, whereas the library API works for even moduli).  Sample 4 uses the same powm kernel to implement Miller-Rabin prime testing.

For reference documentation, please see the CGBN.md file in the docs directory.

### Limitations

The CGBN APIs currently have a number of limitations:

*  CGBN currently requires 4, 8, 16 or 32 thread per CGBN group.
*  Only sizes up to 32K bits are supported.  The size must be evenly divisible by 32.
*  Each cgbn_env_t can only be instantiated at a fixed size.  There is no support for mixing sizes within an environment (other than the cgbn_wide_t).
*  You can instantiate two cgbn_env_t instances in the same kernel with different sizes, but copying values between them is very slow.
*  Performance of some APIs (such as GCD) are not optimal.  Performance will likely improve in future releases.

### Questions and Comments

If you have any questions or comments, please drop a line to nemmart@nvidia.com and jluitjens@nvidia.com.  We look forward to your feedback.

