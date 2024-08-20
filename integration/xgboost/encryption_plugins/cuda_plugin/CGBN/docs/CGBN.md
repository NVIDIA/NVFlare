# Concepts

The typical use case for CGBN is to build kernels that process large arrays of problem instances where each problem instance requires unsigned multiple precision arithmetic computations.   A typical CUDA approach would be to assign a single thread to each problem instance, however, we find that multiple precision arithmetic requires a lot of register resources, and thus it's more efficient to spread a multiple precision value across a group of contiguous threads within a warp.  This is the premise behind CGBN.  

We begin with a sample kernel that is passed an array of problem instances and for each instance, the kernel computes the sum of **_a + b_** and stores the result in **_r_**, where **_a_**, **_b_**, and **_r_** are 1024-bit numbers.

```
#include "cgbn/cgbn.h"

// define a struct to hold each problem instance
typedef struct {  
   cgbn_mem_t<1024> a;
   cgbn_mem_t<1024> b;
   cgbn_mem_t<1024> r;
} problem_instance_t;

#define TPI 8  // threads per instance (can be 4, 8, 16 or 32)
               // IMPORTANT: do not define TPI before including "cgbn/cgbn.h", it'll cause compilation errors
                
// helpful typedefs for kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, 1024> env1024_t;

// define the kernel
__global__ void add_kernel(problem_instance_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context();                                 // create a CGBN context
  env1024_t         bn1024_env(bn_context);                       // construct a bn environment for 1024 bit math
  env1024_t::cgbn_t a, b, r;                                      // three 1024-bit values (spread across a warp)
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=count) return;                                  // return if my_instance is not valid
  
  // load a and b from global memory
  bn1024_env.load(a, &(problem_instances[my_instance]).a));
  bn1024_env.load(b, &(problem_instances[my_instance]).b));
  
  // add them!
  bn1024_env.add(r, a, b);
  
  // store the result back to global memory
  bn1024_env.store(&(problem_instances[my_instance].r), r);
}
```

This is about the simplest kernel one can write using CGBN.   In this example, the context object exists only to construct the bn1024\_env, but in more complex examples, it serves as a central object to report any CGBN errors that occur, such as division by zero, invalid Montgomery modulus, etc.

The bn1024\_env is a templated object that is instantiated with a context class and a number of bits.  It generates a full set of math APIs for doing unsigned integer arithmetic on big number of that size (1024 bits in this example).  It is important to note that in general, CGBN does not allow mixing precisions in a single API call.  So a bn1024\_env can't be used to add a 1024 bit number to a 512 bit number.   There are some exceptions to this rule, which are documented in the APIs below.

The threads of the kernel all start off by computing the **_my\_instance_** value, which is the absolute thread ID, `blockIdx.x * blockDim.x + threadIdx.x` divided by the number of threads per problem instance (TPI).   If my\_instance is beyond the end of the array, we just return.

Next, the kernel loads the 1024-bit input arguments from GPU global memory using `bn1024_env.load(...)`, and then runs `bn1024_env.add(...)` to compute the unsigned multiple precision addition of the arguments and `bn1024_env.store(...)` to store the resulting 1024-bit value back to GPU global memory.   

The code `bn1024_env.add(r, a, b)` probably looks a bit strange, in that the add method returns its result in **_r_** and doesn't make any changes to the bn1024\_env instance variable.   Essentially the bn1024\_env provides a set of _static_ method for doing arithmetic on the big numbers.  Unfortunately, they can't actually be declared as static since the environment needs instance variables in order to report errors back to the user.

After the alpha release, we received feedback that some developers would be more comfortable with a C style interface.  So the same kernel can be written as follows:

```
__global__ void add_kernel(problem_instance_t *problem_instances, uint32_t instance_count) {
  context_t         bn_context();                                 // create a CGBN context
  env1024_t         bn1024_env(bn_context);                       // construct a bn environment for 1024 bit math
  env1024_t::cgbn_t a, b, r;                                      // three 1024-bit values (spread across a warp)
  
  int32_t my_instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;  // determine my instance number
  
  if(my_instance>=count) return;                                  // return if my_instance is not valid
  
  cgbn_load(bn1024_env, a, &(problem_instances[my_instance]).a));
  cgbn_load(bn1024_env, b, &(problem_instances[my_instance]).b));
  
  cgbn_add(bn1024_env, r, a, b);
  
  cgbn_store(bn1024_env, &(problem_instances[my_instance].r), r);
}
```

Here we're calling C wrappers cgbn\_load, cgbn\_add, and cgbn\_store and passing the CGBN environment as the first argument, rather invoking the load, add and store methods of the CGBN environment.   Going forward, we will support both the C wrappers and directly invoking the CGBN environment methods, however, the documentation and samples have been updated to reflect the C style wrappers.

Hopefully, the rest of the example kernel is self explanatory.

The next step is to call the kernel.  This can be done as follows:

```
  #define TPB 128    // the number of threads per block to launch (must be divisible by 32
  
  ... assume instance_count has the number of problem instances ...
  ... assume gpu_instances is a pointer to the problem instances in GPU memory ...

  uint32_t IPB=TPB/TPI;   // instances per block (in this case 128/8 = 16)
  
  // we round up the number of required blocks
  add_kernel<<<(instance_count+IPB-1)/IPB, TPB>>>(gpu_instances, instance_count);
  
  ... synchronize the device, copy the problem instances back to CPU memory ...
```

For more complete examples, including proper error handling, please see the samples.

# Math API: cgbn_env_t calls

### Basic Arithmetic on CGBNs

##### Set (copy)

`void cgbn_set(cgbn_env_t env, cgbn_t &r, const cgbn_t &a)`

Copies the CGBN value of **_a_** into **_r_**. &nbsp; No return value.

##### Swap

`void cgbn_swap(cgbn_env_t env, cgbn_t &r, cgbn_t &a)`

Swaps the CGBN value of **_a_** and **_r_**. &nbsp; No return value.

##### Addition and Subtraction

`int32_t cgbn_add(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes **_a + b_** and stores the result in **_r_**.  &nbsp; If the sum resulted in a carry out, then 1 is returned to all threads in the group, otherwise return 0. 

---

`int32_t cgbn_sub(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes **_a - b_** and stores the result in **_r_**. &nbsp; If **_b>a_** then -1 is returned to all threads, otherwise return 0.

##### Multiplication

`void cgbn_mul(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes the low half of the product of **_a \* b_**, the upper half of the product is discarded.  &nbsp; This is the CGBN equivalent of unsigned multiplication in C.

---

`void cgbn_mul_high(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes the high half of the product of **_a \* b_**, the lower half of the product is discarded. 

---

`void cgbn_sqr(cgbn_env_t env, cgbn_t &r, const cgbn_t &a)`

Computes the low half of the product of **_a \* a_**, the upper half of the product is discarded.

---

`void cgbn_sqr_high(cgbn_env_t env, cgbn_t &r, const cgbn_t &a)`

Computes the high half of the product of **_a \* a_**, the lower half of the product is discarded. 

##### Division and Remainder

`void cgbn_div(cgbn_env_t env, cgbn_t &q, const cgbn_t &num, const cgbn_t &denom)`

Divide **_num_** by **_denom_** and store the resulting quotient into **_q_**.   This is the CGBN equivalent of unsigned division in C.

---

`void cgbn_rem(cgbn_env_t env, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom)`

Computes the remainder of **_num_** divided by **_denom_** and store the result into **_r_**, where **_0 <= r < denom_**.

---

`void cgbn_div_rem(cgbn_env_t env, cgbn_t &q, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom)`

Computes both the quotient and remainder of **_num_** divided by **_denom_**, such that **_num = q \* denom + r_**, where **_0 <= r < denom_**. This is typically faster than computing **_q_** and **_r_** separately. 

##### Square Root

`void cgbn_sqrt(cgbn_env_t env, cgbn_t &s, const cgbn_t &a)`

Computes the rounded-down square root of **_a_** and stores the result in **_s_**.

---

`void cgbn_sqrt_rem(cgbn_env_t env, cgbn_t &s, cgbn_t &r, const cgbn_t &a)`

Computes the rounded-down square root and remainder of **_a_**, such that **_a = s \* s + r_**, where **_0 <= r <= 2\*s_**.

##### Comparisons

`bool cgbn_equals(cgbn_env_t env, const cgbn_t &a, const cgbn_t &b)`

Returns true to all threads in the CGBN if **_a = b_**, false otherwise.

---

`int32_t cgbn_compare(cgbn_env_t env, const cgbn_t &a, const cgbn_t &b)`

Returns 1 to all threads in the CGBN if **_a > b_**, 0 if **_a = b_** and -1 if **_a < b_**.

##### Bitfield Extract and Insert

`void cgbn_extract_bits(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len)`

This operation is used to extract a bitfield from **_a_** and return the result in **_r_**.  The bitfield is defined by a zero-based **_start_** bit and field length, **_len_**.  

---

`void cgbn_insert_bits(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len, const cgbn_t &value)`

This operation copies **_a_** into **_r_**.  Then inserts **_value_** into the bitfield of **_r_** defined by **_start_** and **_len_**.   Note, if **_value_** is longer than **_len_**, only **_len_** bits are copied.

### CGBN Arithmetic with 32-bit Unsigned Integers

For correct operation of the APIs that follow, all threads in the group representing each CGBN must pass the same values for the `uint32_t` arguments, otherwise the results shall be undefined.   Likewise, all APIs will return the same `uint32_t` value to all threads in the CGBN group.

##### Get / Set

`uint32_t cgbn_get_ui32(cgbn_env_t env, const cgbn_t &a)`

Returns the least significant 32 bits of **_a_** to all threads in the group.

---

`void cgbn_set_ui32(cgbn_env_t env, cgbn_t &r, const uint32_t value)`

Sets **_r_** to **_value_**.

##### Addition and Subtraction

`int32_t cgbn_add_ui32(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t add)`

Computes **_a + add_** and returns the result in **_r_**.  If the addition results in a carry out, the call returns 1 to all threads in the CGBN group, otherwise returns 0.

---

`int32_t cgbn_sub_ui32(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t sub)`

Computes **_a - sub_** and returns the result in **_r_**.  Returns -1 to all threads in the group if **_a < sub_**, otherwise returns 0.

##### Multiplication

`uint32_t cgbn_mul_ui32(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t mul)`

Computes **_mul \* a_** which is returned in **_r_**.   The high word of the product, i.e., **_(mul \* a)>>bits_**, is returned to all threads in the CGBN group.

##### Division and Remainder

`uint32_t cgbn_div_ui32(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t div)`

Sets **_r = a / div_**, rounded down, and returns the remainder to all threads in the CGBN.

---

`uint32_t cgbn_rem_ui32(cgbn_env_t env, const cgbn_t &a, const uint32_t div)`

Returns the remainder of **_a / div_** to all threads in the CGBN.

##### Comparisons

`bool cgbn_equals_ui32(cgbn_env_t env, const cgbn_t &a, const uint32_t value)`

Returns true to all threads in the CGBN if **_a = value_**, false otherwise.

---

`int32_t cgbn_compare_ui32(cgbn_env_t env, const cgbn_t &a, const uint32_t value)`

Returns 1 to all threads in the CGBN if **_a > value_**, 0 if **_a = value_**, and -1 if **_a < value_**.

##### Bitfield Extract and Insert

`uint32_t cgbn_extract_bits_ui32(cgbn_env_t env, const cgbn_t &a, const uint32_t start, const uint32_t len)`

This operation is used to extract a bitfield from **_a_** and return the result to all threads in the CGBN group.  The bitfield is defined by a zero-based **_start_** bit and field length, **_len_**.  At most 32 bits are extracted.

---

`void cgbn_insert_bits_ui32(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t start, const uint32_t len, const uint32_t value)`

This operation copies **_a_** into **_r_**.  Then inserts **_value_** into the bitfield of **_r_** defined by **_start_** and **_len_**.   Note, if **_value_** is longer than **_len_**, only **_len_** bits are copied.

##### Binary Inverse

`uint32_t cgbn_binary_inverse_ui32(cgbn_env_t env, const uint32_t n0)`

Computes the 32-bit binary inverse of **_n0_** which must be odd.  This is a very fast routine and is used for computing in conjection with Montgomery reductions.

##### GCD

`uint32_t cgbn_gcd_ui32(cgbn_env_t env, const cgbn_t &a, const uint32_t value)`

Computes the GCD of **_a_** and **_value_**.  There are two special cases.  If **_a_ = 0** the routine returns **_value_**.  If **_value = 0_**, the routine returns zero.

### Wide Arithmetic Routines

There are several multiple precision arithmetic APIs which can be naturally expressed with twice the precision.  We call these the _wide_ routines and are described below.

##### Multiplication

`void cgbn_mul_wide(cgbn_env_t env, cgbn_wide_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes the full product (both low and high halves) of **_a \* b_** and stores the result in **_r_**.

---

`void cgbn_sqr_wide(cgbn_env_t env, cgbn_wide_t &r, const cgbn_t &a)`

Computes the full square product (both low and high halves) of **_a \* a_** and stores the result in **_r_**.

##### Division and Remainder 

`void cgbn_div_wide(cgbn_env_t env, cgbn_t &q, const cgbn_wide_t &num, const cgbn_t &denom)`

Computes the quotient, **_q = num / denom_**.  Note, the caller must ensure than the high CGBN of **_num_** is less than the denominator, **_denom_**.

---

`void cgbn_rem_wide(cgbn_env_t env, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom)`

Computes the remainder of **_num_** divided by **_denom_**.  Note, the caller must ensure that the high CGBN of **_num_** is less than the denominator, **_denom_**.

---

`void cgbn_div_rem_wide(cgbn_env_t env, cgbn_t &q, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom)`

Computes the quotient and remainder of **_num_** divided by **_denom_**.  Note, the caller must ensure that the high CGBN of **_num_** is less than the denominator, **_denom_**.

##### Square Root 

`void cgbn_sqrt_wide(cgbn_env_t env, cgbn_t &s, const cgbn_wide_t &a)`

Computes the square root of the wide value in **_a_** and stores the result in **_s_**.
---

`void cgbn_sqrt_rem_wide(cgbn_env_t env, cgbn_t &s, cgbn_wide_t &r, const cgbn_wide_t &a)`

Computes the square root of the wide value in **_a_** and the remainder **_r = a - s \* s_**.

### Masking and Logical Shifting 

`void cgbn_bitwise_and(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Compute the logical `and` of **_a_** and **_b_** and return the result in **_r_**.

---

`void cgbn_bitwise_ior(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Compute the logical inclusive `or` of **_a_** and **_b_** and return the result in **_r_**.

---

`void cgbn_bitwise_xor(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Compute the logical exclusive `or` of **_a_** and **_b_** and return the result in **_r_**.

---

`void cgbn_bitwise_complement(cgbn_env_t env, cgbn_t &r, const cgbn_t &a)`

Compute the logical complement of **_a_** and return the result in **_r_**.

---

`void cgbn_bitwise_mask_copy(cgbn_env_t env, cgbn_t &r, const int32_t numbits)`

This routine constructs a bitmask in **_r_** as follows.  If **_numbits_** is positive, then the least significant **_numbits_** of **_r_** will be set to one, and the rest set to zero.  If **_numbits_** is negative, then the most significant **_-numbits_** of **_r_** are set to one and the least significant bits set to zero.

---

`void cgbn_bitwise_mask_and(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const int32_t numbits)`

Constructs a mask using `bitwise_mask_copy` then computes the logical `and` of the mask and **_a_**, returning the result in **_r_**.

---

`void cgbn_bitwise_mask_ior(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const int32_t numbits)`

Constructs a mask using `bitwise_mask_copy` then computes the logical inclusive `or` of the mask and **_a_**, returning the result in **_r_**.

---

`void cgbn_bitwise_mask_xor(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const int32_t numbits)`

Constructs a mask using `bitwise_mask_copy` then computes the logical exclusive `or` of the mask and **_a_**, returning the result in **_r_**.

---

`void cgbn_shift_left(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t numbits)`

Shifts **_a_** to the left by **_numbits_**, filling with zeroes, and stores the result in **_r_**.

---

`void cgbn_shift_right(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t numbits)`

Shifts **_a_** to the right by **_numbits_**, filling with zeroes, and stores the result in **_r_**.

---

`void cgbn_rotate_left(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t numbits)`

Performs a circular rotate of **_a_** to the left by **_numbits_**, and stores the result in **_r_**.

---

`void cgbn_rotate_right(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const uint32_t numbits)`

Performs a circular rotate of **_a_** to the right by **_numbits_**, and stores the result in **_r_**.


### Bit Counting Routines

`uint32_t cgbn_pop_count(cgbn_env_t env, const cgbn_t &a)`

Counts the number of one bits in **_a_**, returns the count to all threads in the CGBN group.

---

`uint32_t cgbn_clz(cgbn_env_t env, const cgbn_t &a)`

Counts the number of leading (most significant) zero bits in **_a_**, returns the count to all threads in the CGBN group.

---

`uint32_t cgbn_ctz(cgbn_env_t env, const cgbn_t &a)`

Counts the number of trailing (least significant) zero bits in **_a_**, returns the count to all threads in the CGBN group.


### Accumulator Routines

One pattern that occurs frequently is computing the sum of a sequence of CGBNs.  Repeated calls to `add` and `sub` can often be accelerated by introducing an accumulator.   The accumulator uses a redundant representation, which is left unresolved until a call to `resolve`.

---

`void cgbn_set(cgbn_env_t env, cgbn_accumulator_t &accumulator, const cgbn_t &value)`

Used to set or initialize an accumulator to **_value_**.
---

`void cgbn_add(cgbn_env_t env, cgbn_accumulator_t &accumulator, const cgbn_t &value)`

Adds **_value_** to the accumulator.

---

`void cgbn_sub(cgbn_env_t env, cgbn_accumulator_t &accumulator, const cgbn_t &value)`

Subtracts **_value_** from the accumulator.

---

`void cgbn_set_ui32(cgbn_env_t env, cgbn_accumulator_t &accumulator, const uint32_t value)`

Used to set or initialize an accumulator to a 32-bit unsigned value.

---

`void cgbn_add_ui32(cgbn_env_t env, cgbn_accumulator_t &accumulator, const uint32_t value)`

Adds an unsigned 32-bit value to an accumulator.

---

`void cgbn_sub_ui32(cgbn_env_t env, cgbn_accumulator_t &accumulator, const uint32_t value)`

Subtracts an unsigned 32-bit value from an accumulator.

---

`int32_t cgbn_resolve(cgbn_env_t env, cgbn_t &sum, const cgbn_accumulator_t &accumulator)`

Compute the final value of the accumulator and return the result in **_sum_**.  Internally this routine resolves the redundant representation of the accumulator.   The return value is the sum of all the carry-outs and borrow-outs, and can be thought of as the high word of the accumulator.

### Number Theoretic Functions

The number theoretic functions are currently a little bit slow and should be used sparingly.  We expect the performance of these routines to improve significantly in future releases.

`void cgbn_binary_inverse(cgbn_env_t env, cgbn_t &r, const cgbn_t &x)`

Computes the modular inverse of **_x_** mod **_2<sup>bits</sup>_**.  Requires that **_x_** is odd.

---

`void cgbn_gcd(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b)`

Computes the GCD of **_a_** and **_b_**.  Return **_a_** if **_b = 0_**, returns **_b_** if **_a = 0_**.

---

`bool cgbn_modular_inverse(cgbn_env_t env, cgbn_t &r, const cgbn_t &x, const cgbn_t &m)`

Computes the modular inverse of **_x_** mod **m**.  Returns true if the inverse exists, false if not.  **_r_** is undefined if the inverse does not exist.

---

`void cgbn_modular_power(cgbn_env_t env, cgbn_t &r, const cgbn_t &x, const cgbn_t &e, const cgbn_t &m)`

Computes **_r = x^e_** modulo the modulus, **_m_**.  Requires that **_x < m_**.  

### Montgomery Reduction Routines (Common Modulus)

Montgomery reductions are a technique accelerate modular arithmetic, when the modulus is common across many operations.  The CGBN APIs provide routines to convert to and from Montgomery space, and to compute products in Montgomery space.

For example, assume we have two values, **_a_**, **_b_** and an odd modulus **_m_**.  We can compute the product of **_a \* b mod m_** as follows:
```
  cgbn_t   r, a, b, m;
  uint32_t np0;

  // convert a and b to Montgomery space
  np0=cgbn_bn2mont(bn_env, a, a, m);
  cgbn_bn2mont(bn_env, b, b, m);

  cgbn_mont_mul(bn_env, r, a, b, m, np0);
  
  // convert r back to normal space
  cgbn_mont2bn(bn_env, r, r, m, np0);
```
where bn_env is our CGBN environment.

The `bn2mont` routine returns the Montgomery value np0, which is required for the other API calls.  As an alternative, np0 can also be computed as follows: 
```
   np0 = -cgbn_binary_inverse(bn_env, cgbn_get_ui32(bn_env, m));
```

---

`uint32_t cgbn_bn2mont(cgbn_env_t env, cgbn_t &mont, const cgbn_t &bn, const cgbn_t &n)`

Converts an **_bn_** to Montgomery space using the modulus **_n_**.  **_n_** must be odd.  Requires that **_bn < n_**.  Returns np0 for future calls.

---

`void cgbn_mont2bn(cgbn_env_t env, cgbn_t &bn, const cgbn_t &mont, const cgbn_t &n, const uint32_t np0)`

Converts a value in Montgomery space back to normal space.  Requires the np0 value returned by the `bn2mont` call.

---

`void cgbn_mont_mul(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &b, const cgbn_t &n, const uint32_t np0)`

Computes the Montgomery product of **_a \* b mod n_**.   Requires the np0 value returned by the `bn2mont` call.

---

`void cgbn_mont_sqr(cgbn_env_t env, cgbn_t &r, const cgbn_t &a, const cgbn_t &n, const uint32_t np0)`

Computes the Montgomery product of **_a \* a mod n_**.   Requires the np0 value returned by the `bn2mont` call.

---

`void cgbn_mont_reduce_wide(cgbn_env_t env, cgbn_t &r, const cgbn_wide_t &a, const cgbn_t &n, const uint32_t np0)`

Takes a CGBN wide value, **_a_** and divides it by **_2<sup>bits</sup>_** modulu **_n_**.

### Barrett Reduction Routines (Common Divisors or Moduli)

The Barrett reduction routines can be used to accelerate computations where the same divisor is used repeatedly.  They based on the following idea.  We can precompute an approximation to the inverse of the denominator, which allows the code to swap multiplications for divisions, which tend to be faster.

For example, we can use the APIs to compute a Barrett reduction, **_a \* b mod d_** as follows:
```
  cgbn_t      r, a, b, d, approx;
  cgbn_wide_t w;
  uint32_t    clz_count;
  
  // assume d is a non-zero divisor, and a and b are less than d
  
  // compute the approximation of the inverse
  clz_count=cgbn_barrett_approximation(bn_env, approx, d);
  
  // compute the wide product of a*b
  cgbn_mul_wide(bn_env, w, a, b);
  
  // compute r=a*b mod d.  Pass the clz_count returned by the approx routine.
  cgbn_barrett_rem_wide(bn_env, r, w, d, approx, clz_count);
```

---

`uint32_t cgbn_barrett_approximation(cgbn_env_t env, cgbn_t &approx, const cgbn_t &denom)`

Computes the approximation of the inverse required for the other Barrett routines and returns `clz` of **_denom_**.

---

`void cgbn_barrett_div(cgbn_env_t env, cgbn_t &q, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz)`

Computes the quotient using the Barrett inverse.  Semantics are the same as `div(q, num, denom)`.  Pass the **_denom\_count_** returned by the barrett_approximation.

---

`void cgbn_barrett_rem(cgbn_env_t env, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz)`

Computes the remainder using a Barrett reduction.  Semantics are the same as `rem(r, num, denom)`.

---

`void cgbn_barrett_div_rem(cgbn_env_t env, cgbn_t &q, cgbn_t &r, const cgbn_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz)`

Computes both the quotient and remainder using the Barrett inverse.  Semantics are the same as `div_rem(q, r, num, denom)`.

---

`void cgbn_barrett_div_wide(cgbn_env_t env, cgbn_t &q, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz)`

Computes the quotient using the Barrett inverse.  Semantics are the same as `div_wide(q, num, denom)`. 

---

`void cgbn_barrett_rem_wide(cgbn_env_t env, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz)`

Computes the remainder using a Barrett reduction.  Semantics are the same as `rem_wide(r, num, denom)`.

---

`void cgbn_barrett_div_rem_wide(cgbn_env_t env, cgbn_t &q, cgbn_t &r, const cgbn_wide_t &num, const cgbn_t &denom, const cgbn_t &approx, const uint32_t denom_clz)`

Computes both the quotient and remainder using the Barrett inverse.  Semantics are the same as `div_rem_wide(q, r, num, denom)`.
