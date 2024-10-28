# DAM (Direct-Accessible Marshaller)

A simple serialization library that doesn't have dependencies,  and the data
is directly accessible in C/C++ without copying.

To make the data accessible in C, following rules must be followed,

1. Numeric values must be stored in native byte-order.
2. Numeric values must start at the 64-bit boundaries (8-bytes)
