<!--
SPDX-FileCopyrightText: 2024 University of Rochester

SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Semantics of Tensor Cores

This repository contains tests that evaluate the behavior of tensor cores on NVIDIA GPUs.

The tests themselves can be found in the `hw-tests` directory. To build, simply type `make`.
This will compile using `-arch=sm_70` by default. This can be overridden by passing ARCH=... when invoking make.
E.g., to explicitly compile for `sm_75`, invoke make as: `make ARCH=SM_75`

This will create a file, `tc_numeric_test.out` that can be used to examine the behavior of the tensor cores under various inputs.
For NaN tests, there is an additional target that will exhaustively test all NaN bit patterns. `make exhaustive_nan_test.out` will build this target.

When running the experiments, you must specify the device you wish to target as an argument. By default, it will assume a device with sm_70. To instead run on a different device, e.g., Turing,
invoke via `./tc_numeric_test.out 75`
The code will look for the first available cuda device that matches the provided SM, and will use it for execution.

Note that it is not necessary to recompile the cuda code for each sm. The cuda runtime will compile the device code on the fly when the sm version is not present.


The `smt` directory contains a sample implementation for the V100 gpu.