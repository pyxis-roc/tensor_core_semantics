#!/usr/bin/env python

# Author:       Benjamin Valpey
# Date:         21 Mar 2023
# Filename:     markidi.py
# Last Edited:  Thu 23 Mar 2023 11:58:44 AM EDT
# Description:  Implementation of Markidi's algorithm using the Volta TC semantics.

# We are computing the maximal error after just one step of markidi

from cvc5.pythonic import *
import functools
from fp_add import fp_add


def exponent_in_range(val, low, hi):
    lowest = FPVal(2 ** (low - 1), Float32())
    highest = FPVal(2 ** (hi + 1), Float32())
    # Assert that the value is not nan, and its exponent is less than max exponent
    # and its exponent is greater than minimum exponent
    return And(Not(fpIsNaN(val)), fpGT(fpAbs(val), lowest), fpLT(fpAbs(val), highest))


def markidi():

    # The exponent is in range of fp16.
    solver = Solver()
    solver.setOption("fp-exp", True)
    vals = []
    f16_vals = []
    delta_vals = []

    # 4 vals for a, 4 vals for b
    for _ in range(8):
        new_val = FreshConst(Float32())
        new_val_f16 = fpToFP(RTZ(), new_val, Float16())
        new_delta_val = fpSub(RTZ(), new_val, fpToFP(RNE(), new_val_f16, Float32()))
        vals.append(new_val)
        f16_vals.append(new_val_f16)
        delta_vals.append(new_delta_val)

    # now, compute the multiplication in fp32.

    a_mul_b = []
    delta_a_mul_b = []
    a_mul_delta_b = []
    delta_a_mul_delta_b = []
    for i in range(4):
        a_mul_b.append(
            fpMul(
                RNE(),
                fpToFP(RNE(), f16_vals[i], Float32()),
                fpToFP(RNE(), f16_vals[i + 4], Float32()),
            )
        )
        delta_a_mul_b.append(
            fpMul(
                RNE(),
                fpToFP(RNE(), delta_vals[i], Float32()),
                fpToFP(RNE(), vals[i + 4], Float32()),
            )
        )
        a_mul_delta_b.append(
            fpMul(
                RNE(),
                fpToFP(RNE(), vals[i], Float32()),
                fpToFP(RNE(), delta_vals[i + 4], Float32()),
            )
        )
        delta_a_mul_delta_b.append(
            fpMul(
                RNE(),
                fpToFP(RNE(), delta_vals[i], Float32()),
                fpToFP(RNE(), delta_vals[i + 4], Float32()),
            )
        )
    # we have all of the multiplications ready.  Now we do accumulation.
    # First, we have to have numbers in BV32 form
    step1_bv_muls = []
    step2_bv_muls = []
    step3_bv_muls = []
    step4_bv_muls = []
    for i in range(4):
        term1 = FreshConst(BitVecSort(32))
        term2 = FreshConst(BitVecSort(32))
        term3 = FreshConst(BitVecSort(32))
        term4 = FreshConst(BitVecSort(32))
        solver.add(fpBVToFP(term1, Float32()) == a_mul_b[i])
        solver.add(fpBVToFP(term2, Float32()) == delta_a_mul_b[i])
        solver.add(fpBVToFP(term3, Float32()) == a_mul_delta_b[i])
        solver.add(fpBVToFP(term4, Float32()) == delta_a_mul_delta_b[i])
        step1_bv_muls.append(term1)
        step2_bv_muls.append(term2)
        step3_bv_muls.append(term3)
        step4_bv_muls.append(term4)
    # C starts at 0.
    step1_bv_muls.append(BitVecVal(0, 32))

    # first accumulate step.
    step1_accumulate = fp_add(5, 3, step1_bv_muls)
    step2_bv_muls.append(step1_accumulate)
    step2_accumulate = fp_add(5, 3, step2_bv_muls)
    step3_bv_muls.append(step2_accumulate)
    step3_accumulate = fp_add(5, 3, step3_bv_muls)
    step4_bv_muls.append(step3_accumulate)
    markidi_res = fp_add(5, 3, step4_bv_muls)

    simt_res = fpMul(RNE(), vals[0], vals[4])

    simt_real_res = fpMul(
        RNE(), fpToFP(RTZ(), vals[0], Float64()), fpToFP(RTZ(), vals[4], Float64())
    )

    for i in range(1, 4):
        simt_res = fpFMA(RNE(), vals[i], vals[i + 4], simt_res)
        simt_real_res = fpFMA(
            RNE(),
            fpToFP(RTZ(), vals[i], Float64()),
            fpToFP(RTZ(), vals[i + 4], Float64()),
            simt_real_res,
        )


    # we multiply
    # We compute maximum relative error 
    markidi_abs_error = fpSub(RNE(), simt_real_res, fpToFP(RNE(), markidi_res, Float64()))
    simt_abs_error = fpSub(RNE(), simt_real_res, fpToFP(RNE(), simt_res, Float64()))

    # Ensure result is nonzero
    solver.add(Not(fpIsZero(simt_real_res)))

    markidi_error = fpDiv(RNE(), markidi_abs_error, simt_real_res)
    simt_error = fpDiv(RNE(), simt_abs_error, simt_real_res)

    # check if error can be greater than 10^-8

    # goal: get maximal error.

    
    # prove maximal error is less than 10 power -8
    # assert type 1 for all.
    for val in vals:
        solver.add(exponent_in_range(val, -15, 14))

    error_bound = FPVal(10**-24, Float64())
    if solver.check(fpGT(markidi_error, error_bound)) == sat:
        print("Markidi error greater than 10 pow -6")
    else:
        print("Markidi error less than 10 pow -6")
    # first, check if the error of markidi is less than that of simt
    if solver.check(fpLT(markidi_error, simt_error)) == sat:
        print("Markidi error less than simt error")
    else:
        print("Markidi error can never be less than simt error")


    


if __name__ == "__main__":
    markidi()
