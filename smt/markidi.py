#!/usr/bin/env python

# Author:       Benjamin Valpey
# Date:         21 Mar 2023
# Filename:     markidi.py
# Last Edited:  Fri 28 Apr 2023 03:01:21 PM EDT
# Description:  Implementation of Markidi's algorithm using the Volta TC semantics.

# We are computing the maximal error after just one step of markidi

from cvc5.pythonic import *
import functools
from fp_add import fp_add
import time


def exponent_in_range(val, low, hi):
    lowest = FPVal(2 ** (low - 1), Float32())
    highest = FPVal(2 ** (hi + 1), Float32())
    # Assert that the value is not nan, and its exponent is less than max exponent
    # and its exponent is greater than minimum exponent
    return And(Not(fpIsNaN(val)), fpGT(fpAbs(val), lowest), fpLT(fpAbs(val), highest))


def markidi_v_ootomo():

    # The exponent is in range of fp16.
    solver = Solver()
    solver.setOption("fp-exp", True)
    # the original 32 bit values (Expressed as Float32)
    vals = []
    # The values obtained from converting Float32 to Float16 in RTZ
    f16_vals = []
    # the delta values in markidi, obtained from converting f16_vals back to f32, and subtracting from original val,
    # then converting back to f16
    delta_vals = []
    # the delta vals in ootomo, obtained from converting f16_vals back to f32, subtracting from original val,
    # multiplying by 2^11, then converting back to f16
    ootomo_delta_vals = []

    # 4 vals for a, 4 vals for b
    for _ in range(8):
        # v_16 = ToFP16(v)
        new_val = FreshConst(Float32())
        solver.add(exponent_in_range(new_val, -35, -15))
        # Round original val to f16
        new_val_f16 = fpToFP(RTZ(), new_val, Float16())
        # dv = ToFP16(v - ToFP32(v_16))
        new_delta_val = fpToFP(
            RTZ(),
            fpSub(RTZ(), new_val, fpToFP(RNE(), new_val_f16, Float32())),
            Float16(),
        )
        # dv_ootomo = ToFP16((v - ToFP32(v_16)) * 2048)
        new_ootomo_delta_val = fpToFP(
            RTZ(),
            fpMul(
                RTN(),
                fpSub(RTZ(), new_val, fpToFP(RTZ(), new_val_f16, Float32())),
                FPVal(2048, Float32()),
            ),
            Float16(),
        )
        vals.append(new_val)
        f16_vals.append(new_val_f16)
        delta_vals.append(new_delta_val)
        ootomo_delta_vals.append(new_ootomo_delta_val)

    # now, compute the multiplication in fp32.

    original_a_vals = []
    original_b_vals = []
    for i in range(4):
        original_a_vals.append(FreshConst(BitVecSort(32)))
        original_b_vals.append(FreshConst(BitVecSort(32)))
        solver.add(fpBVToFP(original_a_vals[i], Float32()) == vals[i])
        solver.add(fpBVToFP(original_b_vals[i], Float32()) == vals[i+4])



    a_mul_b = []
    ootomo_da_mul_b = []
    ootomo_a_mul_db = []
    delta_a_mul_b = []

    a_mul_delta_b = []
    delta_a_mul_delta_b = []
    for i in range(4):
        # solver.add(Not(Or(fpIsNaN(vals[i]), fpIsNaN(vals[i+4]), fpIsInf(vals[i+4]), fpIsInf(vals[i]))))
        a_mul_b.append(
            fpMul(
                RNE(),
                fpToFP(RNE(), f16_vals[i], Float32()),
                fpToFP(RNE(), f16_vals[i + 4], Float32()),
            )
        )
        # rounding mode of multiplications shouldn't matter since values are all fp16
        ootomo_a_mul_db.append(
            fpMul(
                RNE(),
                fpToFP(RNE(), f16_vals[i], Float32()),
                fpToFP(RTZ(), ootomo_delta_vals[i + 4], Float32()),
            )
        )
        ootomo_da_mul_b.append(
            fpMul(
                RNE(),
                fpToFP(RNE(), ootomo_delta_vals[i], Float32()),
                fpToFP(RTZ(), f16_vals[i + 4], Float32()),
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

    ootomo_term1_vals = []
    ootomo_term2_vals = []
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

        # ootomo results
        ootomo_term1 = FreshConst(BitVecSort(32))
        ootomo_term2 = FreshConst(BitVecSort(32))

        # ootomo does da mul b first
        solver.add(fpBVToFP(ootomo_term1, Float32()) == ootomo_da_mul_b[i])
        solver.add(fpBVToFP(ootomo_term2, Float32()) == ootomo_a_mul_db[i])
        ootomo_term1_vals.append(ootomo_term1)
        ootomo_term2_vals.append(ootomo_term2)
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

    # Ootoma: compute a part of eq 24
    # c starts at 0
    ootomo_term1_vals.append(BitVecVal(0, 32))
    ootomo_term2_vals.append(fp_add(5, 3, ootomo_term1_vals))
    # line 25 of ootomo's code listing 3
    ootomo_dc = fp_add(5, 3, ootomo_term2_vals)

    # now, we compute the accumulation of a mul b

    ootomo_dc_res_float = FreshConst(Float32())
    ootomo_frag_res_float = FreshConst(Float32())
    solver.add(fpBVToFP(ootomo_dc, Float32()) == ootomo_dc_res_float)
    # step1_accumulate is the same, since it just multiplies a by b
    solver.add(fpBVToFP(step1_accumulate, Float32()) == ootomo_frag_res_float)

    # now do the accumulation.  But we multilpy dc res by 2^-11 (same as dividing by 2^11)
    ootomo_res = fpAdd(
        RNE(), ootomo_frag_res_float, fpMul(RTN(), ootomo_dc_res_float, 2**-11)
    )

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

    # step 1, multiply A and B, get result.
    # step 2,

    # we multiply
    # We compute maximum relative error
    markidi_abs_error = fpAbs(
        fpSub(RNE(), simt_real_res, fpToFP(RNE(), markidi_res, Float64()))
    )
    ootomo_abs_error = fpAbs(
        fpSub(RNE(), simt_real_res, fpToFP(RNE(), ootomo_res, Float64()))
    )
    simt_abs_error = fpAbs(
        fpSub(RNE(), simt_real_res, fpToFP(RNE(), simt_res, Float64()))
    )

    # Ensure result is nonzero
    solver.add(Not(fpIsZero(simt_real_res)))
    solver.add(Not(fpIsInf(simt_real_res)))
    solver.add(Not(fpIsInf(fpToFP(RNE(), ootomo_res, Float64()))))
    solver.add(Not(fpIsZero(fpToFP(RNE(), ootomo_res, Float64()))))
    solver.add(Not(fpIsInf(fpToFP(RNE(), markidi_res, Float64()))))

    markidi_error = fpDiv(RNE(), markidi_abs_error, simt_real_res)
    simt_error = fpDiv(RNE(), simt_abs_error, simt_real_res)

    start = time.time()
    res = solver.check(fpGT(ootomo_abs_error, markidi_abs_error))
    end = time.time()
    with open("markidi_type2_result.txt", "a") as outFile:
        print(f"Query finished after {end - start}s", file=outFile)
        if res == sat:
            m = solver.model()
            print("="*50, file=outFile)
            print("Ootomo's error can be greater than Markidis'. Below are example values", file=outFile)
            for i in range(4):
                print(f"a_{i}: {int(str(m.eval(original_a_vals[i]))):08x}", file=outFile)
                print(f"b_{i}: {int(str(m.eval(original_b_vals[i]))):08x}", file=outFile)
            print("Real result:", str(m.eval(simt_real_res)), file=outFile)
            print("Ootomo result:", str(m.eval(fpToFP(RNE(), ootomo_res, Float64()))), file=outFile)
            print("Markidis result:", str(m.eval(fpToFP(RNE(), markidi_res, Float64()))), file=outFile)

        elif res == unsat:
            print("Ootomo's error can never be greater than Markidi's.", file=outFile)
        else:
            print("Solver returned unknown:", solver.reason_unknown())


if __name__ == "__main__":
    markidi_v_ootomo()
