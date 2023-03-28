#!/usr/bin/env python

# Author:       Benjamin Valpey
# Date:         21 Feb 2023
# Filename:     fp_add.py
# Last Edited:  Tue 28 Mar 2023 02:47:54 PM EDT
# Description:

from typing import List, Optional
from cvc5.pythonic import *

import functools
import itertools

FpClass = Datatype("FpClass")
FpClass.declare("fp_infinity")
FpClass.declare("fp_nan")
FpClass.declare("fp_normal")
FpClass.declare("fp_subnormal")
FpClass.declare("fp_zero")
FpClass = FpClass.create()


NaN = BitVecVal(0x7FC00000, 32)


def fpClassify(a: BitVecRef):
    exp = Extract(30, 23, a)
    mantissa = Extract(22, 0, a)
    return If(
        exp == BitVecVal(0xFF, 8),
        If(mantissa == BitVecVal(0, 23), FpClass.fp_infinity, FpClass.fp_nan),
        If(
            exp == BitVecVal(0, 8),
            If(mantissa == BitVecVal(0, 23), FpClass.fp_zero, FpClass.fp_subnormal),
            FpClass.fp_normal,
        ),
    )


def print_literal_as_hex(a: BitVecNumRef):
    print(f"{int(a.sexpr().replace('#b', ''), 2):#08x}")


def fp_add(nterms: int, num_extra_bits: int, terms: Optional[List] = None):
    """
    Generates the n-term floating point addition in smtlib.  Assumes round-to-zero, and no normaliation of intermediates

    :param nterms: The number of terms in the addition.  All but the last terms are assumed to be normal in FP32.
    There must be at least 3 terms.
    """
    assert nterms >= 3
    cls_list = []

    if terms is None:
        terms = []
        make_terms = True
    else:
        # ensure all of the terms are bitvectors of size 32.
        # also ensure we have the proper number of terms
        assert len(terms) == nterms and all(
            map(lambda x: isinstance(x, BitVecRef) and x.size() == 32, terms)
        )
        make_terms = False

    exponents = []
    mantissas = []

    for i in range(nterms):
        if make_terms:
            terms.append(Const(chr(97 + i), BitVecSort(32)))
        cls_list.append(fpClassify(terms[i]))

    for i in range(nterms - 1):
        mantissas.append(Concat(BitVecVal(1, 1), Extract(22, 0, terms[i])))
        exponents.append(
            If(
                FpClass.is_fp_zero(cls_list[i]),
                BitVecVal(1, 8),
                Extract(30, 23, terms[i]),
            )
        )

    exponents.append(
        If(
            Or(FpClass.is_fp_zero(cls_list[-1]), FpClass.is_fp_subnormal(cls_list[-1])),
            BitVecVal(1, 8),
            Extract(30, 23, terms[-1]),
        )
    )
    mantissas.append(
        Concat(
            If(FpClass.is_fp_subnormal(cls_list[-1]), BitVecVal(0, 1), BitVecVal(1, 1)),
            Extract(22, 0, terms[-1]),
        )
    )

    # we do reduce? UGT(a, b)
    largest_exponent = functools.reduce(
        lambda x, y: If(UGE(x, y), x, y), exponents[1:], exponents[0]
    )
    shifted_mantissas = []
    # now, shift all of the mantissas right appropriately
    for i in range(nterms):
        shifted_mantissas.append(
            LShR(
                mantissas[i],
                ZeroExt(
                    mantissas[i].size() - exponents[i].size(),
                    largest_exponent - exponents[i],
                ),
            )
        )

    # now, negate any of the terms that were negative.
    negated_mantissas = []

    # we always need at least 1 extra bit for the sign bit. This isn't used for carry in
    for i in range(nterms):
        negated_mantissas.append(
            If(
                Extract(31, 31, terms[i]) == BitVecVal(1, 1),
                SignExt(num_extra_bits + 1, -mantissas[i]),
                ZeroExt(num_extra_bits + 1, mantissas[i]),
            )
        )
    # for i in mantissas:
    # print_literal_as_hex(evaluate(i))

    mantissa_sum = functools.reduce(lambda x, y: x + y, negated_mantissas)
    # now we have mantissa sum.  If sign bit is set, then we do 2's complement
    sign_bit = Extract(mantissa_sum.size() - 1, mantissa_sum.size() - 1, mantissa_sum)
    twos_complement_mantissa = Extract(
        mantissa_sum.size() - 1,
        0,
        If(sign_bit == BitVecVal(1, 1), -mantissa_sum, mantissa_sum),
    )

    # Creates if statements that checks if the highest bit is set, and if so returns that bit position,
    # otherwise, checks if next highest bit is set, and so on.
    # Does not check against zero. That check must come before this.
    bit_pos: BitVecRef = functools.reduce(
        lambda x, y: If(
            twos_complement_mantissa
            & BitVecVal(1 << y, twos_complement_mantissa.size())
            == BitVecVal(1 << y, twos_complement_mantissa.size()),
            y + 1,
            x,
        ),
        range(1, twos_complement_mantissa.size()),
        BitVecVal(1, twos_complement_mantissa.size()),
    )

    # calculate the exponent.  To do this, we see what the largest exponent is.

    shift_amount = bit_pos - BitVecVal(24, bit_pos.size())

    # compute the new exponent and mantissa.
    # note that the exponent will always be in range.
    new_exponent = largest_exponent + Extract(7, 0, shift_amount)
    mantissa = If(
        bit_pos > 24,
        Extract(22, 0, twos_complement_mantissa >> (bit_pos - 24)),
        Extract(22, 0, twos_complement_mantissa << 24 - bit_pos),
    )

    any_negative_zero = Or(*list(map(lambda y: y == BitVecVal(0x80000000, 32), terms)))

    all_but_final_zero = And(*list(map(lambda y: FpClass.is_fp_zero(y), cls_list[:-1])))

    # if all terms except for the last are zero and the last term is nonzero, return the last term.
    # otherwise, if all terms are zero, then return positive zero unless there is at least one negative zero
    # otherwise (one of the terms excluding the last is nonzero), return the computed addition.

    compute_addition = If(
        all_but_final_zero,
        If(
            And(FpClass.is_fp_zero(cls_list[-1]), any_negative_zero),
            BitVecVal(0x80000000, 32),
            terms[-1],
        ),
        Concat(sign_bit, new_exponent, mantissa),
    )

    any_negative_infinity = Or(
        *list(map(lambda y: y == BitVecVal(0xFF800000, 32), terms))
    )
    any_positive_infinity = Or(
        *list(map(lambda y: y == BitVecVal(0x7F800000, 32), terms))
    )
    any_nan = Or(*list(map(lambda x: FpClass.is_fp_nan(x), cls_list)))
    infinity_check = If(
        And(any_positive_infinity, any_negative_infinity),
        NaN,
        If(
            any_positive_infinity,
            BitVecVal(0x7F800000, 32),
            If(any_negative_infinity, BitVecVal(0xFF800000, 32), compute_addition),
        ),
    )
    nan_check = If(any_nan, NaN, infinity_check)

    return nan_check


def check_accuracy():
    """
    Sanity check for correctness of smt implementation.
    Checks that adding five 1s results in 5
    """
    terms = []
    for _ in range(5):
        terms.append(BitVecVal(0x3F800000, 32))
    if is_sat(fp_add(5, 2, terms) == BitVecVal(0x40A00000, 32)):
        print("Sanity check (adding five 1s yields 5)".ljust(50, "."), "[PASS]")
    else:
        print("Sanity check (adding five 1s yields 5)".ljust(50, "."), "[FAIL]")

    terms[1] = BitVecVal(0xBF800000, 32)

    if is_sat(fp_add(5, 2, terms) == BitVecVal(0x40400000, 32)):
        print("Sanity check (adding four 1s and -1 yields 3)".ljust(50, "."), "[PASS]")
    else:
        print("Sanity check (adding four 1s and -1 yields 3)".ljust(50, "."), "[FAIL]")
    terms = []
    for _ in range(5):
        terms.append(BitVecVal(0xBF800000, 32))
    if is_sat(fp_add(5, 2, terms) == BitVecVal(0xC0A00000, 32)):
        print("Sanity check(adding five -1s yields -5)".ljust(50, "."), "[PASS]")
    else:
        print("Sanity check(adding five -1s yields -5)".ljust(50, "."), "[FAIL]")


def check_mul_in_fp16(s: Solver):
    print("[Exact Product]")
    a = Const("a", Float16())
    b = Const("b", Float16())

    # ensure a and b are not NaN or INF
    s.add(Not(Or(fpIsInf(a), fpIsNaN(a), fpIsInf(b), fpIsNaN(b))))

    prod_f16 = fpMul(RTZ(), a, b)
    prod_f32 = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    result = s.check(prod_f32 != fpToFP(RTZ(), prod_f16, Float32()))
    if result == sat:
        m = s.model()
        print(f"a is: {m.eval(a)}, b is: {m.eval(b)}")
        print(
            f"Exact result is: {m.eval(prod_f32)}, Fp16 result is: {m.eval(prod_f16)}"
        )
    elif result == unsat:
        print("Fp16 mul is always exact")
    else:
        print("Solver returned unknown:", s.reason_unknown())

    # produce numbers that, if the product was computed in fp16 would not be correct if done in fp32


def check_rounding_of_accumulator(s: Solver):
    a = Const("a", Float16())
    b = Const("b", Float16())
    c = Const("c", Float16())
    d = Const("d", Float16())

    s.add(
        Not(
            Or(
                fpIsInf(d),
                fpIsNaN(d),
                fpIsZero(d),
                fpIsInf(c),
                fpIsInf(a),
                fpIsInf(b),
                fpIsNaN(a),
                fpIsZero(a),
                fpIsZero(b),
                fpIsZero(c),
                fpIsNaN(b),
                fpIsNaN(c),
            )
        )
    )

    prod_ab = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    prod_cd = fpMul(RTZ(), fpToFP(RTZ(), c, Float32()), fpToFP(RTZ(), d, Float32()))

    RNE_res = fpAdd(RNE(), prod_ab, prod_cd)
    RTZ_res = fpAdd(RTZ(), prod_ab, prod_cd)
    RTN_res = fpAdd(RTN(), prod_ab, prod_cd)
    RTP_res = fpAdd(RTP(), prod_ab, prod_cd)

    rm_dict = {"RNE": RNE_res, "RTP": RTP_res, "RTZ": RTZ_res, "RTN": RTN_res}

    print("[Rounding mode of accumulation]")
    for pair in itertools.combinations(rm_dict.keys(), 2):
        print(f"\t[{pair[0]} vs {pair[1]}]")

        res = s.check(rm_dict[pair[0]] != rm_dict[pair[1]])
        if res == sat:
            m = s.model()
            print(f"\t\tValues: a = {m.eval(a)}, b = {m.eval(b)}, c = {m.eval(c)}, d = {m.eval(d)}")
            print(f"\t\tResult if {pair[0]}: {m.eval(rm_dict[pair[0]])}")
            print(f"\t\tResult if {pair[1]}: {m.eval(rm_dict[pair[1]])}")
        elif res == unsat:
            print("\t\tNo difference between {pair[0]} and {pair[1]}")
        else:
            print("\t\tSolver returned unknown:", s.reason_unknown())
    


def check_rounding_of_final(s: Solver):
    c = Const("c", Float16())
    a = Const("a", Float16())
    b = Const("b", Float16())

    # give me numbers that would be unique for each rounding mode.

    # assume multiplication is done in fp32
    s.add(
        Not(
            Or(
                fpIsInf(c),
                fpIsInf(a),
                fpIsInf(b),
                fpIsNaN(a),
                fpIsZero(a),
                fpIsZero(b),
                fpIsZero(c),
                fpIsNaN(b),
                fpIsNaN(c),
            )
        )
    )

    # check values that differ between RTZ and RTP
    # check values that differ between RNE and RTZ
    RNE_res = fpFPToFP(
        RNE(),
        fpAdd(
            RNE(),
            fpToFP(RTZ(), c, Float32()),
            fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())),
        ),
        Float16(),
    )
    RTZ_res = fpFPToFP(
        RTZ(),
        fpAdd(
            RTZ(),
            fpToFP(RTZ(), c, Float32()),
            fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())),
        ),
        Float16(),
    )
    RTP_res = fpFPToFP(
        RTP(),
        fpAdd(
            RTP(),
            fpToFP(RTZ(), c, Float32()),
            fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())),
        ),
        Float16(),
    )
    RTN_res = fpFPToFP(
        RTN(),
        fpAdd(
            RTN(),
            fpToFP(RTZ(), c, Float32()),
            fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())),
        ),
        Float16(),
    )
    rm_dict = {"RNE": RNE_res, "RTP": RTP_res, "RTZ": RTZ_res, "RTN": RTN_res}
    print("[Rounding mode of final result]")
    for pair in itertools.combinations(rm_dict.keys(), 2):
        print(f"\t[{pair[0]} vs {pair[1]}]")
        s.push()
        res = s.check(rm_dict[pair[0]] != rm_dict[pair[1]])
        s.pop()
        if res == sat:
            m = s.model()
            print(f"\t\tValues: a = {m.eval(a)}, b = {m.eval(b)}, c = {m.eval(c)}")
            print(f"\t\tResult if {pair[0]}: {m.eval(rm_dict[pair[0]])}")
            print(f"\t\tResult if {pair[1]}: {m.eval(rm_dict[pair[1]])}")
        elif res == unsat:
            print("\t\tNo difference between {pair[0]} and {pair[1]}")
        else:
            print("\t\tSolver returned unknown:", s.reason_unknown())
    with open("RM.smt2", "w") as outFile:
        print(s.)


def check_exact_in_fp16(s: Solver):
    """
    Determines inputs that test if the accumulation step of TCs uses exact addition or not,
    even when the accumulator is Fp16.
    Produces four inputs, a1, b1, a2, b2, such that a1*b1 + a2*b2 is exactly representable in fp16,
    but would not be exact if done in fp16.
    """
    print("[Exact Sum when Accumulator is FP16]")
    a1, b1, a2, b2 = Consts("a1 b1 a2 b2", Float16())
    # ensure none of the inputs are NaN or infinity.
    s.add(*list(map(lambda x: And(Not(fpIsNaN(x)), Not(fpIsInf(x))), [a1, b1, a2, b2])))

    ab1_16 = fpMul(RTZ(), a1, b1)
    ab1_32 = fpMul(RTZ(), fpToFP(RTZ(), a1, Float32()), fpToFP(RTZ(), b1, Float32()))
    ab2_16 = fpMul(RTZ(), a2, b2)
    ab2_32 = fpMul(RTZ(), fpToFP(RTZ(), a2, Float32()), fpToFP(RTZ(), b2, Float32()))

    exact_result = fpAdd(RTZ(), ab1_32, ab2_32)
    fp16_result = fpAdd(RTZ(), ab1_16, ab2_16)
    # assert ab_32 is in range

    # check if ab_32 is distinct from ab_16
    result_is_different = fpToFP(RTZ(), fp16_result, Float32()) != exact_result

    product_is_reprensetable_in_f16 = (
        fpToFP(RTZ(), fpToFP(RTZ(), exact_result, Float16()), Float32()) == exact_result
    )
    res = s.check(And(product_is_reprensetable_in_f16, result_is_different))
    if res == sat:
        m = s.model()
        print(f"a1 = {m.eval(a1)}, b1 = {m.eval(b1)}")
        print(f"a2 = {m.eval(a2)}, b2 = {m.eval(b2)}")
        print(
            f"Result in full precision: {m.eval(exact_result)}, result in fp16: {m.eval(fp16_result)}"
        )
    elif res == unsat:
        print("Unsat!")
    else:
        print("Solver reported unknown:", s.reason_unknown())


def check_carry_bits(s: Solver):
    """
    Checks the amount of carry bits used by tensor cores. Assumes no normalization
    """
    terms = []
    for i in range(5):
        terms.append(Const(chr(97 + i), BitVecSort(32)))

    a_inputs = []
    b_inputs = []
    for i in range(4):
        a = FreshConst(Float16())
        b = FreshConst(Float16())
        a_inputs.append(a)
        b_inputs.append(b)
        s.add(fpIsPositive(a))
        s.add(fpIsPositive(b))
        s.add(And(Not(fpIsNaN(a)), Not(fpIsInf(a)), Not(fpIsZero(a))))
        s.add(And(Not(fpIsNaN(b)), Not(fpIsInf(b)), Not(fpIsZero(b))))
        s.add(
            fpBVToFP(terms[i], Float32())
            == fpMul(
                RTZ(), fpFPToFP(RTZ(), a, Float32()), fpFPToFP(RTZ(), b, Float32())
            )
        )
    c = FreshConst(Float32())
    s.add(fpBVToFP(terms[-1], Float32()) == c)
    s.add(And(Not(fpIsNaN(c)), Not(fpIsInf(c)), Not(fpIsZero(c))))
    s.add(fpIsPositive(c))

    # we have our assertions on the inputs to

    res_2_bits = fp_add(5, 2, terms)
    # need inputs which is exact in fp16, but for which you need extra bits
    res_3_bits = fp_add(5, 3, terms)

    result = s.check(res_2_bits != res_3_bits)
    if result == sat:
        m = s.model()
        for i in range(len(a_inputs)):
            print(
                f"a_{i} = {m.eval(a_inputs[i])}".ljust(30),
                f"b_{i} = {m.eval(b_inputs[i])}",
            )
        print(f"c = {m.eval(c)}")
        print("res_2_bits is:", m.eval(fpBVToFP(res_2_bits, Float32())))
        print("res_3_bits is:", m.eval(fpBVToFP(res_3_bits, Float32())))
    elif result == unsat:
        print("No values yield different results between 2 and 3 bits of carry.")
    else:
        # result is unknown
        print("Solver reported unknown:", s.reason_unknown())


if __name__ == "__main__":
    # let's do 0.5 + 0.25 + 0.125 + 0.
    # first, check the addition of 1 holds
    s = Solver()
    s.setOption("fp-exp", True)
    check_accuracy()
    check_rounding_of_accumulator(s)

    # check_carry_bits(s)
    # check_rounding_of_final(s)
    # check_mul_in_fp16(s)
    # check_exact_in_fp16(s)

    # without normalization, we need the precise amount of extra bits in order to have sensible results.
    # otherwise, you can do things like adding a bunch of 1s together and get an incorrect result.
    # The amount of extra bits needed is at most log2(terms), although this is possibly too many, since

    # adding 4 1s and a -1 gives 3
    # print(is_sat(fp_add(5, 2, terms) == BitVecVal(0x40400000, 32)))

    # reset the main context so that we can add options to our solver.
    # main_ctx().reset()

    # now, give me 8 constants that

    # This is sat.  Print the values for a-d that result in this
