#!/usr/bin/env python

# SPDX-FileCopyrightText: 2023-2024 University of Rochester
#
# SPDX-License-Identifier: LGPL-3.0-or-later

__author__ = "Benjamin Valpey"
__license__ = "LGPL-3.0-or-later"

from typing import List, Optional, Tuple
# from cvc5.pythonic import *

from z3 import *

import functools
import itertools
import time

FpClass = Datatype("FpClass")
FpClass.declare("fp_infinity")
FpClass.declare("fp_nan")
FpClass.declare("fp_normal")
FpClass.declare("fp_subnormal")
FpClass.declare("fp_zero")
FpClass = FpClass.create()


NaN = BitVecVal(0x7FC00000, 32)

DEBUGPRINT = False


def resetAssertions(func):
    def wrapper(s, *args, **kwargs):
        if getattr(s, "resetAssertions", None) is not None:
            s.resetAssertions()
        res = func(s, *args, **kwargs)

    return wrapper


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


class Term:
    def __init__(self, val, extra_bits: int):
        self.sign = Extract(val.size() - 1, val.size() - 1, val)
        self.exponent = Extract(30, 23, val)
        self.mantissa: BitVecRef = ZeroExt(
            extra_bits,
            Concat(
                If(self.exponent == 0, BitVecVal(0, 1), BitVecVal(1, 1)),
                Extract(22, 0, val),
            ),
        )

    def __str__(self):
        return f"{hex(int(str(evaluate(self.sign))))} {hex(int(str(evaluate(self.exponent))))} {hex(int(str(evaluate(self.mantissa))))}"


def dprint(*args):
    if DEBUGPRINT:
        print(*args)


# This is the algorithm used by NVIDIA Tensor Cores
# Any shifted bits are lost
def fp_add_two_truncate_shift(a: Term, bval, num_extra_bits):
    """
    Drops the significand when results are shifted. 
    This means that result is not correctly rounded with RTZ.
    If it was, then subtracting any nonzero number from a number would reduce (increase) the exponent
    E.g. in RTZ, subtracting 2^-24 from 2 would set exponent to 2^1 and have every bit in the significand set.
    """
    b = Term(bval, num_extra_bits)
    exponent_diff = ZeroExt(a.mantissa.size() - a.exponent.size(), a.exponent - b.exponent)

    shifted_b: BitVecRef = LShR(
        b.mantissa,
        exponent_diff,
    )

    is_add = Or(a.sign == b.sign, a.mantissa == 0)

    lsig = If(a.mantissa > shifted_b, a.mantissa, shifted_b)
    ssig = If(a.mantissa > shifted_b, shifted_b, a.mantissa)

    assert isinstance(lsig, BitVecRef) and isinstance(ssig, BitVecRef)

    result = If(is_add, lsig + ssig, lsig - ssig)
    lsign = If(
        And(a.mantissa == 0, b.mantissa == 0),
        a.sign | b.sign,
        If(
            # Adding the same mantissa results in a negative result.
            And(a.mantissa == shifted_b, Not(is_add)),
            BitVecVal(1, 1),
            If(a.mantissa > shifted_b, a.sign, b.sign),
        ),
    )
    return result, lsign


def fp_add_two_simple(a: Term, bval, num_extra_bits):
    b = Term(bval, num_extra_bits)
    # dprint("Adding two terms:", a, b)
    is_add = Or(a.sign == b.sign, a.mantissa == 0)

    exponent_diff = ZeroExt(a.mantissa.size() - a.exponent.size(), a.exponent - b.exponent)
    # dprint("Exponent diff is:", hex(int(str(evaluate(exponent_diff)))))
    # exponent_diff_is_zero = a.exponent - b.exponent == 0
    # exponent_diff_is_one = a.exponent - b.exponent == 1
    exponent_diff_gt_precision_p1 = UGT(a.exponent - b.exponent, 24)
    exponent_diff_gt_precision = UGT(a.exponent - b.exponent, 23)

    shifted_b: BitVecRef = LShR(
        b.mantissa,
        exponent_diff,
    )
    lsig = If(a.mantissa > shifted_b, a.mantissa, shifted_b)
    ssig = If(shifted_b > a.mantissa, a.mantissa, shifted_b)
    # dprint("lsig is:", hex(int(str(evaluate(lsig)))))
    # dprint("ssig is:", hex(int(str(evaluate(ssig)))))

    lsign = If(
        And(a.mantissa == 0, b.mantissa == 0),
        a.sign | b.sign,
        If(
            And(a.mantissa == shifted_b, Not(is_add)),
            BitVecVal(1, 1),
            If(a.mantissa > shifted_b, a.sign, b.sign),
        ),
    )

    # if a is zero, then we return shifted b and b's sign

    assert isinstance(lsig, BitVecRef) and isinstance(ssig, BitVecRef)

    add_result = lsig + ssig
    sub_result = If(
        a.mantissa == 0,
        shifted_b,
        If(
            a.mantissa == shifted_b,
            BitVecVal(0, a.mantissa.size()),
            lsig + (-b.mantissa),
        ),
    )
    # Why is it when we negate first and then add, we are off by one bit...
    # Let's see.... If we

    sub_result_negate_first = lsig + ((-b.mantissa) >> exponent_diff)

    sub_result = If(
        And(exponent_diff_gt_precision_p1, b.mantissa != 0),
        sub_result + BitVecVal(-1, a.mantissa.size()),
        sub_result,
    )

    shiftedStickyBit = If(
        exponent_diff_gt_precision,
        BitVecVal(1, 1),
        If(
            And(exponent_diff_gt_precision_p1, b.mantissa != 0),
            BitVecVal(1, 1),
            BitVecVal(0, 1),
        ),
    )

    sticky = If(
        Or(is_add, Extract(23, 23, sub_result) != 0),
        BitVecVal(0, 1),
        If(
            And(exponent_diff_gt_precision_p1, b.mantissa != 0),
            BitVecVal(1, 1),
            shiftedStickyBit,
        ),
    )

    result = If(is_add, add_result, sub_result_negate_first)

    return result, lsign, sticky


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
        assert len(terms) == nterms, "Incorrect number of terms"
        assert all(map(lambda x: isinstance(x, BitVecRef) and x.size() == 32, terms)), "Incorrect type of terms"
        make_terms = False

    for i in range(nterms):
        if make_terms:
            terms.append(Const(chr(97 + i), BitVecSort(32)))
        cls_list.append(fpClassify(terms[i]))

    # first, take the largest element.

    # this is the maximum Value.

    maxVal = functools.reduce(
        lambda x, y: If(
            UGT(Extract(30, 23, x), Extract(30, 23, y)),
            x,
            If(
                Extract(30, 23, x) == Extract(30, 23, y),
                If(UGE(Extract(22, 0, x), Extract(22, 0, y)), x, y),
                y,
            ),
        ),
        terms,
    )

    # give me index of maximum term
    maxValFound = BoolVal(False)
    maxIndex = BitVecVal(0, 32)
    for pos, t in enumerate(terms):
        maxIndex = If(maxValFound, maxIndex, If(maxVal == t, BitVecVal(pos, 32), maxIndex))
        maxValFound = Or(maxValFound, maxVal == t)

    # to start, find the term with the largest exponent.

    # now, we start summation
    accumulated = Term(maxVal, num_extra_bits)
    for pos, t in enumerate(terms):
        # dprint("Accumulated [before] is: ", accumulated)
        acc_mantissa, acc_sign = fp_add_two_truncate_shift(
            accumulated,
            If(
                maxIndex == BitVecVal(pos, 32),
                BitVecVal(0, terms[0].size()),
                t,
            ),
            num_extra_bits,
        )
        accumulated.mantissa = acc_mantissa
        accumulated.sign = acc_sign
        # dprint("Accumulated [after] is: ", accumulated)

    # print("Accumulated is:", accumulated)

    # Creates if statements that checks if the highest bit is set, and if so returns that bit position,
    # otherwise, checks if next highest bit is set, and so on.
    # Does not check against zero. That check must come before this.
    bit_pos: BitVecRef = functools.reduce(
        lambda x, y: If(
            accumulated.mantissa & BitVecVal(1 << y, accumulated.mantissa.size())
            == BitVecVal(1 << y, accumulated.mantissa.size()),
            y + 1,
            x,
        ),
        range(1, accumulated.mantissa.size()),
        BitVecVal(1, accumulated.mantissa.size()),
    )

    # dprint("accumulated mantissa is:", hex(int(str(evaluate(accumulated.mantissa)))))
    # print("accumulated mantissa is:", hex(int(str(evaluate(accumulated.mantissa)))))

    # calculate the exponent.  To do this, we see what the largest exponent is.

    shift_amount = bit_pos - BitVecVal(24, bit_pos.size())
    # dprint("shift_amount is:", int(str(evaluate(shift_amount))))

    # compute the new exponent and mantissa.
    # note that the exponent will always be in range.
    # dprint("Shift magnitude:", int(str(evaluate(shift_magnitude))))
    new_exponent = If(accumulated.mantissa == 0, 0, accumulated.exponent + Extract(7, 0, shift_amount))
    new_mantissa = If(
        bit_pos == 24,
        Extract(22, 0, accumulated.mantissa),
        If(
            bit_pos > 24,
            Extract(22, 0, accumulated.mantissa >> (bit_pos - 24)),
            Extract(22, 0, accumulated.mantissa << 24 - bit_pos),
        ),
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
        Concat(accumulated.sign, new_exponent, new_mantissa),
    )

    any_negative_infinity = Or(*list(map(lambda y: y == BitVecVal(0xFF800000, 32), terms)))
    any_positive_infinity = Or(*list(map(lambda y: y == BitVecVal(0x7F800000, 32), terms)))
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


def exact_float(term):
    return term.ast.getFloatingPointValue()[-1].toPythonObj()


class AccuracyTest:
    def __init__(
        self,
        name: str,
        expected,
        a=0,
        b=0,
        c=0,
        d=0,
        e=0,
    ):
        self.terms = [
            BitVecVal(a, 32),
            BitVecVal(b, 32),
            BitVecVal(c, 32),
            BitVecVal(d, 32),
            BitVecVal(e, 32),
        ]
        self.expected = BitVecVal(expected, 32)
        self.name = name
        self.result = "None"

    def check_sat(self):
        if is_sat((result := fp_add(5, 3, self.terms)) == self.expected):
            print(f"Sanity check {self.name}".ljust(50, "."), "[PASS]")
        else:
            print(
                f"Sanity check {self.name}".ljust(50, "."),
                "[FAIL]",
                "have:",
                hex(int(str(evaluate(result)))),
                "expect:",
                hex(int(str(evaluate(self.expected)))),
            )


def check_accuracy():
    """
    Sanity check for correctness of smt implementation.
    Checks that adding five 1s results in 5
    """
    one = 0x3F800000
    neg_one = 0xBF800000
    test_case = AccuracyTest("(adding five 1s yields 5)", 0x40A00000, one, one, one, one, one)
    test_case.check_sat()

    
    test_case = AccuracyTest(
        "(Xinyi test)",
        0x3F000001,
        one,
        0xBEFFFFFF
    )
    test_case.check_sat()

    test_case = AccuracyTest(
        "(adding four 1s and -1 yields 3)",
        0x40400000,
        one,
        neg_one,
        one,
        one,
        one,
    )
    test_case.check_sat()

    test_case = AccuracyTest(
        "(adding five -1s yields -5)",
        0xC0A00000,
        neg_one,
        neg_one,
        neg_one,
        neg_one,
        neg_one,
    )
    test_case.check_sat()

    test_case = AccuracyTest(
        "(adding 1, -1, 1, -1, 0 yields 0)",
        0x80000000,
        one,
        neg_one,
        one,
        neg_one,
    )
    test_case.check_sat()

    # test_case = AccuracyTest(
    #     "proper RTZ 1",
    #     0x38400000,
    #     0xB9800000,
    #     e=0x39980000,
    # )
    # test_case.check_sat()
    test_case = AccuracyTest("Proper RTZ 2", 0x517FFFFF, 0xC5800000, e=0x51800000)
    test_case.check_sat()

    test_case = AccuracyTest("Proper RTZ 3", 0xC0D7FFFF, 0xC0D80000, e=0x1)
    test_case.check_sat()

    test_case = AccuracyTest("Proper RTZ 4", 0x3EDFFE77, 0x3EDFFF3C, e=0xB6C4FFFF)
    test_case.check_sat()

    test_case = AccuracyTest("Proper RTZ 5", 0x37FFFFFE, 0x38000000, e=0xAC000010)
    test_case.check_sat()

    test_case = AccuracyTest("Proper RTZ 6", 0x7EFFFFFF, 0xA7800000, e=0x7F000000)
    test_case.check_sat()


@resetAssertions
def prove_exact_addition(s: Solver):
    terms = []
    a = FreshConst(Float16())
    a_bv = FreshConst(BitVecSort(16))
    b = FreshConst(Float16())
    b_bv = FreshConst(BitVecSort(16))
    c = FreshConst(Float32())
    c_bv = FreshConst(BitVecSort(32))
    ab = FreshConst(BitVecSort(32))
    res = FreshConst(BitVecSort(32))
    terms.append(ab)
    for i in range(3):
        terms.append(BitVecVal(0, 32))
    terms.append(c_bv)

    ab_prod = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    real_result = fpAdd(RNE(), ab_prod, c)
    real_result_bv = Const("real_result_bv", BitVecSort(32))

    for term in {a, b, c}:
        s.add(Not(fpIsInf(term)))
        s.add(Not(fpIsNaN(term)))
    s.add(Not(Or(fpIsNaN(fpBVToFP(ab, Float32())), ab == BitVecVal(0x7FFFFFFF, 32))))
    s.add(fpBVToFP(real_result_bv, Float32()) == real_result)
    s.add(fpBVToFP(ab, Float32()) == ab_prod)
    s.add(fpBVToFP(c_bv, Float32()) == c)
    s.add(fpBVToFP(a_bv, Float16()) == a)
    s.add(fpBVToFP(b_bv, Float16()) == b)
    res = fp_add(5, 3, terms)
    result_as_bv = fpBVToFP(res, Float32())
    s.add(Not(fpEQ(result_as_bv, real_result)))
    result = s.check()
    if result == sat:
        m = s.model()
        print("Not equivalent")
        print(f"Value for a: {m.eval(a)}")
        print("Value for a_bv:", hex(int(str(m.eval(a_bv)))))
        print(f"Value for b: {m.eval(b)}")
        print("Value for b_bv:", hex(int(str(m.eval(b_bv)))))
        print(f"Value for ab:", hex(int(str(m.eval(ab)))))
        print(f"ab_to_fp", m.eval(fpBVToFP(ab, Float32())))
        print(f"Value for ab_prod:", m.eval(ab_prod))
        print(f"Value for c: {m.eval(c)}")
        print(f"Value for c_bv:", hex(int(str(m.eval(c_bv)))))
        print(f"Our result: {m.eval(result_as_bv)}")
        print(f"Expected result: {m.eval(real_result)}")
        print(f"Expected result (bv):", hex(int(str(m.eval(real_result_bv)))))
    elif result == unsat:
        print("Equivalent")
    else:
        print("Solver responded with unknown:", s.reason_unknown())


@resetAssertions
def check_mul_in_fp16(s: Solver):
    print("[Exact Product]")
    a = Const("a", Float16())
    b = Const("b", Float16())

    # ensure a and b are not NaN or INF
    s.add(Not(Or(fpIsInf(a), fpIsNaN(a), fpIsInf(b), fpIsNaN(b))))

    prod_f16 = fpMul(RTZ(), a, b)
    prod_f32 = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    start = time.time()
    result = s.check(prod_f32 != fpToFP(RTZ(), prod_f16, Float32()))
    end = time.time()
    print("Query time: ", end - start, "s", sep="")
    if result == sat:
        m = s.model()
        print(f"a is: {exact_float(m.eval(a))} ({m.eval(a)}), b is: {exact_float(m.eval(b))} ({m.eval(b)})")
        print(f"Exact result is: {m.eval(prod_f32)}, Fp16 result is: {m.eval(prod_f16)}")
    elif result == unsat:
        print("Fp16 mul is always exact")
    else:
        print("Solver returned unknown:", s.reason_unknown())

    # produce numbers that, if the product was computed in fp16 would not be correct if done in fp32


@resetAssertions
def check_rounding_accumulator_abcd(s: Solver):
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
    total_queries = 0
    total_time = 0

    print("[Rounding mode of accumulation]")
    for pair in itertools.combinations(rm_dict.keys(), 2):
        print(f"\t[{pair[0]} vs {pair[1]}]")
        start = time.time()
        res = s.check(rm_dict[pair[0]] != rm_dict[pair[1]])
        end = time.time()
        total_time += end - start
        total_queries += 1
        if res == sat:
            m = s.model()
            print(f"\t\tValues: a = {m.eval(a)}, b = {m.eval(b)}, c = {m.eval(c)}, d = {m.eval(d)}")
            print(f"\t\tResult if {pair[0]}: {m.eval(rm_dict[pair[0]])}")
            print(f"\t\tResult if {pair[1]}: {m.eval(rm_dict[pair[1]])}")
        elif res == unsat:
            print("\t\tNo difference between {pair[0]} and {pair[1]}")
        else:
            print("\t\tSolver returned unknown:", s.reason_unknown())
    print("Query time: ", total_time / total_queries, "s", sep="")


@resetAssertions
def check_rounding_of_final_f16(s: Solver):
    a = FreshConst(Float16())
    b = FreshConst(Float16())
    s.add(Not(Or(fpIsInf(a), fpIsInf(b), fpIsNaN(a), fpIsNaN(b))))

    res = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    RNE_res = fpToFP(RNE(), res, Float16())
    RTZ_res = fpToFP(RTZ(), res, Float16())
    RTP_res = fpToFP(RTP(), res, Float16())
    RTN_res = fpToFP(RTN(), res, Float16())

    rm_dict = {"RNE": RNE_res, "RTP": RTP_res, "RTZ": RTZ_res, "RTN": RTN_res}
    total_queries = 0
    total_time = 0
    print("[Rounding mode of final result]")
    for pair in itertools.combinations(rm_dict.keys(), 2):
        print(f"\t[{pair[0]} vs {pair[1]}]")
        start = time.time()
        res = s.check(rm_dict[pair[0]] != rm_dict[pair[1]])
        end = time.time()
        total_time += end - start
        total_queries += 1
        if res == sat:
            m = s.model()
            print(
                f"\t\tValues: a = {m.eval(a)} ({exact_float(m.eval(a)):#06x}), "
                f"b = {m.eval(b)} ({exact_float(m.eval(b)):#06x})"
            )
            print(f"\t\tResult if {pair[0]}: {m.eval(rm_dict[pair[0]])} ({exact_float(m.eval(rm_dict[pair[0]])):#06x})")
            print(f"\t\tResult if {pair[1]}: {m.eval(rm_dict[pair[1]])} ({exact_float(m.eval(rm_dict[pair[1]])):#06x})")

        elif res == unsat:
            print("\t\tNo difference between {pair[0]} and {pair[1]}")
        else:
            print("\t\tSolver returned unknown:", s.reason_unknown())
    print("Query time: ", total_time / total_queries, "s", sep="")


@resetAssertions
def check_rounding_of_accumulator_abc(s: Solver):
    c = FreshConst(Float16())
    a = FreshConst(Float16())
    b = FreshConst(Float16())

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
    RNE_res = fpAdd(
        RNE(),
        fpToFP(RTZ(), c, Float32()),
        fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())),
    )
    RTZ_res = fpAdd(
        RTZ(),
        fpToFP(RTZ(), c, Float32()),
        fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())),
    )
    RTP_res = fpAdd(
        RTP(),
        fpToFP(RTZ(), c, Float32()),
        fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())),
    )
    RTN_res = fpAdd(
        RTN(),
        fpToFP(RTZ(), c, Float32()),
        fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())),
    )
    rm_dict = {"RNE": RNE_res, "RTP": RTP_res, "RTZ": RTZ_res, "RTN": RTN_res}
    total_time = 0
    total_queries = 0
    print("[Rounding mode of final result]")
    for pair in itertools.combinations(rm_dict.keys(), 2):
        print(f"\t[{pair[0]} vs {pair[1]}]")
        start = time.time()
        res = s.check(rm_dict[pair[0]] != rm_dict[pair[1]])
        end = time.time()
        total_time += end - start
        total_queries += 1
        if res == sat:
            m = s.model()
            print(
                f"\t\tValues: a = {m.eval(a)} ({exact_float(m.eval(a)):#06x}), "
                f"b = {m.eval(b)} ({exact_float(m.eval(b)):#06x}),"
                f"c = {m.eval(c)} ({exact_float(m.eval(c)):#06x})"
            )
            print(f"\t\tResult if {pair[0]}: {m.eval(rm_dict[pair[0]])} ({exact_float(m.eval(rm_dict[pair[0]])):#06x})")
            print(f"\t\tResult if {pair[1]}: {m.eval(rm_dict[pair[1]])} ({exact_float(m.eval(rm_dict[pair[1]])):#06x})")
        elif res == unsat:
            print("\t\tNo difference between {pair[0]} and {pair[1]}")
        else:
            print("\t\tSolver returned unknown:", s.reason_unknown())

    print("[Rounding mode of final result (positive inputs)]")
    for pair in itertools.combinations(rm_dict.keys(), 2):
        print(f"\t[{pair[0]} vs {pair[1]}]")
        start = time.time()
        res = s.check(
            rm_dict[pair[0]] != rm_dict[pair[1]],
            And(fpIsPositive(a), fpIsPositive(b), fpIsPositive(c)),
        )
        end = time.time()
        total_time += end - start
        total_queries += 1
        if res == sat:
            m = s.model()
            print(
                f"\t\tValues: a = {m.eval(a)} ({exact_float(m.eval(a)):#06x}), "
                f"b = {m.eval(b)} ({exact_float(m.eval(b)):#06x}),"
                f"c = {m.eval(c)} ({exact_float(m.eval(c)):#06x})"
            )
            print(f"\t\tResult if {pair[0]}: {m.eval(rm_dict[pair[0]])} ({exact_float(m.eval(rm_dict[pair[0]])):#06x})")
            print(f"\t\tResult if {pair[1]}: {m.eval(rm_dict[pair[1]])} ({exact_float(m.eval(rm_dict[pair[1]])):#06x})")
        elif res == unsat:
            print("\t\tNo difference between {pair[0]} and {pair[1]}")
        else:
            print("\t\tSolver returned unknown:", s.reason_unknown())

    print("Query time:", total_time / total_queries)


@resetAssertions
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

    product_is_reprensetable_in_f16 = fpToFP(RTZ(), fpToFP(RTZ(), exact_result, Float16()), Float32()) == exact_result
    start = time.time()
    res = s.check(And(product_is_reprensetable_in_f16, result_is_different))
    end = time.time()
    print("Query time:", end - start)
    if res == sat:
        m = s.model()
        print(f"a1 = {m.eval(a1)} ({exact_float(m.eval(a1)):#06x}), b1 = {m.eval(b1)} ({exact_float(m.eval(b1)):#06x})")
        print(f"a2 = {m.eval(a2)} ({exact_float(m.eval(a2)):#06x}), b2 = {m.eval(b2)} ({exact_float(m.eval(b2)):#06x})")
        print(
            f"Result in full precision: {m.eval(exact_result)} ({exact_float(m.eval(exact_result)):#010x}), "
            f"result in fp16: {m.eval(fp16_result)} ({exact_float(m.eval(fp16_result)):#06x})"
        )
    elif res == unsat:
        print("Unsat!")
    else:
        print("Solver reported unknown:", s.reason_unknown())


@resetAssertions
def check_accumulation_order(s: Solver):
    """
    Checks the accumulation order used by tensor cores.
    """

    a1 = FreshConst(Float16())
    a2 = FreshConst(Float16())
    b1 = FreshConst(Float16())
    b2 = FreshConst(Float16())
    c = FreshConst(Float32())

    s.add(
        Not(
            Or(
                fpIsNaN(a1),
                fpIsNaN(a2),
                fpIsNaN(b1),
                fpIsNaN(b2),
                fpIsNaN(c),
                fpIsInf(a1),
                fpIsInf(a2),
                fpIsInf(b1),
                fpIsInf(b2),
                fpIsInf(c),
            )
        )
    )
    s.add(
        And(
            fpIsPositive(a1),
            fpIsPositive(a2),
            fpIsPositive(b1),
            fpIsPositive(b2),
            fpIsPositive(c),
        )
    )

    ab1_prod = fpMul(
        RTZ(),
        fpToFP(RTZ(), a1, Float32()),
        fpToFP(RTZ(), b1, Float32()),
    )
    ab2_prod = fpMul(
        RTZ(),
        fpToFP(RTZ(), a2, Float32()),
        fpToFP(RTZ(), b2, Float32()),
    )

    # now, check
    res_ab_first = fpAdd(RTZ(), fpAdd(RTZ(), ab1_prod, ab2_prod), c)
    res_ab_second = fpAdd(RTZ(), ab1_prod, fpAdd(RTZ(), ab2_prod, c))

    start = time.time()
    res = s.check(res_ab_first != res_ab_second)
    end = time.time()
    print("Query time: ", end - start, "s", sep="")
    if res == sat:
        m = s.model()
        print(f"a1 = {m.eval(a1)} (), b1 = {m.eval(b1)} ({exact_float(m.eval(b1)):#06x})")
        print(f"a2 = {m.eval(a2)} ({exact_float(m.eval(a2)):#06x}), b2 = {m.eval(b2)} ({exact_float(m.eval(b2)):#06x})")
        print(f"c = {m.eval(c)} ({exact_float(m.eval(c)):#010x})")
        print(f"Result if (a+b) + c: {m.eval(res_ab_first)} ({exact_float(m.eval(res_ab_first)):#06x}), result if a+(b+c): {m.eval(res_ab_second)} ({exact_float(m.eval(res_ab_second)):#06x})")
    elif res == unsat:
        print("Accumulation is associative")
    else:
        print("Solver returned unknown:", s.reason_unknown())

@resetAssertions
def check_normalization(s: Solver):
    a1 = FreshConst(Float16())
    a2 = FreshConst(Float16())
    b1 = FreshConst(Float16())
    b2 = FreshConst(Float16())
    c = FreshConst(Float32())

    s.add(
        Not(
            Or(
                fpIsNaN(a1),
                fpIsNaN(a2),
                fpIsNaN(b1),
                fpIsNaN(b2),
                fpIsNaN(c),
                fpIsInf(a1),
                fpIsInf(a2),
                fpIsInf(b1),
                fpIsInf(b2),
                fpIsInf(c),
            )
        )
    )
    s.add(
        And(
            fpIsPositive(a1),
            fpIsPositive(a2),
            fpIsPositive(b1),
            fpIsPositive(b2),
            fpIsPositive(c),
        )
    )

    ab1_prod = fpMul(
        RTZ(),
        fpToFP(RTZ(), a1, Float32()),
        fpToFP(RTZ(), b1, Float32()),
    )
    ab1_prod_bv = FreshConst(BitVecSort(32))
    ab2_prod_bv = FreshConst(BitVecSort(32))
    s.add(fpBVToFP(ab1_prod_bv, Float32()) == ab1_prod)
    ab2_prod = fpMul(
        RTZ(),
        fpToFP(RTZ(), a2, Float32()),
        fpToFP(RTZ(), b2, Float32()),
    )
    s.add(fpBVToFP(ab2_prod_bv, Float32()) == ab2_prod)
    c_bv = FreshConst(BitVecSort(32))
    s.add(fpBVToFP(c_bv, Float32()) == c)

    terms = [ab1_prod_bv, ab2_prod_bv, BitVecVal(0, 32), BitVecVal(0, 32), c_bv]
    res = fp_add(5, 3, terms)
    
    res_assoc1 = fpAdd(RTZ(), fpAdd(RTZ(), ab1_prod, ab2_prod), c)
    s.add(fpBVToFP(res, Float32()) != res_assoc1)
    # res_assoc2 = fpAdd(RTZ(), ab1_prod, fpAdd(RTZ(), ab2_prod, c))
    # res_assoc3 = fpAdd(RTZ(), ab2_prod, fpAdd(RTZ(), ab1_prod, c))


    start = time.time()
    res = s.check()
    end = time.time()
    print("Query time: ", end - start, "s", sep="")
    if res == sat:
        m = s.model()
        print(f"a1 = {m.eval(a1)} (), b1 = {m.eval(b1)} ({exact_float(m.eval(b1)):#06x})")
        print(f"a2 = {m.eval(a2)} ({exact_float(m.eval(a2)):#06x}), b2 = {m.eval(b2)} ({exact_float(m.eval(b2)):#06x})")
        print(f"c = {m.eval(c)} ({exact_float(m.eval(c)):#010x})")
        print(f"Result if (a+b) + c: {m.eval(res_assoc1)} ({exact_float(m.eval(res_assoc1)):#06x})")
    elif res == unsat:
        print("Normalization is equivalent")
    else:
        print("Solver returned unknown:", s.reason_unknown())


@resetAssertions
def check_carry_bits(s: Solver, n: int, nbits: int):
    """
    Checks the amount of carry bits used by tensor cores. Assumes no normalization.
    
    :param s: The solver to use.
    :param n: The number of terms to be accumulated.
    :param nbits: The number of carry out bits to test against. (Checks `nbits` and `nbits-1`)
    """
    terms = []
    for i in range(n):
        terms.append(Const(chr(97 + i), BitVecSort(32)))

    a_inputs = []
    b_inputs = []
    for i in range(n):
        a = FreshConst(Float16())
        b = FreshConst(Float16())
        a_inputs.append(a)
        b_inputs.append(b)
        # NaNs and Infs poision our result, since they never compare equal.
        s.add(fpIsPositive(a))
        s.add(fpIsPositive(b))
        s.add(And(Not(fpIsNaN(a)), Not(fpIsInf(a)), Not(fpIsZero(a))))
        s.add(And(Not(fpIsNaN(b)), Not(fpIsInf(b)), Not(fpIsZero(b))))
        s.add(
            fpBVToFP(terms[i], Float32()) == fpMul(RTZ(), fpFPToFP(RTZ(), a, Float32()), fpFPToFP(RTZ(), b, Float32()))
        )
    c = FreshConst(Float32())
    s.add(fpBVToFP(terms[-1], Float32()) == c)
    s.add(And(Not(fpIsNaN(c)), Not(fpIsInf(c)), Not(fpIsZero(c))))
    s.add(fpIsPositive(c))

    # we have our assertions on the inputs to

    res_insufficient_bits = fp_add(n, nbits-1, terms)
    # need inputs which is exact in fp16, but for which you need extra bits
    res = fp_add(n, nbits, terms)

    start = time.time()
    result = s.check(res_insufficient_bits != res)
    end = time.time()

    print("Query time: ", end - start, "s", sep="")
    if result == sat:
        m = s.model()
        for i in range(len(a_inputs)):
            print(
                f"a_{i} = {m.eval(a_inputs[i])}".ljust(30),
                f"b_{i} = {m.eval(b_inputs[i])}",
            )
        c = m.eval(c)
        print(f"c = {c}")
    elif result == unsat:
        print(f"No values yield different results between {b} and {b-1} bits of carry.")
    else:
        # result is unknown
        print("Solver reported unknown:", s.reason_unknown())


@resetAssertions
def negative_zero_result(s: Solver):
    a = FreshConst(Float16())
    b = FreshConst(Float16())
    res = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    s.add(Not(fpIsZero(a)))
    s.add(Not(fpIsZero(b)))
    s.add(fpIsZero(fpToFP(RNE(), res, Float16())))
    s.add(fpIsNegative(fpToFP(RNE(), res, Float16())))
    start = time.time()
    result = s.check()
    end = time.time()
    print("Query time: ", end - start, "s", sep="")
    if response == sat:
        m = s.model()
        print("a", m.eval(a))
        print("b", m.eval(b))
    elif response == unsat:
        print("Unsat:")
    else:
        print("Solver reported unknown:", s.reason_unknown())


@resetAssertions
def prove_subnormal_fp16_inputs(s: Solver):
    a = FreshConst(Float16())
    b = FreshConst(Float16())
    res = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    s.add(fpIsSubnormal(a))
    s.add(fpIsSubnormal(b))
    s.add(Not(fpIsSubnormal(res)))
    start = time.time()
    response = s.check()
    end = time.time()

    print("Query time: ", end - start, "s", sep="")
    if response == sat:
        m = s.model()
        print("a", m.eval(a))
        print("b", m.eval(b))
        print("expected result: ", m.eval(res))
    elif response == unsat:
        print("Unsat:")
    else:
        print("Solver reported unknown:", s.reason_unknown())

@resetAssertions
def prove_subnormal_fp16_outputs(s: Solver):
    a = FreshConst(Float16())
    b = FreshConst(Float16())
    s.add(fpIsNormal(a))
    s.add(fpIsNormal(b))
    # If desired, make this rounding-mode agnostic by changing RTN() like in the mul tests
    res = fpToFP(RTN(), fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32())), Float16())
    s.add(fpIsSubnormal(res))
    
    start = time.time()
    response = s.check()
    end = time.time()

    print("Query time: ", end - start, "s", sep="")
    if response == sat:
        m = s.model()
        print("a", m.eval(a))
        print("b", m.eval(b))
        print("expected result: ", m.eval(res))
    elif response == unsat:
        print("Unsat:")
    else:
        print("Solver reported unknown:", s.reason_unknown())

@resetAssertions
def prove_subnormal_fp32_inputs(s: Solver):
    a = FreshConst(Float16())
    b = FreshConst(Float16())
    s.add(fpIsNormal(a))
    s.add(fpIsNormal(b))
    c = FreshConst(Float32())
    s.add(fpIsSubnormal(c))
    ab = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    res = fpAdd(RTZ(), ab, c)
    s.add(fpIsNormal(res))


    start = time.time()
    response = s.check()
    end = time.time()

    print("Query time: ", end - start, "s", sep="")
    if response == sat:
        m = s.model()
        print("a", m.eval(a))
        print("b", m.eval(b))
        print("expected result: ", m.eval(res))
    elif response == unsat:
        print("Unsat:")
    else:
        print("Solver reported unknown:", s.reason_unknown())


@resetAssertions
def prove_subnormal_fp32_outputs(s: Solver):
    a = FreshConst(Float16())
    b = FreshConst(Float16())
    s.add(fpIsNormal(a))
    s.add(fpIsNormal(b))
    c = FreshConst(Float32())
    s.add(fpIsNormal(c))
    ab = fpMul(RTZ(), fpToFP(RTZ(), a, Float32()), fpToFP(RTZ(), b, Float32()))
    fp32_res = fpAdd(RTZ(), ab, c)
    s.add(fpIsSubnormal(fp32_res))

    start = time.time()
    response = s.check()
    end = time.time()

    print("Query time: ", end - start, "s", sep="")
    if response == sat:
        m = s.model()
        print("a", m.eval(a))
        print("b", m.eval(b))
        print("c", m.eval(c))
        print("expected result: ", m.eval(fp32_res))
    elif response == unsat:
        # impossible to have normal inputs that have subnormal output
        print("Unsat:")
    else:
        print("Solver reported unknown:", s.reason_unknown())

def main():
    s = Solver()

    if getattr(s, "setOption", None) is not None:
        s.setOption("fp-exp", True)

    # check_accuracy()


    # negative_zero_result(s)
    # exit()
    # prove_exact_addition(s)

    # check_accuracy()
    # check_accumulation_order(s)

    # check_carry_bits(s, 5, 3)
    # check_carry_bits(s, 9, 2)
    # check_rounding_of_final_f16(s)
    # check_mul_in_fp16(s)
    # check_exact_in_fp16(s)
    # check_normalization(s)

    prove_subnormal_fp16_inputs(s)
    # prove_subnormal_fp16_outputs(s)
    # prove_subnormal_fp32_inputs(s)
    # prove_subnormal_fp32_outputs(s)

    # without normalization, we need the precise amount of extra bits in order to have sensible results.
    # otherwise, you can do things like adding a bunch of 1s together and get an incorrect result.
    # adding 4 1s and a -1 gives 3
    # print(is_sat(fp_add(5, 2, terms) == BitVecVal(0x40400000, 32)))


if __name__ == "__main__":
    main()

