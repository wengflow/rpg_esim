#pragma once

#include <esim/common/types.hpp>

namespace exp_converging_line {

using Scalar = FloatType;

/**
 * Evaluates the exponentially-converging line equation, given by
 * y = mx + c(1 - exp(-wx)) & returns y.
 **/
Scalar eval(Scalar x, Scalar m, Scalar w, Scalar c);

/**
 * Evaluates & returns the exponentially-converging offset of the
 * exponentially-converging line equation.
 **/
Scalar evalOffset(Scalar x, Scalar w, Scalar c);

/**
 * Checks whether the exponentially-converging line equation has a critical
 * point y with zero gradient at some x >= 0, assuming w > 0.
 **/
Scalar hasCriticalPt(Scalar m, Scalar w, Scalar c);

/**
 * Evaluates & returns the critical point y of the exponentially-converging
 * line equation, assuming it exists.
 **/
Scalar evalCriticalPt(Scalar m, Scalar w, Scalar c);

/**
 * Solves the exponentially-converging line equation & returns the smallest
 * satisfying x >= 0, assuming real x & y, w > 0, (m != 0) || (c != 0), and a
 * solution exists.
 **/
Scalar solve(Scalar y, Scalar m, Scalar w, Scalar c);

}   // namespace exp_converging_line