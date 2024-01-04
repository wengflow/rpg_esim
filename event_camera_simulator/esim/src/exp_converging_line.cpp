#include <esim/esim/exp_converging_line.hpp>
#include <cmath>
#include <boost/math/special_functions/lambert_w.hpp>

namespace exp_converging_line {

Scalar eval(Scalar x, Scalar m, Scalar w, Scalar c) {
  return m * x + c * (1 - std::exp(-w * x));
};

Scalar evalOffset(Scalar x, Scalar w, Scalar c) {
  return c * (1 - std::exp(-w * x));    // eval(x, 0, w, c)
}

Scalar hasCriticalPt(Scalar m, Scalar w, Scalar c) {
  return (w * c / m <= -1);
}

Scalar evalCriticalPt(Scalar m, Scalar w, Scalar c) {
  const Scalar val = -m / (w * c);
  return c * (val * std::log(val) + (1 - val));
}

Scalar solve(Scalar y, Scalar m, Scalar w, Scalar c) {
  // asserts
  CHECK_GT(w, 0);
  CHECK(m != 0 || c != 0);

  // special cases (c or m == 0, or std::isinf(w))
  if (c == 0)
    return y / m;
  else if (std::isinf(w))
    return (y - c) / m;
  else if (m == 0)
    return 1/w * std::log(c / (c - y));

  // else {
  // non-special cases (c and m != 0, and !std::isinf(w))
  const Scalar val = w * c / m;
  const Scalar offset_val = val - w * y / m;
  const Scalar z = val * std::exp(offset_val);

  /**
   * NOTE:
   *    When `offset_val` is very negative or very positive, `z` saturates to 0
   *    or infinity and may yield grossly incorrect `x_wm1` or `x_w0` solutions,
   *    respectively, due to limited precision. We alleviate this issue by
   *    approximating the exponentially-converging line equation using 2 line
   *    segments below, as necessary:
   *      1. y = (m + wc)x, for x <= 1/w (linearization at x = 0)
   *      2. y = mx + c   , for x >  1/w (limit as x -> inf)
   **/
  if (std::isinf(z)) {
    // `x_0 = (y - c) / m = -1/w * offset_val < 0`, as `offset_val > 0`
    return y / (m + w * c);
  } else if (z > 0) {
    return 1/w * (boost::math::lambert_w0(z) - offset_val);
  } else if (z >= -std::numeric_limits<Scalar>::min()) {
    const Scalar x_w0 = 1/w * (boost::math::lambert_w0(z) - offset_val);          // close to `-1/w * offset_val`
    const Scalar x_wm1 = y / (m + w * c);

    return (x_w0 >= 0 && (x_wm1 < 0 || x_w0 <= x_wm1)) ? x_w0 : x_wm1;
  } else {
    const Scalar x_w0 = 1/w * (boost::math::lambert_w0(z) - offset_val);
    const Scalar x_wm1 = 1/w * (boost::math::lambert_wm1(z) - offset_val);

    return (x_w0 >= 0 && (x_wm1 < 0 || x_w0 <= x_wm1)) ? x_w0 : x_wm1;
  }
}

}   // namespace exp_converging_line