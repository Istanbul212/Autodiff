Playing around with implementations of symbolic differentiaton + autodiff.
Tests should give an idea.

TODO (roughly in this order):
  1. better handling of multi-variable functions. currently only supports single variable functions.
  2. more optimal implementations for higher order functions
      - Likely using sophisticated matrix algebra techniques + SIMD 
  3. avoid false sharing (step 1: sit down and actually think about whether it's actually a problem here)
