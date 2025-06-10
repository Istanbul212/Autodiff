#include "../autodiff/graph.h"
#include "iostream"
#include "math.h"

#define IS_TRUE(x)                                                                    \
    {                                                                                 \
        if (!(x))                                                                     \
            std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; \
    }

#define IS_EQ(x, y)                                                                   \
    {                                                                                 \
        if (x != y)                                                                   \
            std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; \
    }

#define IS_APPROX_EQ(x, y)                                                            \
    {                                                                                 \
        if (abs(x - y) > 1e-15)                                                       \
            std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; \
    }

void test_const()
{
    var constant(5);

    // evaluate
    IS_EQ(constant(-5), 5);
    IS_EQ(constant(0), 5);
    IS_EQ(constant(5), 5);

    // symbolic derivative
    IS_EQ(d(constant)(-5), 0);
    IS_EQ(d(constant)(0), 0);
    IS_EQ(d(constant)(5), 0);

    // autodiff
    IS_EQ(d(constant, -5.0), 0);
    IS_EQ(d(constant, 0.0), 0);
    IS_EQ(d(constant, 5.0), 0);
}

void test_add()
{
    var x;
    var f = x + x + 17.0;

    // evaluate
    IS_EQ(f(-5), 7);
    IS_EQ(f(0), 17);
    IS_EQ(f(5), 27);

    // symbolic derivative
    IS_EQ(d(f)(-5), 2);
    IS_EQ(d(f)(0), 2);
    IS_EQ(d(f)(5), 2);

    // autodiff
    IS_EQ(d(f, -5.0), 2);
    IS_EQ(d(f, 0.0), 2);
    IS_EQ(d(f, 5.0), 2);
}

void test_sub()
{
    var x;
    var f = 3.0 * x - x - 17.0;

    // evaluate
    IS_EQ(f(-5), -27);
    IS_EQ(f(0), -17);
    IS_EQ(f(5), -7);

    // symbolic derivative
    IS_EQ(d(f)(-5), 2);
    IS_EQ(d(f)(0), 2);
    IS_EQ(d(f)(5), 2);

    // autodiff
    IS_EQ(d(f, -5.0), 2);
    IS_EQ(d(f, 0.0), 2);
    IS_EQ(d(f, 5.0), 2);
}

void test_mul()
{
    var x;
    var f = x * x * x + 12.5 * x + 35.2;

    // evaluate
    IS_EQ(f(-5), -152.3);
    IS_EQ(f(0), 35.2);
    IS_EQ(f(5), 222.7);

    // symbolic derivative
    IS_EQ(d(f)(-5), 87.5);
    IS_EQ(d(f)(0), 12.5);
    IS_EQ(d(f)(5), 87.5);

    // autodiff
    IS_EQ(d(f, -5.0), 87.5);
    IS_EQ(d(f, 0.0), 12.5);
    IS_EQ(d(f, 5.0), 87.5);
}

void test_div()
{
    var x;
    var f = 1.0 / x + 1.0 / (x * x);

    // evaluate
    IS_EQ(f(-5), -0.16);
    IS_EQ(f(0), INFINITY);
    IS_EQ(f(8), 0.140625);

    // symbolic derivative
    IS_EQ(d(f)(-5), -0.024);
    IS_TRUE(std::isnan(d(f)(0)));
    IS_EQ(d(f)(5), -0.056);

    // autodiff
    IS_EQ(d(f, -5.0), -0.024);
    IS_TRUE(std::isnan(d(f, 0.0)));
    IS_EQ(d(f, 5.0), -0.056);
}

void test_sin()
{
    var x;
    var f = sin(2.0 * x);

    // evaluate
    IS_APPROX_EQ(f(0), 0);
    IS_APPROX_EQ(f(M_PI / 8), 1 / sqrt(2));
    IS_APPROX_EQ(f(M_PI / 4), 1);
    IS_APPROX_EQ(f(M_PI / 2), 0);

    // symbolic derivative
    IS_APPROX_EQ(d(f)(0), 2);
    IS_APPROX_EQ(d(f)(M_PI / 8), sqrt(2));
    IS_APPROX_EQ(d(f)(M_PI / 4), 0);
    IS_APPROX_EQ(d(f)(M_PI / 2), -2);

    // autodiff
    IS_APPROX_EQ(d(f, 0.0), 2);
    IS_APPROX_EQ(d(f, M_PI / 8), sqrt(2));
    IS_APPROX_EQ(d(f, M_PI / 4), 0);
    IS_APPROX_EQ(d(f, M_PI / 2), -2);
}

void test_cos()
{
    var x;
    var f = cos(2.0 * x);

    // evaluate
    IS_APPROX_EQ(f(0), 1);
    IS_APPROX_EQ(f(M_PI / 8), 1 / sqrt(2));
    IS_APPROX_EQ(f(M_PI / 4), 0);
    IS_APPROX_EQ(f(M_PI / 2), -1);

    // symbolic derivative
    IS_APPROX_EQ(d(f)(0), 0);
    IS_APPROX_EQ(d(f)(M_PI / 8), -sqrt(2));
    IS_APPROX_EQ(d(f)(M_PI / 4), -2);
    IS_APPROX_EQ(d(f)(M_PI / 2), 0);

    // autodiff
    IS_APPROX_EQ(d(f, 0.0), 0);
    IS_APPROX_EQ(d(f, M_PI / 8), -sqrt(2));
    IS_APPROX_EQ(d(f, M_PI / 4), -2);
    IS_APPROX_EQ(d(f, M_PI / 2), 0);
}

void test_natural_log()
{
    var x;
    var f = ln(M_E * x + M_E);

    // evaluate
    IS_EQ(f(-1), -INFINITY);
    IS_APPROX_EQ(f(0), 1);
    IS_APPROX_EQ(f(M_E - 1), 2);

    // symbolic derivative
    IS_EQ(d(f)(-1), INFINITY);
    IS_APPROX_EQ(d(f)(M_E / 2), 2 / (M_E + 2));
    IS_APPROX_EQ(d(f)(M_E * M_E / 2), 2 / (M_E * M_E + 2));

    // autodiff
    IS_EQ(d(f, -1.0), INFINITY);
    IS_APPROX_EQ(d(f, M_E / 2), 2 / (M_E + 2));
    IS_APPROX_EQ(d(f, M_E * M_E / 2), 2 / (M_E * M_E + 2));
}

void test_pow()
{
    var x;
    var f = pow(x, ln(x));

    // evaluate
    IS_APPROX_EQ(f(1), 1);
    IS_APPROX_EQ(f(M_E), M_E);
    IS_APPROX_EQ(f(M_E * M_E), pow(M_E, 4));

    // symbolic derivative
    IS_APPROX_EQ(d(f)(1), 0);
    IS_APPROX_EQ(d(f)(M_E), 2);
    IS_APPROX_EQ(d(f)(M_E * M_E), 4 * M_E * M_E);

    // autodiff
    IS_APPROX_EQ(d(f, 1.0), 0);
    IS_APPROX_EQ(d(f, M_E), 2);
    IS_APPROX_EQ(d(f, M_E * M_E), 4 * M_E * M_E);
}

int main(void)
{
    test_const();
    test_add();
    test_sub();
    test_mul();
    test_div();
    test_sin();
    test_cos();
    test_natural_log();
    test_pow();
}
