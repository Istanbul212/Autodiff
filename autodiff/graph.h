#include "math.h"
#include "memory"

template <typename T>
struct Expr;

template <typename T>
using ExprPtr = std::shared_ptr<Expr<T>>;

template <typename T>
struct Expr
{
    virtual ~Expr() {}

    virtual const T evaluate(T x) const = 0;

    /// @brief This function finds the symbolic derivative for this expression using repeated invocations of the chain rule. Note: this is susceptible to "expression swell".
    /// @return The symbolic derivative for this expression.
    virtual const ExprPtr<T> derivative() const = 0;

    /// @brief This function finds the derivative for this expression at a specific point using automatic differentiation.
    /// @param x The point at which the derivative is evaluated.
    /// @return The derivative of this expression evaluated at x.
    virtual const T derivative(T x) const = 0;

    T operator()(T x) { return evaluate(x); };
};

template <typename T>
struct ConstantExpr : Expr<T>
{
    T val;

    ConstantExpr(T x) : val(x) {}

    const T evaluate(T x) const override { return val; }

    const ExprPtr<T> derivative() const override { return std::make_shared<ConstantExpr<T>>(0); }

    const T derivative(T x) const override { return 0; }
};

template <typename T>
struct VariableExpr : Expr<T>
{
    VariableExpr() {}

    const T evaluate(T x) const override { return x; }

    const ExprPtr<T> derivative() const override { return std::make_shared<ConstantExpr<T>>(1); }

    const T derivative(T x) const override { return 1; }
};

template <typename T>
struct UnaryExpr : Expr<T>
{
    ExprPtr<T> f;

    UnaryExpr(const ExprPtr<T> &ff) : f(ff) {}
};

template <typename T>
struct NegateExpr : UnaryExpr<T>
{
    using UnaryExpr<T>::f;
    using UnaryExpr<T>::UnaryExpr;

    const T evaluate(T x) const override { return -(*f)(x); }

    const ExprPtr<T> derivative() const override { return -d(f); }

    const T derivative(T x) const override { return -d(f, x); }
};

template <typename T>
struct SinExpr : UnaryExpr<T>
{
    using UnaryExpr<T>::f;
    using UnaryExpr<T>::UnaryExpr;

    const T evaluate(T x) const override { return std::sin((*f)(x)); }

    const ExprPtr<T> derivative() const override { return cos(f) * d(f); }

    const T derivative(T x) const override { return std::cos((*f)(x)) * d(f, x); }
};

template <typename T>
struct CosExpr : UnaryExpr<T>
{
    using UnaryExpr<T>::f;
    using UnaryExpr<T>::UnaryExpr;

    const T evaluate(T x) const override { return std::cos((*f)(x)); }

    const ExprPtr<T> derivative() const override { return -sin(f) * d(f); }

    const T derivative(T x) const override { return -std::sin((*f)(x)) * d(f, x); }
};

template <typename T>
struct LogNaturalExpr : UnaryExpr<T>
{
    using UnaryExpr<T>::f;
    using UnaryExpr<T>::UnaryExpr;

    const T evaluate(T x) const override { return std::log((*f)(x)); }

    const ExprPtr<T> derivative() const override { return d(f) / f; }

    const T derivative(T x) const override { return d(f, x) / (*f)(x); }
};

template <typename T>
struct BinaryExpr : Expr<T>
{
    ExprPtr<T> f, g;

    BinaryExpr(const ExprPtr<T> &ff, const ExprPtr<T> &gg) : f(ff), g(gg) {}
};

template <typename T>
struct AddExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::f;
    using BinaryExpr<T>::g;
    using BinaryExpr<T>::BinaryExpr;

    const T evaluate(T x) const override { return (*f)(x) + (*g)(x); }

    const ExprPtr<T> derivative() const override { return d(f) + d(g); }

    const T derivative(T x) const override { return d(f, x) + d(g, x); }
};

template <typename T>
struct MulExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::f;
    using BinaryExpr<T>::g;
    using BinaryExpr<T>::BinaryExpr;

    const T evaluate(T x) const override { return (*f)(x) * (*g)(x); }

    const ExprPtr<T> derivative() const override { return d(f) * g + f * d(g); }

    const T derivative(T x) const override { return d(f, x) * (*g)(x) + (*f)(x)*d(g, x); }
};

template <typename T>
struct DivExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::f;
    using BinaryExpr<T>::g;
    using BinaryExpr<T>::BinaryExpr;

    const T evaluate(T x) const override { return (*f)(x) / (*g)(x); }

    const ExprPtr<T> derivative() const override { return (d(f) * g - f * d(g)) / (g * g); }

    const T derivative(T x) const override { return (d(f, x) * (*g)(x) - (*f)(x)*d(g, x)) / ((*g)(x) * (*g)(x)); }
};

template <typename T>
struct PowExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::f;
    using BinaryExpr<T>::g;
    using BinaryExpr<T>::BinaryExpr;

    const T evaluate(T x) const override { return std::pow((*f)(x), (*g)(x)); }

    // TODO: is there a way to use "this" in place of "pow(f, g)" to minimize number of operations?
    const ExprPtr<T> derivative() const override { return pow(f, g) * (d(f) * g / f + d(g) * ln(f)); }

    const T derivative(T x) const override { return evaluate(x) * (d(f, x) * (*g)(x) / (*f)(x) + d(g, x) * std::log((*f)(x))); }
};

template <typename T>
struct Variable
{
    ExprPtr<T> expr;

    Variable() : expr(std::make_shared<VariableExpr<T>>()) {};

    Variable(T val) : expr(std::make_shared<ConstantExpr<T>>(val)) {};

    Variable(const ExprPtr<T> &e) : expr(e) {};

    operator const ExprPtr<T> &() const { return expr; }

    T operator()(T x) { return expr->evaluate(x); }
};

// OPERATORS
// ExprPtr
template <typename T>
ExprPtr<T> operator-(const ExprPtr<T> &e) { return std::make_shared<NegateExpr<T>>(e); }

// Variable
template <typename T>
ExprPtr<T> operator-(const Variable<T> &v) { return -v.expr; }

// ExprPtr - ExprPtr
template <typename T>
ExprPtr<T> operator+(const ExprPtr<T> &l, const ExprPtr<T> &r) { return std::make_shared<AddExpr<T>>(l, r); }
template <typename T>
ExprPtr<T> operator-(const ExprPtr<T> &l, const ExprPtr<T> &r) { return std::make_shared<AddExpr<T>>(l, -r); }
template <typename T>
ExprPtr<T> operator*(const ExprPtr<T> &l, const ExprPtr<T> &r) { return std::make_shared<MulExpr<T>>(l, r); }
template <typename T>
ExprPtr<T> operator/(const ExprPtr<T> &l, const ExprPtr<T> &r) { return std::make_shared<DivExpr<T>>(l, r); }

// Const - ExprPtr
template <typename T>
ExprPtr<T> operator+(T l, const ExprPtr<T> &r) { return Variable(l) + r; }
template <typename T>
ExprPtr<T> operator-(T l, const ExprPtr<T> &r) { return Variable(l) - r; }
template <typename T>
ExprPtr<T> operator*(T l, const ExprPtr<T> &r) { return Variable(l) * r; }
template <typename T>
ExprPtr<T> operator/(T l, const ExprPtr<T> &r) { return Variable(l) / r; }

// ExprPtr - Const
template <typename T>
ExprPtr<T> operator+(const ExprPtr<T> &l, T r) { return l + Variable(r); }
template <typename T>
ExprPtr<T> operator-(const ExprPtr<T> &l, T r) { return l - Variable(r); }
template <typename T>
ExprPtr<T> operator*(const ExprPtr<T> &l, T r) { return l * Variable(r); }
template <typename T>
ExprPtr<T> operator/(const ExprPtr<T> &l, T r) { return l / Variable(r); }

// Variable - ExprPtr
template <typename T>
ExprPtr<T> operator+(const Variable<T> &l, const ExprPtr<T> &r) { return l.expr + r; }
template <typename T>
ExprPtr<T> operator-(const Variable<T> &l, const ExprPtr<T> &r) { return l.expr - r; }
template <typename T>
ExprPtr<T> operator*(const Variable<T> &l, const ExprPtr<T> &r) { return l.expr * r; }
template <typename T>
ExprPtr<T> operator/(const Variable<T> &l, const ExprPtr<T> &r) { return l.expr / r; }

// ExprPtr - Variable
template <typename T>
ExprPtr<T> operator+(const ExprPtr<T> &l, const Variable<T> &r) { return l + r.expr; }
template <typename T>
ExprPtr<T> operator-(const ExprPtr<T> &l, const Variable<T> &r) { return l - r.expr; }
template <typename T>
ExprPtr<T> operator*(const ExprPtr<T> &l, const Variable<T> &r) { return l * r.expr; }
template <typename T>
ExprPtr<T> operator/(const ExprPtr<T> &l, const Variable<T> &r) { return l / r.expr; }

// Variable - Variable
template <typename T>
ExprPtr<T> operator+(const Variable<T> &l, const Variable<T> &r) { return l.expr + r.expr; }
template <typename T>
ExprPtr<T> operator-(const Variable<T> &l, const Variable<T> &r) { return l.expr - r.expr; }
template <typename T>
ExprPtr<T> operator*(const Variable<T> &l, const Variable<T> &r) { return l.expr * r.expr; }
template <typename T>
ExprPtr<T> operator/(const Variable<T> &l, const Variable<T> &r) { return l.expr / r.expr; }

// Const - Variable
template <typename T>
ExprPtr<T> operator+(T l, const Variable<T> &r) { return l + r.expr; }
template <typename T>
ExprPtr<T> operator-(T l, const Variable<T> &r) { return l - r.expr; }
template <typename T>
ExprPtr<T> operator*(T l, const Variable<T> &r) { return l * r.expr; }
template <typename T>
ExprPtr<T> operator/(T l, const Variable<T> &r) { return l / r.expr; }

// Variable - Const
template <typename T>
ExprPtr<T> operator+(const Variable<T> &l, T r) { return l.expr + r; }
template <typename T>
ExprPtr<T> operator-(const Variable<T> &l, T r) { return l.expr - r; }
template <typename T>
ExprPtr<T> operator*(const Variable<T> &l, T r) { return l.expr * r; }
template <typename T>
ExprPtr<T> operator/(const Variable<T> &l, T r) { return l.expr / r; }

// FUNCTIONS
// ExprPtr
template <typename T>
ExprPtr<T> d(const ExprPtr<T> &e) { return e->derivative(); }
template <typename T>
T d(const ExprPtr<T> &e, T x) { return e->derivative(x); }
template <typename T>
ExprPtr<T> sin(const ExprPtr<T> &e) { return std::make_shared<SinExpr<T>>(e); }
template <typename T>
ExprPtr<T> cos(const ExprPtr<T> &e) { return std::make_shared<CosExpr<T>>(e); }
template <typename T>
ExprPtr<T> ln(const ExprPtr<T> &e) { return std::make_shared<LogNaturalExpr<T>>(e); }

// ExprPtr - ExprPtr
template <typename T>
ExprPtr<T> pow(const ExprPtr<T> &l, const ExprPtr<T> &r) { return std::make_shared<PowExpr<T>>(l, r); }

// ExprPtr - Variable
template <typename T>
ExprPtr<T> pow(const ExprPtr<T> &l, const Variable<T> &r) { return pow(l, r.expr); }

// Variable - ExprPtr
template <typename T>
ExprPtr<T> pow(const Variable<T> &l, const ExprPtr<T> &r) { return pow(l.expr, r); }

// Variable - Variable
template <typename T>
ExprPtr<T> pow(const Variable<T> &l, const Variable<T> &r) { return pow(l.expr, r.expr); }

// Variable
template <typename T>
Variable<T> d(const Variable<T> &v) { return v.expr->derivative(); }
template <typename T>
T d(const Variable<T> &v, T x) { return v.expr->derivative(x); }
template <typename T>
ExprPtr<T> sin(const Variable<T> &v) { return sin(v.expr); }
template <typename T>
ExprPtr<T> cos(const Variable<T> &v) { return cos(v.expr); }
template <typename T>
ExprPtr<T> ln(const Variable<T> &v) { return ln(v.expr); }

using var = Variable<double>;
