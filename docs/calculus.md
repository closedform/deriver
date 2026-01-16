# Calculus Functions

## Differentiation

### D - Derivative

Compute derivatives of expressions.

```python
from derive import *

x = Symbol('x')

# First derivative
D(x**3, x)  # 3*x**2

# Higher derivatives
D(x**5, (x, 3))  # 60*x**2

# Partial derivatives
y = Symbol('y')
D(x**2 * y**3, x)  # 2*x*y**3
D(x**2 * y**3, y)  # 3*x**2*y**2

# Chain rule
D(Sin(x**2), x)  # 2*x*cos(x**2)
```

## Integration

### Integrate - Symbolic Integration

```python
# Indefinite integrals
Integrate(x**2, x)  # x**3/3
Integrate(Sin(x), x)  # -cos(x)

# Definite integrals
Integrate(x**2, (x, 0, 1))  # 1/3

# Multiple integrals
Integrate(x*y, (x, 0, 1), (y, 0, 1))  # 1/4

# Special integrals
Integrate(Exp(-x**2), (x, -Infinity, Infinity))  # sqrt(pi)
```

### NIntegrate - Numerical Integration

```python
# Numerical integration
NIntegrate(Sin(x**2), (x, 0, 10))  # 0.6460...
```

## Limits

### Limit - Compute Limits

```python
# Basic limits
Limit(Sin(x)/x, x, 0)  # 1

# One-sided limits
Limit(1/x, x, 0, '+')  # Infinity
Limit(1/x, x, 0, '-')  # -Infinity

# Limits at infinity
Limit((1 + 1/n)**n, n, Infinity)  # E
```

## Series

### Series - Taylor/Laurent Expansion

```python
# Taylor series around x=0
Series(Exp(x), (x, 0, 5))
# 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)

Series(Sin(x), (x, 0, 7))
# x - x**3/6 + x**5/120 - x**7/5040 + O(x**8)

# Series around other points
Series(Log(x), (x, 1, 4))
# (x-1) - (x-1)**2/2 + (x-1)**3/3 - (x-1)**4/4 + O((x-1)**5)
```

## Summation

### Sum - Symbolic and Numeric Sums

```python
i = Symbol('i')

# Finite sums
Sum(i, (i, 1, 10))  # 55
Sum(i**2, (i, 1, 10))  # 385

# Symbolic sums
n = Symbol('n')
Sum(i, (i, 1, n))  # n*(n+1)/2

# Infinite series
Sum(1/n**2, (n, 1, Infinity))  # pi**2/6
```

### Product - Symbolic Products

```python
# Factorial via product
Product(i, (i, 1, 5))  # 120

# Symbolic products
Product(i, (i, 1, n))  # factorial(n)
```

## Integral Transforms

### FourierTransform - Fourier Transform

Compute the Fourier transform of an expression.

```python
from derive import *

t, omega = symbols('t omega')

# Fourier transform of Gaussian
FourierTransform(Exp(-t**2), t, omega)  # sqrt(pi)*exp(-omega**2/4)
```

### InverseFourierTransform - Inverse Fourier Transform

```python
# Inverse transform
InverseFourierTransform(Sqrt(Pi)*Exp(-omega**2/4), omega, t)  # exp(-t**2)
```

### LaplaceTransform - Laplace Transform

```python
t, s = symbols('t s')

# Laplace transform of exponential
LaplaceTransform(Exp(-t), t, s, noconds=True)  # 1/(s + 1)

# Laplace transform of sine
LaplaceTransform(Sin(t), t, s, noconds=True)  # 1/(s**2 + 1)
```

### InverseLaplaceTransform - Inverse Laplace Transform

```python
# Inverse Laplace transform
InverseLaplaceTransform(1/(s + 1), s, t)  # exp(-t)*Heaviside(t)
```

### Convolve - Convolution

Compute the convolution integral of two functions.

```python
t = Symbol('t')

# Convolution of two functions
Convolve(Exp(-t)*Heaviside(t), Exp(-t)*Heaviside(t), t)
```

## Change of Variables

### ChangeVariables - Variable Substitution in Integrals

Transform integrals using u-substitution.

```python
x, u = symbols('x u')

# Change x to u where x = u^2
ChangeVariables(x**2, x, u, u**2)  # 2*u**5

# Definite integral with bounds transformation
ChangeVariables(Sqrt(x), x, u, u**2, bounds=(0, 4))
# Returns (Integral(2*u**2, (u, 0, 2)), {0: 0, 4: 2})
```

### IntegrateWithSubstitution - Integration via Substitution

Combine variable change with integration.

```python
x, u = symbols('x u')

# Integrate sin(x^2) * 2x dx with u = x^2
IntegrateWithSubstitution(Sin(x**2) * 2*x, x, u, Sqrt(u))  # -Cos(u)
```

## Variational Calculus

### VariationalDerivative - Functional Derivative

Compute the variational (Euler-Lagrange) derivative of a Lagrangian.

```python
x, t = symbols('x t')
phi = Function('phi')(x, t)
m = Symbol('m')

# Klein-Gordon Lagrangian: L = (1/2)(d_t phi)^2 - (1/2)(d_x phi)^2 - (1/2)m^2 phi^2
L = Rational(1,2)*D(phi, t)**2 - Rational(1,2)*D(phi, x)**2 - Rational(1,2)*m**2*phi**2

# Get the equation of motion
eq = VariationalDerivative(L, phi, [x, t])
# Result: d_t^2 phi - d_x^2 phi + m^2 phi (equals 0)
```

### EulerLagrangeEquation - Euler-Lagrange Equations

Alias for VariationalDerivative, following standard naming convention.

```python
# Same as VariationalDerivative
eq = EulerLagrangeEquation(L, phi, [x, t])
```

## Green's Functions

Green's functions solve L[G] = delta, useful for solving inhomogeneous PDEs.

### GreenFunctionPoisson1D

Green's function for 1D Poisson equation -d^2u/dx^2 = f on [0, L].

```python
x, xp, L = symbols('x xp L', positive=True)

# Green's function with Dirichlet BCs
G = GreenFunctionPoisson1D(x, xp, L)
# G(x, x') = x(L-x')/L if x < x', else x'(L-x)/L
```

### GreenFunctionHelmholtz1D

Green's function for 1D Helmholtz equation (d^2/dx^2 + k^2)u = f.

```python
x, xp, k = symbols('x xp k')

G = GreenFunctionHelmholtz1D(x, xp, k)
# G(x, x') = -(i/2k) * exp(ik|x - x'|)
```

### GreenFunctionLaplacian3D

Green's function for 3D Laplacian.

```python
x, y, z = symbols('x y z')
r = Sqrt(x**2 + y**2 + z**2)

G = GreenFunctionLaplacian3D(r)  # -1/(4*pi*r)
```

### GreenFunctionWave1D

Green's function for 1D wave equation.

```python
x, t, xp, tp, c = symbols('x t xp tp c', real=True)

G = GreenFunctionWave1D(x, t, xp, tp, c)
# G = (1/2c) * Heaviside(t - t' - |x - x'|/c)
```
