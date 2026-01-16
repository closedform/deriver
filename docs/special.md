# Special Functions

Derive provides extensive special function support for physics and mathematics applications.

## Bessel Functions

### Bessel Functions of the First Kind

```python
from derive import *

x = Symbol('x')

# J_n(x) - Bessel function of first kind
BesselJ(0, x)  # J_0(x)
BesselJ(1, 2.0)  # Numeric evaluation

# Numeric values
float(BesselJ(0, 1).evalf())  # 0.7651976865579666
```

### Bessel Functions of the Second Kind

```python
# Y_n(x) - Neumann function
BesselY(0, x)
BesselY(1, 2.0)
```

### Modified Bessel Functions

```python
# I_n(x) - Modified Bessel of first kind
BesselI(0, x)

# K_n(x) - Modified Bessel of second kind
BesselK(0, x)
```

### Hankel Functions

```python
# H^(1)_n(x), H^(2)_n(x)
HankelH1(0, x)
HankelH2(0, x)
```

### Spherical Bessel Functions

```python
# j_n(x), y_n(x) - spherical Bessel functions
SphericalBesselJ(1, x)
SphericalBesselY(1, x)
```

## Orthogonal Polynomials

### Legendre Polynomials

```python
# P_n(x) - Legendre polynomial
LegendreP(0, x)  # 1
LegendreP(1, x)  # x
LegendreP(2, x)  # (3*x**2 - 1)/2

# P^m_n(x) - Associated Legendre
AssociatedLegendreP(1, 1, x)  # -sqrt(1-x**2)
```

### Chebyshev Polynomials

```python
# T_n(x) - First kind
ChebyshevT(0, x)  # 1
ChebyshevT(1, x)  # x

# U_n(x) - Second kind
ChebyshevU(0, x)  # 1
ChebyshevU(1, x)  # 2*x
```

### Hermite Polynomials

Used in quantum mechanics (harmonic oscillator).

```python
HermiteH(0, x)  # 1
HermiteH(1, x)  # 2*x
HermiteH(2, x)  # 4*x**2 - 2
```

### Laguerre Polynomials

Used in hydrogen atom wavefunctions.

```python
LaguerreL(0, x)  # 1
LaguerreL(1, x)  # -x + 1

# Associated Laguerre
AssociatedLaguerreL(1, 1, x)
```

### Gegenbauer Polynomials

Ultraspherical polynomials, generalization of Chebyshev and Legendre.

```python
# C_n^alpha(x) - Gegenbauer polynomial
GegenbauerC(2, Rational(1, 2), x)  # Reduces to Legendre when alpha=1/2
GegenbauerC(2, 1, x)  # Related to Chebyshev U
```

### Jacobi Polynomials

Most general classical orthogonal polynomials.

```python
# P_n^(alpha, beta)(x) - Jacobi polynomial
JacobiP(2, 0, 0, x)  # Reduces to Legendre when alpha=beta=0
JacobiP(2, Rational(1,2), Rational(1,2), x)  # Related to Chebyshev
```

## Spherical Harmonics

Critical for angular momentum in quantum mechanics.

```python
theta, phi = symbols('theta phi')

# Y^m_l(theta, phi)
SphericalHarmonicY(0, 0, theta, phi)  # 1/(2*sqrt(pi))
SphericalHarmonicY(1, 0, theta, phi)
SphericalHarmonicY(1, 1, theta, phi)
```

## Elliptic Integrals

Important for classical mechanics and string theory.

```python
k = Symbol('k')

# Complete elliptic integrals
EllipticK(k)  # K(k) - first kind
EllipticE(k)  # E(k) - second kind

# Incomplete elliptic integrals
EllipticF(phi, k)  # F(phi, k)
EllipticPi(n, phi, k)  # Pi(n; phi, k)

# Special values
EllipticK(0)  # pi/2
EllipticE(0)  # pi/2
```

## Error Functions

```python
# Error function
Erf(x)
Erf(0)  # 0

# Complementary error function
Erfc(x)
Erfc(0)  # 1

# Identity: Erf(x) + Erfc(x) = 1

# Imaginary error function
Erfi(x)
```

## Airy Functions

Solutions to y'' - xy = 0.

```python
AiryAi(x)  # Ai(x)
AiryBi(x)  # Bi(x)

# Numeric values
float(AiryAi(0).evalf())  # 0.35502805388781724
```

## Gamma and Related Functions

```python
# Gamma function
Gamma(5)  # 24 (= 4!)
Gamma(Rational(1, 2))  # sqrt(pi)

# Beta function
Beta(2, 3)

# Factorial and Binomial
Factorial(5)  # 120
Binomial(10, 3)  # 120
```

## Zeta and Polylogarithm

Important in QFT and string theory.

```python
# Riemann zeta function
Zeta(2)  # pi**2/6
Zeta(4)  # pi**4/90

# Polylogarithm
PolyLog(2, x)  # Li_2(x)
```

## Hypergeometric Functions

Very general class of special functions.

```python
# 2F1 - Gauss hypergeometric
Hypergeometric2F1([a, b], [c], x)

# Generalized pFq
HypergeometricPFQ([a1, a2], [b1, b2], x)

# Meijer G-function (most general)
MeijerG([[a1], [a2]], [[b1], [b2]], x)
```

## Exponential Integrals

```python
# Ei(x)
ExpIntegralEi(x)

# Si(x), Ci(x) - sine and cosine integrals
SinIntegral(x)
CosIntegral(x)

# Li(x) - logarithmic integral
LogIntegral(x)
```

## Fresnel Integrals

Used in optics and wave propagation.

```python
FresnelS(x)  # S(x) = integral(sin(pi*t^2/2), t=0..x)
FresnelC(x)  # C(x) = integral(cos(pi*t^2/2), t=0..x)
```
