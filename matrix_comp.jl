#
# This computes the traces from section 4 of "On a Fiber Conjecture of Wan"
#
# See here () for the arxiv link.
#
# To compute the traces in Lemma 4.4, set Xi = 1.
# To compute the traces in Lemma 4.10, set Xi = teich_lift(xi + 2).
#

using Nemo
using Plots, LazySets
using Printf
using Base.Iterators
using AbstractAlgebra

# Define $\mathbb{F}_{p^a}$
p = 5
a = 2
Fq, xi = FiniteField(p, a, "xi")

# Matrices related to Dwork theory will be computed to precision x precison size.
matrix_size = 20

# All power series will also be computed to precision ps_prec
# Compute the Artin-Hasse exponential E(x)=\sum_{k=0}^ps_prec u_k x^k
#
# All p-adics will be computed to precision padic_prec.
AH_prec = p*matrix_size
ps_prec = p*matrix_size
padic_prec = 4

ZZq, q = QadicField(p, a, padic_prec)

# Computes the Teichmuller lift of an element of Fq, g.
function teich_lift(g::fq_nmod)
    ZZqY, Y = PolynomialRing(ZZq, "Y")
    fq_poly = sum([ZZq(coeff(g, i)) * Y^i for i = 0:(a-1)])

    return teichmuller(fq_poly(q))
end

# The array u contains the Artin-Hasse coefficients.
# Note that the 0th coefficient is u[1] by Julia numbering.
u = [ZZq(1), ZZq(1)]

function un_comp(n)
    un = 0
    k = Int(floor(log(p, n)))

    for i = 0:n
        index = n - p^i + 1
        if index > 0
            un = un + u[index]
        else
            break
        end
    end

    un = un // n
    push!(u, un)
end

for n = 2:AH_prec
    un_comp(n)
end

R, xpi = PolynomialRing(ZZq, "xpi")
Zpi = ResidueRing(R, xpi^(p - 1) + p)

Zpi_Fq, W = PolynomialRing(Fq, "W")
L, x = PowerSeriesRing(Zpi, ps_prec, "x")

Xi = 1
#Xi = teich_lift(xi + 2)

B = sum(u[i+1] * (xpi * Xi * x^8)^(i) for i = 0:ps_prec-1) *
    sum(u[i+1] * (xpi * Xi * x^6)^(i) for i = 0:ps_prec-1) *
    sum(u[i+1] * (xpi * Xi * x^2)^(i) for i = 0:ps_prec-1)

const xpi_T = AbstractAlgebra.Generic.Poly{qadic}
const Zpi_T = AbstractAlgebra.Generic.Res{xpi_T}

#Computes the pi-adic valuation of a power series.
function pi_val(g)
    vals = []

    if g == 0
        return 1000
    end

    for i = 0:degree(g.data)
        if coeff(g.data, i) != 0
            push!(vals, valuation(coeff(g.data, i)) + i / (p - 1))
        end
    end

    return minimum(vals)
end

function tau_inv(g, k = 1)
    return Zpi(map_coefficients(c -> frobenius(c, -k), g.data))
end

# Returns tau^(-k)M, some matrix M of size mxm and integer k
function matrix_tau_inv_k(Mat, m::Int64, k::Int64)
    nM::Array = Array{typeof(Zpi(xpi))}(undef, m * m)

    for i = 1:m
        for j = 1:m
            nM[m*(i-1)+(j-1)+1] = tau_inv(getindex(Mat, i, j), k)
        end
    end

    return Matrix_sp(nM)
end

# Prints an element g in Fp[pi] in a Latex friendly format. Coefficients are mod p^m.
Qw, w = FlintQQ["w"]
function print_lifted(g, m)
    lifted = 0

    for i in 1:length(g.data.coeffs)
        fmp_coeff = lift(Qw, g.data.coeffs[i])
        int_coeff = BigInt(numerator(fmp_coeff(0)))
        if (int_coeff % p^m) == 0
            continue
        end

        if (int_coeff % p^m) == 1
            print("\\pi^{", i-1, "}")
        else
            t = int_coeff % p^m
            print(t, "\\pi^{", i-1, "}")
        end

        if i != length(g.data.coeffs)
            print(" + ")
        end
    end
end

# This constructs the matrix nM = [F_{pi-j}]_{i,j}
nM = Array{typeof(Zpi(xpi))}(undef, matrix_size * matrix_size)
for i = 1:matrix_size
    for j = 1:matrix_size
        if p * i - j < 0
            nM[matrix_size*(i-1)+(j-1)+1] = Zpi(0)
        else
            nM[matrix_size*(i-1)+(j-1)+1] = ((Zpi(B.coeffs[p*i-j+1])))
        end
    end
end

Matrix_sp = MatrixSpace(Zpi, matrix_size, matrix_size)
MMM = Matrix_sp(nM) * matrix_tau_inv_k(Matrix_sp(nM), matrix_size, 1)

c1 = -tr(MMM)
c2 = -(1 // 2) * tr(MMM^2) + (1 // 2) * (tr(MMM))^2
c3 =
    -(1 // 6) * tr(MMM)^3 + (1 // 2) * tr(MMM^2) * tr(MMM) -
    (1 // 3) * (tr(MMM^3))

println("*******************")
print("\\Tr(\\phi) &\\equiv ")
print_lifted(tr(MMM),3)
print("\\bmod p\\\\\n\\\\\n")
print("\\Tr(\\phi^2) &\\equiv ")
print_lifted(tr(MMM^2),3)
print("\\bmod p^2\\\\\n\\\\\n")
print("\\Tr(\\phi^3) &\\equiv ")
print_lifted(tr(MMM^3),3)
print("\\bmod p^3\n")

print("\nc_1 = ")
print_lifted(c1,1)
print("\nc_2 = ")
print_lifted(c2,2)
print("\nc_3 = ")
print_lifted(c3,3)

println("\n\nord c_1 = ", pi_val(c1))
println("ord c_2 = ", pi_val(c2))
println("ord c_3 = ", pi_val(c3))
