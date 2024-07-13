# This Julia code uses the Nemo algebra package to implement the algorithm
# of Lauder and Wan to compute the Newton polygon of the L-function
# of an Artin-Schreier curve.
#
# See Alan G. B. Lauder and Daqing Wan,
#     Computing Zeta Functions of Artin–schreier Curves over Finite Fields,
#     LMS Journal of Computation and Mathematics, Volume 5, 2002, pp. 34 - 55.
#

using Nemo
using Plots, LazySets
using Base.Iterators
using AbstractAlgebra

using Distributed

TAU_ON = true   # Setting to false removes Galois-action on the frobenius matrix

# When true, this allows the manual setting of the p-adic and
# power series precision. The default settings from [LW]
# are not optimal, as mentioned in their paper.
DIRTY = true

# These are the manual settings. Depending on p and a, they will need
# to be adjusted experimentally.
dirty_padic_prec = 180
dirty_ps_prec = 100

##################################################################
#
#   Framework
#
##################################################################

const Zpi_t = AbstractAlgebra.Generic.Res
const Zpi_powerseries = AbstractAlgebra.Generic.RelSeries

const xpi_T = AbstractAlgebra.Generic.Poly{qadic}
const Zpi_T = AbstractAlgebra.Generic.Res{xpi_T}

mutable struct Dwork_Framework
    p::Int64
    a::Int64
    d::Int64

    Fq::FqNmodFiniteField
    xi::fq_nmod
    ZZq::FlintQadicField
    q::qadic
    ZZqpoly::AbstractAlgebra.Generic.PolyRing{qadic}
    X::AbstractAlgebra.Generic.Poly{qadic}

    R::AbstractAlgebra.Generic.PolyRing{qadic}
    xpi::xpi_T
    Zpi::AbstractAlgebra.Generic.ResRing{xpi_T}
    Lpoly::AbstractAlgebra.Generic.PolyRing{Zpi_T}
    y::AbstractAlgebra.Generic.Poly{Zpi_T}
    L::AbstractAlgebra.Generic.RelSeriesRing{Zpi_T}
    x::AbstractAlgebra.Generic.RelSeries{Zpi_T}

    M::AbstractAlgebra.Generic.MatSpace{Zpi_T}

    C::AbstractAlgebra.Generic.RelSeriesRing{Zpi_T}
    T::AbstractAlgebra.Generic.RelSeries{Zpi_T}

    xpi_inv::xpi_T

    Dwork_Framework() = new()
end

function Dwork_Framework(p::Int64, a::Int64, d::Int64)
    DF = Dwork_Framework()

    DF.p = p
    DF.a = a
    DF.d = d

    epsilon = 4
    if DF.p == 2
        epsilon = 4 * d + 1
    end
    N = Int(floor((DF.p - 1) * (d - 1) * (1 + (DF.a / 2)) + 1))

    if DIRTY == false
        #This can be improved but we need to change the precision mid computation.
        DF.ZZq, DF.q = QadicField(p, a, epsilon * (N + 1))
        println("power series prec = ", epsilon * N * DF.p * d)
        println("p-adic prec = ", epsilon * (N + 1))
    else
        DF.ZZq, DF.q = QadicField(p, a, dirty_padic_prec)
    end

    DF.Fq, DF.xi = FiniteField(p, a, "xi")
    DF.ZZqpoly, DF.X = PolynomialRing(DF.ZZq, "X")

    DF.R, DF.xpi = PolynomialRing(DF.ZZq, "xpi")
    DF.xpi_inv = -1 * DF.xpi^(p - 2) * (1 // p)

    DF.Zpi = ResidueRing(DF.R, DF.xpi^(p - 1) + p)
    DF.Lpoly, DF.y = PolynomialRing(DF.Zpi, "y")

    if DIRTY == false
        DF.L, DF.x = PowerSeriesRing(DF.Zpi, epsilon * N * DF.p * d, "x")
    else
        DF.L, DF.x = PowerSeriesRing(DF.Zpi, dirty_ps_prec, "x")
    end
    DF.M = MatrixSpace(DF.Zpi, d - 1, d - 1)

    return DF
end

# Returns tau^(-k)g, some element of Zpi and integer k
#
# Takes an element of Zpi and applies the inverse frobenius
# to each term, leaving pi constant.
function tau_inv(
    DF::Dwork_Framework,
    g::AbstractAlgebra.Generic.Res,
    k::Int64 = 1,
)
    lift_g::AbstractAlgebra.Generic.Poly = data(g)

    if TAU_ON
        lift_g = map_coefficients(c -> frobenius(c, -k), lift_g)
    end

    return DF.Zpi(lift_g)
end

# Returns tau^(-k)M, some matrix M of size mxm and integer k
function matrix_tau_inv_k(
    DF::Dwork_Framework,
    Mat::AbstractAlgebra.Generic.MatSpaceElem,
    m::Int64,
    k::Int64,
)
    nM::Array = Array{Zpi_T}(undef, m * m)

    for i = 1:m
        for j = 1:m
            nM[m*(i-1)+(j-1)+1] = tau_inv(DF, getindex(Mat, i, j), k)
        end
    end

    return DF.M(nM)
end

# Computes the Teichmuller lift of an element of Fq, g.
function teich_lift(DF::Dwork_Framework, g::fq_nmod)
    ZZqY, Y = PolynomialRing(DF.ZZq, "Y")
    fq_poly = sum([DF.ZZq(coeff(g, i)) * Y^i for i = 0:(DF.a-1)])

    return teichmuller(fq_poly(DF.q))
end

##################################################################
#
#   Monomial Reduction
#
##################################################################

wt(d::Int, u::Int) = ceil(u / d)

function ps_degree(g::Zpi_powerseries)
    if g == 0
        return 0
    else
        return g.length + g.val - 1
    end
end

# Let u\geq d. Returns monomial cohomologous to pi^wt(u)x^u
function reduced_monomial(
    DF::Dwork_Framework,
    u::Int64,
    fpoly::Zpi_powerseries,
    HX::Zpi_powerseries,
)

    fpoly_d::Int = ps_degree(fpoly)

    Hd_1::Zpi_powerseries =
        HX - DF.xpi * fpoly_d * coeff(fpoly, fpoly_d) * DF.x^fpoly_d

    mono::Zpi_powerseries =
        (coeff(fpoly, fpoly_d) * fpoly_d)^(-1) *
        DF.xpi^(Int(wt(fpoly_d, u)) - 1) *
        DF.x^(u - fpoly_d)

    return -1 * (Hd_1 * mono + DF.x * derivative(mono))
end

# Returns the reduced form of a series g.
function normal_form_reduction(
    DF::Dwork_Framework,
    g::Zpi_powerseries,
    fpoly::Zpi_powerseries,
    HX::Zpi_powerseries)

    fpoly_d::Int = ps_degree(fpoly)

    while ps_degree(g) > fpoly_d - 1
        g_deg::Int = ps_degree(g)
        coe::Zpi_t = g.coeffs[g.length]

        g =
            g - coe * DF.x^g_deg +
            coe *
            DF.xpi_inv^(Int(wt(fpoly_d, g_deg))) *
            reduced_monomial(DF, g_deg, fpoly, HX)
    end

    return g
end

##################################################################
#
#   Dwork Theory
#
##################################################################

# Applies U_p to an element of L
function U_p(DF::Dwork_Framework, g::Zpi_powerseries)
    gp::Zpi_powerseries = DF.L(0)

    for i = 0:DF.L.prec_max
        if i % DF.p == 0
            gp = gp + tau_inv(DF, coeff(g, i)) * DF.x^(Int(i / DF.p))
        end
    end

    return gp
end

function compute_Lfunc(
    DF::Dwork_Framework,
    fpoly::AbstractAlgebra.Generic.Poly{qadic},
)

    if fpoly == 0
        return 1
    end

    if degree(fpoly) != DF.d
        println("Dwork Framework not compatible with given fpoly:
                    $(degree(fpoly)) != $(DF.d).")
        return 1
    end

    fmono::Array = [DF.Zpi(fpoly.coeffs[i+1]) * DF.x^i for i = 0:DF.d]
    fpoly_lift = sum(fmono)

    F = 1
    for mono in fmono
        F = F * exp(DF.xpi * (mono - mono^DF.p))
    end

    H::Zpi_powerseries = DF.xpi * fpoly_lift
    HX::Zpi_powerseries = DF.x * derivative(H)
    M1::Array = Array{Zpi_T}(undef, (DF.d - 1) * (DF.d - 1))

    # Lauder and Wan define the matrix entires m_ij in this way.
    # It's opposite the normal notation
    for j = 1:(DF.d-1)
        expansion = normal_form_reduction(
            DF,
            U_p(DF, F * DF.xpi * DF.x^j),
            fpoly_lift,
            HX,
        )

        for i = 1:(DF.d-1)
            M1[(DF.d-1)*(i-1)+(j-1)+1] = DF.xpi_inv * coeff(expansion, i)
        end
    end

    A::AbstractAlgebra.Generic.MatSpaceElem = DF.M(M1)

    M_a::AbstractAlgebra.Generic.MatSpaceElem = DF.M(1)
    for i = 0:(DF.a-1)
        M_a = M_a * DF.M(matrix_tau_inv_k(DF, A, DF.d - 1, i))
    end

    DF.C, DF.T = PowerSeriesRing(DF.Zpi, DF.d + 100, "T")
    partsum::AbstractAlgebra.Generic.RelSeries = DF.C(0)

    # Note that the returned power series may not be of degree (d-1) as
    # expected, but the first d-1 will be exactly what they should be.
    #
    # If the desired concise polynomial is required, increase (d+1) to 2*d or more
    for i = 1:(DF.d+1)
        partsum = partsum + (tr(M_a^(i))) * DF.ZZq(1 // i) * DF.T^i
    end

    return exp(-1 * partsum)
end

# This can be used to compute zeta function
function eta_action(DF::Dwork_Framework, g, k::Int64)
    gp::Zpi_powerseries = DF.C(0)
    eta = teich_lift(DF.xi^(Nemo.ZZ((DF.p^DF.a - 1) / (DF.p - 1))))

    for i = 0:power_series_prec
        g_c = data(coeff(g, i))

        new_coeff = 0

        for j = 1:g_c.length
            new_coeff =
                new_coeff +
                DF.ZZq(eta^(k * (j - 1))) * g_c.coeffs[j] * DF.xpi^(j - 1)
        end

        gp = gp + DF.Zpi(new_coeff) * DF.T^i
    end

    return gp
end

##################################################################
#
#   Newton Polygon Computation
#
##################################################################

#Computes the pi-adic valuation of a power series.
function pi_val(p::Int64, g::Zpi_t)
    poly = data(g)
    vals = []

    for i = 0:(p-1)
        push!(vals, valuation(coeff(poly, i)) + i / (p - 1))
    end

    return minimum(vals)
end

# Compute without hull: for each vertex, compute slopes between it and every
# other point. Take the one with the lowest slope
# Use this to generate lower convex hull and elimiate Julia dependency
#
# This returns a list which consists of pairs (slope, multiplicity).
function get_next_slope(slope_list::Array, verts::Array)
    if length(verts) == 1
        return slope_list
    end

    single = verts[1]
    slopes = [
        (verts[j][2] - single[2]) / (verts[j][1] - single[1]) for
        j = 2:(length(verts))
    ]
    (s, index) = findmin(reverse(slopes))

    n_verts = verts[(length(verts)-index+1):length(verts)]

    push!(slope_list, [s, length(verts) - index])

    return get_next_slope(slope_list, n_verts)
end

function get_verts_from_slopes(slopes::Array)
    verts = Vector{Float64}[[0, 0]]

    current = [0, 0]
    for s in slopes
        v = [current[1] + s[2], s[1] * s[2] + current[2]]
        push!(verts, v)

        current = v
    end

    return verts
end

# If we  only want the slopes of the NP, this is a quick wrapper.
function compute_newton_polygon(DF::Dwork_Framework, fpoly)
    d = ps_degree(fpoly)
    DetM = compute_Lfunc(DF, fpoly)

    if DetM == 1
        return [0]
    end

    verts = [[i, pi_val(DF.p, coeff(DetM, i))] for i = 0:(d-1)]
    slopes = []
    get_next_slope(slopes, verts)

    return slopes
end

##################################################################
#
#   Computations
#
##################################################################

default(titlefontsize = 8)

# Computes the NP's of lambda*fpoly_1 for each lambda in lambda_set.
#
# If quick = true, then function will product no output and only
# return true or false depending on if the NP changed.
#
# When quick = false, this returns a dictionary with keys the NP's.
# In the dictionary, each NP will give the lambda values which yield it.
function test_poly(DF::Dwork_Framework, fpoly_1, lambda_set, quick = false)
    base = []
    base_NP = []
    isotropy = Dict()

    d = degree(fpoly_1)
    if DF.d != d
        return
    end

    @sync @distributed for lambda in lambda_set
        if lambda == 0
            continue
        end

        fpoly = (teich_lift(DF, lambda)) * (fpoly_1)
        DetM = compute_Lfunc(DF, fpoly)
        if DetM == 1
            continue
        end

        verts = [[i, pi_val(DF.p, coeff(DetM, i))] for i = 0:(d-1)]
        slopes = []
        get_next_slope(slopes, verts)
        hull = get_verts_from_slopes(slopes)

        if !quick
            println("\n")
            println("lambda = ", lambda)
            println("verts = ", verts)
            println("slopes = ", slopes)
            println("NP = ", hull)
        end

        if lambda == DF.Fq(1)
            base = verts
            base_NP = hull
        end
        if hull != base_NP
            println("The Newton polygon has changed!")
            if quick
                return true
            end
        end

        if hull in keys(isotropy)
            push!(isotropy[hull], lambda)
        else
            isotropy[hull] = [lambda]
        end

        plt_title = string(
            "p = ",
            DF.p,
            ", a = ",
            DF.a,
            ", d = ",
            d,
            ", λ = ",
            lambda,
        )
        p_plt = plot(
            [Singleton(vi) for vi in verts],
            title = plt_title,
            legend = false,
        )

        HP_x_coord = 0:(d-1)
        HP_y_coord = [n * (n - 1) * DF.a // (2 * d) for n = 1:(d)]

        if !quick
            plot!(p_plt, HP_x_coord, HP_y_coord, label = "HP")
            plot!(p_plt, [x[1] for x in hull], [x[2] for x in hull], label = "NP")

            display(p_plt)
        end
    end

    if quick
        return false
    end

    return isotropy
end

result(ss, n) = Iterators.product(ntuple(i -> ss, n)...)

# This does an inexhaustive search for changing NP's for a prime p
# and functions f with degrees between f_upper and f_lower
function test_family(p, f_lower, f_upper, a)
    counter = 0
    for f_deg in (f_lower:f_upper)
        DF = Dwork_Framework(p, a, f_deg)
        Fqn = result([DF.Fq(0), DF.Fq(1)], f_deg-2)

        for fpoly_var in Fqn
            fpoly_1 = DF.X^f_deg +
                sum(teich_lift(DF, fpoly_var[i]) * DF.X^i for i in (1:f_deg-2))

                    if test_poly(DF, fpoly_1, DF.Fq, true)
                        println("fpoly = ", fpoly_1)
                    end
                    counter = counter + 1
                    println("counter = ", counter, " ", fpoly_1)
                end
        end
end

##################################################################
#
#   Example
#
##################################################################

p = 5
a = 2
d = 8
DF_g = Dwork_Framework(p,a, d)

println(" p = ", p)
println(" a = ", a)

println("f = x^8 + x^6 + x^2 *********************************")
@time test_poly(
    DF_g,
    DF_g.X^8 +  DF_g.X^6+  DF_g.X^2,
    [k for k in DF_g.Fq], false)
