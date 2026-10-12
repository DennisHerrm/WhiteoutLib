// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_linalg.h
 * @brief The dense algebra the soft body's fit and its modes need, and nothing
 *        more: a Cholesky factorisation for a few hundred rows, a symmetric
 *        eigen solve, and 3x3 helpers.
 *
 * Private to the soft code. A matrix is row-major in a flat vector.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

namespace whiteout {
namespace models {
namespace wem {
namespace softdetail {

/// @p a, symmetric positive definite and @p n square, factored in place to its
/// lower triangle `L` (`a = L * L^T`). False where a pivot is not positive: the
/// caller's regulariser is what keeps that from happening.
inline bool Cholesky(std::vector<f64>& a, std::size_t n) {
    for (std::size_t j = 0; j < n; ++j) {
        f64 d = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k) {
            d -= a[j * n + k] * a[j * n + k];
        }
        if (!(d > 0.0)) {
            return false;
        }
        d = std::sqrt(d);
        a[j * n + j] = d;
        for (std::size_t i = j + 1; i < n; ++i) {
            f64 s = a[i * n + j];
            const f64* ri = &a[i * n];
            const f64* rj = &a[j * n];
            for (std::size_t k = 0; k < j; ++k) {
                s -= ri[k] * rj[k];
            }
            a[i * n + j] = s / d;
        }
    }
    return true;
}

/// `L * L^T * x = b` solved in place over @p b, @p stride apart: one column of
/// a right-hand side kept as rows.
inline void CholeskySolve(const std::vector<f64>& l, std::size_t n, f64* b, std::size_t stride = 1) {
    for (std::size_t i = 0; i < n; ++i) {
        f64 s = b[i * stride];
        const f64* row = &l[i * n];
        for (std::size_t k = 0; k < i; ++k) {
            s -= row[k] * b[k * stride];
        }
        b[i * stride] = s / row[i];
    }
    for (std::size_t i = n; i-- > 0;) {
        f64 s = b[i * stride];
        for (std::size_t k = i + 1; k < n; ++k) {
            s -= l[k * n + i] * b[k * stride];
        }
        b[i * stride] = s / l[i * n + i];
    }
}

/**
 * @brief Every eigenvalue and eigenvector of @p a, symmetric and @p n square.
 *
 * The textbook route: Householder reflections bring the matrix to tridiagonal
 * form, and the QL iteration with implicit shifts diagonalises that, each
 * rotation carried into the vectors. On return @p values holds the
 * eigenvalues ascending and @p a the eigenvectors, one a ROW, orthonormal.
 * False when an eigenvalue did not settle in 64 sweeps, which a symmetric
 * matrix of finite numbers does not do.
 */
inline bool SymmetricEigen(std::vector<f64>& a, std::size_t n, std::vector<f64>& values) {
    values.assign(n, 0.0);
    if (n == 0) {
        return true;
    }
    // The reflections' product, grown as they are made: column j is the j-th
    // basis vector of the tridiagonal form.
    std::vector<f64> q(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        q[i * n + i] = 1.0;
    }
    std::vector<f64> v(n), p(n), off(n, 0.0);
    for (std::size_t k = 0; k + 2 < n; ++k) {
        // The reflection that leaves column k with one entry under its diagonal.
        f64 norm = 0.0;
        for (std::size_t i = k + 1; i < n; ++i) {
            norm += a[i * n + k] * a[i * n + k];
        }
        norm = std::sqrt(norm);
        if (!(norm > 0.0)) {
            continue;
        }
        const f64 alpha = a[(k + 1) * n + k] > 0.0 ? -norm : norm;
        f64 length = 0.0;
        for (std::size_t i = k + 1; i < n; ++i) {
            v[i] = a[i * n + k];
        }
        v[k + 1] -= alpha;
        for (std::size_t i = k + 1; i < n; ++i) {
            length += v[i] * v[i];
        }
        if (!(length > 0.0)) {
            continue;
        }
        const f64 beta = 2.0 / length;
        // A <- H A H over the trailing block, H = I - beta v v^T.
        f64 spread = 0.0;
        for (std::size_t i = k + 1; i < n; ++i) {
            f64 sum = 0.0;
            const f64* row = &a[i * n];
            for (std::size_t j = k + 1; j < n; ++j) {
                sum += row[j] * v[j];
            }
            p[i] = beta * sum;
            spread += p[i] * v[i];
        }
        const f64 half = 0.5 * beta * spread;
        for (std::size_t i = k + 1; i < n; ++i) {
            p[i] -= half * v[i];
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            f64* row = &a[i * n];
            for (std::size_t j = k + 1; j < n; ++j) {
                row[j] -= v[i] * p[j] + p[i] * v[j];
            }
        }
        a[(k + 1) * n + k] = a[k * n + k + 1] = alpha;
        for (std::size_t i = k + 2; i < n; ++i) {
            a[i * n + k] = a[k * n + i] = 0.0;
        }
        for (std::size_t r = 0; r < n; ++r) {
            f64 sum = 0.0;
            f64* row = &q[r * n];
            for (std::size_t j = k + 1; j < n; ++j) {
                sum += row[j] * v[j];
            }
            sum *= beta;
            for (std::size_t j = k + 1; j < n; ++j) {
                row[j] -= sum * v[j];
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = a[i * n + i];
        off[i] = i + 1 < n ? a[(i + 1) * n + i] : 0.0;
    }
    // QL with implicit shifts on the diagonal `values` and the sub-diagonal
    // `off`, one eigenvalue at a time from the top.
    bool settled = true;
    for (std::size_t l = 0; l < n; ++l) {
        for (int sweep = 0;; ++sweep) {
            // The first small sub-diagonal at or after l splits the problem.
            std::size_t m = l;
            for (; m + 1 < n; ++m) {
                const f64 scale = std::abs(values[m]) + std::abs(values[m + 1]);
                if (std::abs(off[m]) <= 1e-15 * scale) {
                    break;
                }
            }
            if (m == l) {
                break;
            }
            if (sweep >= 64) {
                settled = false;
                break;
            }
            // Wilkinson's shift, from the leading 2x2.
            f64 g = (values[l + 1] - values[l]) / (2.0 * off[l]);
            f64 r = std::hypot(g, 1.0);
            g = values[m] - values[l] + off[l] / (g + (g >= 0.0 ? r : -r));
            f64 s = 1.0, c = 1.0, carried = 0.0;
            bool broke = false;
            for (std::size_t i = m; i-- > l;) {
                const f64 f = s * off[i];
                const f64 b = c * off[i];
                r = std::hypot(f, g);
                off[i + 1] = r;
                if (r == 0.0) {
                    // Underflow: the problem split under the rotation.
                    values[i + 1] -= carried;
                    off[m] = 0.0;
                    broke = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = values[i + 1] - carried;
                r = (values[i] - g) * s + 2.0 * c * b;
                carried = s * r;
                values[i + 1] = g + carried;
                g = c * r - b;
                for (std::size_t row = 0; row < n; ++row) {
                    const f64 upper = q[row * n + i + 1];
                    q[row * n + i + 1] = s * q[row * n + i] + c * upper;
                    q[row * n + i] = c * q[row * n + i] - s * upper;
                }
            }
            if (broke) {
                continue;
            }
            values[l] -= carried;
            off[l] = g;
            off[m] = 0.0;
        }
    }
    // Ascending, the vectors as rows.
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) {
        order[i] = i;
    }
    for (std::size_t i = 1; i < n; ++i) {
        const std::size_t at = order[i];
        std::size_t j = i;
        for (; j > 0 && values[order[j - 1]] > values[at]; --j) {
            order[j] = order[j - 1];
        }
        order[j] = at;
    }
    std::vector<f64> sorted(n);
    for (std::size_t i = 0; i < n; ++i) {
        sorted[i] = values[order[i]];
        for (std::size_t row = 0; row < n; ++row) {
            a[i * n + row] = q[row * n + order[i]];
        }
    }
    values = std::move(sorted);
    return settled;
}

/**
 * @brief `SymmetricEigen`'s vectors made a function of the matrix alone.
 *
 * A vector's sign, and which basis spans the vectors of one eigenvalue, are
 * the iteration's whim: they move with a rounding. Each run of @p values that
 * agree is given the basis its own space makes of the coordinate axes, taken
 * in order: the axis's shadow on the space, less the ones before it. They stay
 * orthonormal eigenvectors, so nothing that sums over them moves; what reads
 * them one by one no longer depends on the solver.
 */
inline void SettleEigenBasis(std::vector<f64>& vectors, std::size_t n, const std::vector<f64>& values) {
    if (n == 0 || values.size() != n || vectors.size() != n * n) {
        return;
    }
    const f64 near = 1e-11 * std::max(std::abs(values.front()), std::abs(values.back()));
    std::vector<f64> made, shadow(n);
    for (std::size_t first = 0; first < n;) {
        std::size_t last = first + 1;
        while (last < n && values[last] - values[last - 1] <= near) {
            ++last;
        }
        const std::size_t count = last - first;
        made.clear();
        for (std::size_t axis = 0; axis < n && made.size() < count * n; ++axis) {
            // The shadow's own length, squared: an axis the space has no part of.
            f64 part = 0.0;
            for (std::size_t k = first; k < last; ++k) {
                part += vectors[k * n + axis] * vectors[k * n + axis];
            }
            if (!(part > 1e-12)) {
                continue;
            }
            std::fill(shadow.begin(), shadow.end(), 0.0);
            for (std::size_t k = first; k < last; ++k) {
                const f64* vector = &vectors[k * n];
                for (std::size_t c = 0; c < n; ++c) {
                    shadow[c] += vector[axis] * vector[c];
                }
            }
            // Twice: the first pass's rounding is what the second removes.
            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t taken = 0; taken * n < made.size(); ++taken) {
                    const f64* other = &made[taken * n];
                    f64 along = 0.0;
                    for (std::size_t c = 0; c < n; ++c) {
                        along += shadow[c] * other[c];
                    }
                    for (std::size_t c = 0; c < n; ++c) {
                        shadow[c] -= along * other[c];
                    }
                }
            }
            f64 length = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                length += shadow[c] * shadow[c];
            }
            length = std::sqrt(length);
            if (!(length > 1e-6)) {
                continue;
            }
            for (std::size_t c = 0; c < n; ++c) {
                made.push_back(shadow[c] / length);
            }
        }
        if (made.size() == count * n) {
            std::copy(made.begin(), made.end(), vectors.begin() + static_cast<std::ptrdiff_t>(first * n));
        }
        first = last;
    }
}

using Mat3 = std::array<std::array<f64, 3>, 3>;

inline Mat3 Identity3() {
    return Mat3{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
}

inline Mat3 Mul(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    return out;
}

inline Mat3 Transposed(const Mat3& a) {
    Mat3 out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            out[i][j] = a[j][i];
        }
    }
    return out;
}

inline f64 Determinant(const Mat3& m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

/// The inverse, or the identity for a matrix with none.
inline Mat3 Inverse(const Mat3& m) {
    const f64 det = Determinant(m);
    if (!(std::abs(det) > 1e-300)) {
        return Identity3();
    }
    const f64 inv = 1.0 / det;
    Mat3 out{};
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv;
    return out;
}

/// The rotation nearest @p m (its polar factor), by Higham's iteration. The
/// identity for a matrix that mirrors or has collapsed.
inline Mat3 NearestRotation(const Mat3& m) {
    if (!(Determinant(m) > 1e-12)) {
        return Identity3();
    }
    Mat3 r = m;
    for (int pass = 0; pass < 24; ++pass) {
        // Scaled by its determinant first: unscaled, a pass only halves what a
        // matrix flattened along one axis is off by, and these are not enough.
        const f64 det = std::abs(Determinant(r));
        const f64 by = det > 1e-300 ? std::cbrt(1.0 / det) : 1.0;
        const Mat3 it = Transposed(Inverse(r));
        f64 moved = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const f64 next = 0.5 * (by * r[i][j] + it[i][j] / by);
                moved = std::max(moved, std::abs(next - r[i][j]));
                r[i][j] = next;
            }
        }
        if (moved < 1e-12) {
            break;
        }
    }
    return r;
}

/// The turn by rotation vector @p w (its length the angle), as a matrix that
/// turns ROW vectors: `v * R` is `v` turned.
inline Mat3 TurnOf(const std::array<f64, 3>& w) {
    const f64 angle = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (!(angle > 1e-14)) {
        return Identity3();
    }
    const f64 x = w[0] / angle, y = w[1] / angle, z = w[2] / angle;
    const f64 c = std::cos(angle), s = std::sin(angle), t = 1.0 - c;
    // The column-vector matrix, transposed.
    return Mat3{{{t * x * x + c, t * x * y + s * z, t * x * z - s * y},
                 {t * x * y - s * z, t * y * y + c, t * y * z + s * x},
                 {t * x * z + s * y, t * y * z - s * x, t * z * z + c}}};
}

inline std::array<f64, 3> RowTimes(const std::array<f64, 3>& v, const Mat3& m) {
    return {v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0], v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
            v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2]};
}

} // namespace softdetail
} // namespace wem
} // namespace models
} // namespace whiteout
