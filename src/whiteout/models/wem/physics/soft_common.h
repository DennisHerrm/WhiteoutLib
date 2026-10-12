// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/// What the soft body sources share: a mesh's positions and its triangles, and
/// a travelling Bulge's line.

#include <algorithm>
#include <cmath>
#include <span>

#include <whiteout/models/wem/geometry/attributes.h>
#include <whiteout/models/wem/geometry/mesh.h>

namespace whiteout {
namespace models {
namespace wem {
namespace softdetail {

/// A weight under this share of a point's skin rides nothing.
inline constexpr f32 kNoWeight = 1e-6f;

inline std::span<const Vector3f> PositionsOf(const Mesh& mesh) {
    return mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
}

/// @p fn(face, a, b, c) for every triangle of @p mesh, a polygon as a fan from
/// its first corner.
template <class Fn>
void ForEachTriangle(const Mesh& mesh, Fn&& fn) {
    const geom::FaceSet& faces = mesh.faceSet();
    std::size_t corner = 0;
    for (std::size_t f = 0; f < faces.faceCount(); ++f) {
        const u32 valence = faces.faceValence[f];
        for (u32 k = 1; k + 1 < valence; ++k) {
            fn(static_cast<u32>(f), faces.cornerVertex[corner], faces.cornerVertex[corner + k],
               faces.cornerVertex[corner + k + 1]);
        }
        corner += valence;
    }
}

/// Under this share of its widest, a carrier's thinnest way through is flat.
inline constexpr f64 kFlatCarrier = 0.05;

/// Whether @p carrier has scaled its points to nothing, which is how a part is
/// hidden: under a hundredth along each axis by its volume, or flat, under a
/// twentieth of its widest along one (a rig hides a part by one axis as often
/// as by three). Such a matrix has no way back through it worth the name: a
/// fit through it is the solve's noise, a thousand times over along the axes
/// it kept. So a lump it carries is left at rest.
inline bool Collapsed(const Matrix44f& carrier) {
    const auto& m = carrier.data;
    const f64 det = static_cast<f64>(m[0][0]) * (static_cast<f64>(m[1][1]) * m[2][2] - static_cast<f64>(m[1][2]) * m[2][1]) -
                    static_cast<f64>(m[0][1]) * (static_cast<f64>(m[1][0]) * m[2][2] - static_cast<f64>(m[1][2]) * m[2][0]) +
                    static_cast<f64>(m[0][2]) * (static_cast<f64>(m[1][0]) * m[2][1] - static_cast<f64>(m[1][1]) * m[2][0]);
    if (std::abs(det) < 1e-6) {
        return true;
    }
    // Its thinnest way through: the volume over the two longest of its rows.
    f64 rows[3];
    for (int r = 0; r < 3; ++r) {
        rows[r] = std::sqrt(static_cast<f64>(m[r][0]) * m[r][0] + static_cast<f64>(m[r][1]) * m[r][1] +
                            static_cast<f64>(m[r][2]) * m[r][2]);
    }
    std::sort(rows, rows + 3);
    return std::abs(det) < kFlatCarrier * rows[2] * rows[2] * rows[1];
}

/// How far from a travelling Bulge's line its swell reaches, in radii (§8.2).
inline constexpr f32 kTravelReach = 3.0f;

/// The nearest point to @p x on the line through @p points in order, and how
/// far along the line it is.
inline Vector3f OnLine(std::span<const Vector3f> points, const Vector3f& x, f32& along) {
    Vector3f best = points.empty() ? x : points.front();
    f32 least = (x - best).length();
    f32 walked = 0.0f;
    along = 0.0f;
    for (std::size_t k = 0; k + 1 < points.size(); ++k) {
        const Vector3f edge = points[k + 1] - points[k];
        const f32 length2 = edge.dot(edge);
        const f32 t = length2 > 0.0f ? std::clamp((x - points[k]).dot(edge) / length2, 0.0f, 1.0f) : 0.0f;
        const Vector3f on = points[k] + edge * t;
        const f32 d = (x - on).length();
        if (d < least) {
            least = d;
            best = on;
            along = walked + std::sqrt(length2) * t;
        }
        walked += std::sqrt(length2);
    }
    return best;
}

} // namespace softdetail
} // namespace wem
} // namespace models
} // namespace whiteout
