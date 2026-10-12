// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_fields.h
 * @brief What the effects' fields are built from: a soft body's volume, the
 *        exact cubic an offset makes of it, distance over its surface, and a
 *        place on it (EDIT_MODE_SOFTBODY_DESIGN.md §5.4, §7, §8.2).
 *
 * **The volume** is the outermost layer's (§5.4): of the region's shells,
 * those no other shell encloses, each wound outward whatever its faces say,
 * and each open border closed by a fan to its middle. Layers drawn inside one
 * another are counted once.
 *
 * **An offset's volume is an exact cubic** (§7): moving every point along a
 * fixed direction by one number makes each signed tetrahedron a cubic in that
 * number, and so their sum. `OffsetCubic` gives the four coefficients in one
 * pass; the wanted offset is a root.
 */

#include <array>
#include <span>
#include <utility>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "../model.h"
#include "soft.h"
#include "soft_fit.h"

namespace whiteout {
namespace models {
namespace wem {

/// The surface a soft body's volume is read from, in its rig's points.
struct SoftVolume {
    /// The outermost layer's faces, each wound outward.
    std::vector<std::array<u32, 3>> triangles;
    /// The open borders, each a loop of points in the order that closes its
    /// shell outward when fanned to the loop's middle.
    std::vector<std::vector<u32>> fans;
    u32 shells = 0; ///< Connected pieces of the region's surface.
    u32 outer = 0;  ///< Those no other encloses.
    f32 rest = 0.0f; ///< The volume at the bind.
    /// Whether that is a volume at all: a flat patch encloses almost none,
    /// and an Amount said as a share of it means nothing (§8.1).
    bool encloses = false;
};

SoftVolume MakeSoftVolume(const SoftRig& rig);

/// @p volume's with its points at @p positions.
f64 SoftVolumeAt(const SoftVolume& volume, std::span<const Vector3f> positions);

/// What the volume changes by for a unit move of each point: the cubic's
/// linear term is its sum over the points with their moves.
std::vector<Vector3f> SoftVolumeGradient(const SoftVolume& volume, std::span<const Vector3f> positions);

/// The volume with every point at `positions[i] + d * moves[i]`, as the
/// coefficients of `c0 + c1 d + c2 d^2 + c3 d^3`.
std::array<f64, 4> OffsetCubic(const SoftVolume& volume, std::span<const Vector3f> positions,
                               std::span<const Vector3f> moves);

/// The offset nearest 0, within @p limit either way, that brings @p cubic to
/// @p target; the end of that range nearest the target when it holds no root.
f64 SolveOffset(const std::array<f64, 4>& cubic, f64 target, f64 limit);

/// Each point's distance over the surface from @p seeds (a point and the
/// distance it starts at), by the region's face edges at the bind; a point
/// past @p cutoff, or that no edge reaches, is at `cutoff`.
std::vector<f32> SurfaceDistances(const SoftRig& rig, std::span<const std::pair<u32, f32>> seeds, f32 cutoff);

/// A place found on a body at the bind (§8.2).
struct SoftPlaceAt {
    bool valid = false;
    Vector3f at{0, 0, 0};
    Vector3f normal{0, 0, 1}; ///< Outward.
    /// The face's corners as points of the rig, and the place's weights on them.
    std::array<u32, 3> points{kInvalidIndex, kInvalidIndex, kInvalidIndex};
    Vector3f weights{1, 0, 0};
    u32 bone = kInvalidNode; ///< The bone that carries most of it.
};

/// @p place on @p rig's body, read from @p model's mesh at the bind. Not
/// valid when the face is not there or no corner of it is the body's.
SoftPlaceAt ResolveSoftPlace(const Model& model, const SoftRig& rig, const SoftPlace& place);

/// Where @p place is with the points at @p positions.
Vector3f SoftPlacePosition(const SoftPlaceAt& place, std::span<const Vector3f> positions);

} // namespace wem
} // namespace models
} // namespace whiteout
