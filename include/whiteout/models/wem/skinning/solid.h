// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file solid.h
 * @brief A model's solid as voxels, and distances walked through it
 *        (EDIT_MODE_SKIN_DESIGN.md §8.5 steps 1-3).
 *
 * A game mesh is dozens of open shells, so "inside" cannot be read from one
 * surface. The solid is what a flood from outside cannot reach, joined with
 * the cells two of three parity lines call inside; a walk through it goes
 * through the model and never round it. The geodesic weight solver measures
 * through it, and so do the soft bodies: where their lumps go, which way is
 * out of the body, and how far a ripple has run.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "../geometry/mesh.h"

namespace whiteout {
namespace models {
namespace wem {
namespace skinning {

/// The grid: `origin` is cell (0, 0, 0)'s low corner.
struct SolidGrid {
    Vector3f origin{0, 0, 0};
    f32 cell = 1.0f;
    std::array<i32, 3> size{0, 0, 0};

    std::size_t count() const {
        return static_cast<std::size_t>(size[0]) * static_cast<std::size_t>(size[1]) *
               static_cast<std::size_t>(size[2]);
    }
    std::size_t index(i32 x, i32 y, i32 z) const {
        return (static_cast<std::size_t>(z) * static_cast<std::size_t>(size[1]) +
                static_cast<std::size_t>(y)) *
                   static_cast<std::size_t>(size[0]) +
               static_cast<std::size_t>(x);
    }
    /// The cell at @p index, as (x, y, z).
    std::array<i32, 3> at(std::size_t index) const {
        return {static_cast<i32>(index % static_cast<std::size_t>(size[0])),
                static_cast<i32>((index / static_cast<std::size_t>(size[0])) % static_cast<std::size_t>(size[1])),
                static_cast<i32>(index / (static_cast<std::size_t>(size[0]) * static_cast<std::size_t>(size[1])))};
    }
    bool contains(i32 x, i32 y, i32 z) const {
        return x >= 0 && y >= 0 && z >= 0 && x < size[0] && y < size[1] && z < size[2];
    }
    Vector3f centre(i32 x, i32 y, i32 z) const {
        return Vector3f{origin.x + (static_cast<f32>(x) + 0.5f) * cell,
                        origin.y + (static_cast<f32>(y) + 0.5f) * cell,
                        origin.z + (static_cast<f32>(z) + 0.5f) * cell};
    }
    /// The cell @p position falls in, which may be outside the grid -- held to
    /// one cell past each side, so that a joint far off a small grid cannot
    /// overflow the cast; every caller asks `contains` anyway.
    std::array<i32, 3> cellOf(const Vector3f& position) const {
        const auto along = [this](f32 offset, i32 axis) {
            const f32 at = std::floor(offset / cell);
            return static_cast<i32>(std::clamp(at, -1.0f, static_cast<f32>(size[static_cast<std::size_t>(axis)])));
        };
        return {along(position.x - origin.x, 0), along(position.y - origin.y, 1),
                along(position.z - origin.z, 2)};
    }
    /// The parameters along `from + t · (to - from)`, t in [0, 1], inside the
    /// grid's box; false when the segment misses it.
    bool clip(const Vector3f& from, const Vector3f& to, f32& t0, f32& t1) const;
};

/// Empty cells round the model on every side of a `MakeSolidGrid` grid, so
/// that the outside is one connected region a flood can go all the way round.
inline constexpr i32 kSolidPad = 2;

/// The grid over the box @p low .. @p high, its longest side cut into
/// @p resolution cells (at least 4), padded by `kSolidPad`.
SolidGrid MakeSolidGrid(const Vector3f& low, const Vector3f& high, u32 resolution);

/// The solid of @p meshes on @p grid: 1 where a walk may go. A null mesh is
/// skipped. The grid must be padded, as `MakeSolidGrid` pads it.
std::vector<u8> BuildSolid(const SolidGrid& grid, std::span<const Mesh* const> meshes);

/// A cell a walk starts from, and the distance it starts at.
struct SolidSeed {
    std::size_t cell = 0;
    f32 distance = 0.0f;
};

/// What a walk found: each cell's distance, infinite where it did not reach,
/// and the box of cells it reached (`high[0] < 0` when it reached none).
struct SolidWalk {
    std::vector<f32> distance;
    std::array<i32, 3> low{0, 0, 0};
    std::array<i32, 3> high{-1, -1, -1};

    bool reachedAny() const {
        return high[0] >= 0;
    }
};

/// Dijkstra from @p seeds through @p solid over the 26 neighbours, no farther
/// than @p cutoff. A seed may sit outside the solid; a step never enters a
/// cell outside it.
SolidWalk WalkSolid(const SolidGrid& grid, std::span<const u8> solid, std::span<const SolidSeed> seeds,
                    f32 cutoff = std::numeric_limits<f32>::infinity());

/// @p walk read at @p position: trilinear over the eight cells round it, from
/// the ones the walk reached, re-weighted over those. False when it reached
/// none of them.
bool ReadSolidWalk(const SolidGrid& grid, const SolidWalk& walk, const Vector3f& position, f32& distance);

} // namespace skinning
} // namespace wem
} // namespace models
} // namespace whiteout
