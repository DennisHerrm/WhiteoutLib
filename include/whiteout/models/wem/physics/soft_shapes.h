// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_shapes.h
 * @brief Three of a soft body's fields: Blobify's blob, Melt's map and Ripple's
 *        wave (EDIT_MODE_SOFTBODY_DESIGN.md §8.5, §8.8, §8.9).
 *
 * Each is a function of where the points are and of a time or a progress, with
 * no memory: `SoftStepper::step` calls them in the list's order, as it does
 * Inflate and Bulge.
 *
 * **The blob** is a level of a sum of balls [Blinn 1982], one at each lump, set
 * inward by half the body's thickness there. It asks nothing of the mesh.
 *
 * **The melt** maps the body into the pool a thick liquid spreads as
 * [Huppert 1982], `h = h0 * (1 - r^2 / R^2)^(1/3)`: a point keeps its bearing,
 * its height becomes its depth in the pool and its distance out is kept in
 * order, so nothing folds. The pool's size is the setting and its depth is
 * read from the volume.
 *
 * **The wave** is a height by the distance from a place and the time.
 */

#include <span>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "soft.h"
#include "soft_fields.h"
#include "soft_fit.h"

namespace whiteout {
namespace models {
namespace wem {

/// One ball of a blob: where it stands and how far its field reaches.
struct SoftBall {
    Vector3f at{0, 0, 0};
    f32 reach = 0.0f;
};

/// How large each lump of @p rig is: half the body's thickness under one on
/// the surface, and as far as its points stand from one at a bone. 0 for a
/// lump no point rides. What a ball's size is, and what a swell of a lump's
/// scale is measured against.
std::vector<f32> SoftLumpRadii(const SoftRig& rig);

/// The balls of @p rig under @p skin (§8.5): one a lump, at its anchor as the
/// pose carries it, set inward by its depth where it is on the surface.
/// @p roundness takes each ball's reach from twice its size, which hugs the
/// body, through the region's half way, to eight regions, which is a ball.
std::vector<SoftBall> SoftBlobBalls(const SoftRig& rig, std::span<const Matrix44f> skin, f32 roundness);

/// The sum of the balls' fields at @p at, and the way it grows fastest there:
/// toward the inside of the blob. Zero where no ball reaches.
f32 SoftBlobField(std::span<const SoftBall> balls, const Vector3f& at, Vector3f* inward = nullptr);

/// What `SoftBlobTo` did.
struct SoftBlobReport {
    f32 level = 0.0f; ///< The level of the field the blob is.
    /// The share of the points that were inside it: the inner side of a part
    /// that fused, sent most of the way out and kept in order under the surface.
    f32 buried = 0.0f;
    /// How near two parts of the body come before the blob joins them, model
    /// units. Between two alike the field in the middle is twice what each has
    /// there, so they join where each has half the level left: twice how far
    /// out of the blob's surface that is, over the area. At least that: where
    /// the field reaches past what was read of it, as far as was read, which
    /// is more than the body's size across. 0 when no ball reaches the body.
    f32 fuses = 0.0f;
};

/**
 * @brief @p points moved onto the blob of @p balls (§8.5), by @p amount (0 the
 *        model, 1 the blob) times each point's softness.
 *
 * Each point steps along the field's gradient to the level. With
 * @p keepVolume, and a body that encloses one, the level is the one that
 * keeps the volume the points had, and @p swell more of it.
 */
SoftBlobReport SoftBlobTo(const SoftRig& rig, const SoftVolume& volume, std::span<const SoftBall> balls,
                           std::vector<Vector3f>& points, f32 amount, f32 swell, bool keepVolume);

/// Each point's place in the body's height where it stands, at the bind: how
/// far up its column of the solid it is (0 its underside, 1 its top), and how
/// thick that column is. What a melt keeps the order of.
struct SoftColumns {
    std::vector<f32> share;
    std::vector<f32> thickness;
};

SoftColumns SoftMeltColumns(const SoftRig& rig);

/// What a melt comes to: the pool it ends as.
struct SoftPool {
    Vector3f centre{0, 0, 0}; ///< Of the footprint, on the floor.
    f32 radius = 0.0f;        ///< The pool's, at the end.
    f32 depth = 0.0f;         ///< Read: how deep the body ends, by its volume.
    f32 height = 0.0f;        ///< The body's above the floor, before it melts.
};

/// The pool @p points would end as on @p floor (a height) under @p settings.
SoftPool SoftMeltPool(const SoftRig& rig, const SoftVolume& volume, const SoftColumns& columns,
                      std::span<const Vector3f> points, f32 floor, const SoftMelt& settings);

/**
 * @brief @p points melted to @p progress (0 the body, 1 the pool) on @p floor
 *        (§8.8), each by its softness. @p seed turns which parts go a little
 *        sooner under *Uneven*.
 *
 * A column of the body keeps its order at every progress: a layer thins where
 * the melt has reached it and stays over the ones below, so nothing passes
 * through anything. The volume is kept at every progress, by the height above
 * the floor. Returns the pool the body is on its way to.
 */
SoftPool SoftMeltTo(const SoftRig& rig, const SoftVolume& volume, const SoftColumns& columns,
                    std::vector<Vector3f>& points, f32 floor, const SoftMelt& settings, f32 progress, u32 seed);

/// A Ripple's train or swell as lengths: what the settings' shares of the
/// region come to on a body of @p size.
struct SoftWave {
    f32 height = 0.0f;
    f32 length = 1.0f;
    f32 speed = 1.0f;         ///< Of the crests, a second.
    f32 envelope = 1.0f;      ///< The swell's own speed over the crests': 1 unless *Water*.
    f32 fadeDistance = 0.0f;  ///< Per length travelled: `exp(-fade * d)`.
    f32 fadeTime = 0.0f;      ///< Seconds to a third; 0 does not fade.
};

SoftWave SoftWaveOf(const SoftRipple& settings, f32 amount, f32 size);

/// A swell that set out @p since seconds ago, at @p distance from its place
/// (§8.9, *Once*): a crest at `speed * since`, a trough either side of it.
f32 SoftSwellHeight(const SoftWave& wave, f32 distance, f32 since);

/// A steady train at @p distance, @p turns swings into its cycle (§8.9,
/// *Loops*): the place swings as `sin`, and a crest moves outward.
f32 SoftTrainHeight(const SoftWave& wave, f32 distance, f32 turns);

/// Every point's distance from @p place (§8.9, *Through*): over the skin, or
/// through the solid. Measured at the bind. A point the walk does not reach
/// (another shell, over the skin) is at infinity.
std::vector<f32> SoftRippleDistances(const SoftRig& rig, const SoftPlaceAt& place, SoftThrough through);

/// Each lump's distance, from its points' (@p distances, one a point): theirs
/// on the whole, by what each weighs on it. Infinite where none was reached.
std::vector<f32> SoftLumpDistances(const SoftRig& rig, std::span<const f32> distances);

} // namespace wem
} // namespace models
} // namespace whiteout
