// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_contact.h
 * @brief The colliders a soft body meets (EDIT_MODE_SOFTBODY_DESIGN.md §8.2,
 *        §8.3).
 *
 * A soft body lists cloth colliders (`SoftBody::colliders`): a capsule or a
 * sphere on a bone, or a plane. Each has an exact distance, so meeting one is
 * a point moved to its surface: one pass, and nothing that solves.
 */

#include <span>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "../model.h"
#include "soft.h"

namespace whiteout {
namespace models {
namespace wem {

/// A collider as a pose has it, model space.
struct SoftCollider {
    bool plane = false;
    Vector3f a{0, 0, 0}; ///< A capsule's one end; or a point of the plane.
    Vector3f b{0, 0, 0}; ///< Its other end, the same for a sphere; or the plane's unit normal.
    f32 radius = 0.0f;
};

/// @p body's colliders under @p skin, a pose's skinning matrices by node, in
/// the order it lists them. One the model no longer has is left out.
std::vector<SoftCollider> SoftCollidersAt(const Model& model, const SoftBody& body, std::span<const Matrix44f> skin);

/// How deep @p at is in @p collider: positive inside, and under 0 its
/// distance to the surface. @p way is the unit way out of it there.
f32 SoftColliderDepth(const SoftCollider& collider, const Vector3f& at, Vector3f& way);

/// How far along the unit @p way a point @p at inside @p collider is from its
/// surface: what moves it out by a side that is said, and not by the nearest.
f32 SoftColliderExit(const SoftCollider& collider, const Vector3f& at, const Vector3f& way);

} // namespace wem
} // namespace models
} // namespace whiteout
