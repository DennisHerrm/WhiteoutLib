// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_contact.h>

#include <algorithm>
#include <cmath>

namespace whiteout {
namespace models {
namespace wem {

namespace {

/// @p node's frame at the bind: where a collider's own frame is said from.
Matrix44f RestFrame(const NodeTree& tree, u32 node) {
    if (node >= tree.size()) {
        return Matrix44f::identity();
    }
    if (tree.rig != RigConvention::PivotRelative) {
        return ToMatrix(tree.worldBind(node));
    }
    Matrix44f frame = Matrix44f::identity();
    frame.data[3][0] = tree.nodes[node].pivot.x;
    frame.data[3][1] = tree.nodes[node].pivot.y;
    frame.data[3][2] = tree.nodes[node].pivot.z;
    return frame;
}

} // namespace

std::vector<SoftCollider> SoftCollidersAt(const Model& model, const SoftBody& body, std::span<const Matrix44f> skin) {
    std::vector<SoftCollider> out;
    for (const u32 id : body.colliders) {
        const ClothCollider* collider = model.physics.collider(id);
        if (collider == nullptr) {
            continue;
        }
        // Its own frame, then its node's at the bind, then the pose.
        Matrix44f frame = collider->transform * RestFrame(model.nodes, collider->node);
        if (collider->node < skin.size()) {
            frame = frame * skin[collider->node];
        }
        const auto row = [&](int r) { return Vector3f{frame.data[r][0], frame.data[r][1], frame.data[r][2]}; };
        const Vector3f origin = row(3), axis = row(2);
        SoftCollider made;
        if (collider->kind == ClothColliderKind::Plane) {
            made.plane = true;
            made.a = origin;
            made.b = axis.length() > 1e-9f ? axis * (1.0f / axis.length()) : Vector3f{0, 0, 1};
        } else {
            // A capsule runs along its own +Z, centred; a sphere is one of no length.
            const Vector3f half = axis * (0.5f * collider->length);
            made.a = origin - half;
            made.b = origin + half;
            made.radius = collider->radius * std::max(row(0).length(), row(1).length());
        }
        out.push_back(made);
    }
    return out;
}

f32 SoftColliderDepth(const SoftCollider& collider, const Vector3f& at, Vector3f& way) {
    if (collider.plane) {
        way = collider.b;
        return -(at - collider.a).dot(collider.b);
    }
    const Vector3f axis = collider.b - collider.a;
    const f32 length2 = axis.dot(axis);
    const f32 t = length2 > 1e-12f ? std::clamp((at - collider.a).dot(axis) / length2, 0.0f, 1.0f) : 0.0f;
    const Vector3f from = at - (collider.a + axis * t);
    const f32 far = from.length();
    if (far > 1e-9f) {
        way = from * (1.0f / far);
    } else {
        // On the axis itself every way across it is as short.
        const Vector3f beside = std::abs(axis.x) < 0.9f * std::sqrt(length2) ? Vector3f{1, 0, 0} : Vector3f{0, 1, 0};
        const Vector3f across = cross(axis, beside);
        way = across.length() > 1e-9f ? across * (1.0f / across.length()) : Vector3f{0, 0, 1};
    }
    return collider.radius - far;
}

f32 SoftColliderExit(const SoftCollider& collider, const Vector3f& at, const Vector3f& way) {
    if (collider.plane) {
        const f32 depth = -(at - collider.a).dot(collider.b);
        return depth > 0.0f ? depth / std::max(way.dot(collider.b), 1e-3f) : 0.0f;
    }
    // The ball round the nearest point of its axis.
    const Vector3f axis = collider.b - collider.a;
    const f32 length2 = axis.dot(axis);
    const f32 t = length2 > 1e-12f ? std::clamp((at - collider.a).dot(axis) / length2, 0.0f, 1.0f) : 0.0f;
    const Vector3f from = at - (collider.a + axis * t);
    const f32 inside = collider.radius * collider.radius - from.dot(from);
    if (!(inside > 0.0f)) {
        return 0.0f;
    }
    const f32 along = from.dot(way);
    return -along + std::sqrt(along * along + inside);
}

} // namespace wem
} // namespace models
} // namespace whiteout
