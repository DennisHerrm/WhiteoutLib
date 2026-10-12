// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_fit.h
 * @brief A soft body ready to be posed, and the fit that makes its lumps
 *        carry its points to a goal (EDIT_MODE_SOFTBODY_DESIGN.md §7).
 *
 * `SoftRig` is a body's welded points with the lanes a file holds for them,
 * as slots: a lump, or a bone as the pose has it. It poses them, and it fits:
 * given where the effects ask each point to be, the translation, turn and
 * scale of every lump that best carry the points there through the lanes.
 *
 * **The fit is a least-squares problem whose matrix is the bind's alone**
 * (the plan's P5). The unknown is each lump's whole posed matrix, and a point
 * is linear in those by its lanes and its bind position, whatever the pose.
 * So that matrix is built and factored once. A step is then:
 *
 * 1. the factored solve for a 3x4 matrix a lump, which is exact for any goal
 *    the lumps could meet as free matrices;
 * 2. each taken back through its carrier and projected to a turn and a scale
 *    along the model's axes, which is all a lump may hold (§6.5); and, with
 *    those settled, where every lump stands solved again, all at once;
 * 3. a few damped Gauss-Newton steps of nine numbers a lump, all lumps at
 *    once. `Fast` solves each step by conjugate gradients with a nine-by-nine
 *    block a lump; `Dense` factors it whole, and is the reference the fast one
 *    is held to.
 *
 * It is a function of the goal alone: it starts from the linear solve, never
 * from the step before, so a scrub gives what a play gives.
 *
 * No lump has a stiffness of its own: every lump is pulled toward its carrier
 * by the same small share of its own points' weight, which only settles a
 * lump whose points are few or lie in a line.
 */

#include <array>
#include <span>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "../model.h"
#include "../nodes/node.h"
#include "soft.h"
#include "soft_lumps.h"

namespace whiteout {
namespace models {
namespace wem {

/// What carries a share of a point: a lump, or a bone as the pose has it.
struct SoftSlot {
    u32 lump = kInvalidIndex; ///< Into the body's lumps; none is a bone alone.
    u32 node = kInvalidNode;  ///< The bone whose matrix it rides: the lump's family, or the bone.
    f32 weight = 0.0f;
};

enum class SoftFitMethod : u8 {
    Fast,  ///< The factored linear solve, projected, then swept a lump at a time.
    Dense, ///< Every lump's nine numbers at once, damped: the reference.
};

/// What a fit left.
struct SoftFitReport {
    f32 largestMove = 0.0f; ///< The farthest the goal has any point from where its skin carries it.
    f32 worstMiss = 0.0f;   ///< The farthest any point lands from its goal.
    f32 meanMiss = 0.0f;    ///< Over the area.
    /// *Follows within* (§6.6): the worst miss as a share of the largest move.
    f32 follows() const {
        return largestMove > 1e-12f ? worstMiss / largestMove : 0.0f;
    }
};

/// A soft body's points, lanes and surface, ready to pose and to fit. A cursor
/// over the model, as `Animator` is: it must not outlive it, an edit of its
/// skin, or a remake.
class SoftRig {
public:
    /// @p body is a soft body's id. Empty when the model has none of that id,
    /// or the body holds no point.
    SoftRig(const Model& model, u32 body);

    bool empty() const {
        return body_ == nullptr || points_.points.empty();
    }
    const SoftBody& body() const {
        return *body_;
    }
    const SoftPoints& points() const {
        return points_;
    }
    u32 pointCount() const {
        return static_cast<u32>(points_.points.size());
    }
    u32 lumpCount() const {
        return body_ ? static_cast<u32>(body_->lumps.size()) : 0;
    }

    /// Point @p point's slots: they sum to 1.
    std::span<const SoftSlot> slotsOf(u32 point) const {
        return std::span<const SoftSlot>(slots_).subspan(slotStart_[point], slotStart_[point + 1] - slotStart_[point]);
    }
    /// The points @p point shares a face edge with, ascending.
    std::span<const u32> ringOf(u32 point) const {
        return std::span<const u32>(ring_).subspan(ringStart_[point], ringStart_[point + 1] - ringStart_[point]);
    }
    /// The region's faces as triangles of points: those whose corners are all
    /// the body's, by the faces' own winding.
    std::span<const std::array<u32, 3>> triangles() const {
        return triangles_;
    }

    /// Where @p skin (a pose's skinning matrices by node) carries each point
    /// with every lump at rest, which is the model's own skinning; and, when
    /// asked, the matrix each point is carried by.
    void carry(std::span<const Matrix44f> skin, std::vector<Vector3f>& positions,
               std::vector<Matrix44f>* carriedBy = nullptr) const;

    /// Where the points are with each lump departed by @p departures, one a
    /// lump; a span that ends early reads as rest from there.
    void pose(std::span<const Matrix44f> skin, std::span<const Transform> departures,
              std::vector<Vector3f>& positions) const;

    /// The departures that best carry the points to @p goal (one a point,
    /// posed space) under @p skin: @p departures is replaced. A lump no point
    /// rides stays at rest.
    SoftFitReport fit(std::span<const Matrix44f> skin, std::span<const Vector3f> goal,
                      std::vector<Transform>& departures, SoftFitMethod method = SoftFitMethod::Fast) const;

    /// How @p departures do against @p goal, as `fit` reports it.
    SoftFitReport measure(std::span<const Matrix44f> skin, std::span<const Vector3f> goal,
                          std::span<const Transform> departures) const;

    /// Whether lump @p lump may scale, and may move: a rigid part's lump never
    /// scales, and a bone used as it is that has a bone of the body below it
    /// only turns (§6.5).
    bool scales(u32 lump) const {
        return lump < free_.size() && (free_[lump] & 1) != 0;
    }
    bool moves(u32 lump) const {
        return lump < free_.size() && (free_[lump] & 2) != 0;
    }

private:
    const SoftBody* body_ = nullptr;
    SoftPoints points_;
    std::vector<SoftSlot> slots_;
    std::vector<u32> slotStart_;
    std::vector<u32> ring_;
    std::vector<u32> ringStart_;
    std::vector<std::array<u32, 3>> triangles_;
    std::vector<u8> free_; ///< Per lump: bit 0 it scales, bit 1 it moves.
    std::vector<u8> ridden_; ///< Per lump: a point rides it.
    /// The linear problem's matrix, four rows a lump: as the blocks two lumps
    /// share (by row, then column), and factored with the pull on its diagonal.
    std::vector<u32> blockStart_;
    std::vector<u32> blockLump_;
    std::vector<std::array<f64, 16>> blocks_;
    std::vector<f64> factor_;
    std::vector<f64> pull_; ///< Per row, the pull toward the carrier.
    /// The same problem for where the lumps stand alone, one row a lump, factored.
    std::vector<f64> standing_;
    bool factored_ = false;
};

} // namespace wem
} // namespace models
} // namespace whiteout
