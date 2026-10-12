// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_fit.h>

#include <algorithm>
#include <cmath>
#include <map>

#include "soft_common.h"
#include "soft_linalg.h"

namespace whiteout {
namespace models {
namespace wem {

namespace {

using softdetail::ForEachTriangle;
using softdetail::Mat3;

/// The share of a lump's own weight it is pulled toward its carrier by: the
/// same for every lump, and only enough to settle one whose points are few or
/// lie in a line. A goal the lumps can meet is missed by about this share.
constexpr f64 kPull = 1e-6;
/// How dear it is for two lumps of a point to turn or scale apart: as if the
/// point were moved by their difference over a length, whose square is this
/// share of the area a lump has. A seventh of a lump's width, about.
constexpr f64 kAlike = 0.02;
/// A lump counts as held by its points by this much at the least (the mean of
/// its lanes' weights, by those weights): under it, it is asked alone for the
/// rest, and is sent five times as far as a miss at the most.
constexpr f64 kLeastHold = 0.2;
constexpr f64 kLeastScale = 0.05;
constexpr f64 kMostScale = 20.0;
/// Gauss-Newton steps after the linear solve, and what ends them early.
constexpr int kFastSteps = 6;
constexpr int kDenseSteps = 24;
constexpr int kGradientSteps = 32;
constexpr f64 kSettled = 1e-5;

/// A lump's departure while it is being fitted: `p' = ((p - c) * s) * r + c + t`.
struct LumpState {
    std::array<f64, 3> t{0, 0, 0};
    Mat3 r = softdetail::Identity3();
    std::array<f64, 3> s{1, 1, 1};
};

Transform ToTransform(const LumpState& state) {
    Matrix44f m = Matrix44f::identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m.data[i][j] = static_cast<f32>(state.s[i] * state.r[i][j]);
        }
    }
    Transform out = FromMatrix(m);
    out.translation = Vector3f{static_cast<f32>(state.t[0]), static_cast<f32>(state.t[1]), static_cast<f32>(state.t[2])};
    out.scale = Vector3f{static_cast<f32>(state.s[0]), static_cast<f32>(state.s[1]), static_cast<f32>(state.s[2])};
    return out;
}

std::array<f64, 3> Cross(const std::array<f64, 3>& a, const std::array<f64, 3>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

/// A lump's posed matrix as the linear problem holds it, twelve numbers: the
/// three rows that take a point's offset from the lump's pivot, then where the
/// pivot itself lands.
using Posed = std::array<f64, 12>;
/// How those twelve move with the lump's nine: its translation, a small turn,
/// its three scales.
using Slope = std::array<std::array<f64, 9>, 12>;

} // namespace

SoftRig::SoftRig(const Model& model, u32 body) {
    body_ = model.physics.softBody(body);
    if (body_ == nullptr) {
        return;
    }
    points_ = GatherSoftPoints(model, body, std::max<u32>(1, body_->influences));
    const u32 count = pointCount();
    if (count == 0) {
        return;
    }
    const u32 nodes = model.nodes.size();
    const std::span<const SoftLump> lumps = body_->lumps;
    std::vector<u32> boneLump(nodes, kInvalidIndex);
    std::vector<u8> ofBody(nodes, 0);
    for (u32 k = 0; k < lumps.size(); ++k) {
        if (lumps[k].kind == SoftLumpKind::Bone && lumps[k].node < nodes) {
            boneLump[lumps[k].node] = k;
        }
        if (lumps[k].family() < nodes) {
            ofBody[lumps[k].family()] = 1;
        }
    }

    // The lanes a file holds, as slots.
    SoftSkinner skinner(model, *body_, SoftWeights::Float);
    slotStart_.reserve(count + 1);
    slotStart_.push_back(0);
    ridden_.assign(lumps.size(), 0);
    for (const SoftPoint& point : points_.points) {
        f32 sum = 0.0f;
        const std::size_t first = slots_.size();
        for (const geom::Influence& influence : skinner.of(point.mesh, point.vertex)) {
            if (!(influence.weight > 0.0f)) {
                continue;
            }
            SoftSlot slot;
            slot.weight = influence.weight;
            if (influence.bone >= skinner.lumpBase()) {
                slot.lump = influence.bone - skinner.lumpBase();
                slot.node = slot.lump < lumps.size() ? lumps[slot.lump].family() : kInvalidNode;
            } else {
                slot.node = influence.bone;
                slot.lump = influence.bone < nodes ? boneLump[influence.bone] : kInvalidIndex;
            }
            if (slot.lump < lumps.size()) {
                ridden_[slot.lump] = 1;
            } else {
                slot.lump = kInvalidIndex;
            }
            sum += slot.weight;
            slots_.push_back(slot);
        }
        for (std::size_t s = first; s < slots_.size() && sum > 0.0f; ++s) {
            slots_[s].weight /= sum;
        }
        slotStart_.push_back(static_cast<u32>(slots_.size()));
    }

    // The surface: the faces whose corners are all the body's.
    std::vector<std::vector<u32>> rings(count);
    for (u32 m = 0; m < model.meshes.size() && m < points_.pointOfVertex.size(); ++m) {
        const std::vector<u32>& entryOf = points_.pointOfVertex[m];
        if (entryOf.empty()) {
            continue;
        }
        ForEachTriangle(model.meshes[m], [&](u32, u32 a, u32 b, u32 c) {
            if (a >= entryOf.size() || b >= entryOf.size() || c >= entryOf.size()) {
                return;
            }
            const std::array<u32, 3> corners{entryOf[a], entryOf[b], entryOf[c]};
            if (corners[0] == kInvalidIndex || corners[1] == kInvalidIndex || corners[2] == kInvalidIndex ||
                corners[0] == corners[1] || corners[1] == corners[2] || corners[0] == corners[2]) {
                return;
            }
            triangles_.push_back(corners);
            for (int k = 0; k < 3; ++k) {
                rings[corners[k]].push_back(corners[(k + 1) % 3]);
                rings[corners[k]].push_back(corners[(k + 2) % 3]);
            }
        });
    }
    ringStart_.reserve(count + 1);
    ringStart_.push_back(0);
    for (std::vector<u32>& ring : rings) {
        std::sort(ring.begin(), ring.end());
        ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
        ring_.insert(ring_.end(), ring.begin(), ring.end());
        ringStart_.push_back(static_cast<u32>(ring_.size()));
    }

    // What each lump is free to do (§6.5).
    free_.assign(lumps.size(), 3);
    for (u32 k = 0; k < lumps.size(); ++k) {
        const SoftLump& lump = lumps[k];
        bool scales = !lump.rigid;
        bool moves = true;
        if (lump.kind == SoftLumpKind::Bone && lump.node < nodes) {
            // A bone with bones below it would hand them its scale.
            for (u32 n = 0; n < nodes; ++n) {
                const Node& child = model.nodes.nodes[n];
                scales = scales && !(child.parent == lump.node && child.kind == NodeKind::Bone && !child.removed);
            }
            // A chain bone only turns.
            const u32 parent = model.nodes.nodes[lump.node].parent;
            moves = !(body_->chainsAsTheyAre && parent < nodes && ofBody[parent] != 0);
        }
        free_[k] = static_cast<u8>((scales ? 1 : 0) | (moves ? 2 : 0));
    }

    // A lump its points hold weakly is asked alone as well, by what its hold
    // lacks: to take those points where they are asked, as if it had them
    // whole. Its hold is the mean of its lanes' weights, by those weights.
    alone_.assign(lumps.size(), 0.0);
    {
        std::vector<f64> once(lumps.size(), 0.0), twice(lumps.size(), 0.0);
        for (u32 p = 0; p < count; ++p) {
            const f64 area = std::max(static_cast<f64>(points_.points[p].area), 1e-12);
            for (const SoftSlot& slot : slotsOf(p)) {
                if (slot.lump != kInvalidIndex) {
                    once[slot.lump] += area * slot.weight;
                    twice[slot.lump] += area * slot.weight * slot.weight;
                }
            }
        }
        for (std::size_t k = 0; k < lumps.size(); ++k) {
            alone_[k] = once[k] > 0.0 ? std::max(0.0, kLeastHold - twice[k] / once[k]) : 0.0;
        }
    }

    // The linear problem: each lump's posed matrix, as `[q, 1] * Y` with `q`
    // a point's offset from the lump's pivot at the bind. Its matrix holds the
    // lanes and the bind positions and nothing of the pose.
    const std::size_t n = lumps.size() * 4;
    if (n == 0) {
        return;
    }
    std::map<u64, std::array<f64, 16>> shared;
    // The area a lump has, were they shared out evenly: the square of a length.
    const f64 patch = static_cast<f64>(points_.area) / static_cast<f64>(lumps.size());
    std::map<u64, f64> apart;
    for (u32 p = 0; p < count; ++p) {
        const SoftPoint& point = points_.points[p];
        const f64 area = std::max(static_cast<f64>(point.area), 1e-12);
        std::array<std::array<f64, 4>, 8> feature{};
        std::array<u32, 8> of{};
        u32 used = 0;
        for (const SoftSlot& slot : slotsOf(p)) {
            if (slot.lump == kInvalidIndex || used >= feature.size()) {
                continue;
            }
            const Vector3f& pivot = lumps[slot.lump].rest;
            feature[used] = {slot.weight * (static_cast<f64>(point.at.x) - pivot.x),
                             slot.weight * (static_cast<f64>(point.at.y) - pivot.y),
                             slot.weight * (static_cast<f64>(point.at.z) - pivot.z), slot.weight};
            of[used++] = slot.lump;
        }
        for (u32 a = 0; a < used; ++a) {
            for (u32 b = 0; b < used; ++b) {
                std::array<f64, 16>& block = shared[(static_cast<u64>(of[a]) << 32) | of[b]];
                // Alone, a lane counts by its weight and not by its square.
                const f64 both = a == b && feature[a][3] > 0.0 ? 1.0 + alone_[of[a]] / feature[a][3] : 1.0;
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        block[i * 4 + j] += area * feature[a][i] * feature[b][j] * both;
                    }
                }
                // Two lumps of the point turn and scale alike: the three rows
                // of each that are not where it stands, one less the other.
                if (of[a] != of[b]) {
                    const f64 alike = kAlike * patch * area * feature[a][3] * feature[b][3];
                    apart[(static_cast<u64>(of[a]) << 32) | of[b]] += alike;
                    std::array<f64, 16>& own = shared[(static_cast<u64>(of[a]) << 32) | of[a]];
                    for (std::size_t i = 0; i < 3; ++i) {
                        own[i * 4 + i] += alike;
                        block[i * 4 + i] -= alike;
                    }
                }
            }
        }
    }
    blockStart_.assign(lumps.size() + 1, 0);
    for (const auto& [key, block] : shared) {
        ++blockStart_[(key >> 32) + 1];
        blockLump_.push_back(static_cast<u32>(key & 0xFFFFFFFFu));
        blocks_.push_back(block);
        const auto pair = apart.find(key);
        alike_.push_back(pair != apart.end() ? pair->second : 0.0);
    }
    for (std::size_t k = 0; k < lumps.size(); ++k) {
        blockStart_[k + 1] += blockStart_[k];
    }

    factor_.assign(n * n, 0.0);
    for (std::size_t k = 0; k < lumps.size(); ++k) {
        for (u32 e = blockStart_[k]; e < blockStart_[k + 1]; ++e) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    factor_[(k * 4 + i) * n + blockLump_[e] * 4 + j] = blocks_[e][i * 4 + j];
                }
            }
        }
    }
    f64 largest = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        largest = std::max(largest, factor_[i * n + i]);
    }
    pull_.assign(n, 0.0);
    for (std::size_t k = 0; k < lumps.size(); ++k) {
        const f64 turn = (factor_[(k * 4) * n + k * 4] + factor_[(k * 4 + 1) * n + k * 4 + 1] +
                          factor_[(k * 4 + 2) * n + k * 4 + 2]) / 3.0;
        const f64 move = factor_[(k * 4 + 3) * n + k * 4 + 3];
        for (int i = 0; i < 4; ++i) {
            // A lump no point rides is held at rest by the floor alone.
            pull_[k * 4 + i] = kPull * (i < 3 ? turn : move) + 1e-9 * largest + 1e-30;
            factor_[(k * 4 + i) * n + k * 4 + i] += pull_[k * 4 + i];
        }
    }
    // Where the lumps stand, alone: the rows and columns of the pivots.
    const std::size_t lumpCount = lumps.size();
    standing_.assign(lumpCount * lumpCount, 0.0);
    for (std::size_t k = 0; k < lumpCount; ++k) {
        for (u32 e = blockStart_[k]; e < blockStart_[k + 1]; ++e) {
            standing_[k * lumpCount + blockLump_[e]] = blocks_[e][15];
        }
        standing_[k * lumpCount + k] += pull_[k * 4 + 3];
    }
    factored_ = softdetail::Cholesky(factor_, n) && softdetail::Cholesky(standing_, lumpCount);
}

void SoftRig::carry(std::span<const Matrix44f> skin, std::vector<Vector3f>& positions,
                    std::vector<Matrix44f>* carriedBy) const {
    const u32 count = pointCount();
    positions.assign(count, Vector3f{0, 0, 0});
    if (carriedBy != nullptr) {
        carriedBy->assign(count, Matrix44f::identity());
    }
    for (u32 p = 0; p < count; ++p) {
        const SoftPoint& point = points_.points[p];
        Matrix44f blend = Matrix44f::identity();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                blend.data[i][j] = 0.0f;
            }
        }
        f32 sum = 0.0f;
        for (const SoftSlot& slot : slotsOf(p)) {
            if (slot.node >= skin.size()) {
                continue;
            }
            const Matrix44f& m = skin[slot.node];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    blend.data[i][j] += m.data[i][j] * slot.weight;
                }
            }
            sum += slot.weight;
        }
        if (!(sum > 0.0f)) {
            blend = Matrix44f::identity();
        } else if (std::abs(sum - 1.0f) > 1e-6f) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    blend.data[i][j] /= sum;
                }
            }
        }
        positions[p] = transform_point(point.at, blend);
        if (carriedBy != nullptr) {
            (*carriedBy)[p] = blend;
        }
    }
}

void SoftRig::pose(std::span<const Matrix44f> skin, std::span<const Transform> departures,
                   std::vector<Vector3f>& positions) const {
    const u32 count = pointCount();
    positions.assign(count, Vector3f{0, 0, 0});
    if (count == 0) {
        return;
    }
    const std::span<const SoftLump> lumps = body_->lumps;
    std::vector<Matrix44f> moved(lumps.size(), Matrix44f::identity());
    for (u32 k = 0; k < lumps.size(); ++k) {
        const u32 family = lumps[k].family();
        const Matrix44f carried = family < skin.size() ? skin[family] : Matrix44f::identity();
        moved[k] = k < departures.size() ? SoftDepartureMatrix(lumps[k].rest, departures[k]) * carried : carried;
    }
    for (u32 p = 0; p < count; ++p) {
        const SoftPoint& point = points_.points[p];
        Vector3f at{0, 0, 0};
        f32 sum = 0.0f;
        for (const SoftSlot& slot : slotsOf(p)) {
            if (slot.lump < moved.size()) {
                at = at + transform_point(point.at, moved[slot.lump]) * slot.weight;
            } else if (slot.node < skin.size()) {
                at = at + transform_point(point.at, skin[slot.node]) * slot.weight;
            } else {
                at = at + point.at * slot.weight;
            }
            sum += slot.weight;
        }
        positions[p] = sum > 0.0f ? at * (1.0f / sum) : point.at;
    }
}

SoftFitReport SoftRig::measure(std::span<const Matrix44f> skin, std::span<const Vector3f> goal,
                               std::span<const Transform> departures) const {
    SoftFitReport report;
    const u32 count = pointCount();
    if (count == 0 || goal.size() < count) {
        return report;
    }
    std::vector<Vector3f> carried, posed;
    carry(skin, carried);
    pose(skin, departures, posed);
    f64 missed = 0.0, area = 0.0;
    for (u32 p = 0; p < count; ++p) {
        const f32 miss = (posed[p] - goal[p]).length();
        report.largestMove = std::max(report.largestMove, (goal[p] - carried[p]).length());
        report.worstMiss = std::max(report.worstMiss, miss);
        missed += static_cast<f64>(miss) * points_.points[p].area;
        area += points_.points[p].area;
    }
    report.meanMiss = area > 0.0 ? static_cast<f32>(missed / area) : 0.0f;
    return report;
}

SoftFitReport SoftRig::fit(std::span<const Matrix44f> skin, std::span<const Vector3f> goal,
                           std::vector<Transform>& departures, SoftFitMethod method) const {
    const u32 count = pointCount();
    const std::size_t lumpCount = this->lumpCount();
    departures.assign(lumpCount, Transform{});
    if (count == 0 || goal.size() < count || lumpCount == 0 || !factored_) {
        return measure(skin, goal, departures);
    }
    const std::span<const SoftLump> lumps = body_->lumps;
    const std::size_t n = lumpCount * 4;

    // Each lump's carrier, as the pose has it: `p * K = p * a + tau`.
    std::vector<Mat3> carrier(lumpCount, softdetail::Identity3()), back(lumpCount, softdetail::Identity3());
    std::vector<std::array<f64, 3>> tau(lumpCount, std::array<f64, 3>{0, 0, 0});
    std::vector<u8> collapsed(lumpCount, 0);
    for (std::size_t k = 0; k < lumpCount; ++k) {
        const u32 family = lumps[k].family();
        if (family < skin.size()) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    carrier[k][i][j] = skin[family].data[i][j];
                }
                tau[k][i] = skin[family].data[3][i];
            }
        }
        back[k] = softdetail::Inverse(carrier[k]);
        // A bone scaled to nothing (how a part is hidden) has no way back: its
        // lumps are left at rest, where any departure is as good, and none is
        // keyed for a clip to ease into.
        collapsed[k] = family < skin.size() && softdetail::Collapsed(skin[family]) ? 1 : 0;
    }
    // A lump at rest, as the linear problem holds it.
    const auto resting = [&](std::size_t k) {
        Posed y{};
        const Vector3f& c = lumps[k].rest;
        const std::array<f64, 3> pivot = softdetail::RowTimes({c.x, c.y, c.z}, carrier[k]);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                y[i * 3 + j] = carrier[k][i][j];
            }
            y[9 + i] = pivot[i] + tau[k][i];
        }
        return y;
    };

    // What the lumps have to supply: each point's move from where its skin
    // carries it. Said as a move, a goal that is the pose asks for nothing at
    // all, and the lumps stay at rest to the bit.
    std::vector<Vector3f> carried;
    carry(skin, carried);
    std::vector<f64> wanted(n * 3, 0.0);
    for (u32 p = 0; p < count; ++p) {
        const SoftPoint& point = points_.points[p];
        const f64 area = std::max(static_cast<f64>(point.area), 1e-12);
        const Vector3f move = goal[p] - carried[p];
        const std::array<f64, 3> left{move.x, move.y, move.z};
        for (const SoftSlot& slot : slotsOf(p)) {
            if (slot.lump == kInvalidIndex) {
                continue;
            }
            const Vector3f& pivot = lumps[slot.lump].rest;
            const std::array<f64, 4> feature{slot.weight * (static_cast<f64>(point.at.x) - pivot.x),
                                             slot.weight * (static_cast<f64>(point.at.y) - pivot.y),
                                             slot.weight * (static_cast<f64>(point.at.z) - pivot.z), slot.weight};
            // Together, and alone.
            const f64 asked = area * (1.0 + alone_[slot.lump]);
            for (int i = 0; i < 4; ++i) {
                for (int c = 0; c < 3; ++c) {
                    wanted[(slot.lump * 4 + i) * 3 + c] += asked * feature[i] * left[c];
                }
            }
        }
    }
    // Each lump at rest: what the unknowns are counted from.
    std::vector<f64> rest(n * 3, 0.0);
    for (std::size_t k = 0; k < lumpCount; ++k) {
        const Posed y = resting(k);
        std::copy(y.begin(), y.end(), rest.begin() + static_cast<std::ptrdiff_t>(k * 12));
    }
    // *Alike* is asked of what each lump does past its carrier, and the blocks
    // hold it as the bind has it. Whether this pose turns any carrier at all.
    bool turned = false;
    for (std::size_t k = 0; k < lumpCount && !turned; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                turned = turned || std::abs(carrier[k][i][j] - (i == j ? 1.0 : 0.0)) > 1e-9;
            }
        }
    }
    // The problem's matrix times @p y, with or without the pull.
    const auto apply = [&](const std::vector<f64>& y, std::vector<f64>& out, bool pulled) {
        out.assign(n * 3, 0.0);
        for (std::size_t k = 0; k < lumpCount; ++k) {
            for (u32 e = blockStart_[k]; e < blockStart_[k + 1]; ++e) {
                const std::array<f64, 16>& block = blocks_[e];
                const f64* from = &y[blockLump_[e] * 12];
                f64* to = &out[k * 12];
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        const f64 w = block[i * 4 + j];
                        to[i * 3] += w * from[j * 3];
                        to[i * 3 + 1] += w * from[j * 3 + 1];
                        to[i * 3 + 2] += w * from[j * 3 + 2];
                    }
                }
            }
        }
        // Two lumps' rows, each taken back through its own carrier, in place
        // of the rows as they are: `B_k^T (B_k y_k - B_l y_l) - (y_k - y_l)`.
        for (std::size_t k = 0; turned && k < lumpCount; ++k) {
            for (u32 e = blockStart_[k]; e < blockStart_[k + 1]; ++e) {
                const std::size_t l = blockLump_[e];
                if (!(alike_[e] > 0.0) || collapsed[k] != 0 || collapsed[l] != 0) {
                    continue;
                }
                for (int j = 0; j < 3; ++j) {
                    std::array<f64, 3> differ{0, 0, 0};
                    for (int i = 0; i < 3; ++i) {
                        for (int m = 0; m < 3; ++m) {
                            differ[i] += back[k][i][m] * y[k * 12 + m * 3 + j] - back[l][i][m] * y[l * 12 + m * 3 + j];
                        }
                    }
                    for (int i = 0; i < 3; ++i) {
                        const f64 through =
                            back[k][0][i] * differ[0] + back[k][1][i] * differ[1] + back[k][2][i] * differ[2];
                        const f64 asItIs = y[k * 12 + i * 3 + j] - y[l * 12 + i * 3 + j];
                        out[k * 12 + i * 3 + j] += alike_[e] * (through - asItIs);
                    }
                }
            }
        }
        for (std::size_t i = 0; pulled && i < n * 3; ++i) {
            out[i] += pull_[i / 3] * y[i];
        }
    };

    // 1. A 3x4 matrix a lump, by the factored solve: the lumps' departure
    // from rest. Two more passes take out what the pull held back; where the
    // pose turns the carriers, four, for what the bind's matrix does not hold.
    std::vector<f64> solved = wanted;
    for (int c = 0; c < 3; ++c) {
        softdetail::CholeskySolve(factor_, n, solved.data() + c, 3);
    }
    {
        std::vector<f64> left, more;
        for (int pass = 0; pass < (turned ? 4 : 2); ++pass) {
            apply(solved, left, false);
            more.resize(n * 3);
            for (std::size_t i = 0; i < n * 3; ++i) {
                more[i] = wanted[i] - left[i];
            }
            for (int c = 0; c < 3; ++c) {
                softdetail::CholeskySolve(factor_, n, more.data() + c, 3);
            }
            for (std::size_t i = 0; i < n * 3; ++i) {
                solved[i] += more[i];
            }
        }
    }
    for (std::size_t i = 0; i < n * 3; ++i) {
        solved[i] += rest[i];
    }

    // 2. Each taken back through its carrier, and projected to a turn and a
    // scale along the model's axes.
    std::vector<LumpState> state(lumpCount);
    for (std::size_t k = 0; k < lumpCount; ++k) {
        if (ridden_[k] == 0 || collapsed[k] != 0) {
            continue;
        }
        Mat3 posed{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                posed[i][j] = solved[(k * 4 + i) * 3 + j];
            }
        }
        const Mat3 linear = softdetail::Mul(posed, back[k]);
        LumpState& lump = state[k];
        lump.r = softdetail::NearestRotation(linear);
        if (scales(static_cast<u32>(k))) {
            const Mat3 stretch = softdetail::Mul(linear, softdetail::Transposed(lump.r));
            for (int i = 0; i < 3; ++i) {
                lump.s[i] = std::clamp(stretch[i][i], kLeastScale, kMostScale);
            }
        }
        if (moves(static_cast<u32>(k))) {
            const std::array<f64, 3> at = softdetail::RowTimes({solved[(k * 4 + 3) * 3] - tau[k][0],
                                                                 solved[(k * 4 + 3) * 3 + 1] - tau[k][1],
                                                                 solved[(k * 4 + 3) * 3 + 2] - tau[k][2]},
                                                                back[k]);
            const Vector3f& c = lumps[k].rest;
            lump.t = {at[0] - c.x, at[1] - c.y, at[2] - c.z};
        }
    }

    // A lump's twelve numbers from its nine, and how they move with them.
    const auto posedOf = [&](std::size_t k, const LumpState& lump, Slope* slope) {
        Posed y{};
        const Vector3f& c = lumps[k].rest;
        std::array<std::array<f64, 3>, 3> v{};
        for (int a = 0; a < 3; ++a) {
            v[a] = {lump.s[a] * lump.r[a][0], lump.s[a] * lump.r[a][1], lump.s[a] * lump.r[a][2]};
            const std::array<f64, 3> row = softdetail::RowTimes(v[a], carrier[k]);
            y[a * 3] = row[0];
            y[a * 3 + 1] = row[1];
            y[a * 3 + 2] = row[2];
        }
        const std::array<f64, 3> pivot =
            softdetail::RowTimes({c.x + lump.t[0], c.y + lump.t[1], c.z + lump.t[2]}, carrier[k]);
        for (int i = 0; i < 3; ++i) {
            y[9 + i] = pivot[i] + tau[k][i];
        }
        if (slope == nullptr) {
            return y;
        }
        for (std::array<f64, 9>& row : *slope) {
            row.fill(0.0);
        }
        const bool moving = moves(static_cast<u32>(k));
        const bool scaling = scales(static_cast<u32>(k));
        for (int j = 0; j < 3; ++j) {
            // Its translation moves where the pivot lands, through the carrier.
            for (int i = 0; i < 3 && moving; ++i) {
                (*slope)[9 + i][j] = carrier[k][j][i];
            }
            // A small turn about axis j: each row `v` becomes `v + e_j x v`.
            std::array<f64, 3> axis{0, 0, 0};
            axis[j] = 1.0;
            for (int a = 0; a < 3; ++a) {
                const std::array<f64, 3> turned = softdetail::RowTimes(Cross(axis, v[a]), carrier[k]);
                for (int i = 0; i < 3; ++i) {
                    (*slope)[a * 3 + i][3 + j] = turned[i];
                }
            }
            // Its scale along axis j stretches row j.
            if (scaling) {
                const std::array<f64, 3> stretched =
                    softdetail::RowTimes({lump.r[j][0], lump.r[j][1], lump.r[j][2]}, carrier[k]);
                for (int i = 0; i < 3; ++i) {
                    (*slope)[j * 3 + i][6 + j] = stretched[i];
                }
            }
        }
        return y;
    };
    // The lumps' departure from rest as the linear problem holds it, and the
    // matrix times it.
    std::vector<f64> y(n * 3), ny(n * 3);
    // Half the weighted squared miss, less what does not move with the lumps.
    // The pull is no part of it: here it only damps a step, so a goal the
    // lumps can meet stays met.
    const auto cost = [&](const std::vector<LumpState>& lumpsNow) {
        for (std::size_t k = 0; k < lumpCount; ++k) {
            const Posed one = posedOf(k, lumpsNow[k], nullptr);
            for (int i = 0; i < 12; ++i) {
                y[k * 12 + static_cast<std::size_t>(i)] = one[i] - rest[k * 12 + static_cast<std::size_t>(i)];
            }
        }
        apply(y, ny, false);
        f64 sum = 0.0;
        for (std::size_t i = 0; i < n * 3; ++i) {
            sum += y[i] * (0.5 * ny[i] - wanted[i]);
        }
        return sum;
    };
    const auto moved = [&](const std::vector<LumpState>& from, const std::vector<f64>& delta, f64 scale) {
        std::vector<LumpState> out = from;
        for (std::size_t k = 0; k < lumpCount; ++k) {
            if (collapsed[k] != 0) {
                continue;
            }
            const f64* d = &delta[k * 9];
            LumpState& lump = out[k];
            if (moves(static_cast<u32>(k))) {
                for (int c = 0; c < 3; ++c) {
                    lump.t[c] += scale * d[c];
                }
            }
            lump.r = softdetail::Mul(lump.r, softdetail::TurnOf({scale * d[3], scale * d[4], scale * d[5]}));
            if (scales(static_cast<u32>(k))) {
                for (int c = 0; c < 3; ++c) {
                    lump.s[c] = std::clamp(lump.s[c] + scale * d[6 + c], kLeastScale, kMostScale);
                }
            }
        }
        return out;
    };

    // 2b. A lump's turn and scale are not the matrix the solve gave it, so
    // where it stood no longer suits them. With the turns and scales settled,
    // where every lump stands is a linear problem again.
    {
        for (std::size_t k = 0; k < lumpCount; ++k) {
            const Posed one = posedOf(k, state[k], nullptr);
            for (int i = 0; i < 12; ++i) {
                // Its turn and scale as they are; its pivot where its carrier has it.
                y[k * 12 + static_cast<std::size_t>(i)] = i < 9 ? one[i] - rest[k * 12 + static_cast<std::size_t>(i)] : 0.0;
            }
        }
        apply(y, ny, false);
        std::vector<f64> stand(lumpCount * 3);
        for (std::size_t k = 0; k < lumpCount; ++k) {
            for (int c = 0; c < 3; ++c) {
                stand[k * 3 + static_cast<std::size_t>(c)] = wanted[(k * 4 + 3) * 3 + static_cast<std::size_t>(c)] -
                                                            ny[(k * 4 + 3) * 3 + static_cast<std::size_t>(c)];
            }
        }
        for (int c = 0; c < 3; ++c) {
            softdetail::CholeskySolve(standing_, lumpCount, stand.data() + c, 3);
        }
        for (std::size_t k = 0; k < lumpCount; ++k) {
            if (ridden_[k] == 0 || collapsed[k] != 0 || !moves(static_cast<u32>(k))) {
                continue;
            }
            // Back through the carrier: `(c + t) * a + tau` is where the pivot lands.
            const std::array<f64, 3> landed{rest[k * 12 + 9] + stand[k * 3] - tau[k][0],
                                            rest[k * 12 + 10] + stand[k * 3 + 1] - tau[k][1],
                                            rest[k * 12 + 11] + stand[k * 3 + 2] - tau[k][2]};
            const std::array<f64, 3> at = softdetail::RowTimes(landed, back[k]);
            const Vector3f& c = lumps[k].rest;
            state[k].t = {at[0] - c.x, at[1] - c.y, at[2] - c.z};
        }
    }

    // 3. Gauss-Newton on the nine numbers a lump, all lumps at once.
    const std::size_t m = lumpCount * 9;
    std::vector<Slope> slopes(lumpCount);
    std::vector<f64> gradient(m), delta(m), scratch(n * 3), spread(n * 3);
    f64 before = cost(state);
    const int steps = method == SoftFitMethod::Dense ? kDenseSteps : kFastSteps;
    for (int pass = 0; pass < steps; ++pass) {
        for (std::size_t k = 0; k < lumpCount; ++k) {
            const Posed one = posedOf(k, state[k], &slopes[k]);
            for (int i = 0; i < 12; ++i) {
                y[k * 12 + static_cast<std::size_t>(i)] = one[i] - rest[k * 12 + static_cast<std::size_t>(i)];
            }
        }
        apply(y, ny, false);
        for (std::size_t k = 0; k < lumpCount; ++k) {
            for (int a = 0; a < 9; ++a) {
                f64 g = 0.0;
                for (int i = 0; i < 12; ++i) {
                    const std::size_t at = k * 12 + static_cast<std::size_t>(i);
                    g += slopes[k][i][a] * (ny[at] - wanted[at]);
                }
                gradient[k * 9 + a] = g;
            }
        }
        // The step's matrix times @p v: through the slopes, the problem's
        // matrix, and back.
        const auto hessian = [&](const std::vector<f64>& v, std::vector<f64>& out) {
            for (std::size_t k = 0; k < lumpCount; ++k) {
                for (int i = 0; i < 12; ++i) {
                    f64 s = 0.0;
                    for (int a = 0; a < 9; ++a) {
                        s += slopes[k][i][a] * v[k * 9 + a];
                    }
                    spread[k * 12 + i] = s;
                }
            }
            apply(spread, scratch, true);
            out.assign(m, 0.0);
            for (std::size_t k = 0; k < lumpCount; ++k) {
                for (int a = 0; a < 9; ++a) {
                    f64 s = 0.0;
                    for (int i = 0; i < 12; ++i) {
                        s += slopes[k][i][a] * scratch[k * 12 + i];
                    }
                    out[k * 9 + a] = s;
                }
            }
        };
        // Each lump's own nine by nine: what conjugate gradients are
        // preconditioned by, and the diagonal the dense solve is damped on.
        std::vector<std::vector<f64>> own(lumpCount, std::vector<f64>(81, 0.0));
        f64 meanDiagonal = 0.0;
        for (std::size_t k = 0; k < lumpCount; ++k) {
            const std::array<f64, 16>* self = nullptr;
            for (u32 e = blockStart_[k]; e < blockStart_[k + 1]; ++e) {
                if (blockLump_[e] == k) {
                    self = &blocks_[e];
                }
            }
            for (int a = 0; a < 9; ++a) {
                for (int b = 0; b < 9; ++b) {
                    f64 s = 0.0;
                    for (int i = 0; i < 4; ++i) {
                        for (int c = 0; c < 3; ++c) {
                            f64 row = pull_[k * 4 + i] * slopes[k][i * 3 + c][b];
                            for (int j = 0; j < 4 && self != nullptr; ++j) {
                                row += (*self)[i * 4 + j] * slopes[k][j * 3 + c][b];
                            }
                            s += slopes[k][i * 3 + c][a] * row;
                        }
                    }
                    own[k][a * 9 + b] = s;
                }
                meanDiagonal += own[k][a * 9 + a];
            }
        }
        meanDiagonal = meanDiagonal / static_cast<f64>(m) + 1e-30;
        // A number the lump may not change, or one nothing moves, stands still.
        for (std::size_t k = 0; k < lumpCount; ++k) {
            for (int a = 0; a < 9; ++a) {
                own[k][a * 9 + a] += 1e-9 * meanDiagonal + 1e-30;
            }
        }

        bool solvedStep = false;
        if (method == SoftFitMethod::Dense) {
            std::vector<f64> whole(m * m, 0.0);
            for (std::size_t k = 0; k < lumpCount; ++k) {
                for (u32 e = blockStart_[k]; e < blockStart_[k + 1]; ++e) {
                    const std::size_t l = blockLump_[e];
                    for (int a = 0; a < 9; ++a) {
                        for (int b = 0; b < 9; ++b) {
                            f64 s = 0.0;
                            for (int i = 0; i < 4; ++i) {
                                for (int j = 0; j < 4; ++j) {
                                    const f64 w = blocks_[e][i * 4 + j];
                                    for (int c = 0; c < 3; ++c) {
                                        s += slopes[k][i * 3 + c][a] * w * slopes[l][j * 3 + c][b];
                                    }
                                }
                            }
                            whole[(k * 9 + a) * m + l * 9 + b] += s;
                        }
                    }
                }
                for (int a = 0; a < 9; ++a) {
                    for (int b = 0; b < 9; ++b) {
                        f64 s = 0.0;
                        for (int i = 0; i < 12; ++i) {
                            s += slopes[k][i][a] * pull_[k * 4 + i / 3] * slopes[k][i][b];
                        }
                        whole[(k * 9 + a) * m + k * 9 + b] += s;
                    }
                    whole[(k * 9 + a) * m + k * 9 + a] += 1e-9 * meanDiagonal + 1e-30;
                }
            }
            for (std::size_t i = 0; i < m; ++i) {
                delta[i] = -gradient[i];
            }
            if (softdetail::Cholesky(whole, m)) {
                softdetail::CholeskySolve(whole, m, delta.data());
                solvedStep = true;
            }
        } else {
            std::vector<u8> sound(lumpCount, 0);
            for (std::size_t k = 0; k < lumpCount; ++k) {
                sound[k] = softdetail::Cholesky(own[k], 9) ? 1 : 0;
            }
            const auto precondition = [&](const std::vector<f64>& r, std::vector<f64>& z) {
                z = r;
                for (std::size_t k = 0; k < lumpCount; ++k) {
                    if (sound[k] != 0) {
                        softdetail::CholeskySolve(own[k], 9, &z[k * 9]);
                    }
                }
            };
            std::vector<f64> residual(m), z, direction, hd;
            for (std::size_t i = 0; i < m; ++i) {
                residual[i] = -gradient[i];
                delta[i] = 0.0;
            }
            precondition(residual, z);
            direction = z;
            f64 rz = 0.0, start = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                rz += residual[i] * z[i];
            }
            start = rz;
            for (int it = 0; it < kGradientSteps && rz > 1e-7 * start && rz > 0.0; ++it) {
                hessian(direction, hd);
                f64 dhd = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    dhd += direction[i] * (hd[i] + 1e-9 * meanDiagonal * direction[i]);
                }
                if (!(dhd > 0.0)) {
                    break;
                }
                const f64 alpha = rz / dhd;
                for (std::size_t i = 0; i < m; ++i) {
                    delta[i] += alpha * direction[i];
                    residual[i] -= alpha * (hd[i] + 1e-9 * meanDiagonal * direction[i]);
                }
                precondition(residual, z);
                f64 next = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    next += residual[i] * z[i];
                }
                const f64 beta = next / rz;
                rz = next;
                for (std::size_t i = 0; i < m; ++i) {
                    direction[i] = z[i] + beta * direction[i];
                }
            }
            solvedStep = true;
        }
        if (!solvedStep) {
            break;
        }
        // The step, or as much of it as lowers the cost.
        bool better = false;
        f64 scale = 1.0;
        for (int attempt = 0; attempt < 5; ++attempt, scale *= 0.5) {
            std::vector<LumpState> tried = moved(state, delta, scale);
            const f64 after = cost(tried);
            if (after < before) {
                const bool settled = before - after < kSettled * (std::abs(before) + 1e-30);
                state = std::move(tried);
                before = after;
                better = !settled;
                break;
            }
        }
        if (!better) {
            break;
        }
    }

    for (std::size_t k = 0; k < lumpCount; ++k) {
        if (ridden_[k] != 0) {
            departures[k] = ToTransform(state[k]);
        }
    }
    return measure(skin, goal, departures);
}

} // namespace wem
} // namespace models
} // namespace whiteout
