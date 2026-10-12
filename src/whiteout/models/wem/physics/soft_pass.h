// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_pass.h
 * @brief The soft bodies' part of a clip's bake (EDIT_MODE_SOFTBODY_DESIGN.md
 *        §11): private to `BakeClip`, which writes what this finds.
 */

#include <vector>

#include <whiteout/models/wem/physics/bake.h>

namespace whiteout {
namespace models {
namespace wem {
namespace softdetail {

/// A bone used as it is, as the pass leaves it. Its keys are the bone's own,
/// so `BakeClip` splices them in place, as it does a body's node. And a bone
/// below one, keyed so that it stands where the bodies were stepped on it.
struct SoftBoneKeys {
    u32 body = 0; ///< The soft body it is a lump of, or its parent is.
    u32 node = kInvalidNode;
    /// Per step, the bone's local with its lump's departure on it, and without
    /// what the bone above it departs by; and what the clip plays there.
    std::vector<Transform> samples;
    std::vector<Transform> plays;
    std::vector<u8> departs; ///< Per step, whether the two differ.
    bool moves = false;
    bool turns = false;
    bool scales = false;
    f32 reach = 0.0f; ///< How far its points lie from its pivot: what a turn or a scale moves.
};

struct SoftPass {
    /// The copies' and the surface lumps' tracks, each on its lump's channel,
    /// thinned, with a key at rest at each edge of where the lump departs.
    std::vector<SubTrack> tracks;
    /// The channels it added to the model's table.
    std::vector<u32> declared;
    std::vector<SoftBoneKeys> bones;
    std::vector<BakeReport::SoftLine> lines;
};

/**
 * @brief Every soft body of @p clip's model stepped over the clip as its keys
 *        now play it, @p steps steps of @p dt, and its lumps' departures made
 *        tracks within @p tolerance (model units; 0 keeps a key every
 *        @p stride-th step, the clip's key rate).
 *
 * Declares the lump channels it keys. Writes no track and no key: the caller
 * holds the clip's bake record.
 */
SoftPass RunSoftPass(Document& document, u32 clip, u32 steps, f32 dt, f32 tolerance, u32 stride,
                     SoftModesCache* modes = nullptr);

/// @p local, a node's own transform in a pose, with a lump's @p departure on
/// it: applied before the node's, about its pivot.
Transform WithDeparture(const NodeTree& tree, u32 node, const Transform& local, const Transform& departure);

} // namespace softdetail
} // namespace wem
} // namespace models
} // namespace whiteout
