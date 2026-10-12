// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_bake.h
 * @brief A soft body as a file holds it: bones for its lumps
 *        (EDIT_MODE_SOFTBODY_DESIGN.md §12).
 */

#include <vector>

#include <whiteout/common_types.h>

#include "../diagnostics.h"
#include "../document.h"
#include "../profile.h"
#include "soft_lumps.h"

namespace whiteout {
namespace models {
namespace wem {

/// How a file for @p target holds a point's weights: a classic Warcraft III
/// group's equal shares, bytes that sum to 255, or floats where the target is
/// no game's.
SoftWeights SoftWeightsFor(ProfileId target);

/// What `ExpandSoftToBones` did.
struct SoftExpansion {
    u32 bones = 0;    ///< Bones made: one a copy and one a surface lump.
    u32 points = 0;   ///< Points whose skin names a lump's bone.
    u32 rebound = 0;  ///< Soft bodies whose lanes were solved again for the target's weights a point.
    u32 reduced = 0;  ///< Points moved onto their bones' own lumps to keep a geoset in 256 groups.
    u32 keys = 0;     ///< Keys the made bones took from the lumps' baked tracks.
    /// Per model, per soft body in the set's order, per lump: the bone that
    /// is the lump, made or its own.
    std::vector<std::vector<std::vector<u32>>> boneOf;
};

/**
 * @brief Each soft body's lumps made bones, on the export's copy (§12).
 *
 * - **A bone per copy and per surface lump,** a child of its family's bone at
 *   the lump's pivot. A copy is named for its bone, `<bone>_soft`; a surface
 *   lump `<Name>_01` and on. A bone used as it is needs none.
 * - **The skin** by `SoftSkinner`, for the target's weights. A body solved for
 *   more weights a point than the target holds, or for other shares, is bound
 *   again first.
 * - **Keys:** a baked clip's lump tracks become the made bones' translation,
 *   rotation and scale keys, at the same times. A clip not baked gets none and
 *   plays as it did: a lump at rest stands where its family's bone has it.
 *
 * The records, layers and channels stay for `DropSoftForProfile`.
 */
SoftExpansion ExpandSoftToBones(Document& document, ProfileId target, Diagnostics& out);

} // namespace wem
} // namespace models
} // namespace whiteout
