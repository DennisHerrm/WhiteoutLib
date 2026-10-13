// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file prune.h
 * @brief What a model need not hold: the records that do nothing in the game.
 *
 * `makeLoadable` (`loadable.h`) repairs what Warcraft III's loader requires.
 * This removes what it merely tolerates, each by what the 3.0 client reads:
 *
 * - **A geoset animation that does nothing.** The game multiplies a record's
 *   static colour and alpha into its geoset (`BuildAllGeosets`), plays its two
 *   tracks, and reads one bit of its flags word, `DropShadow`, through a bone
 *   that links it (`IAnimCreateObjects`). A record that is white, opaque,
 *   unkeyed, casts no shadow and is linked by no bone states nothing.
 * - **A bone's link to a record that is always visible.** The link makes the
 *   bone follow the record's alpha, and one that never leaves 1 hides nothing.
 * - **A geoset with no faces**, which draws nothing.
 * - **A material identical to another**, and one nothing draws with.
 * - **A global sequence no track runs on.**
 *
 * An export does this (`MdxConverter::exportToBytes`, and the host's own
 * conversion). Re-saving a model as it was read does not: that is the same
 * model in another container, and what its author left in it is theirs.
 */

#include <vector>

#include "structures.h"
#include "types.h"

namespace whiteout {
namespace mdx {

/// What `pruneModel` removed.
struct PruneReport {
    u32 boneLinks = 0;        ///< Links to an always-visible record, cleared.
    u32 geosetAnimations = 0; ///< Records that did nothing.
    u32 geosets = 0;          ///< Geosets with no faces.
    u32 materialsMerged = 0;  ///< Materials identical to an earlier one.
    u32 materialsUnused = 0;  ///< Materials nothing draws with.
    u32 globalSequences = 0;  ///< Global sequences no track runs on.
    /// Records a bone links that sit past the byte the game keeps the link in
    /// (255 means none), after they were moved to the front: more than 255
    /// such records. Not a removal -- a limit the model is over.
    u32 linksPastByte = 0;

    bool empty() const {
        return boneLinks + geosetAnimations + geosets + materialsMerged + materialsUnused +
                   globalSequences + linksPastByte ==
               0;
    }
};

/// Everything below, in the order that lets each see what the last left.
PruneReport pruneModel(Model& model);

/// Clears each bone's link to a geoset animation that is always visible, then
/// removes the records that do nothing, and puts the records bones still link
/// first when one would otherwise fall past the byte. @p report takes the
/// counts.
void pruneGeosetAnimations(Model& model, PruneReport& report);

/// Removes the geosets that have no faces, with their geoset animations, and
/// moves the later geosets' numbers down. One stays while a bone is linked to
/// its record: the record needs a geoset to name. And all of them stay in a
/// model whose geosets are on more than one level of detail, where the game
/// matches the levels by how many geosets each holds. Returns how many went.
u32 pruneEmptyGeosets(Model& model);

/// Points every geoset and ribbon emitter at the first material identical to
/// its own. Two are identical when the writer produces the same bytes for
/// them, so nothing the file states can differ. Returns how many materials
/// that left with nothing drawing with them; `pruneUnusedMaterials` removes
/// those.
u32 mergeIdenticalMaterials(Model& model);

/// Removes the materials no geoset and no ribbon emitter names, and renumbers
/// the ones that do. Returns how many went.
u32 pruneUnusedMaterials(Model& model);

/// Removes the global sequences no track and no event object runs on, and
/// renumbers the ids of those that do. Returns how many went.
u32 pruneUnusedGlobalSequences(Model& model);

/// Removes the geoset animations @p remove marks, one flag a record, and moves
/// every bone's link with the records that stay. A bone linked to a removed
/// record is linked to none.
void removeGeosetAnimations(Model& model, const std::vector<u8>& remove);

} // namespace mdx
} // namespace whiteout
