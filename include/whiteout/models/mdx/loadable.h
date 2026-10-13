// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file loadable.h
 * @brief What Warcraft III's loader requires of a model, in one place.
 *
 * The game's reader is strict where the format looks loose. A chunk reader
 * that fails fails the whole model (`BinToModelData`), and several of the
 * things it fails on are things a model built in memory can easily hold: a
 * track with no keys, an event object that fires at no time, object ids that
 * are not the order the nodes are read in. Others it does not check at all and
 * follows out of bounds: a material, a texture, a geoset animation.
 *
 * `makeLoadable` holds every such rule, each read off the 3.0 client, and
 * repairs a model against them. It is the last step before bytes on every
 * route: `Writer` runs it unless told not to, so a route that builds a model
 * by hand, converts one, or re-saves one it read cannot skip it.
 *
 * It repairs what the loader REQUIRES and nothing else. What a model merely
 * need not hold -- a geoset animation that does nothing, a material nothing
 * draws with -- is the export's to remove (`prune.h`), and is left alone
 * here, so that a model that is already fine comes back untouched.
 */

#include <string>
#include <vector>

#include "structures.h"
#include "types.h"

namespace whiteout {
namespace mdx {

/// A rule of the loader's, and so one kind of thing `makeLoadable` finds.
enum class LoadRule : u32 {
    /// A track whose key count and key arrays disagree. Cut to the whole keys
    /// it holds.
    TrackKeyCount,
    /// An interpolation type past Bezier. The game reads tangents for a type
    /// above Linear (`IReadBinGeosetAnim` sizes a key by `type <= 1`) and this
    /// library writes none for one it has no name for, so the track's bytes
    /// would be out of step. Made Linear.
    TrackInterpolation,
    /// An integer track keyed with tangents. The game reads a time and a value
    /// a key whatever the header says (`ReadBinUintKeyFrames`). Made stepped.
    IntegerTrackTangents,
    /// A key holding a NaN or an infinity. Dropped.
    NonFiniteKey,
    /// A used track with no keys, which fails the model (`ReadBinGenObject`:
    /// every key reader answers 0 for a count of 0). Marked unused.
    EmptyTrack,
    /// A track or event object on a global sequence the model does not have.
    /// Put on the sequences' clock.
    GlobalSequenceRange,
    /// An event object that fires at no time: `KEVT` with a count of 0 fails
    /// the model, and without `KEVT` the next object's size is read for a tag.
    /// Removed, or made a helper when a node hangs from it.
    KeylessEvent,
    /// Object ids that are not the order the nodes are read in. The game
    /// ignores the stored id and numbers a node as it reads it, so every
    /// parent, pivot and skin entry then names another node. Renumbered.
    ObjectOrder,
    /// A parent that is no node of the model. Made a root.
    ParentRange,
    /// A node that is its own ancestor. The cycle is cut where it closes.
    ParentCycle,
    /// Fewer pivot points than nodes. Padded with the origin.
    PivotCount,
    /// A geoset or ribbon emitter naming a material the model does not have,
    /// which the game indexes unchecked. Pointed at a blank material.
    MaterialRange,
    /// A layer or particle emitter naming a texture the model does not have,
    /// which the game indexes unchecked. Pointed at a blank texture.
    TextureRange,
    /// A layer naming a texture animation the model does not have. Unlinked.
    TextureAnimationRange,
    /// A layer reading a UV set a geoset it draws does not hold. Reads the
    /// first.
    UvSetRange,
    /// A geoset animation naming a geoset the model does not have: the game
    /// grows a table to the id (`AnimAddGeoset`). Removed.
    GeosetAnimationTarget,
    /// A bone naming a geoset animation the model does not have, which the
    /// game indexes unchecked (`IAnimCreateObjects`). Unlinked.
    GeosetAnimationRange,
    /// Attachment ids that do not climb through the chunk. The game sizes its
    /// id table by the last record's id and fills it by every record's
    /// (`ProcessAttachments`). Numbered by place.
    AttachmentIds,
    /// A negative or non-finite emission rate or lifespan, static or keyed:
    /// the game sizes the particle pool by their product, cast to unsigned
    /// (`CParticleEmitter2::SyncAllocation`). Made 0.
    EmitterPool,
    /// A collision box whose corners are not its minimum and its maximum,
    /// which the game's ray test takes them for. Ordered.
    BoxCorners,
    /// More than 255 sequences: the game counts them in a byte. Reported, not
    /// repaired -- which ones go is the author's to say.
    SequenceCount,
    /// A geoset of more than 65,535 vertices: the game builds none
    /// (`CreateGeoset`) and reads through the null one. Reported, not
    /// repaired -- `toMdx` cuts such a section before it gets here.
    GeosetVertexCount,
};

/// One rule that fired, over every place it did.
struct LoadFinding {
    LoadRule rule = LoadRule::TrackKeyCount;
    u32 count = 0;        ///< How many tracks, keys or records.
    bool repaired = true; ///< False for a rule that is reported and left.
    std::string first;    ///< The first place, by a name a reader can find.
};

/// What `makeLoadable` found. Empty for a model the game loads as it is.
struct LoadReport {
    std::vector<LoadFinding> findings; ///< One a rule, in the order found.

    bool empty() const {
        return findings.empty();
    }
    /// Whether the model was changed.
    bool repaired() const;
    /// The finding for @p rule, or null.
    const LoadFinding* find(LoadRule rule) const;
};

/// @p finding as one line of plain words, for a log.
std::string describe(const LoadFinding& finding);

/**
 * @brief Makes @p model one Warcraft III loads, and says what that took.
 *
 * Every repair keeps as much of what the model says as the loader allows: a
 * track with no keys is one that animates nothing, so it is marked unused; a
 * geoset naming a material that does not exist is still drawn, with a blank
 * one. A model with nothing to repair is not written to.
 */
LoadReport makeLoadable(Model& model);

} // namespace mdx
} // namespace whiteout
