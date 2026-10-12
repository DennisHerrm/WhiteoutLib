// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_drip.h
 * @brief Drip: strands that grow from where the surface would shed a liquid
 *        (EDIT_MODE_SOFTBODY_DESIGN.md §8.6).
 *
 * **Sites, by where a liquid would run.** Every point of the body is walked
 * down its surface until it reaches a place it cannot leave: a low point, or a
 * face past the *Overhang*. What arrives at a place is its catchment
 * [O'Callaghan 1984]: the largest drip first, and thickest. A place with no
 * room under it for a strand is passed over while another has some: the
 * ground is the model's, at a height of nothing.
 *
 * **A strand is geometry the effect adds:** a six-sided tube of four rings
 * with a rounded tip, all the strands in one mesh. **Its three lumps are three
 * bones** (base, neck, tip), side by side under the bone that carries its
 * site, added with it. They are points of the body like any other, so its
 * lumps, its lanes, its bake and its export take them with nothing of their
 * own; and each is used as it is, so a file has three bones a strand.
 *
 * **A strand over time** is a function of how long ago it set out, with no
 * memory: its tip falls at *Speed* to its reach, its neck thins by the
 * material's law, and before it sets out it is a hundredth of itself, sunk
 * under the surface.
 *
 * ***Sway*** is a function of the time too: a pendulum of the strand's
 * length, in whole swings a loop. (The design has it a hanging chain in
 * Jiggle's model, which the body's motion would drive. Here the way down is
 * the world's, so a strand stays plumb as its carrier tilts, and the swing is
 * its own.)
 *
 * ***Pinch off*** is a cycle, a function of the time as well: the strand
 * grows, its neck thins, its swollen tip is let go, and it draws back to a
 * stub to grow again. **The droplet is one more bone and a small ball,**
 * built with the strand: it falls from where the tip was let go, lies flat on
 * the floor and shrinks away before the next. In a loop a cycle is a whole
 * share of the loop, each strand from a phase of its own.
 */

#include <span>
#include <string>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "../model.h"
#include "../nodes/node.h"
#include "soft.h"
#include "soft_fit.h"

namespace whiteout {
namespace models {
namespace wem {

/// Where a strand hangs from.
struct SoftDripSite {
    u32 point = kInvalidIndex; ///< Of the rig, at the bind.
    Vector3f at{0, 0, 0};
    Vector3f normal{0, 0, -1};
    u32 bone = kInvalidNode; ///< The bone that carries most of it.
    /// The share of the body's skin that runs to it. A site placed by hand has none.
    f32 catchment = 0.0f;
    f32 width = 0.0f; ///< Its strand's radius, model units.
    /// The surface's first UV under it: a strand has the colour it runs from.
    Vector2f uv{0, 0};
};

/// @p rig's sites under @p settings (§8.6), the largest catchment first: by
/// where a liquid would run, kept *Spacing* apart, *Sites* of them; or the
/// ones placed by hand.
std::vector<SoftDripSite> SoftDripSites(const Model& model, const SoftRig& rig, const SoftDrip& settings);

/// How wide a strand's neck is, of its own width, when it has fallen the
/// share @p fallen of its reach: from 1 down to @p narrowest, by the
/// material's law.
f32 SoftNeckWidth(SoftNeckLaw law, f32 narrowest, f32 fallen);

/// How much of its swell a strand's tip has, 0 to 1 (§8.6): **the tip takes
/// what the neck gives up, in step.** It is the share of the volume the
/// thinning will have taken out of the strand at its reach that it has taken
/// out now, with the strand @p length of the way there and its neck @p neck
/// wide. So a neck that snaps swells its tip as it snaps, and one that thins
/// early swells it early. The volume is kept to the last where *Tip* is the
/// size that pays the neck back; no size is solved for. A neck that does not
/// thin swells its tip by the strand's length.
f32 SoftTipSwell(const SoftDrip& settings, f32 length, f32 neck);

/// @p preset's settings: its neck's law, its speed, its thickness and its
/// stagger over @p from, the rest of which is kept.
SoftDrip SoftDripPresetOf(SoftDripPreset preset, const SoftDrip& from = {});

/// How long a strand's tube is, in its widths, before it is stretched.
constexpr f32 kSoftStrandRest = 4.0f;
/// What a strand is built at, of itself: a hundredth, under the surface, so a
/// clip that does not key it draws nothing. (The design says a thousandth:
/// at that, a thin strand's ring is under the tolerance points are welded at.)
constexpr f32 kSoftStrandShrink = 0.01f;

/// Where in its cycle a strand that pinches off lets its tip go.
constexpr f32 kSoftPinchAt = 0.6f;
/// What it draws back to afterwards, of its length.
constexpr f32 kSoftPinchStub = 0.25f;
/// How much of the way to the ground a strand hangs before it pinches off:
/// the rest is its droplet's fall.
constexpr f32 kSoftPinchReach = 0.6f;
/// How flat a droplet lies on the floor, of its own height.
constexpr f32 kSoftDropFlat = 0.35f;
/// How fast a droplet falls, in the body's heights a second squared: the
/// Jiggle's own gravity.
constexpr f32 kSoftDropGravity = 5.0f;

/// A strand that pinches off (§8.6), at one moment of its cycle.
struct SoftPinch {
    f32 length = 1.0f;  ///< Of its reach: it grows to 1, lets go, and draws back to its stub.
    f32 thinned = 0.0f; ///< How far its neck's law has run, 0 to 1.
    f32 swollen = 0.0f; ///< How much of its swollen tip it has, 0 to 1: none once it let go.
    bool away = false;  ///< Its droplet is away,
    f32 since = 0.0f;   ///< ... for this many seconds,
    f32 fade = 1.0f;    ///< ... and this much of itself: it shrinks away before the next.
};

/**
 * @brief Where a strand is in its pinch-off cycle.
 *
 * @p turn counts cycles: its whole part is which, its fraction how far
 * through. @p period is a cycle's seconds. @p grows is the share of a cycle
 * the strand takes to reach its length, `kSoftPinchAt` at most: after that it
 * hangs and thins until it lets go. The first cycle of a clip played once
 * grows from nothing (@p first); every other from the stub.
 */
SoftPinch SoftPinchAt(f32 turn, f32 period, f32 grows, bool first);

/// The share of a cycle strand @p strand of an effect seeded @p seed starts
/// its own at, in a loop: no two let go together, and the same in every clip.
f32 SoftPinchPhase(u32 seed, u32 strand);

/// One strand's three bones at a time: each a transform about its own pivot
/// along the model's axes, as a lump's departure is. At rest it is as built.
struct SoftStrandPose {
    Transform base;
    Transform neck;
    Transform tip;
    f32 length = 0.0f; ///< From its site to its tip's pivot.
};

/**
 * @brief A strand of radius @p width that set out @p since seconds ago
 *        (negative: not yet), falling toward a tip @p reach below its site.
 *
 * @p height is the body's, which *Speed* is a share of. @p hangs is a looping
 * clip's strand: it is at its reach throughout. @p down is the way it falls
 * and @p sunk the way from its base bone out to its site, both in the frame
 * its bones' departures are said in.
 *
 * @p swing is *Sway*: how far its tip has swung off the way down, radians
 * about two axes square to it. The strand bends as a hanging chain of two
 * links, the upper half as far as the lower, and is no longer or shorter.
 *
 * @p pinch, when given, is where it is in its pinch-off cycle: its length,
 * its neck and its tip are the cycle's, and @p reach is where it lets go.
 */
SoftStrandPose SoftStrandAt(const SoftDrip& settings, f32 width, f32 reach, f32 height, f32 since, bool hangs,
                            const Vector3f& down, const Vector3f& sunk, const Vector2f& swing = Vector2f{0, 0},
                            const SoftPinch* pinch = nullptr);

/// Where the tip of a strand @p length long hangs from its site, swung by
/// @p swing as `SoftStrandAt` swings it: in the frame @p down is said in.
Vector3f SoftStrandTip(const Vector3f& down, f32 length, const Vector2f& swing);

/**
 * @brief A droplet's bone (§8.6): a transform about its pivot, as a strand's.
 *
 * At rest, which is as built (a hundredth, under the surface), until
 * @p pinch has it away. Then it falls from @p from, which is where the tip
 * was let go, from its bone's pivot: freely, by `kSoftDropGravity` of the
 * body's @p height, down @p down, which here is the world's way down said in
 * its bone's frame at its true length, so a fall is a fall in the model.
 * @p room is how far under @p from the floor is: there it lies flat, the
 * volume kept, and shrinks away by the cycle's end. @p width is its strand's.
 */
Transform SoftDropletAt(const SoftDrip& settings, f32 width, f32 room, f32 height, const SoftPinch& pinch,
                        const Vector3f& down, const Vector3f& from);

/// How fast a strand @p reach long swings on a body @p height high, swings a
/// second: as a pendulum's rate goes, by the root of its length. A strand as
/// long as the body takes four thirds of a second.
f32 SoftSwayRate(f32 reach, f32 height);

/// *Sway* (§8.6): where strand @p strand of an effect seeded @p seed has
/// swung to at @p seconds, as `SoftStrandAt` takes it, at @p rate swings a
/// second. Each strand swings from a phase of its own, the same in every
/// clip. @p sway is the setting, 0 to 1: at 1 it swings twenty degrees.
Vector2f SoftStrandSwing(f32 sway, u32 seed, u32 strand, f32 rate, f32 seconds);

/// What `BuildSoftStrands` made.
struct SoftStrandsBuilt {
    u32 strands = 0;
    u32 bones = 0;
    u32 mesh = kInvalidIndex;
};

/**
 * @brief @p effect's strands added to @p model: three bones a site, and one
 *        mesh for them all, its points the body's (§8.6).
 *
 * The mesh is named for the body, takes the surface's material at the first
 * site (or the effect's own), and joins the body's meshes; `SoftBody::drips`
 * names it. A strand's corners all hold its site's UV, and its points are
 * marked painted, so a remake's blend leaves them wholly soft. The effect's
 * `strandNodes`, `strandAt` and `strandWidth` are
 * filled. The caller makes the body's lumps again afterwards. Nothing is
 * made for no site, or for an effect that already has its strands.
 *
 * An effect that pinches off (`SoftDrip::pinches`) gets a droplet a strand as
 * well: one more bone under the site's, and a small ball in the same mesh,
 * built as the strand is. `dropNodes` names them.
 */
SoftStrandsBuilt BuildSoftStrands(Model& model, SoftBody& body, SoftEffect& effect,
                                  std::span<const SoftDripSite> sites);

} // namespace wem
} // namespace models
} // namespace whiteout
