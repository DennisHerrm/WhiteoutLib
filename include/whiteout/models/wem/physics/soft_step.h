// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_step.h
 * @brief When an effect acts, and one step of a soft body
 *        (EDIT_MODE_SOFTBODY_DESIGN.md §7, §9).
 *
 * **When.** A span is where an effect acts in a clip, with an ease in and an
 * ease out. Spans are keys of the effect's `SoftAmount` channel: each run of
 * keys above 0 is one. A clip that keys nothing takes the effect's
 * *Everywhere*.
 *
 * **A step** is §7's steps that have no memory, at one time of a clip: the
 * pose carries the points; the shape effects move them in the list's order,
 * each by its move times the point's softness; the travelling effects add
 * theirs; the fit finds the lumps' departures; the effects said along the
 * bones scale the lumps on top. It is a function of the pose and the time, so
 * it scrubs.
 */

#include <functional>
#include <span>
#include <utility>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "../document.h"
#include "soft.h"
#include "soft_fields.h"
#include "soft_fit.h"
#include "soft_shapes.h"

namespace whiteout {
namespace models {
namespace wem {

/// Where an effect acts in a clip, seconds.
struct SoftSpan {
    f32 start = 0.0f;
    f32 end = 0.0f;
    f32 easeIn = 0.0f;
    f32 easeOut = 0.0f;

    bool operator==(const SoftSpan&) const = default;
};

/// The id of the channel that holds effect @p effect of soft body @p body's
/// spans, or `kInvalidIndex` when none was made.
u32 SoftAmountChannel(const Model& model, u32 body, u32 effect);

/// The effect's spans in clip @p clip, by their starts.
std::vector<SoftSpan> SoftSpansOf(const Document& document, u32 model, u32 clip, u32 body, u32 effect);

/// Whether an effect of a soft body of @p clip's model acts in it: one that is
/// on, and is on wherever nothing keys it or has a span there. Such a clip
/// has something to bake whether or not its physics is set up.
bool SoftActsIn(const Document& document, u32 clip);

/// Replaces the effect's spans in clip @p clip, making its channel when there
/// is none: four keys a span, fewer where it meets the clip's ends at full.
/// No span takes the clip's keys of it away.
void SetSoftSpans(Document& document, u32 model, u32 clip, u32 body, u32 effect, std::span<const SoftSpan> spans);

/// What an effect reads of a clip at one time (§9).
struct SoftTiming {
    f32 level = 0.0f;    ///< 0 to 1: the ease in, the hold, the ease out; *Everywhere* where nothing is keyed.
    f32 progress = 0.0f; ///< 0 at a span's start to 1 at its end, held after it.
    bool within = false; ///< A span holds the time.
    /// Seconds since each span's start at or before the time, newest first:
    /// what *Times* fires on.
    std::vector<f32> sinceStart;
};

SoftTiming SoftTimingAt(const Document& document, u32 model, u32 clip, const SoftBody& body,
                        const SoftEffect& effect, f32 seconds);

/// @p rate (cycles a second) as a looping clip of @p duration plays it: the
/// whole number of cycles a loop nearest it, and never under one (§9).
f32 SoftWholeCycles(f32 rate, f32 duration, bool looping);

/// *Wind-up*'s moment (§8.7): how far to either side of a time a bone's
/// travel is read, seconds. Twice the *Smoothing*, a quarter of the clip at most.
f32 SoftWindUpWindow(f32 smoothing, f32 duration);

/// What a bone's middle band asks of a body of @p size: the band itself where
/// it is small, and never as far as half the region, so a cut in the
/// animation is not taken for a move.
Vector3f SoftWindUpLean(const Vector3f& band, f32 size);

/// *Pull the surface* (§8.6): how far round a strand's site the skin comes
/// down with it, and how far at the site at full, both in the strand's widths.
constexpr f32 kSoftPullReach = 4.0f;
constexpr f32 kSoftPullDepth = 1.5f;

/// Where a step is: a clip and a time in it. No clip is the bind's time 0.
struct SoftClock {
    u32 clip = kInvalidIndex;
    f32 seconds = 0.0f;
};

/// The pose at another time of the clip being stepped: a pose's skinning
/// matrices by node. What reads the body's own motion asks through it.
using SoftPoseAt = std::function<void(f32 seconds, std::vector<Matrix44f>& skin)>;

struct SoftStepResult {
    std::vector<Transform> departures; ///< One a lump, from its family's bone.
    std::vector<Vector3f> carried;     ///< Where the pose has each point.
    std::vector<Vector3f> goal;        ///< Where the fields ask it to be: what the fit is held to.
    std::vector<Vector3f> posed;       ///< Where the lumps put it.
    SoftFitReport fit;
    bool moved = false; ///< Any effect asked for anything.
    /// What each Blobify that acted read of its blob (§8.5): its effect's id,
    /// and how near two parts come before it joins them (`SoftBlobReport::fuses`).
    std::vector<std::pair<u32, f32>> fuses;
};

/// One soft body, ready to step. A cursor over the document, as `Animator`
/// is: it must not outlive it or an edit of the body, its skin or its clips'
/// spans.
class SoftStepper {
public:
    SoftStepper(const Document& document, u32 model, u32 body);

    bool empty() const {
        return rig_.empty();
    }
    const SoftRig& rig() const {
        return rig_;
    }
    const SoftVolume& volume() const {
        return volume_;
    }

    /// One step at @p clock under @p skin (the pose there). @p poseAt may be
    /// empty: what reads the motion then reads none. @p drivingOnly leaves out
    /// the effects that do not drive the Jiggle (§7): what it gives is what
    /// the body's swing is thrown by.
    void step(std::span<const Matrix44f> skin, const SoftClock& clock, const SoftPoseAt& poseAt,
              SoftStepResult& out, SoftFitMethod method = SoftFitMethod::Fast, bool drivingOnly = false) const;

    /// §7's limit on @p departures, one a lump: a lump's travel inward stops
    /// at half the clear space under it (§5.4).
    void keepClear(std::span<Transform> departures) const;

    /// §7's volume trim on @p departures, one a lump, under @p skin: an even
    /// change of the scales of the lumps that scale, which takes the posed
    /// surface's volume @p share of the way to @p asked (1: all of it), found
    /// in a pass or two. Nothing for a body that encloses no volume.
    void trimVolume(std::span<const Matrix44f> skin, f64 asked, f32 share, std::span<Transform> departures) const;

private:
    /// Each point's way out from its lumps' pivots as @p departures have them:
    /// what an even change of the lumps' scales moves it along, each lump by
    /// its @p share.
    void awayOf(std::span<const Matrix44f> skin, std::span<const Transform> departures, std::span<const f32> share,
                std::vector<Vector3f>& away) const;

    struct PlaceCache {
        u32 effect = 0;
        SoftPlace place;
        SoftPlaceAt at;
        f32 radius = 0.0f;
        std::vector<f32> distance; ///< Over the surface from the place, to 3 radii.
    };
    const PlaceCache& placeOf(const SoftEffect& effect) const;
    /// A Ripple's distances from its place, measured once at the bind (§8.9).
    struct RippleCache {
        u32 effect = 0;
        SoftPlace place;
        SoftThrough through = SoftThrough::Skin;
        bool valid = false;
        std::vector<f32> points; ///< Per point; infinite where the walk did not reach.
        std::vector<f32> lumps;  ///< Per lump, from its points'.
    };
    const RippleCache& rippleOf(const SoftEffect& effect) const;
    /// A Drip's strands as the body's surface knows them (§8.6), found once.
    struct DripCache {
        u32 effect = 0;
        std::vector<Vector3f> at; ///< The sites it was found for, and their widths.
        std::vector<f32> width;
        std::vector<u32> nearest; ///< Per strand, the body's own point nearest its site.
        /// What *Pull the surface* moves: per strand, from `start[s]` to
        /// `start[s + 1]`, a point and how much of the pull it takes.
        std::vector<u32> start;
        std::vector<std::pair<u32, f32>> points;
    };
    const DripCache& dripOf(const SoftEffect& effect) const;
    Vector3f axisOf(SoftAxis axis, u32 bone, std::span<const Matrix44f> skin) const;

    const Document* document_ = nullptr;
    u32 model_ = 0;
    SoftRig rig_;
    SoftVolume volume_;
    Vector3f longest_{0, 0, 1};       ///< The region's longest axis at the bind.
    std::vector<Vector3f> pivot_;     ///< Per node, its pivot at the bind.
    std::vector<u32> child_;          ///< Per node, its first child bone, or none.
    std::vector<Vector3f> reach_;     ///< Per node, from its pivot to the middle of the body's points on it.
    std::vector<Vector3f> lumpOut_;   ///< Per lump, outward over its points; zero where they face every way.
    std::vector<f32> lumpRadius_;     ///< Per lump, its size: what a swell of its scale is measured against.
    std::vector<u32> lumpOfNode_;     ///< Per node, the lump that is it or copies it, or none.
    mutable std::vector<PlaceCache> places_;
    mutable std::vector<RippleCache> ripples_;
    mutable std::vector<DripCache> drips_;
    mutable SoftColumns columns_; ///< A melt's, found when one first asks.
    mutable bool columned_ = false;
};

} // namespace wem
} // namespace models
} // namespace whiteout
