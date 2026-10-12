// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_step.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include <whiteout/models/wem/anim/track_read.h>
#include <whiteout/models/wem/physics/soft_contact.h>
#include <whiteout/models/wem/physics/soft_drip.h>
#include <whiteout/models/wem/physics/soft_shapes.h>

#include "soft_common.h"

namespace whiteout {
namespace models {
namespace wem {

namespace {

using softdetail::Collapsed;
using softdetail::kTravelReach;
using softdetail::OnLine;

constexpr f32 kTau = 6.2831853f;
/// The least a key of a span stands from the next, seconds: an ease of none.
constexpr f32 kSpanStep = 0.001f;
// What Squash & stretch reads as "at speed", "a stop" and "a swing", in
// region sizes a second, sizes a second squared, and radians a second.
constexpr f32 kSpeed = 3.0f;
constexpr f32 kStop = 40.0f;
constexpr f32 kSwing = 8.0f;
/// The most a Wobble on *Jelly* stretches the body, as a logarithm: to four
/// times its height.
constexpr f32 kJellyMost = 1.3862944f;
/// The most a Ripple said along the bones takes from a bone's lump's scale in
/// a trough, and adds to it on a crest.
constexpr f32 kRippleTrough = 0.5f;
constexpr f32 kRippleCrest = 1.0f;

const SubTrack* AmountTrack(const Document& document, u32 model, u32 clip, u32 body, u32 effect) {
    if (model >= document.models.size() || clip >= document.clips.size()) {
        return nullptr;
    }
    const u32 channel = SoftAmountChannel(document.models[model], body, effect);
    if (channel == kInvalidIndex) {
        return nullptr;
    }
    for (const SubTrackContainer& container : document.clips[clip].containers) {
        for (const SubTrack& track : container.subTracks) {
            if (track.channel == channel && !track.times.empty() &&
                track.values.size() >= track.times.size() * sizeof(f32)) {
                return &track;
            }
        }
    }
    return nullptr;
}

f32 KeyOf(const SubTrack& track, std::size_t k) {
    f32 value = 0.0f;
    std::memcpy(&value, track.values.data() + k * sizeof(f32), sizeof value);
    return value;
}

/// The track's value at @p seconds: linear between its keys, held past its ends.
f32 LevelOf(const SubTrack& track, f32 seconds) {
    const std::size_t count = track.times.size();
    if (seconds <= track.times.front()) {
        return KeyOf(track, 0);
    }
    for (std::size_t k = 1; k < count; ++k) {
        if (seconds <= track.times[k]) {
            const f32 span = track.times[k] - track.times[k - 1];
            const f32 t = span > 0.0f ? (seconds - track.times[k - 1]) / span : 1.0f;
            return KeyOf(track, k - 1) + (KeyOf(track, k) - KeyOf(track, k - 1)) * t;
        }
    }
    return KeyOf(track, count - 1);
}

Vector3f Unit(const Vector3f& v, const Vector3f& otherwise) {
    const f32 length = v.length();
    return length > 1e-12f ? v * (1.0f / length) : otherwise;
}

/// A smooth bump that ends at 1 (§8.2's Round), or a rim round a dent whose
/// integral over a plane is nothing (Ring), at @p u radii from the place.
f32 Profile(SoftBulgeShape shape, f32 u) {
    if (shape == SoftBulgeShape::Ring) {
        if (u >= 2.5f) {
            return 0.0f;
        }
        const f32 x2 = (u / 0.5f) * (u / 0.5f);
        return (1.0f - 0.5f * x2) * std::exp(-0.5f * x2);
    }
    if (u >= 1.0f) {
        return 0.0f;
    }
    const f32 inside = 1.0f - u * u;
    return inside * inside;
}

/// Inflate's rhythm on its Amount: 1 at the top of a beat, `1 - depth` between.
f32 PulseOf(const SoftInflate& inflate, f32 seconds, f32 duration, bool looping) {
    if (inflate.pulse == SoftPulse::None) {
        return 1.0f;
    }
    const f32 rate = SoftWholeCycles(inflate.rate, duration, looping);
    f32 phase = rate * seconds;
    phase -= std::floor(phase);
    const auto bump = [](f32 x) {
        x -= std::floor(x);
        return x < 0.18f ? 0.5f * (1.0f - std::cos(kTau * x / 0.18f)) : 0.0f;
    };
    // A heartbeat is two bumps a cycle, a third of a cycle apart.
    const f32 shape = inflate.pulse == SoftPulse::Heartbeat ? std::max(bump(phase), 0.6f * bump(phase - 1.0f / 3.0f))
                                                            : 0.5f * (1.0f - std::cos(kTau * phase));
    return 1.0f - std::clamp(inflate.depth, 0.0f, 1.0f) * (1.0f - shape);
}

/// @p x stretched by @p k along unit @p axis about @p anchor, and thinned
/// across so that the volume is kept.
Vector3f Stretched(const Vector3f& x, const Vector3f& anchor, const Vector3f& axis, f32 k) {
    const Vector3f rel = x - anchor;
    const f32 along = rel.dot(axis);
    const Vector3f across = rel - axis * along;
    return anchor + axis * (along * k) + across * (1.0f / std::sqrt(std::max(k, 1e-4f)));
}

} // namespace

u32 SoftAmountChannel(const Model& model, u32 body, u32 effect) {
    const u32 sub = SoftSub(body, effect);
    for (const AnimChannel& channel : model.animChannels.channels) {
        if (channel.target.kind == TrackTarget::Kind::Physics && channel.target.channel == Channel::SoftAmount &&
            channel.target.sub == sub) {
            return channel.id;
        }
    }
    return kInvalidIndex;
}

std::vector<SoftSpan> SoftSpansOf(const Document& document, u32 model, u32 clip, u32 body, u32 effect) {
    std::vector<SoftSpan> out;
    const SubTrack* track = AmountTrack(document, model, clip, body, effect);
    if (track == nullptr) {
        return out;
    }
    const std::size_t count = track->times.size();
    std::size_t k = 0;
    while (k < count) {
        if (!(KeyOf(*track, k) > 0.0f)) {
            ++k;
            continue;
        }
        // A run of keys above 0: from the 0 before it to the 0 after.
        std::size_t last = k;
        f32 most = 0.0f;
        while (last + 1 < count && KeyOf(*track, last + 1) > 0.0f) {
            ++last;
        }
        for (std::size_t i = k; i <= last; ++i) {
            most = std::max(most, KeyOf(*track, i));
        }
        std::size_t firstFull = k, lastFull = last;
        while (firstFull < last && KeyOf(*track, firstFull) < most - 1e-6f) {
            ++firstFull;
        }
        while (lastFull > firstFull && KeyOf(*track, lastFull) < most - 1e-6f) {
            --lastFull;
        }
        SoftSpan span;
        span.start = k > 0 ? track->times[k - 1] : track->times[k];
        span.end = last + 1 < count ? track->times[last + 1] : track->times[last];
        span.easeIn = track->times[firstFull] - span.start;
        span.easeOut = span.end - track->times[lastFull];
        // An ease of one step is none.
        span.easeIn = span.easeIn <= kSpanStep * 1.5f ? 0.0f : span.easeIn;
        span.easeOut = span.easeOut <= kSpanStep * 1.5f ? 0.0f : span.easeOut;
        out.push_back(span);
        k = last + 1;
    }
    return out;
}

bool SoftActsIn(const Document& document, u32 clip) {
    if (clip >= document.clips.size() || document.clips[clip].model >= document.models.size()) {
        return false;
    }
    const u32 model = document.clips[clip].model;
    for (const SoftBody& body : document.models[model].physics.soft) {
        for (const SoftEffect& effect : body.effects) {
            if (effect.enabled && (effect.everywhere || !SoftSpansOf(document, model, clip, body.id, effect.id).empty())) {
                return true;
            }
        }
    }
    return false;
}

void SetSoftSpans(Document& document, u32 model, u32 clip, u32 body, u32 effect, std::span<const SoftSpan> spans) {
    if (model >= document.models.size() || clip >= document.clips.size()) {
        return;
    }
    Model& owner = document.models[model];
    Clip& target = document.clips[clip];
    u32 channel = SoftAmountChannel(owner, body, effect);
    const auto holds = [&](const SubTrack& track) { return track.channel == channel; };
    if (channel != kInvalidIndex) {
        for (SubTrackContainer& container : target.containers) {
            std::erase_if(container.subTracks, holds);
        }
    }
    std::vector<SoftSpan> sorted(spans.begin(), spans.end());
    std::erase_if(sorted, [](const SoftSpan& span) { return !(span.end > span.start); });
    std::sort(sorted.begin(), sorted.end(), [](const SoftSpan& a, const SoftSpan& b) { return a.start < b.start; });
    if (sorted.empty()) {
        return;
    }
    if (channel == kInvalidIndex) {
        AnimChannel made;
        made.id = owner.animChannels.nextFreeId();
        made.target.kind = TrackTarget::Kind::Physics;
        made.target.channel = Channel::SoftAmount;
        made.target.sub = SoftSub(body, effect);
        made.valueType = geom::AttrType::F32;
        owner.animChannels.add(made);
        channel = made.id;
    }
    SubTrack track;
    track.channel = channel;
    track.interp = Interpolation::Linear;
    const f32 duration = target.duration;
    const auto key = [&](f32 time, f32 value) {
        // Times only go forward: a span that starts where the last ended shares its 0.
        if (!track.times.empty() && time <= track.times.back()) {
            if (time == track.times.back() && value == 0.0f && KeyOf(track, track.times.size() - 1) == 0.0f) {
                return;
            }
            time = track.times.back() + kSpanStep;
        }
        track.times.push_back(time);
        const u8* at = reinterpret_cast<const u8*>(&value);
        track.values.insert(track.values.end(), at, at + sizeof value);
    };
    for (const SoftSpan& asked : sorted) {
        const f32 start = std::max(asked.start, 0.0f);
        const f32 end = duration > 0.0f ? std::min(asked.end, duration) : asked.end;
        if (!(end > start)) {
            continue;
        }
        const f32 room = end - start;
        const f32 easeIn = std::clamp(asked.easeIn, 0.0f, room);
        const f32 easeOut = std::clamp(asked.easeOut, 0.0f, room - easeIn);
        // At full from the clip's first frame, or to its last: no 0 to rise from.
        if (start <= 0.0f && easeIn <= 0.0f) {
            key(0.0f, 1.0f);
        } else {
            key(start, 0.0f);
            key(start + std::max(easeIn, kSpanStep), 1.0f);
        }
        if (duration > 0.0f && end >= duration && easeOut <= 0.0f) {
            key(duration, 1.0f);
        } else {
            key(std::max(end - std::max(easeOut, kSpanStep), track.times.back() + kSpanStep), 1.0f);
            key(end, 0.0f);
        }
    }
    // Both of the clip's edges are keyed. A track that starts or ends inside
    // its clip is read there by its reader's own rule, and the rules disagree:
    // the step holds its first and last keys, and a detach (a clip's first
    // bake) keys the edges as a game wraps a sequence, the last key round to
    // the first, which brings a span that starts late on at the clip's start.
    if (!track.times.empty() && track.times.front() > 0.0f) {
        const f32 none = 0.0f;
        const u8* at = reinterpret_cast<const u8*>(&none);
        track.times.insert(track.times.begin(), 0.0f);
        track.values.insert(track.values.begin(), at, at + sizeof none);
    }
    if (!track.times.empty() && duration > 0.0f && track.times.back() < duration) {
        key(duration, 0.0f);
    }
    if (target.containers.empty()) {
        target.containers.emplace_back();
    }
    target.containers.front().subTracks.push_back(std::move(track));
}

SoftTiming SoftTimingAt(const Document& document, u32 model, u32 clip, const SoftBody& body,
                        const SoftEffect& effect, f32 seconds) {
    SoftTiming out;
    const SubTrack* track = AmountTrack(document, model, clip, body.id, effect.id);
    if (track == nullptr) {
        // Nothing keyed: on everywhere, or not at all (§9).
        out.level = effect.everywhere ? 1.0f : 0.0f;
        out.within = effect.everywhere;
        if (effect.everywhere && clip < document.clips.size() && document.clips[clip].duration > 0.0f) {
            out.progress = std::clamp(seconds / document.clips[clip].duration, 0.0f, 1.0f);
            out.sinceStart.push_back(std::max(seconds, 0.0f));
        }
        return out;
    }
    out.level = std::clamp(LevelOf(*track, seconds), 0.0f, 1.0f);
    for (const SoftSpan& span : SoftSpansOf(document, model, clip, body.id, effect.id)) {
        if (seconds < span.start) {
            break;
        }
        out.sinceStart.insert(out.sinceStart.begin(), seconds - span.start);
        out.within = seconds <= span.end;
        // A span of one key has no length: it is over as it starts.
        out.progress = span.end > span.start ? std::clamp((seconds - span.start) / (span.end - span.start), 0.0f, 1.0f)
                                             : 1.0f;
    }
    return out;
}

f32 SoftWindUpWindow(f32 smoothing, f32 duration) {
    return std::clamp(2.0f * smoothing, 1.0f / 60.0f, std::max(duration * 0.25f, 1.0f / 60.0f));
}

Vector3f SoftWindUpLean(const Vector3f& band, f32 size) {
    const f32 most = 0.5f * size;
    return most > 0.0f ? band * (1.0f / std::sqrt(1.0f + band.dot(band) / (most * most))) : Vector3f{0, 0, 0};
}

f32 SoftWholeCycles(f32 rate, f32 duration, bool looping) {
    if (!looping || !(duration > 0.0f) || !(rate > 0.0f)) {
        return std::max(rate, 0.0f);
    }
    return std::max(1.0f, std::round(rate * duration)) / duration;
}

SoftStepper::SoftStepper(const Document& document, u32 model, u32 body)
    : document_(&document), model_(model), rig_(document.models[model], body) {
    if (rig_.empty()) {
        return;
    }
    volume_ = MakeSoftVolume(rig_);
    const Model& owner = document.models[model];
    const SoftPoints& points = rig_.points();
    const Vector3f extent = points.high - points.low;
    longest_ = extent.x >= extent.y && extent.x >= extent.z ? Vector3f{1, 0, 0}
               : extent.y >= extent.z                       ? Vector3f{0, 1, 0}
                                                            : Vector3f{0, 0, 1};
    const u32 nodes = owner.nodes.size();
    pivot_.resize(nodes);
    child_.assign(nodes, kInvalidNode);
    for (u32 n = 0; n < nodes; ++n) {
        pivot_[n] = SoftBonePivot(owner.nodes, n);
    }
    for (u32 n = 0; n < nodes; ++n) {
        const Node& node = owner.nodes.nodes[n];
        if (node.parent < nodes && node.kind == NodeKind::Bone && !node.removed && child_[node.parent] == kInvalidNode) {
            child_[node.parent] = n;
        }
    }
    // Each bone's reach: to the middle of the body's points on it. And each
    // lump's outward.
    reach_.assign(nodes, Vector3f{0, 0, 0});
    std::vector<f32> held(nodes, 0.0f);
    lumpOut_.assign(rig_.lumpCount(), Vector3f{0, 0, 0});
    for (u32 p = 0; p < rig_.pointCount(); ++p) {
        const SoftPoint& point = points.points[p];
        for (const SoftSlot& slot : rig_.slotsOf(p)) {
            const f32 weight = slot.weight * std::max(point.area, 1e-12f);
            if (slot.node < nodes) {
                reach_[slot.node] = reach_[slot.node] + (point.at - pivot_[slot.node]) * weight;
                held[slot.node] += weight;
            }
            if (slot.lump < lumpOut_.size()) {
                lumpOut_[slot.lump] = lumpOut_[slot.lump] + point.outward * weight;
            }
        }
    }
    for (u32 n = 0; n < nodes; ++n) {
        reach_[n] = held[n] > 0.0f ? reach_[n] * (1.0f / held[n]) : Vector3f{0, 0, 0};
    }
    for (u32 k = 0; k < lumpOut_.size(); ++k) {
        // Points that face every way give no outward: half a unit of agreement is one.
        f32 total = 0.0f;
        for (u32 p = 0; p < rig_.pointCount(); ++p) {
            for (const SoftSlot& slot : rig_.slotsOf(p)) {
                total += slot.lump == k ? slot.weight * std::max(points.points[p].area, 1e-12f) : 0.0f;
            }
        }
        lumpOut_[k] = total > 0.0f && lumpOut_[k].length() > 0.5f * total ? Unit(lumpOut_[k], Vector3f{0, 0, 0})
                                                                         : Vector3f{0, 0, 0};
    }
    lumpRadius_ = SoftLumpRadii(rig_);
    lumpOfNode_.assign(nodes, kInvalidIndex);
    for (u32 k = 0; k < rig_.lumpCount(); ++k) {
        const SoftLump& lump = rig_.body().lumps[k];
        if (lump.kind != SoftLumpKind::Surface && lump.node < nodes && lumpOfNode_[lump.node] == kInvalidIndex) {
            lumpOfNode_[lump.node] = k;
        }
    }
}

const SoftStepper::DripCache& SoftStepper::dripOf(const SoftEffect& effect) const {
    const SoftDrip& drip = *effect.drip;
    DripCache* cache = nullptr;
    for (DripCache& each : drips_) {
        cache = each.effect == effect.id ? &each : cache;
    }
    const auto same = [&](const DripCache& held) {
        if (held.at.size() != drip.strandAt.size() || held.width != drip.strandWidth) {
            return false;
        }
        for (std::size_t s = 0; s < held.at.size(); ++s) {
            if ((held.at[s] - drip.strandAt[s]).length() != 0.0f) {
                return false;
            }
        }
        return true;
    };
    if (cache != nullptr && same(*cache)) {
        return *cache;
    }
    if (cache == nullptr) {
        cache = &drips_.emplace_back();
    }
    *cache = DripCache{};
    cache->effect = effect.id;
    cache->at = drip.strandAt;
    cache->width = drip.strandWidth;
    // A strand's own points are no part of the surface it hangs from.
    std::vector<u8> strand(pivot_.size(), 0);
    for (const SoftEffect& each : rig_.body().effects) {
        if (!each.drip) {
            continue;
        }
        for (const std::vector<u32>* nodes : {&each.drip->strandNodes, &each.drip->dropNodes}) {
            for (const u32 node : *nodes) {
                if (node < strand.size()) {
                    strand[node] = 1;
                }
            }
        }
    }
    const SoftPoints& points = rig_.points();
    const u32 count = rig_.pointCount();
    std::vector<u8> own(count, 1);
    for (u32 p = 0; p < count; ++p) {
        for (const SoftSlot& slot : rig_.slotsOf(p)) {
            if (slot.node < strand.size() && strand[slot.node] != 0) {
                own[p] = 0;
            }
        }
    }
    const std::size_t strands = std::min(drip.strandAt.size(), drip.strandWidth.size());
    cache->nearest.assign(strands, kInvalidIndex);
    cache->start.assign(strands + 1, 0);
    for (std::size_t s = 0; s < strands; ++s) {
        cache->start[s] = static_cast<u32>(cache->points.size());
        const f32 reach = kSoftPullReach * drip.strandWidth[s];
        f32 least = 0.0f;
        for (u32 p = 0; p < count; ++p) {
            if (own[p] == 0) {
                continue;
            }
            const f32 far = (points.points[p].at - drip.strandAt[s]).length();
            if (cache->nearest[s] == kInvalidIndex || far < least) {
                cache->nearest[s] = p;
                least = far;
            }
            if (far < reach) {
                cache->points.push_back({p, Profile(SoftBulgeShape::Round, far / reach)});
            }
        }
    }
    cache->start[strands] = static_cast<u32>(cache->points.size());
    return *cache;
}

const SoftStepper::RippleCache& SoftStepper::rippleOf(const SoftEffect& effect) const {
    const SoftPlace& place = effect.place;
    const SoftThrough through = effect.ripple ? effect.ripple->through : SoftThrough::Skin;
    RippleCache* cache = nullptr;
    for (RippleCache& each : ripples_) {
        cache = each.effect == effect.id ? &each : cache;
    }
    if (cache != nullptr && cache->through == through && cache->place.mesh == place.mesh &&
        cache->place.face == place.face && cache->place.weights.x == place.weights.x &&
        cache->place.weights.y == place.weights.y && cache->place.weights.z == place.weights.z) {
        return *cache;
    }
    if (cache == nullptr) {
        cache = &ripples_.emplace_back();
    }
    *cache = RippleCache{};
    cache->effect = effect.id;
    cache->place = place;
    cache->through = through;
    const SoftPlaceAt at = ResolveSoftPlace(document_->models[model_], rig_, place);
    cache->valid = at.valid;
    cache->points = SoftRippleDistances(rig_, at, through);
    cache->lumps = SoftLumpDistances(rig_, cache->points);
    return *cache;
}

const SoftStepper::PlaceCache& SoftStepper::placeOf(const SoftEffect& effect) const {
    const SoftPlace& place = effect.place;
    for (PlaceCache& cache : places_) {
        if (cache.effect != effect.id) {
            continue;
        }
        const bool same = cache.place.mesh == place.mesh && cache.place.face == place.face &&
                          cache.place.radius == place.radius && cache.place.weights.x == place.weights.x &&
                          cache.place.weights.y == place.weights.y && cache.place.weights.z == place.weights.z;
        if (same) {
            return cache;
        }
        cache = PlaceCache{};
        cache.effect = effect.id;
    }
    PlaceCache* cache = nullptr;
    for (PlaceCache& each : places_) {
        cache = each.effect == effect.id ? &each : cache;
    }
    if (cache == nullptr) {
        cache = &places_.emplace_back();
        cache->effect = effect.id;
    }
    cache->place = place;
    cache->at = ResolveSoftPlace(document_->models[model_], rig_, place);
    // A place with no radius of its own takes §8.2's default.
    cache->radius = place.radius > 0.0f ? place.radius : 0.3f * rig_.points().size;
    if (cache->at.valid) {
        std::vector<std::pair<u32, f32>> seeds;
        for (const u32 point : cache->at.points) {
            seeds.push_back({point, (rig_.points().points[point].at - cache->at.at).length()});
        }
        cache->distance = SurfaceDistances(rig_, seeds, 3.0f * cache->radius);
    }
    return *cache;
}

Vector3f SoftStepper::axisOf(SoftAxis axis, u32 bone, std::span<const Matrix44f> skin) const {
    switch (axis) {
    case SoftAxis::X:
        return Vector3f{1, 0, 0};
    case SoftAxis::Y:
        return Vector3f{0, 1, 0};
    case SoftAxis::Auto:
        return longest_;
    case SoftAxis::Bone:
        if (bone < pivot_.size() && bone < skin.size()) {
            // Along the bone: to its child, or to its own points.
            const Vector3f along = child_[bone] < pivot_.size() ? pivot_[child_[bone]] - pivot_[bone] : reach_[bone];
            return Unit(transform_normal(along, skin[bone]), Vector3f{0, 0, 1});
        }
        return Vector3f{0, 0, 1};
    default:
        return Vector3f{0, 0, 1};
    }
}

void SoftStepper::keepClear(std::span<Transform> departures) const {
    const SoftBody& body = rig_.body();
    for (std::size_t k = 0; k < departures.size() && k < body.lumps.size(); ++k) {
        const f32 clear = body.lumps[k].clear;
        if (!(clear > 0.0f) || lumpOut_[k].length() < 0.5f) {
            continue;
        }
        const f32 inward = -departures[k].translation.dot(lumpOut_[k]);
        if (inward > 0.5f * clear) {
            departures[k].translation = departures[k].translation + lumpOut_[k] * (inward - 0.5f * clear);
        }
    }
}

void SoftStepper::awayOf(std::span<const Matrix44f> skin, std::span<const Transform> departures,
                         std::span<const f32> share, std::vector<Vector3f>& away) const {
    const SoftBody& body = rig_.body();
    const SoftPoints& points = rig_.points();
    const u32 count = rig_.pointCount();
    const std::size_t lumpCount = std::min<std::size_t>(rig_.lumpCount(), share.size());
    // Each lump as it stands, and its pivot there.
    std::vector<Matrix44f> moved(lumpCount, Matrix44f::identity());
    std::vector<Vector3f> pivot(lumpCount, Vector3f{0, 0, 0});
    for (std::size_t k = 0; k < lumpCount; ++k) {
        if (share[k] == 0.0f) {
            continue;
        }
        const u32 family = body.lumps[k].family();
        const Matrix44f carried = family < skin.size() ? skin[family] : Matrix44f::identity();
        moved[k] = SoftDepartureMatrix(body.lumps[k].rest, k < departures.size() ? departures[k] : Transform{}) * carried;
        pivot[k] = transform_point(body.lumps[k].rest, moved[k]);
    }
    away.assign(count, Vector3f{0, 0, 0});
    std::vector<std::pair<u32, f32>> rides;
    for (u32 p = 0; p < count; ++p) {
        rides.clear();
        for (const SoftSlot& slot : rig_.slotsOf(p)) {
            if (slot.lump < lumpCount && share[slot.lump] != 0.0f) {
                rides.emplace_back(slot.lump, slot.weight);
            }
        }
        // Summed in the lumps' order, whatever order the point's slots are in.
        std::stable_sort(rides.begin(), rides.end(),
                         [](const std::pair<u32, f32>& a, const std::pair<u32, f32>& b) { return a.first < b.first; });
        for (const auto& [k, weight] : rides) {
            away[p] = away[p] + (transform_point(points.points[p].at, moved[k]) - pivot[k]) * (weight * share[k]);
        }
    }
}

void SoftStepper::trimVolume(std::span<const Matrix44f> skin, f64 asked, f32 share,
                             std::span<Transform> departures) const {
    if (!volume_.encloses || !(share > 0.0f)) {
        return;
    }
    const std::size_t lumpCount = std::min<std::size_t>(rig_.lumpCount(), departures.size());
    std::vector<f32> every(lumpCount, 0.0f);
    for (std::size_t k = 0; k < lumpCount; ++k) {
        every[k] = rig_.scales(static_cast<u32>(k)) ? 1.0f : 0.0f;
    }
    std::vector<Vector3f> posed, away;
    f64 target = asked;
    for (int pass = 0; pass < 2; ++pass) {
        rig_.pose(skin, departures, posed);
        if (pass == 0 && share < 1.0f) {
            // Part of the way, from where the surface stands now.
            const f64 now = SoftVolumeAt(volume_, posed);
            target = now + (asked - now) * static_cast<f64>(share);
        }
        awayOf(skin, departures, every, away);
        const f64 even = SolveOffset(OffsetCubic(volume_, posed, away), target, 0.5);
        if (std::abs(even) < 1e-7) {
            break;
        }
        for (std::size_t k = 0; k < lumpCount; ++k) {
            departures[k].scale =
                departures[k].scale * std::clamp(1.0f + static_cast<f32>(even) * every[k], 0.05f, 20.0f);
        }
    }
}

void SoftStepper::step(std::span<const Matrix44f> skin, const SoftClock& clock, const SoftPoseAt& poseAt,
                       SoftStepResult& out, SoftFitMethod method, bool drivingOnly) const {
    out = SoftStepResult{};
    if (rig_.empty()) {
        return;
    }
    const SoftBody& body = rig_.body();
    const SoftPoints& points = rig_.points();
    const u32 count = rig_.pointCount();
    const std::size_t lumpCount = rig_.lumpCount();
    const f32 size = std::max(points.size, 1e-6f);
    const bool clipped = clock.clip < document_->clips.size();
    const f32 duration = clipped ? document_->clips[clock.clip].duration : 0.0f;
    const bool looping = clipped && document_->clips[clock.clip].looping;

    // 1. Carried.
    std::vector<Matrix44f> carriedBy;
    rig_.carry(skin, out.carried, &carriedBy);
    out.goal = out.carried;
    out.departures.assign(lumpCount, Transform{});
    std::vector<Vector3f> outward(count);
    for (u32 p = 0; p < count; ++p) {
        outward[p] = Unit(transform_normal(points.points[p].outward, carriedBy[p]), points.points[p].outward);
    }
    const auto posedPivot = [&](u32 node) {
        return node < pivot_.size() && node < skin.size() ? transform_point(pivot_[node], skin[node]) : Vector3f{0, 0, 0};
    };
    const auto timingOf = [&](const SoftEffect& effect) {
        return SoftTimingAt(*document_, model_, clock.clip, body, effect, clock.seconds);
    };
    // An impact's rise and fall, since the newest mark (§8.2).
    const auto impact = [](const SoftBulge& bulge, std::span<const f32> since) {
        f32 most = 0.0f;
        for (const f32 t : since) {
            const f32 rise = std::max(bulge.rise, 1e-4f), fall = std::max(bulge.fall, 1e-4f);
            const f32 level = t < rise ? t / rise : 1.0f - (t - rise) / fall;
            most = std::max(most, std::clamp(level, 0.0f, 1.0f));
        }
        // Eased, so it neither starts nor stops with a corner.
        return most * most * (3.0f - 2.0f * most);
    };
    const auto envelopeOf = [&](const SoftEffect& effect, const SoftTiming& timing) {
        const SoftBulge& bulge = *effect.bulge;
        if (bulge.from == SoftComesFrom::Times) {
            return impact(bulge, timing.sinceStart);
        }
        if (bulge.from == SoftComesFrom::Events) {
            std::vector<f32> since;
            for (const ClipEvent& event : clipped ? document_->clips[clock.clip].events : std::vector<ClipEvent>{}) {
                if (event.name == bulge.event && event.time <= clock.seconds) {
                    since.push_back(clock.seconds - event.time);
                }
            }
            return impact(bulge, since);
        }
        // Contact is the colliders': it comes with them.
        return bulge.from == SoftComesFrom::Spans ? timing.level : 0.0f;
    };
    // The anchor a stretch holds still.
    const auto anchorOf = [&](SoftAnchor anchor, const Vector3f& axis) {
        Vector3f middle{0, 0, 0};
        f32 area = 0.0f, least = 0.0f;
        for (u32 p = 0; p < count; ++p) {
            const f32 a = std::max(points.points[p].area, 1e-12f);
            middle = middle + out.goal[p] * a;
            area += a;
            const f32 along = out.goal[p].dot(axis);
            least = p == 0 ? along : std::min(least, along);
        }
        middle = middle * (1.0f / std::max(area, 1e-20f));
        if (anchor == SoftAnchor::Centre) {
            return middle;
        }
        if (anchor == SoftAnchor::Pivot) {
            // The bone that carries the most of the body.
            u32 most = kInvalidNode;
            f32 share = -1.0f;
            std::vector<f32> held(pivot_.size(), 0.0f);
            for (u32 p = 0; p < count; ++p) {
                for (const SoftSlot& slot : rig_.slotsOf(p)) {
                    if (slot.node < held.size()) {
                        held[slot.node] += slot.weight * points.points[p].area;
                    }
                }
            }
            for (u32 n = 0; n < held.size(); ++n) {
                if (held[n] > share) {
                    share = held[n];
                    most = n;
                }
            }
            return most != kInvalidNode ? posedPivot(most) : middle;
        }
        return middle - axis * (middle.dot(axis) - least);
    };

    // What the body meets at this step, found when a Bulge first asks.
    std::vector<SoftCollider> colliders;
    bool met = false;

    // 2. The shape effects, in the list's order.
    std::vector<Vector3f> moves(count), directions;
    for (const SoftEffect& effect : body.effects) {
        if (!effect.enabled || (drivingOnly && !effect.drivesJiggle)) {
            continue;
        }
        if (effect.kind == SoftEffectKind::Inflate && effect.inflate && effect.inflate->along == SoftAlong::Surface) {
            const SoftInflate& inflate = *effect.inflate;
            const SoftTiming timing = timingOf(effect);
            const f32 amount = effect.amount * timing.level * PulseOf(inflate, clock.seconds, duration, looping);
            if (amount == 0.0f) {
                continue;
            }
            // Outward, evened so that a crease crosses itself later.
            directions = outward;
            for (u32 pass = 0; pass < inflate.smooth; ++pass) {
                std::vector<Vector3f> next(count);
                for (u32 p = 0; p < count; ++p) {
                    Vector3f sum = directions[p];
                    for (const u32 other : rig_.ringOf(p)) {
                        sum = sum + directions[other];
                    }
                    next[p] = Unit(sum, directions[p]);
                }
                directions.swap(next);
            }
            if (inflate.round > 0.0f && amount > 0.0f) {
                // *Round*: out of the body's blob in place of out of its own
                // face, so a swell fills the creases and does not deepen them.
                const std::vector<SoftBall> balls = SoftBlobBalls(rig_, skin, 0.5f);
                const f32 round = std::min(inflate.round, 1.0f);
                for (u32 p = 0; p < count; ++p) {
                    Vector3f inward{0, 0, 0};
                    SoftBlobField(balls, out.goal[p], &inward);
                    const Vector3f outOf = Unit(inward * -1.0f, directions[p]);
                    // Never against the face's own way out: a point does not sink.
                    if (outOf.dot(directions[p]) > 0.0f) {
                        directions[p] = Unit(directions[p] * (1.0f - round) + outOf * round, directions[p]);
                    }
                }
            }
            const PlaceCache* place = inflate.towardPlace ? &placeOf(effect) : nullptr;
            const f64 before = volume_.encloses ? SoftVolumeAt(volume_, out.goal) : 0.0;
            if (amount < 0.0f) {
                // Deflate goes toward the skeleton: a share of the way to the
                // nearest point of the bone that carries most of each point.
                for (u32 p = 0; p < count; ++p) {
                    const std::span<const SoftSlot> slots = rig_.slotsOf(p);
                    const SoftSlot* heaviest = nullptr;
                    for (const SoftSlot& slot : slots) {
                        heaviest = heaviest == nullptr || slot.weight > heaviest->weight ? &slot : heaviest;
                    }
                    Vector3f bone = out.goal[p];
                    if (heaviest != nullptr && heaviest->node < pivot_.size()) {
                        const Vector3f a = posedPivot(heaviest->node);
                        const u32 child = child_[heaviest->node];
                        const Vector3f line[2] = {a, child < pivot_.size() ? posedPivot(child) : a};
                        f32 along = 0.0f;
                        bone = OnLine(line, out.goal[p], along);
                    }
                    moves[p] = (bone - out.goal[p]) * points.points[p].softness;
                }
                const f64 share = volume_.encloses
                                      ? std::clamp(SolveOffset(OffsetCubic(volume_, out.goal, moves),
                                                               before * (1.0 + std::max(amount, -0.9f)), 0.98),
                                                   0.0, 0.98)
                                      : static_cast<f64>(-amount);
                for (u32 p = 0; p < count; ++p) {
                    out.goal[p] = out.goal[p] + moves[p] * static_cast<f32>(share);
                }
                out.moved = true;
                continue;
            }
            for (u32 p = 0; p < count; ++p) {
                f32 where = 1.0f;
                if (place != nullptr) {
                    where = place->at.valid && p < place->distance.size()
                                ? Profile(SoftBulgeShape::Round, place->distance[p] / place->radius)
                                : 0.0f;
                }
                moves[p] = directions[p] * (points.points[p].softness * where);
            }
            // A flat patch encloses nothing: its Amount is a distance (§8.1).
            const f64 offset = volume_.encloses ? SolveOffset(OffsetCubic(volume_, out.goal, moves),
                                                              before * (1.0 + amount), 2.0 * size)
                                                : static_cast<f64>(amount) * size;
            for (u32 p = 0; p < count; ++p) {
                out.goal[p] = out.goal[p] + moves[p] * static_cast<f32>(offset);
            }
            if (inflate.keepVolume && place != nullptr && volume_.encloses) {
                // What swelled near the place is paid for everywhere.
                for (u32 p = 0; p < count; ++p) {
                    moves[p] = directions[p] * points.points[p].softness;
                }
                const f64 back = SolveOffset(OffsetCubic(volume_, out.goal, moves), before, size);
                for (u32 p = 0; p < count; ++p) {
                    out.goal[p] = out.goal[p] + moves[p] * static_cast<f32>(back);
                }
            }
            out.moved = true;
        } else if (effect.kind == SoftEffectKind::Bulge && effect.bulge && effect.bulge->along == SoftAlong::Surface) {
            const SoftBulge& bulge = *effect.bulge;
            const SoftTiming timing = timingOf(effect);
            if (bulge.from == SoftComesFrom::Contact) {
                // Pressed in by what the body meets (§8.2): a point inside a
                // collider goes to its surface, by the effect's Amount, and
                // with *Keep volume* what that displaces comes back as a rise
                // round the contact. One pass, and nothing that solves.
                const f32 give = std::clamp(effect.amount, 0.0f, 1.0f) * timing.level;
                if (give == 0.0f) {
                    continue;
                }
                if (!met) {
                    colliders = SoftCollidersAt(document_->models[model_], body, skin);
                    met = true;
                }
                if (colliders.empty()) {
                    continue;
                }
                // The rise fades over the place's radius from the contact's edge.
                const f32 band = effect.place.radius > 0.0f ? effect.place.radius : 0.3f * size;
                const f64 before = volume_.encloses ? SoftVolumeAt(volume_, out.goal) : 0.0;
                bool pressed = false;
                for (u32 p = 0; p < count; ++p) {
                    f32 deepest = -std::numeric_limits<f32>::infinity();
                    Vector3f way{0, 0, 0};
                    for (const SoftCollider& collider : colliders) {
                        Vector3f outOf{0, 0, 0};
                        const f32 depth = SoftColliderDepth(collider, out.goal[p], outOf);
                        if (depth > deepest) {
                            deepest = depth;
                            way = outOf;
                        }
                    }
                    const f32 softness = points.points[p].softness;
                    if (deepest > 0.0f) {
                        out.goal[p] = out.goal[p] + way * (deepest * give * softness);
                        pressed = pressed || softness > 0.0f;
                        moves[p] = Vector3f{0, 0, 0};
                    } else {
                        moves[p] = outward[p] * (softness * Profile(SoftBulgeShape::Round, -deepest / band));
                    }
                }
                if (!pressed) {
                    continue;
                }
                if (bulge.keepVolume && volume_.encloses) {
                    const f64 back = SolveOffset(OffsetCubic(volume_, out.goal, moves), before, size);
                    for (u32 p = 0; p < count; ++p) {
                        out.goal[p] = out.goal[p] + moves[p] * static_cast<f32>(back);
                    }
                }
                out.moved = true;
                continue;
            }
            const PlaceCache& place = placeOf(effect);
            const f32 height = effect.amount * envelopeOf(effect, timing) * place.radius;
            if (height == 0.0f) {
                continue;
            }
            const f64 before = volume_.encloses ? SoftVolumeAt(volume_, out.goal) : 0.0;
            std::vector<f32> lifted(count, 0.0f);
            if (bulge.travels && (bulge.chain.size() >= 2 || bulge.line.size() >= 2)) {
                // Along a line: a swell away from its axis, where the span's
                // progress has it.
                std::vector<Vector3f> line;
                for (const u32 bone : bulge.chain) {
                    line.push_back(posedPivot(bone));
                }
                if (line.size() < 2) {
                    line.clear();
                    const Matrix44f carried =
                        place.at.bone < skin.size() ? skin[place.at.bone] : Matrix44f::identity();
                    for (const Vector3f& at : bulge.line) {
                        line.push_back(transform_point(at, carried));
                    }
                }
                f32 length = 0.0f;
                for (std::size_t k = 0; k + 1 < line.size(); ++k) {
                    length += (line[k + 1] - line[k]).length();
                }
                const f32 centre = timing.progress * length;
                for (u32 p = 0; p < count; ++p) {
                    f32 along = 0.0f;
                    const Vector3f on = OnLine(line, out.goal[p], along);
                    const Vector3f away = out.goal[p] - on;
                    const f32 reach = away.length();
                    lifted[p] = reach < kTravelReach * place.radius
                                    ? Profile(SoftBulgeShape::Round, std::abs(along - centre) / place.radius)
                                    : 0.0f;
                    moves[p] = Unit(away, outward[p]) * (points.points[p].softness * lifted[p]);
                }
            } else {
                if (!place.at.valid) {
                    continue;
                }
                // Over the surface, to the points that face as the place does (§8.2).
                for (u32 p = 0; p < count; ++p) {
                    const bool faces = points.points[p].outward.dot(place.at.normal) > 0.0f;
                    lifted[p] = faces ? Profile(bulge.shape, place.distance[p] / place.radius) : 0.0f;
                    moves[p] = outward[p] * (points.points[p].softness * lifted[p]);
                }
            }
            for (u32 p = 0; p < count; ++p) {
                out.goal[p] = out.goal[p] + moves[p] * height;
            }
            if (bulge.keepVolume && volume_.encloses) {
                // Paid for by a slight shrink of the rest.
                for (u32 p = 0; p < count; ++p) {
                    moves[p] = outward[p] * (points.points[p].softness * (1.0f - std::min(std::abs(lifted[p]), 1.0f)));
                }
                const f64 back = SolveOffset(OffsetCubic(volume_, out.goal, moves), before, size);
                for (u32 p = 0; p < count; ++p) {
                    out.goal[p] = out.goal[p] + moves[p] * static_cast<f32>(back);
                }
            }
            out.moved = true;
        } else if (effect.kind == SoftEffectKind::Blobify && effect.blobify) {
            const struct SoftBlobify& blobify = *effect.blobify;
            const f32 amount = std::clamp(effect.amount * timingOf(effect).level, 0.0f, 1.0f);
            if (amount == 0.0f) {
                continue;
            }
            // The balls stand at the lumps as the pose has them, so an arm
            // fuses with the side only where it hangs there (§8.5).
            const SoftBlobReport blob = SoftBlobTo(rig_, volume_, SoftBlobBalls(rig_, skin, blobify.roundness),
                                                   out.goal, amount, blobify.swell, blobify.keepVolume);
            out.fuses.emplace_back(effect.id, blob.fuses);
            out.moved = true;
        } else if (effect.kind == SoftEffectKind::Melt && effect.melt) {
            const SoftMelt& melt = *effect.melt;
            const SoftTiming timing = timingOf(effect);
            // Its span is a progress, held after it. It acts only in a clip
            // that places one: a body that rises out of its pool in Birth is
            // not a pool in Walk. A Decay clip that places none holds a
            // forward melt's end.
            f32 progress = timing.sinceStart.empty() ? 0.0f : timing.progress;
            if (!clipped || AmountTrack(*document_, model_, clock.clip, body.id, effect.id) == nullptr) {
                bool named = clipped && melt.staysInDecay && !melt.backward;
                if (named) {
                    const std::string& name = document_->clips[clock.clip].name;
                    const char* decay = "decay";
                    named = name.size() >= 5;
                    for (std::size_t i = 0; named && i < 5; ++i) {
                        named = std::tolower(static_cast<unsigned char>(name[i])) == decay[i];
                    }
                }
                if (!named) {
                    continue;
                }
                progress = 1.0f;
            }
            progress *= std::clamp(effect.amount, 0.0f, 1.0f);
            // Running backward the body is its pool until its span starts: only
            // a forward melt has nothing to do at 0.
            if (progress <= 0.0f && !melt.backward) {
                continue;
            }
            if (!columned_) {
                columns_ = SoftMeltColumns(rig_);
                columned_ = true;
            }
            SoftMeltTo(rig_, volume_, columns_, out.goal, melt.onGrid ? 0.0f : melt.floor, melt, progress, effect.seed);
            out.moved = true;
        } else if (effect.kind == SoftEffectKind::SquashStretch && effect.squash) {
            const SoftSquash& squash = *effect.squash;
            const SoftTiming timing = timingOf(effect);
            const f32 level = effect.amount * timing.level;
            if (level == 0.0f) {
                continue;
            }
            if (squash.mode == SoftSquashMode::Keyed) {
                // -1 is a full squash, +1 a full stretch.
                const f32 s = std::clamp(level, -1.0f, 1.0f);
                const f32 k = std::max(1.0f + (s >= 0.0f ? squash.stretch : squash.squash) * s, 0.1f);
                const Vector3f axis = axisOf(squash.axis, squash.axisBone, skin);
                const Vector3f anchor = anchorOf(squash.anchor, axis);
                for (u32 p = 0; p < count; ++p) {
                    out.goal[p] =
                        out.goal[p] + (Stretched(out.goal[p], anchor, axis, k) - out.goal[p]) * points.points[p].softness;
                }
                out.moved = true;
                continue;
            }
            if (!poseAt || !clipped || !(duration > 0.0f)) {
                continue;
            }
            // Auto: each bone's own travel and turn, by differences across the
            // smoothing window. It has no memory, so it scrubs.
            const f32 h = std::clamp(squash.smoothing * 0.5f, 1.0f / 240.0f, std::max(duration * 0.25f, 1.0f / 240.0f));
            const auto timeAt = [&](f32 t) {
                if (looping) {
                    t = std::fmod(t, duration);
                    return t < 0.0f ? t + duration : t;
                }
                return std::clamp(t, 0.0f, duration);
            };
            // Played once, a clip's end is no stop and its start no standing
            // start: the window is cut short there, and what is read is the
            // travel over what is left of it.
            const f32 behind = looping ? h : std::min(h, std::max(clock.seconds, 0.0f));
            const f32 ahead = looping ? h : std::min(h, std::max(duration - clock.seconds, 0.0f));
            const f32 across = behind + ahead;
            if (!(across > 0.0f)) {
                continue;
            }
            const bool whole = behind == h && ahead == h;
            std::vector<Matrix44f> earlier, later;
            poseAt(timeAt(clock.seconds - h), earlier);
            poseAt(timeAt(clock.seconds + h), later);
            const u32 nodes = static_cast<u32>(std::min({pivot_.size(), skin.size(), earlier.size(), later.size()}));
            struct Motion {
                Vector3f pivot{0, 0, 0};
                Vector3f along{0, 0, 1};
                f32 stretch = 1.0f;
                Vector3f swing{0, 0, 1};
                f32 swung = 1.0f;
                bool any = false;
            };
            std::vector<Motion> motion(nodes);
            std::vector<u8> used(nodes, 0);
            for (u32 p = 0; p < count; ++p) {
                for (const SoftSlot& slot : rig_.slotsOf(p)) {
                    if (slot.node < nodes) {
                        used[slot.node] = 1;
                    }
                }
            }
            for (u32 n = 0; n < nodes; ++n) {
                if (used[n] == 0) {
                    continue;
                }
                Motion& m = motion[n];
                m.pivot = posedPivot(n);
                const Vector3f before = transform_point(pivot_[n], earlier[n]);
                const Vector3f after = transform_point(pivot_[n], later[n]);
                const Vector3f in = m.pivot - before, outv = after - m.pivot;
                const f32 speed = (after - before).length() / across;
                // How much slower it leaves than it came. With one side cut
                // to nothing there is no telling, and no stop is read.
                constexpr f32 kNoSide = 1e-5f;
                const f32 slowing =
                    whole ? std::max((in.length() - outv.length()) / (h * h), 0.0f)
                          : behind > kNoSide && ahead > kNoSide
                                ? std::max((in.length() / behind - outv.length() / ahead) / (0.5f * across), 0.0f)
                                : 0.0f;
                const f32 fast = squash.stretch * (1.0f - std::exp(-speed / (kSpeed * size)));
                const f32 stop = squash.squash * (1.0f - std::exp(-slowing / (kStop * size)));
                m.along = squash.axis == SoftAxis::Motion ? Unit(after - before, Unit(in, Vector3f{0, 0, 1}))
                                                          : axisOf(squash.axis, squash.axisBone, skin);
                m.stretch = std::max((1.0f + level * fast) * (1.0f - level * stop), 0.1f);
                // A swung part stretches away from the axis it turns about:
                // along its own reach, by how fast it turns.
                const Vector3f reach = transform_normal(reach_[n], skin[n]);
                const Vector3f was = transform_normal(reach_[n], earlier[n]);
                const Vector3f will = transform_normal(reach_[n], later[n]);
                const f32 lengths = was.length() * will.length();
                const f32 turn = lengths > 1e-12f ? std::acos(std::clamp(was.dot(will) / lengths, -1.0f, 1.0f)) / across
                                                 : 0.0f;
                m.swing = Unit(reach, m.along);
                m.swung = 1.0f + level * squash.turning * squash.stretch * (1.0f - std::exp(-turn / kSwing));
                m.any = std::abs(m.stretch - 1.0f) > 1e-6f || std::abs(m.swung - 1.0f) > 1e-6f;
            }
            std::vector<Vector3f> was, will;
            if (squash.drag > 0.0f) {
                rig_.carry(earlier, was);
                rig_.carry(later, will);
            }
            Vector3f drift{0, 0, 0};
            f32 area = 0.0f;
            for (u32 p = 0; squash.drag > 0.0f && p < count; ++p) {
                drift = drift + (will[p] - was[p]) * points.points[p].area;
                area += points.points[p].area;
            }
            drift = area > 0.0f ? drift * (1.0f / area) : drift;
            // *Wind-up*: the middle band of each bone's travel, which is where
            // it stands against where it stood and will stand a moment away.
            // Added back, the body leans away from a move that is coming and
            // carries past one that ended (§8.7). The clip is known, so it
            // needs no memory; in a loop the moment wraps.
            std::vector<Vector3f> band;
            if (squash.windUp > 0.0f) {
                const f32 w = SoftWindUpWindow(squash.smoothing, duration);
                // Played once, a clip starts and ends as the animation has it.
                const f32 fade = looping ? 1.0f
                                         : std::clamp(std::min(clock.seconds, duration - clock.seconds) / w, 0.0f, 1.0f);
                std::vector<Matrix44f> past, coming;
                poseAt(timeAt(clock.seconds - w), past);
                poseAt(timeAt(clock.seconds + w), coming);
                band.assign(nodes, Vector3f{0, 0, 0});
                for (u32 n = 0; n < nodes && n < past.size() && n < coming.size(); ++n) {
                    if (used[n] != 0) {
                        const Vector3f around =
                            (transform_point(pivot_[n], past[n]) + transform_point(pivot_[n], coming[n])) * 0.5f;
                        band[n] = SoftWindUpLean(motion[n].pivot - around, size) * (level * squash.windUp * fade);
                    }
                }
            }
            const Vector3f hold = anchorOf(squash.anchor, Vector3f{0, 0, 1});
            for (u32 p = 0; p < count; ++p) {
                // A point takes its bones' stretch by its weights.
                Vector3f to{0, 0, 0};
                f32 sum = 0.0f;
                for (const SoftSlot& slot : rig_.slotsOf(p)) {
                    Vector3f mapped = out.goal[p];
                    if (slot.node < nodes && motion[slot.node].any) {
                        const Motion& m = motion[slot.node];
                        mapped = Stretched(mapped, m.pivot, m.along, m.stretch);
                        mapped = Stretched(mapped, m.pivot, m.swing, m.swung);
                    }
                    to = to + mapped * slot.weight;
                    sum += slot.weight;
                }
                Vector3f moved = sum > 0.0f ? to * (1.0f / sum) : out.goal[p];
                if (squash.drag > 0.0f) {
                    // The far parts trail the motion.
                    const Vector3f own = (will[p] - was[p]) - drift;
                    const f32 far = std::min((out.goal[p] - hold).length() / size, 1.0f);
                    moved = moved - own * (level * squash.drag * far * squash.smoothing / across);
                }
                if (!band.empty() && sum > 0.0f) {
                    // A lean: nothing at what is held still, all of it a region away.
                    Vector3f lean{0, 0, 0};
                    for (const SoftSlot& slot : rig_.slotsOf(p)) {
                        if (slot.node < nodes) {
                            lean = lean + band[slot.node] * slot.weight;
                        }
                    }
                    moved = moved + lean * (std::min((out.goal[p] - hold).length() / size, 1.0f) / sum);
                }
                out.goal[p] = out.goal[p] + (moved - out.goal[p]) * points.points[p].softness;
            }
            out.moved = true;
        }
    }

    // A Ripple's height at a distance from its place, at this step (§8.9).
    const auto heightOf = [&](const SoftEffect& effect, std::vector<f32>& since, f32& turns, f32& level) {
        const SoftRipple& ripple = *effect.ripple;
        const SoftTiming timing = timingOf(effect);
        const SoftWave wave = SoftWaveOf(ripple, effect.amount, size);
        since.clear();
        turns = 0.0f;
        level = 0.0f;
        if (ripple.plays == SoftRipplePlays::Loops) {
            // A steady train, by whole cycles a loop.
            level = timing.level;
            turns = SoftWholeCycles(wave.speed / wave.length, duration, looping) * clock.seconds;
        } else if (ripple.from == SoftComesFrom::Events) {
            for (const ClipEvent& event : clipped ? document_->clips[clock.clip].events : std::vector<ClipEvent>{}) {
                if (event.name == ripple.event && event.time <= clock.seconds) {
                    since.push_back(clock.seconds - event.time);
                }
            }
        } else {
            // A swell from each span's start.
            since = timing.sinceStart;
        }
        return wave;
    };
    const auto rippleAt = [&](const SoftWave& wave, const SoftRipple& ripple, std::span<const f32> since, f32 turns,
                              f32 level, f32 distance) {
        if (ripple.plays == SoftRipplePlays::Loops) {
            return level * SoftTrainHeight(wave, distance, turns);
        }
        f32 height = 0.0f;
        for (const f32 t : since) {
            height += SoftSwellHeight(wave, distance, t);
        }
        return height;
    };

    // 3. The travelling effects add theirs.
    for (const SoftEffect& effect : body.effects) {
        if (!effect.enabled || (drivingOnly && !effect.drivesJiggle)) {
            continue;
        }
        if (effect.kind == SoftEffectKind::Ripple && effect.ripple && effect.ripple->along == SoftAlong::Surface) {
            const RippleCache& from = rippleOf(effect);
            std::vector<f32> since;
            f32 turns = 0.0f, level = 0.0f;
            const SoftWave wave = heightOf(effect, since, turns, level);
            if (!from.valid || wave.height == 0.0f || (since.empty() && level == 0.0f)) {
                continue;
            }
            for (u32 p = 0; p < count; ++p) {
                const f32 height = rippleAt(wave, *effect.ripple, since, turns, level, from.points[p]);
                out.goal[p] = out.goal[p] + outward[p] * (height * points.points[p].softness);
            }
            out.moved = true;
            continue;
        }
        if (effect.kind != SoftEffectKind::Wobble || !effect.wobble) {
            continue;
        }
        const SoftWobble& wobble = *effect.wobble;
        const SoftTiming timing = timingOf(effect);
        const f32 far = effect.amount * timing.level * size;
        if (far == 0.0f || wobble.pattern == SoftWobblePattern::Random) {
            continue; // Random is the body's own ways of swinging: `SwingSoft` adds it.
        }
        const Vector3f axis = axisOf(wobble.axis, wobble.axisBone, skin);
        f32 low = 0.0f, high = 0.0f;
        Vector3f middle{0, 0, 0};
        f32 area = 0.0f;
        for (u32 p = 0; p < count; ++p) {
            const f32 along = out.goal[p].dot(axis);
            low = p == 0 ? along : std::min(low, along);
            high = p == 0 ? along : std::max(high, along);
            middle = middle + out.goal[p] * points.points[p].area;
            area += points.points[p].area;
        }
        middle = area > 0.0f ? middle * (1.0f / area) : middle;
        const f32 tall = std::max(high - low, 1e-6f);
        const Vector3f base = middle - axis * (middle.dot(axis) - low);
        // Every clip starts a rhythm at one phase; the seed turns which way it leans.
        const f32 phase = kTau * SoftWholeCycles(wobble.speed, duration, looping) * clock.seconds;
        const Vector3f across = Unit(cross(axis, std::abs(axis.z) < 0.9f ? Vector3f{0, 0, 1} : Vector3f{1, 0, 0}),
                                     Vector3f{1, 0, 0});
        const f32 lean = 2.3999632f * static_cast<f32>(effect.seed);
        const Vector3f side = across * std::cos(lean) + cross(axis, across) * std::sin(lean);
        for (u32 p = 0; p < count; ++p) {
            const f32 along = out.goal[p].dot(axis) - low;
            const f32 up = along / tall;
            // The far end trails the near end.
            const f32 at = phase - wobble.overlap * 3.1415927f * up;
            Vector3f moved = out.goal[p];
            if (wobble.pattern == SoftWobblePattern::Jelly) {
                // To four times its height or a quarter at the most: a region
                // thin along the axis would else be asked for a stretch without end.
                moved = Stretched(moved, base, axis,
                                  std::exp(std::clamp((far / tall) * std::sin(at), -kJellyMost, kJellyMost)));
            } else if (wobble.pattern == SoftWobblePattern::Sway) {
                moved = moved + side * (far * up * std::sin(at));
            } else {
                const f32 wave = std::max(wobble.size, 0.5f) * tall;
                moved = moved + side * (far * std::sin(kTau * along / wave - phase));
            }
            out.goal[p] = out.goal[p] + (moved - out.goal[p]) * points.points[p].softness;
        }
        out.moved = true;
    }

    // Each point's way out from its lumps' pivots as they stand: what an even
    // change of the lumps' scales moves it along, each lump by its @p share.
    const auto awayFrom = [&](std::span<const f32> share, std::vector<Vector3f>& away) {
        awayOf(skin, out.departures, share, away);
    };
    const auto grow = [&](std::span<const f32> share, f32 factor) {
        for (std::size_t k = 0; k < lumpCount; ++k) {
            out.departures[k].scale = out.departures[k].scale * std::clamp(1.0f + factor * share[k], 0.05f, 20.0f);
        }
    };

    // Drip (§8.6). One strand at this step: its three bones by how long ago
    // it set out, or by where it is in its cycle, and its droplet. @p rode is
    // how far its site has gone with the body's surface, model space.
    struct Strand {
        u32 lumps[3] = {kInvalidIndex, kInvalidIndex, kInvalidIndex};
        u32 drop = kInvalidIndex; ///< Its droplet's lump, or none.
        SoftStrandPose pose;
        Transform droplet;
        f32 share = 0.0f; ///< How much of its length it hangs by.
        bool acts = false;
    };
    const auto stranded = [](const SoftDrip& drip) {
        const std::size_t strands = drip.strandAt.size();
        return strands != 0 && drip.strandNodes.size() == 3 * strands && drip.strandWidth.size() == strands;
    };
    const auto strandOf = [&](const SoftEffect& effect, const SoftTiming& timing, std::size_t s, const Vector3f& rode) {
        Strand strand;
        const SoftDrip& drip = *effect.drip;
        const std::size_t strands = drip.strandAt.size();
        const u32 base = drip.strandNodes[3 * s];
        bool whole = true;
        for (int i = 0; i < 3; ++i) {
            const u32 node = drip.strandNodes[3 * s + i];
            strand.lumps[i] = node < lumpOfNode_.size() ? lumpOfNode_[node] : kInvalidIndex;
            whole = whole && strand.lumps[i] < lumpCount;
        }
        if (!whole) {
            strand.lumps[0] = kInvalidIndex;
            return strand;
        }
        if (s < drip.dropNodes.size() && drip.dropNodes[s] < lumpOfNode_.size() &&
            lumpOfNode_[drip.dropNodes[s]] < lumpCount) {
            strand.drop = lumpOfNode_[drip.dropNodes[s]];
        }
        // In a loop it hangs wherever the effect is on. Played once, it sets
        // out at the first span's start.
        const bool hangs = looping && timing.level > 0.0f;
        const f32 since = timing.sinceStart.empty() ? -1.0f : timing.sinceStart.back();
        const f32 speed = std::clamp(drip.speed, 0.01f, 10.0f) * size;
        // Each bone's own frame: its family's bone as the pose has it.
        const u32 family = body.lumps[strand.lumps[0]].family();
        const Matrix44f carried = family < skin.size() ? skin[family] : Matrix44f::identity();
        // A way in the model, said in that frame: a turn undone is its transpose.
        const auto inFrame = [&](const Vector3f& way) {
            Vector3f said{0, 0, 0};
            for (int axis = 0; axis < 3; ++axis) {
                const Vector3f row{carried.data[axis][0], carried.data[axis][1], carried.data[axis][2]};
                (axis == 0 ? said.x : axis == 1 ? said.y : said.z) = row.dot(way) / std::max(row.dot(row), 1e-12f);
            }
            return said;
        };
        const Vector3f site = transform_point(drip.strandAt[s], carried) + rode;
        const f32 width = drip.strandWidth[s];
        const f32 swollen = width * std::clamp(drip.tip, 0.2f, 4.0f);
        const Vector3f down = inFrame(Vector3f{0, 0, -1});
        // The strand is built in its bone's frame, and a bone that scales it
        // scales its length: the height to the ground is the model's, so it
        // is said in the frame's units, that many a unit of the model.
        const f32 said = down.length();
        const f32 perUnit = std::abs(said - 1.0f) < 1e-4f ? 1.0f : said;
        // To the ground, short of it by its swollen end; or a distance.
        const f32 reach = std::max((drip.toGround ? site.z * perUnit - swollen : std::max(drip.length, 0.0f) * size) *
                                       std::clamp(effect.amount, 0.0f, 1.0f),
                                   0.0f);
        // Each strand a little after the one before, by *Stagger*: the
        // largest catchment drips first.
        const f32 late = drip.stagger * (static_cast<f32>(s) / static_cast<f32>(strands)) * reach / speed;
        const f32 started = since < 0.0f ? -1.0f : since - late;
        // Its site rides the surface, and the strand with it.
        const Vector3f sunk =
            (base < pivot_.size() ? drip.strandAt[s] - pivot_[base] : Vector3f{0, 0, 0}) + inFrame(rode);
        // *Sway*: a pendulum of its length at the bind, so its rate is one
        // number whatever the pose; in whole swings a loop, so the loop
        // closes. It does not answer the body's motion: the way down,
        // which is the world's, does.
        const f32 hung = drip.toGround ? drip.strandAt[s].z - swollen : std::max(drip.length, 0.0f) * size;
        const f32 swings = SoftWholeCycles(SoftSwayRate(hung, size), duration, looping);
        const Vector2f swing =
            SoftStrandSwing(drip.sway * timing.level, effect.seed, static_cast<u32>(s), swings, clock.seconds);
        if (drip.pinches() && (hangs || started >= 0.0f)) {
            // *Pinch off*: a cycle, a whole share of a loop. It lets go short
            // of the ground, so its droplet has a fall.
            const f32 period = 1.0f / SoftWholeCycles(1.0f / std::max(drip.pinchEvery, 0.05f), duration, looping);
            const f32 from = drip.toGround ? reach * kSoftPinchReach : reach;
            const f32 turn = hangs ? clock.seconds / period + SoftPinchPhase(effect.seed, static_cast<u32>(s))
                                   : started / period;
            const SoftPinch pinch = SoftPinchAt(turn, period, from / (speed * period), !hangs);
            strand.pose = SoftStrandAt(drip, width, from, size, started, hangs, down, sunk, swing, &pinch);
            strand.share = pinch.length;
            if (strand.drop != kInvalidIndex && pinch.away) {
                // From where its tip was then, swung as it was then.
                const Vector2f then = SoftStrandSwing(drip.sway * timing.level, effect.seed, static_cast<u32>(s),
                                                      swings, clock.seconds - pinch.since);
                strand.droplet = SoftDropletAt(drip, width, std::max(site.z * perUnit - from, 0.0f), size, pinch, down,
                                               sunk + SoftStrandTip(down, from, then));
            }
            strand.acts = true;
            return strand;
        }
        strand.pose = SoftStrandAt(drip, width, reach, size, started, hangs, down, sunk, swing);
        strand.share = reach > 0.0f ? std::clamp(strand.pose.length / reach, 0.0f, 1.0f) : 0.0f;
        strand.acts = hangs || since >= late;
        return strand;
    };
    // *Pull the surface*: the skin under each strand comes down with it, by
    // how much of its length the strand hangs by.
    for (const SoftEffect& effect : body.effects) {
        if (!effect.enabled || (drivingOnly && !effect.drivesJiggle) || effect.kind != SoftEffectKind::Drip ||
            !effect.drip || !(effect.drip->pull > 0.0f) || !stranded(*effect.drip)) {
            continue;
        }
        const SoftDrip& drip = *effect.drip;
        const SoftTiming timing = timingOf(effect);
        const DripCache& near = dripOf(effect);
        for (std::size_t s = 0; s + 1 < near.start.size(); ++s) {
            const Strand strand = strandOf(effect, timing, s, Vector3f{0, 0, 0});
            // In a loop the effect's level eases it; played once, the strand's own growth does.
            const f32 depth = kSoftPullDepth * drip.strandWidth[s] * std::min(drip.pull, 1.0f) * strand.share *
                              (looping ? timing.level : 1.0f);
            if (strand.lumps[0] == kInvalidIndex || !strand.acts || !(depth > 0.0f)) {
                continue;
            }
            for (u32 a = near.start[s]; a < near.start[s + 1]; ++a) {
                const u32 p = near.points[a].first;
                out.goal[p].z -= depth * near.points[a].second * points.points[p].softness;
                out.moved = true;
            }
        }
    }

    // 4. The fit; and the volume the fields asked for, met on the skinned
    // result by a last even change of the lumps' scales (§7).
    if (out.moved) {
        out.fit = rig_.fit(skin, out.goal, out.departures, method);
        if (volume_.encloses) {
            trimVolume(skin, SoftVolumeAt(volume_, out.goal), 1.0f, out.departures);
        }
    }

    // 5. The effects said along the bones scale the lumps themselves, on top.
    for (const SoftEffect& effect : body.effects) {
        if (!effect.enabled || (drivingOnly && !effect.drivesJiggle)) {
            continue;
        }
        const bool inflating = effect.kind == SoftEffectKind::Inflate && effect.inflate &&
                               effect.inflate->along == SoftAlong::Bones;
        const bool bulging = effect.kind == SoftEffectKind::Bulge && effect.bulge &&
                             effect.bulge->along == SoftAlong::Bones;
        if (!inflating && !bulging) {
            continue;
        }
        const SoftTiming timing = timingOf(effect);
        const PlaceCache* place = bulging || effect.inflate->towardPlace ? &placeOf(effect) : nullptr;
        // How much of the effect each lump takes: all of it, or the profile
        // at its pivot.
        std::vector<f32> share(lumpCount, 0.0f);
        for (std::size_t k = 0; k < lumpCount; ++k) {
            if (!rig_.scales(static_cast<u32>(k))) {
                continue;
            }
            share[k] = 1.0f;
            if (place != nullptr) {
                const SoftBulgeShape shape = bulging ? effect.bulge->shape : SoftBulgeShape::Round;
                share[k] = place->at.valid
                               ? Profile(shape, (body.lumps[k].rest - place->at.at).length() / place->radius)
                               : 0.0f;
            }
        }
        f32 factor = 0.0f; // What a lump at full share grows by.
        if (bulging) {
            factor = effect.amount * envelopeOf(effect, timing);
        } else {
            const f32 amount =
                effect.amount * timing.level * PulseOf(*effect.inflate, clock.seconds, duration, looping);
            if (amount == 0.0f) {
                continue;
            }
            // Each point moves away from its lumps' pivots: an exact cubic again.
            rig_.pose(skin, out.departures, out.posed);
            std::vector<Vector3f> away;
            awayFrom(share, away);
            if (volume_.encloses) {
                const f64 now = SoftVolumeAt(volume_, out.posed);
                factor = static_cast<f32>(SolveOffset(OffsetCubic(volume_, out.posed, away),
                                                      now * (1.0 + std::max(amount, -0.9f)), 4.0));
            } else {
                factor = amount;
            }
        }
        if (factor == 0.0f) {
            continue;
        }
        grow(share, factor);
        out.moved = true;
    }

    // A Ripple said along the bones is a wave of the lumps themselves (§8.9). A
    // bone's lump scales about the bone, by the wave's height at it over its
    // own size. A lump of the surface stands in the middle of its patch, where
    // a scale would spread the patch and raise nothing: it moves out by the
    // wave's height, which a bone carries as exactly.
    for (const SoftEffect& effect : body.effects) {
        if (!effect.enabled || (drivingOnly && !effect.drivesJiggle) || effect.kind != SoftEffectKind::Ripple ||
            !effect.ripple || effect.ripple->along != SoftAlong::Bones) {
            continue;
        }
        const RippleCache& from = rippleOf(effect);
        std::vector<f32> since;
        f32 turns = 0.0f, level = 0.0f;
        const SoftWave wave = heightOf(effect, since, turns, level);
        if (!from.valid || wave.height == 0.0f || (since.empty() && level == 0.0f)) {
            continue;
        }
        for (std::size_t k = 0; k < lumpCount; ++k) {
            const SoftLump& lump = body.lumps[k];
            if (!(lumpRadius_[k] > 0.0f)) {
                continue;
            }
            const f32 height = rippleAt(wave, *effect.ripple, since, turns, level, from.lumps[k]);
            if (lump.kind == SoftLumpKind::Surface) {
                // Not a rigid part's, and not one whose points face every way.
                if (!lump.rigid && rig_.moves(static_cast<u32>(k))) {
                    out.departures[k].translation = out.departures[k].translation + lumpOut_[k] * height;
                }
            } else if (rig_.scales(static_cast<u32>(k))) {
                // A wave higher than a bone's points stand from it is no scale
                // of that bone: to half its size in a trough, to twice on a crest.
                out.departures[k].scale = out.departures[k].scale *
                                          (1.0f + std::clamp(height / lumpRadius_[k], -kRippleTrough, kRippleCrest));
            }
        }
        out.moved = true;
    }

    // 8. Limit: a lump's travel inward stops at half the clear space (§5.4).
    keepClear(out.departures);

    // Drip's strands: the last word on their bones. A strand is not the body's
    // to swell or to melt, but it hangs from the body's skin, so it rides
    // wherever the fit put the point nearest its site. Where the effect does
    // not act its strands stay as built, which draws nothing.
    bool surfaced = false;
    for (const SoftEffect& effect : body.effects) {
        if (effect.kind != SoftEffectKind::Drip || !effect.drip || !stranded(*effect.drip)) {
            continue;
        }
        const SoftTiming timing = effect.enabled ? timingOf(effect) : SoftTiming{};
        const DripCache* near = out.moved ? &dripOf(effect) : nullptr;
        if (near != nullptr && !surfaced) {
            rig_.pose(skin, out.departures, out.posed);
            surfaced = true;
        }
        for (std::size_t s = 0; s < effect.drip->strandAt.size(); ++s) {
            Vector3f rode{0, 0, 0};
            if (near != nullptr && s < near->nearest.size() && near->nearest[s] < count) {
                rode = out.posed[near->nearest[s]] - out.carried[near->nearest[s]];
            }
            const Strand strand = strandOf(effect, timing, s, rode);
            if (strand.lumps[0] == kInvalidIndex) {
                continue;
            }
            out.departures[strand.lumps[0]] = strand.pose.base;
            out.departures[strand.lumps[1]] = strand.pose.neck;
            out.departures[strand.lumps[2]] = strand.pose.tip;
            if (strand.drop != kInvalidIndex) {
                out.departures[strand.drop] = strand.droplet;
            }
            out.moved = out.moved || strand.acts;
        }
    }
    // A carrier scaled to nothing (how a part is hidden) shows none of what
    // its lumps do. They are left at rest, so no key of theirs stands there
    // for the part to ease out of as it comes back.
    for (std::size_t k = 0; k < lumpCount && k < out.departures.size(); ++k) {
        const u32 family = body.lumps[k].family();
        if (family < skin.size() && Collapsed(skin[family])) {
            out.departures[k] = Transform{};
        }
    }
    rig_.pose(skin, out.departures, out.posed);
}

} // namespace wem
} // namespace models
} // namespace whiteout
