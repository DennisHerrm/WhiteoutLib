// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_shapes.h>

#include <whiteout/models/wem/skinning/solid.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace whiteout {
namespace models {
namespace wem {

namespace {

constexpr f32 kPi = 3.14159265f;
constexpr f32 kTau = 6.2831853f;
/// A buried point is sent this share short of the surface, by how far inside
/// the blob it was: under it, and in its order.
constexpr f32 kBuried = 0.15f;
/// The steps a point's way to the blob is read at, each side of it: close
/// together near the point, since a blob that hugs is near.
constexpr int kRaySteps = 12;

Vector3f Unit(const Vector3f& v, const Vector3f& fallback) {
    const f32 length = v.length();
    return length > 1e-12f ? v * (1.0f / length) : fallback;
}

/// A value between -1 and 1 for a cell of space and a seed: the same every time.
f32 Hashed(i32 x, i32 y, i32 z, u32 seed) {
    u32 h = seed * 0x9E3779B1u;
    for (const i32 value : {x, y, z}) {
        h ^= static_cast<u32>(value) + 0x9E3779B9u + (h << 6) + (h >> 2);
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
    }
    return static_cast<f32>(h & 0xFFFFFFu) / 8388607.5f - 1.0f;
}

/// `Hashed` made smooth over space: cells of side @p cell, blended between.
f32 Noise(const Vector3f& at, f32 cell, u32 seed) {
    const f32 gx = at.x / cell, gy = at.y / cell, gz = at.z / cell;
    const f32 fx = std::floor(gx), fy = std::floor(gy), fz = std::floor(gz);
    const auto ease = [](f32 t) { return t * t * (3.0f - 2.0f * t); };
    const f32 tx = ease(gx - fx), ty = ease(gy - fy), tz = ease(gz - fz);
    const i32 x = static_cast<i32>(fx), y = static_cast<i32>(fy), z = static_cast<i32>(fz);
    f32 sum = 0.0f;
    for (int corner = 0; corner < 8; ++corner) {
        const int dx = corner & 1, dy = (corner >> 1) & 1, dz = (corner >> 2) & 1;
        sum += Hashed(x + dx, y + dy, z + dz, seed) * (dx ? tx : 1.0f - tx) * (dy ? ty : 1.0f - ty) *
               (dz ? tz : 1.0f - tz);
    }
    return sum;
}

} // namespace

std::vector<f32> SoftLumpRadii(const SoftRig& rig) {
    std::vector<f32> radii;
    if (rig.empty()) {
        return radii;
    }
    const SoftBody& body = rig.body();
    const std::vector<SoftPoint>& points = rig.points().points;
    const std::size_t lumps = body.lumps.size();
    std::vector<f32> far(lumps, 0.0f), held(lumps, 0.0f);
    for (u32 p = 0; p < points.size(); ++p) {
        for (const SoftSlot& slot : rig.slotsOf(p)) {
            if (slot.lump < lumps && slot.weight > 0.0f) {
                const f32 share = points[p].area * slot.weight;
                far[slot.lump] += share * (points[p].at - body.lumps[slot.lump].anchor).length();
                held[slot.lump] += share;
            }
        }
    }
    const f32 size = std::max(rig.points().size, 1e-6f);
    radii.assign(lumps, 0.0f);
    for (std::size_t k = 0; k < lumps; ++k) {
        if (!(held[k] > 0.0f)) {
            continue;
        }
        const SoftLump& lump = body.lumps[k];
        const f32 radius = lump.kind == SoftLumpKind::Surface && lump.depth > 0.0f ? lump.depth : far[k] / held[k];
        radii[k] = std::clamp(radius, 1e-3f * size, size);
    }
    return radii;
}

std::vector<SoftBall> SoftBlobBalls(const SoftRig& rig, std::span<const Matrix44f> skin, f32 roundness) {
    std::vector<SoftBall> balls;
    if (rig.empty()) {
        return balls;
    }
    const SoftBody& body = rig.body();
    const std::vector<SoftPoint>& points = rig.points().points;
    const std::size_t lumps = body.lumps.size();
    const std::vector<f32> radii = SoftLumpRadii(rig);
    // Which way is out of each lump, by its points.
    std::vector<Vector3f> out(lumps, Vector3f{0, 0, 0});
    for (u32 p = 0; p < points.size(); ++p) {
        for (const SoftSlot& slot : rig.slotsOf(p)) {
            if (slot.lump < lumps && slot.weight > 0.0f) {
                out[slot.lump] = out[slot.lump] + points[p].outward * (points[p].area * slot.weight);
            }
        }
    }
    const f32 size = std::max(rig.points().size, 1e-6f);
    const f32 round = std::clamp(roundness, 0.0f, 1.0f);
    for (std::size_t k = 0; k < lumps; ++k) {
        const SoftLump& lump = body.lumps[k];
        if (!(radii[k] > 0.0f)) {
            continue; // A lump no point rides stands for nothing of the body.
        }
        const u32 family = lump.family();
        const Matrix44f carried = family < skin.size() ? skin[family] : Matrix44f::identity();
        SoftBall ball;
        // On the surface its ball lies under it, and touches it; at a bone it
        // is round the bone.
        ball.at = lump.kind == SoftLumpKind::Surface
                      ? transform_point(lump.anchor - Unit(out[k], Vector3f{0, 0, 1}) * radii[k], carried)
                      : transform_point(lump.anchor, carried);
        // From twice its own size, which hugs; through the region's, half way,
        // which rounds the body as a whole; to eight regions, so far that the
        // sum's levels are balls round the body's middle.
        ball.reach = round <= 0.5f ? 2.0f * std::pow(radii[k], 1.0f - 2.0f * round) * std::pow(0.5f * size, 2.0f * round)
                                   : size * std::pow(8.0f, 2.0f * round - 1.0f);
        balls.push_back(ball);
    }
    return balls;
}

f32 SoftBlobField(std::span<const SoftBall> balls, const Vector3f& at, Vector3f* inward) {
    f32 field = 0.0f;
    Vector3f slope{0, 0, 0};
    for (const SoftBall& ball : balls) {
        const Vector3f to = ball.at - at;
        const f32 reach2 = ball.reach * ball.reach;
        const f32 u2 = to.dot(to) / std::max(reach2, 1e-30f);
        if (u2 >= 1.0f) {
            continue;
        }
        // (1 - u^2)^3: smooth to nothing at the reach.
        const f32 left = 1.0f - u2;
        field += left * left * left;
        if (inward != nullptr) {
            slope = slope + to * (6.0f * left * left / std::max(reach2, 1e-30f));
        }
    }
    if (inward != nullptr) {
        *inward = slope;
    }
    return field;
}

SoftBlobReport SoftBlobTo(const SoftRig& rig, const SoftVolume& volume, std::span<const SoftBall> balls,
                           std::vector<Vector3f>& points, f32 amount, f32 swell, bool keepVolume) {
    SoftBlobReport report;
    const std::size_t count = points.size();
    if (rig.empty() || balls.empty() || count != rig.pointCount() || amount == 0.0f) {
        return report;
    }
    const std::vector<SoftPoint>& soft = rig.points().points;
    const f32 size = std::max(rig.points().size, 1e-6f);
    // Each point's way in: along the field's gradient where a ball reaches it,
    // and to the nearest ball where none does. The field is read along it once,
    // at steps close together near the point; a level is then a place on it.
    constexpr int kSamples = 2 * kRaySteps + 1;
    const f32 longest = 0.75f * size;
    std::array<f32, kSamples> offsets{};
    for (int j = 0; j < kSamples; ++j) {
        const f32 t = static_cast<f32>(j - kRaySteps) / static_cast<f32>(kRaySteps);
        offsets[static_cast<std::size_t>(j)] = longest * t * std::abs(t);
    }
    std::vector<Vector3f> way(count);
    std::vector<f32> table(count * kSamples);
    std::vector<f32> here(count);
    std::vector<SoftBall> near;
    f64 mean = 0.0, area = 0.0;
    for (std::size_t p = 0; p < count; ++p) {
        // The balls that can reach any place on this point's way.
        near.clear();
        const SoftBall* nearest = nullptr;
        f32 least = std::numeric_limits<f32>::max();
        for (const SoftBall& ball : balls) {
            const f32 d = (ball.at - points[p]).length();
            if (d < ball.reach + longest) {
                near.push_back(ball);
            }
            if (d - ball.reach < least) {
                least = d - ball.reach;
                nearest = &ball;
            }
        }
        Vector3f inward{0, 0, 0};
        here[p] = SoftBlobField(near, points[p], &inward);
        way[p] = Unit(inward, Unit(nearest->at - points[p], Vector3f{0, 0, -1}));
        for (int j = 0; j < kSamples; ++j) {
            table[p * kSamples + static_cast<std::size_t>(j)] =
                j == kRaySteps ? here[p] : SoftBlobField(near, points[p] + way[p] * offsets[static_cast<std::size_t>(j)]);
        }
        mean += static_cast<f64>(here[p]) * soft[p].area;
        area += soft[p].area;
    }
    // Where on its way point p meets the field's value @p wanted: the crossing
    // nearest the point, inward when the field there is under it and outward
    // when over.
    const auto meets = [&](std::size_t p, f32 wanted) {
        const f32* row = table.data() + p * kSamples;
        const int step = row[kRaySteps] < wanted ? 1 : -1;
        bool varies = false;
        for (int j = kRaySteps; j + step >= 0 && j + step < kSamples; j += step) {
            const f32 a = row[j], b = row[j + step];
            varies = varies || a != b;
            if ((a - wanted) * (b - wanted) <= 0.0f && a != b) {
                const f32 t = (wanted - a) / (b - a);
                return offsets[static_cast<std::size_t>(j)] +
                       (offsets[static_cast<std::size_t>(j + step)] - offsets[static_cast<std::size_t>(j)]) * t;
            }
        }
        // Farther than its way was read: as far as that. And where no ball
        // reaches its way at all, it stays.
        return varies ? offsets[static_cast<std::size_t>(step > 0 ? kSamples - 1 : 0)] : 0.0f;
    };
    std::vector<Vector3f> moved(count);
    const auto place = [&](f32 level) {
        for (std::size_t p = 0; p < count; ++p) {
            // Inside the blob already: most of the way out, in its order.
            const f32 wanted = here[p] > level ? level + kBuried * (here[p] - level) : level;
            const f32 share = amount * soft[p].softness;
            moved[p] = points[p] + way[p] * (meets(p, wanted) * share);
        }
    };
    // With nothing to keep, the level the body's own surface stands at on the whole.
    f32 level = area > 0.0 ? static_cast<f32>(mean / area) : 0.0f;
    if (keepVolume && volume.encloses) {
        const f64 wanted = SoftVolumeAt(volume, points) * (1.0 + static_cast<f64>(std::max(swell, 0.0f) * amount));
        // A higher level is a smaller blob. At the field's most among the
        // points every one of them comes in; under its least, every one goes out.
        f32 low = here.front(), high = here.front();
        for (const f32 value : here) {
            low = std::min(low, value);
            high = std::max(high, value);
        }
        low -= high - low;
        for (int pass = 0; pass < 28; ++pass) {
            level = 0.5f * (low + high);
            place(level);
            (SoftVolumeAt(volume, moved) > wanted ? low : high) = level;
        }
        level = 0.5f * (low + high);
    }
    place(level);
    report.level = level;
    // What fuses: how far out of the blob the field falls to half its level,
    // on each point's own way, twice.
    f64 gap = 0.0, read = 0.0;
    for (std::size_t p = 0; p < count; ++p) {
        const f32 surface = meets(p, level), half = meets(p, 0.5f * level);
        // The half level is the farther out; past what was read of the way,
        // as far as was read.
        if (surface > half) {
            gap += 2.0 * static_cast<f64>(surface - half) * soft[p].area;
            read += soft[p].area;
        }
    }
    report.fuses = read > 0.0 ? static_cast<f32>(gap / read) : 0.0f;
    f64 buried = 0.0;
    for (std::size_t p = 0; p < count; ++p) {
        buried += here[p] > level ? soft[p].area : 0.0f;
    }
    report.buried = area > 0.0 ? static_cast<f32>(buried / area) : 0.0f;
    points.swap(moved);
    return report;
}

SoftColumns SoftMeltColumns(const SoftRig& rig) {
    SoftColumns out;
    if (rig.empty()) {
        return out;
    }
    const SoftPoints& points = rig.points();
    const std::size_t count = points.points.size();
    const f32 tall = std::max(points.high.z - points.low.z, 1e-6f);
    out.share.resize(count);
    out.thickness.assign(count, tall);
    const skinning::SolidGrid& grid = points.grid;
    const bool solid = grid.count() != 0 && points.solid.size() == grid.count();
    for (std::size_t p = 0; p < count; ++p) {
        const SoftPoint& point = points.points[p];
        // With no solid to read, by its height in the whole body.
        out.share[p] = std::clamp((point.at.z - points.low.z) / tall, 0.0f, 1.0f);
        if (!solid) {
            continue;
        }
        // The column under and over it, read a little inside the skin.
        const Vector3f inside = point.at - point.outward * (0.5f * grid.cell);
        const std::array<i32, 3> cell = grid.cellOf(inside);
        if (cell[0] < 0 || cell[1] < 0 || cell[0] >= grid.size[0] || cell[1] >= grid.size[1]) {
            continue;
        }
        i32 low = -1, high = -1;
        for (i32 z = 0; z < grid.size[2]; ++z) {
            if (points.solid[grid.index(cell[0], cell[1], z)] != 0) {
                low = low < 0 ? z : low;
                high = z;
            }
        }
        if (low < 0) {
            continue;
        }
        const f32 bottom = grid.origin.z + static_cast<f32>(low) * grid.cell;
        const f32 top = grid.origin.z + static_cast<f32>(high + 1) * grid.cell;
        out.thickness[p] = std::max(top - bottom, grid.cell);
        out.share[p] = std::clamp((point.at.z - bottom) / out.thickness[p], 0.0f, 1.0f);
    }
    return out;
}

namespace {

/// What a melt reads of the body before it moves it.
struct Footprint {
    Vector3f centre{0, 0, 0};
    f32 radius = 0.0f; ///< Nine tenths of the body's skin is within it.
    f32 farthest = 0.0f;
    f32 height = 0.0f;
};

Footprint FootprintOf(const SoftRig& rig, std::span<const Vector3f> points, f32 floor) {
    Footprint out;
    const std::vector<SoftPoint>& soft = rig.points().points;
    f64 x = 0.0, y = 0.0, area = 0.0;
    for (std::size_t p = 0; p < points.size() && p < soft.size(); ++p) {
        const f64 a = std::max(soft[p].area, 1e-12f);
        x += points[p].x * a;
        y += points[p].y * a;
        area += a;
    }
    if (!(area > 0.0)) {
        return out;
    }
    out.centre = Vector3f{static_cast<f32>(x / area), static_cast<f32>(y / area), floor};
    std::vector<std::pair<f32, f32>> out_by; // How far out, and how much skin.
    for (std::size_t p = 0; p < points.size() && p < soft.size(); ++p) {
        const f32 dx = points[p].x - out.centre.x, dy = points[p].y - out.centre.y;
        const f32 d = std::sqrt(dx * dx + dy * dy);
        out_by.emplace_back(d, std::max(soft[p].area, 1e-12f));
        out.farthest = std::max(out.farthest, d);
        out.height = std::max(out.height, points[p].z - floor);
    }
    // An arm held out is not the footprint: the distance nine tenths of the
    // skin is within.
    std::sort(out_by.begin(), out_by.end());
    f64 within = 0.0;
    out.radius = out.farthest;
    for (const auto& [d, a] : out_by) {
        within += a;
        if (within >= 0.9 * area) {
            out.radius = d;
            break;
        }
    }
    return out;
}

/// Huppert's pool: flat on top, its rim steep and round.
f32 PoolHeight(f32 depth, f32 r, f32 radius) {
    const f32 u = std::clamp(r / std::max(radius, 1e-9f), 0.0f, 1.0f);
    return depth * std::cbrt(1.0f - u * u);
}

/// A ramp from nothing at @p line to all a @p band above it, summed up to @p f.
f32 RampTo(f32 f, f32 line, f32 band) {
    if (f <= line) {
        return 0.0f;
    }
    if (f < line + band) {
        return (f - line) * (f - line) / (2.0f * band);
    }
    return f - line - 0.5f * band;
}

/// How much of a column, from its foot up to the share @p f of it, has melted
/// when the line stands at @p line: nothing below the line, all of it a
/// @p band above.
f32 MeltedBelow(f32 f, f32 line, f32 band) {
    return RampTo(f, line, band) - RampTo(0.0f, line, band);
}

/// The body melted to @p melted, before softness: every point of it.
void MeltAll(const SoftRig& rig, const SoftColumns& columns, std::span<const Vector3f> points, f32 floor,
             const SoftMelt& settings, const Footprint& foot, f32 radius, f32 depth, f32 melted, u32 seed,
             std::vector<Vector3f>& moved) {
    const std::vector<SoftPoint>& soft = rig.points().points;
    const std::size_t count = points.size();
    const f32 wide = std::max(foot.farthest, 1e-6f);
    const f32 tall = std::max(foot.height, 1e-6f);
    const f32 size = std::max(rig.points().size, 1e-6f);
    const f32 lean = 2.0f * (std::clamp(settings.solidity, 0.0f, 1.0f) - 0.5f);
    const f32 uneven = std::clamp(settings.uneven, 0.0f, 1.0f);
    constexpr f32 kBand = 0.25f;
    moved.resize(count);
    for (std::size_t p = 0; p < count; ++p) {
        const f32 dx = points[p].x - foot.centre.x, dy = points[p].y - foot.centre.y;
        const f32 out = std::sqrt(dx * dx + dy * dy);
        const f32 rim = std::clamp(out / wide, 0.0f, 1.0f);
        // *Solidity*: ice's middle starts late and its rim ends early; jelly's
        // the other way. Both start with the span and end with it.
        const f32 late = 0.4f * (lean >= 0.0f ? lean * (1.0f - rim) : -lean * rim);
        const f32 early = 0.4f * (lean >= 0.0f ? lean * rim : -lean * (1.0f - rim));
        f32 own = std::clamp((melted - late) / std::max(1.0f - late - early, 1e-3f), 0.0f, 1.0f);
        // *Uneven*: a little sooner or later by the seed, and not at the ends.
        own = std::clamp(own + 0.6f * uneven * Noise(soft[p].at, 0.25f * size, seed) * own * (1.0f - own), 0.0f, 1.0f);
        const f32 eased = own * own * (3.0f - 2.0f * own);

        const f32 share = p < columns.share.size() ? columns.share[p] : 0.0f;
        const f32 thick = std::max(p < columns.thickness.size() ? columns.thickness[p] : tall, 1e-6f);
        const f32 r = radius * (out / wide);
        // What its column ends as thick, over what it is.
        const f32 thin = PoolHeight(depth, r, radius) / thick;
        // How much of the column under the point has melted. A layer keeps its
        // place over the ones below it, so the column stays in its order.
        f32 gone = 0.0f;
        if (settings.order == SoftMeltOrder::Top) {
            gone = MeltedBelow(share, 1.0f - own * (1.0f + kBand), kBand);
        } else if (settings.order == SoftMeltOrder::Bottom) {
            gone = share - MeltedBelow(share, own * (1.0f + kBand) - kBand, kBand);
        } else {
            gone = share * own;
        }
        // Its column's foot comes down to the floor, and what is over the
        // point in it thins.
        const f32 foot_ = points[p].z - floor - thick * share;
        const f32 up = foot_ * (1.0f - eased) + thick * (share - (1.0f - thin) * gone);
        // What is low runs out first.
        const f32 high = std::clamp(up / tall, 0.0f, 1.0f);
        const f32 run = std::pow(eased, 1.0f + 2.0f * high);
        const f32 spread = out > 1e-9f ? 1.0f + (r / out - 1.0f) * run : 1.0f;
        moved[p] = Vector3f{foot.centre.x + dx * spread, foot.centre.y + dy * spread, floor + up};
    }
}

/// @p moved's heights above @p floor scaled to hold @p before: the volume
/// scales as the height does.
f32 KeepVolume(const SoftVolume& volume, f64 before, f32 floor, std::vector<Vector3f>& moved) {
    if (!volume.encloses || !(std::abs(before) > 1e-12)) {
        return 1.0f;
    }
    const f64 now = SoftVolumeAt(volume, moved);
    const f32 by = now * before > 1e-12 ? std::clamp(static_cast<f32>(before / now), 0.1f, 10.0f) : 1.0f;
    for (Vector3f& at : moved) {
        at.z = floor + (at.z - floor) * by;
    }
    return by;
}

} // namespace

SoftPool SoftMeltPool(const SoftRig& rig, const SoftVolume& volume, const SoftColumns& columns,
                      std::span<const Vector3f> points, f32 floor, const SoftMelt& settings) {
    SoftPool pool;
    if (rig.empty() || points.size() != rig.pointCount()) {
        return pool;
    }
    const Footprint foot = FootprintOf(rig, points, floor);
    pool.centre = foot.centre;
    pool.height = foot.height;
    pool.radius = std::clamp(settings.puddle, 1.0f, 8.0f) * std::max(foot.radius, 1e-6f);
    // A pool holds three quarters of pi R^2 h0: where the body ends from.
    // A patch that encloses nothing takes a twentieth of its height.
    const f64 held = volume.encloses ? SoftVolumeAt(volume, points) : 0.0;
    pool.depth = std::abs(held) > 0.0
                     ? static_cast<f32>(std::abs(held) / (0.75 * static_cast<f64>(kPi) * pool.radius * pool.radius))
                     : 0.05f * foot.height;
    // **The depth is read:** the body's own skin, laid into that pool, holds
    // less than the pool does, and its height is what makes up the volume.
    std::vector<Vector3f> ended;
    SoftMelt forward = settings;
    forward.uneven = 0.0f;
    MeltAll(rig, columns, points, floor, forward, foot, pool.radius, pool.depth, 1.0f, 0, ended);
    pool.depth *= KeepVolume(volume, held, floor, ended);
    f32 top = 0.0f;
    for (const Vector3f& at : ended) {
        top = std::max(top, at.z - floor);
    }
    pool.depth = top > 0.0f ? top : pool.depth;
    return pool;
}

SoftPool SoftMeltTo(const SoftRig& rig, const SoftVolume& volume, const SoftColumns& columns,
                    std::vector<Vector3f>& points, f32 floor, const SoftMelt& settings, f32 progress, u32 seed) {
    const SoftPool pool = SoftMeltPool(rig, volume, columns, points, floor, settings);
    const std::size_t count = points.size();
    const f32 melted = std::clamp(settings.backward ? 1.0f - progress : progress, 0.0f, 1.0f);
    if (rig.empty() || count != rig.pointCount() || melted <= 0.0f || !(pool.radius > 0.0f)) {
        return pool;
    }
    const std::vector<SoftPoint>& soft = rig.points().points;
    const Footprint foot = FootprintOf(rig, points, floor);
    const f64 before = volume.encloses ? SoftVolumeAt(volume, points) : 0.0;
    const f32 depth = std::abs(before) > 0.0
                          ? static_cast<f32>(std::abs(before) / (0.75 * static_cast<f64>(kPi) * pool.radius * pool.radius))
                          : 0.05f * foot.height;
    std::vector<Vector3f> moved;
    MeltAll(rig, columns, points, floor, settings, foot, pool.radius, depth, melted, seed, moved);
    // The volume kept at every progress.
    KeepVolume(volume, before, floor, moved);
    for (std::size_t p = 0; p < count; ++p) {
        points[p] = points[p] + (moved[p] - points[p]) * soft[p].softness;
    }
    return pool;
}

SoftWave SoftWaveOf(const SoftRipple& settings, f32 amount, f32 size) {
    SoftWave wave;
    const f32 region = std::max(size, 1e-6f);
    wave.height = amount * region;
    wave.length = std::clamp(settings.wavelength, 0.05f, 8.0f) * region;
    wave.speed = std::clamp(settings.speed, 0.01f, 50.0f) * region;
    wave.envelope = settings.water ? std::clamp(settings.envelope, 0.1f, 4.0f) : 1.0f;
    // By the region travelled: at 1, a third of the height is left a region out.
    wave.fadeDistance = std::clamp(settings.fadeDistance, 0.0f, 1.0f) * 1.1f / region;
    wave.fadeTime = std::max(settings.fadeTime, 0.0f);
    return wave;
}

f32 SoftSwellHeight(const SoftWave& wave, f32 distance, f32 since) {
    if (since < 0.0f || !std::isfinite(distance)) {
        return 0.0f;
    }
    // The swell is one length wide either side of its middle, which moves at
    // its own speed; the crests inside it move at theirs.
    const f32 from = distance - wave.envelope * wave.speed * since;
    if (std::abs(from) >= wave.length) {
        return 0.0f;
    }
    const f32 window = std::cos(0.5f * kPi * from / wave.length);
    const f32 crest = std::cos(kTau * (distance - wave.speed * since) / wave.length);
    f32 height = wave.height * window * window * crest * std::exp(-wave.fadeDistance * distance);
    if (wave.fadeTime > 0.0f) {
        height *= std::exp(-1.1f * since / wave.fadeTime);
    }
    return height;
}

f32 SoftTrainHeight(const SoftWave& wave, f32 distance, f32 turns) {
    if (!std::isfinite(distance)) {
        return 0.0f;
    }
    return wave.height * std::sin(kTau * (turns - distance / wave.length)) * std::exp(-wave.fadeDistance * distance);
}

std::vector<f32> SoftRippleDistances(const SoftRig& rig, const SoftPlaceAt& place, SoftThrough through) {
    const f32 never = std::numeric_limits<f32>::infinity();
    std::vector<f32> distance(rig.pointCount(), never);
    if (rig.empty() || !place.valid) {
        return distance;
    }
    const SoftPoints& points = rig.points();
    if (through == SoftThrough::Body && points.grid.count() != 0 && points.solid.size() == points.grid.count()) {
        // Through the solid: from the place, a cell at a time.
        const std::array<i32, 3> cell = points.grid.cellOf(place.at);
        if (points.grid.contains(cell[0], cell[1], cell[2])) {
            const skinning::SolidSeed seed{points.grid.index(cell[0], cell[1], cell[2]), 0.0f};
            const skinning::SolidWalk walk = skinning::WalkSolid(points.grid, points.solid, {&seed, 1});
            for (u32 p = 0; p < distance.size(); ++p) {
                f32 read = 0.0f;
                // A point is on the solid's skin: read a little inside it.
                const Vector3f at = points.points[p].at - points.points[p].outward * (0.5f * points.grid.cell);
                if (skinning::ReadSolidWalk(points.grid, walk, at, read) ||
                    skinning::ReadSolidWalk(points.grid, walk, points.points[p].at, read)) {
                    distance[p] = read;
                }
            }
            return distance;
        }
    }
    std::vector<std::pair<u32, f32>> seeds;
    for (const u32 point : place.points) {
        if (point < distance.size()) {
            seeds.push_back({point, (points.points[point].at - place.at).length()});
        }
    }
    // A walk's cut-off is what it leaves a point at that it did not reach.
    const f32 cutoff = 1e30f;
    distance = SurfaceDistances(rig, seeds, cutoff);
    for (f32& d : distance) {
        d = d >= cutoff ? never : d;
    }
    return distance;
}

std::vector<f32> SoftLumpDistances(const SoftRig& rig, std::span<const f32> distances) {
    const f32 never = std::numeric_limits<f32>::infinity();
    const std::size_t lumps = rig.lumpCount();
    std::vector<f32> sum(lumps, 0.0f), held(lumps, 0.0f);
    for (u32 p = 0; p < rig.pointCount() && p < distances.size(); ++p) {
        if (!std::isfinite(distances[p])) {
            continue;
        }
        for (const SoftSlot& slot : rig.slotsOf(p)) {
            if (slot.lump < lumps && slot.weight > 0.0f) {
                const f32 share = std::max(rig.points().points[p].area, 1e-12f) * slot.weight;
                sum[slot.lump] += share * distances[p];
                held[slot.lump] += share;
            }
        }
    }
    std::vector<f32> out(lumps, never);
    for (std::size_t k = 0; k < lumps; ++k) {
        if (held[k] > 0.0f) {
            out[k] = sum[k] / held[k];
        }
    }
    return out;
}

} // namespace wem
} // namespace models
} // namespace whiteout
