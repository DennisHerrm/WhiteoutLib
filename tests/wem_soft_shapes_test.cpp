// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

// Blobify, Inflate's Round, Melt and Ripple (EDIT_MODE_SOFTBODY_PLAN.md S6; the
// gate G-S2 for these).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <whiteout/models/wem/anim/animator.h>
#include <whiteout/models/wem/physics/soft_fields.h>
#include <whiteout/models/wem/physics/soft_fit.h>
#include <whiteout/models/wem/physics/soft_shapes.h>
#include <whiteout/models/wem/physics/soft_step.h>

#include "wem_soft_figures.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace whiteout;
using namespace whiteout::models::wem;
using namespace softfig;
using Catch::Approx;

namespace {

std::vector<Matrix44f> Bind(const Model& model) {
    return std::vector<Matrix44f>(model.nodes.size(), Matrix44f::identity());
}

std::vector<Matrix44f> SkinAt(const Document& document, u32 clip, f32 seconds) {
    const Animator animator(document, 0);
    Mix mix;
    mix.plays.push_back(Play{clip, seconds, 1.0f, false});
    Pose pose;
    animator.evaluate(mix, pose);
    return pose.skinning;
}

SoftEffect& AddEffect(Figure& f, SoftEffectKind kind) {
    SoftBody& body = f.model().physics.soft[0];
    SoftEffect effect = DefaultSoftEffect(kind);
    effect.id = body.nextEffect++;
    effect.everywhere = true;
    body.effects.push_back(effect);
    return body.effects.back();
}

/// The body stepped at @p seconds of @p clip, or at the bind for no clip.
SoftStepResult StepAt(const Document& document, u32 clip, f32 seconds) {
    const SoftStepper stepper(document, 0, document.models[0].physics.soft[0].id);
    const SoftPoseAt poseAt = [&](f32 at, std::vector<Matrix44f>& skin) { skin = SkinAt(document, clip, at); };
    SoftStepResult out;
    stepper.step(clip == kInvalidIndex ? Bind(document.models[0]) : SkinAt(document, clip, seconds),
                 SoftClock{clip, seconds}, clip == kInvalidIndex ? SoftPoseAt{} : poseAt, out);
    return out;
}

/// The closed tube, soft throughout.
Figure MakeBody(u32 detail = 12) {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    SoftenAll(f, detail, true);
    return f;
}

/// The face of mesh 0 whose middle is nearest @p at.
u32 FaceNear(const Mesh& mesh, const Vector3f& at) {
    const geom::FaceSet& faces = mesh.faceSet();
    const std::span<const Vector3f> positions =
        mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    u32 best = 0;
    f32 least = 1e30f;
    std::size_t corner = 0;
    for (u32 face = 0; face < faces.faceCount(); ++face) {
        Vector3f middle{0, 0, 0};
        for (u32 k = 0; k < faces.faceValence[face]; ++k) {
            middle = middle + positions[faces.cornerVertex[corner + k]];
        }
        middle = middle * (1.0f / static_cast<f32>(faces.faceValence[face]));
        if ((middle - at).length() < least) {
            least = (middle - at).length();
            best = face;
        }
        corner += faces.faceValence[face];
    }
    return best;
}

} // namespace

TEST_CASE("G-S2 a blob's field is a sum of balls, and its way in is the field's own slope", "[wem][soft][shapes]") {
    const SoftBall one[] = {{Vector3f{1, 2, 3}, 2.0f}};
    CHECK(SoftBlobField(one, Vector3f{1, 2, 3}) == Approx(1.0f));
    CHECK(SoftBlobField(one, Vector3f{3, 2, 3}) == 0.0f);
    CHECK(SoftBlobField(one, Vector3f{9, 9, 9}) == 0.0f);
    const SoftBall two[] = {{Vector3f{0, 0, 0}, 2.0f}, {Vector3f{1.5f, 0, 0}, 1.5f}};
    const Vector3f at{0.6f, 0.3f, -0.2f};
    Vector3f inward{0, 0, 0};
    const f32 here = SoftBlobField(two, at, &inward);
    CHECK(here == Approx(SoftBlobField(two, at)));
    const f32 h = 1e-3f;
    for (int axis = 0; axis < 3; ++axis) {
        Vector3f step{0, 0, 0};
        (axis == 0 ? step.x : axis == 1 ? step.y : step.z) = h;
        const f32 slope = (SoftBlobField(two, at + step) - SoftBlobField(two, at - step)) / (2.0f * h);
        CHECK((axis == 0 ? inward.x : axis == 1 ? inward.y : inward.z) == Approx(slope).margin(2e-3));
    }
}

TEST_CASE("G-S2 Blobify moves the body onto its blob and keeps the volume", "[wem][soft][shapes]") {
    Figure f = MakeBody();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    REQUIRE(volume.encloses);
    const std::vector<Matrix44f> bind = Bind(f.model());
    std::vector<Vector3f> rest;
    rig.carry(bind, rest);
    const f64 before = SoftVolumeAt(volume, rest);
    const f32 size = rig.points().size;

    // A ball a lump, each no larger than the body.
    const std::vector<SoftBall> hugging = SoftBlobBalls(rig, bind, 0.0f);
    const std::vector<SoftBall> round = SoftBlobBalls(rig, bind, 1.0f);
    REQUIRE(hugging.size() == round.size());
    REQUIRE(!hugging.empty());
    for (std::size_t k = 0; k < hugging.size(); ++k) {
        CHECK(hugging[k].reach > 0.0f);
        CHECK(hugging[k].reach <= 2.0f * size + 1e-4f);
        CHECK(round[k].reach == Approx(8.0f * size));
    }

    // Nothing asked, nothing moved: to the bit.
    std::vector<Vector3f> same = rest;
    SoftBlobTo(rig, volume, hugging, same, 0.0f, 0.0f, true);
    CHECK(std::memcmp(same.data(), rest.data(), rest.size() * sizeof(Vector3f)) == 0);

    for (const f32 roundness : {0.0f, 0.4f, 1.0f}) {
        const std::vector<SoftBall> balls = SoftBlobBalls(rig, bind, roundness);
        std::vector<Vector3f> blobbed = rest;
        const SoftBlobReport report = SoftBlobTo(rig, volume, balls, blobbed, 1.0f, 0.0f, true);
        INFO("roundness " << roundness);
        CHECK(SoftVolumeAt(volume, blobbed) == Approx(before).epsilon(0.01));
        CHECK(report.level > 0.0f);
        // A point that was outside the level is on it now.
        f32 off = 0.0f;
        for (std::size_t p = 0; p < blobbed.size(); ++p) {
            if (SoftBlobField(balls, rest[p]) <= report.level) {
                off = std::max(off, std::abs(SoftBlobField(balls, blobbed[p]) - report.level) / report.level);
            }
        }
        CHECK(off < 0.05f);
        // With a swell, that much more volume.
        std::vector<Vector3f> swollen = rest;
        SoftBlobTo(rig, volume, balls, swollen, 1.0f, 0.2f, true);
        CHECK(SoftVolumeAt(volume, swollen) == Approx(before * 1.2).epsilon(0.01));
        // Half way is half way.
        std::vector<Vector3f> half = rest;
        SoftBlobTo(rig, volume, balls, half, 0.5f, 0.0f, true);
        CHECK(SoftVolumeAt(volume, half) == Approx(before).epsilon(0.01));
    }

    // Rounder, the body is nearer a ball: the box round it is nearer a cube.
    const auto longer = [&](f32 roundness) {
        std::vector<Vector3f> blobbed = rest;
        SoftBlobTo(rig, volume, SoftBlobBalls(rig, bind, roundness), blobbed, 1.0f, 0.0f, true);
        Vector3f low = blobbed.front(), high = blobbed.front();
        for (const Vector3f& at : blobbed) {
            low = Vector3f{std::min(low.x, at.x), std::min(low.y, at.y), std::min(low.z, at.z)};
            high = Vector3f{std::max(high.x, at.x), std::max(high.y, at.y), std::max(high.z, at.z)};
        }
        const Vector3f side = high - low;
        return std::max({side.x, side.y, side.z}) / std::min({side.x, side.y, side.z});
    };
    const f32 hugs = longer(0.0f), half = longer(0.5f), rounds = longer(1.0f);
    WARN("Blobify: the box's longest side over its shortest, " << hugs << " hugging, " << half << " half way, "
                                                              << rounds << " round");
    CHECK(hugs > 1.7f);
    CHECK(half < hugs);
    CHECK(rounds < half);
    CHECK(rounds < 1.25f);
}

TEST_CASE("G-S2 a blob reads how near two parts come before it joins them", "[wem][soft][shapes]") {
    Figure f = MakeBody();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const std::vector<Matrix44f> bind = Bind(f.model());
    std::vector<Vector3f> rest;
    rig.carry(bind, rest);
    const std::vector<SoftPoint>& points = rig.points().points;
    f32 last = 0.0f;
    for (const f32 roundness : {0.0f, 0.2f, 0.4f}) {
        const std::vector<SoftBall> balls = SoftBlobBalls(rig, bind, roundness);
        std::vector<Vector3f> blobbed = rest;
        const SoftBlobReport report = SoftBlobTo(rig, volume, balls, blobbed, 1.0f, 0.0f, true);
        INFO("roundness " << roundness);
        REQUIRE(report.fuses > 0.0f);
        // Half that far out of the blob, the field is half its level: over the area.
        f64 sum = 0.0, area = 0.0;
        u32 side = 0;
        for (u32 p = 0; p < blobbed.size(); ++p) {
            Vector3f inward{0, 0, 0};
            SoftBlobField(balls, blobbed[p], &inward);
            if (inward.length() < 1e-9f) {
                continue;
            }
            const Vector3f out = inward * (-1.0f / inward.length());
            sum += SoftBlobField(balls, blobbed[p] + out * (0.5f * report.fuses)) / report.level * points[p].area;
            area += points[p].area;
            // The point farthest along +X, half way up the tube.
            if (std::abs(blobbed[p].z - 3.0f) < 0.4f && blobbed[p].x > blobbed[side].x) {
                side = p;
            }
        }
        REQUIRE(area > 0.0);
        CHECK(sum / area == Approx(0.5).margin(0.08));
        // Two of the body side by side: nearer than that their blobs are one
        // across the middle, and farther they are two.
        const auto between = [&](f32 gap) {
            std::vector<SoftBall> both = balls;
            const Vector3f apart{2.0f * blobbed[side].x + gap, 0, 0};
            for (const SoftBall& ball : balls) {
                both.push_back(SoftBall{ball.at + apart, ball.reach});
            }
            return SoftBlobField(both, Vector3f{blobbed[side].x + 0.5f * gap, blobbed[side].y, blobbed[side].z});
        };
        WARN("roundness " << roundness << ": parts closer than " << report.fuses / rig.points().size
                          << " of the size join; between two bodies " << between(0.6f * report.fuses) / report.level
                          << " of the level at 0.6 of it, " << between(1.5f * report.fuses) / report.level
                          << " at 1.5");
        CHECK(between(0.6f * report.fuses) > report.level);
        CHECK(between(1.5f * report.fuses) < report.level);
        // Rounder, parts farther apart join.
        CHECK(report.fuses > last);
        last = report.fuses;
    }
}

TEST_CASE("G-S2 a layer inside the blob stays under it, in its order", "[wem][soft][shapes]") {
    // A box of side 1 inside a box of side 2 turned inside out: the inner one
    // is buried in the outer one's blob.
    Figure f;
    f.document.profiles = {ProfileId::Wc3Reforged};
    f.document.defaultProfile = ProfileId::Wc3Reforged;
    Model& model = f.document.models.emplace_back();
    f.bones.push_back(AddNode(model.nodes, "root", kInvalidNode, {0, 0, 0}));
    geom::MeshBuilder builder = StartMesh("layers");
    AddBox(builder, {-1, -1, -1}, 2.0f, {{0, 1.0f}}, /*insideOut=*/true);
    AddBox(builder, {-0.5f, -0.5f, -0.5f}, 1.0f, {{0, 1.0f}});
    model.meshes.push_back(builder.build().mesh);
    SoftenAll(f, 8, true);
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const std::vector<Matrix44f> bind = Bind(f.model());
    std::vector<Vector3f> rest;
    rig.carry(bind, rest);
    std::vector<Vector3f> blobbed = rest;
    const SoftBlobReport report = SoftBlobTo(rig, volume, SoftBlobBalls(rig, bind, 0.7f), blobbed, 1.0f, 0.0f, true);
    CHECK(report.buried > 0.0f);
    f32 inner = 0.0f, outer = 1e30f;
    for (std::size_t p = 0; p < rest.size(); ++p) {
        const bool inside = std::abs(rest[p].x) < 0.75f;
        (inside ? inner : outer) = inside ? std::max(inner, blobbed[p].length()) : std::min(outer, blobbed[p].length());
    }
    // No point of the inner layer has come out through the outer one.
    CHECK(inner < outer);
    for (const Vector3f& at : blobbed) {
        CHECK(std::isfinite(at.length()));
    }
}

TEST_CASE("G-S2 Blobify and Inflate's Round through a step", "[wem][soft][shapes]") {
    Figure f = MakeBody();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    {
        SoftEffect& blobify = AddEffect(f, SoftEffectKind::Blobify);
        blobify.amount = 1.0f;
        blobify.blobify->roundness = 0.8f;
        blobify.blobify->swell = 0.0f;
    }
    const SoftStepResult blobbed = StepAt(f.document, kInvalidIndex, 0.0f);
    REQUIRE(blobbed.moved);
    // The step says what its blob fuses, for the effect it is of.
    REQUIRE(blobbed.fuses.size() == 1u);
    CHECK(blobbed.fuses[0].first == f.model().physics.soft[0].effects[0].id);
    CHECK(blobbed.fuses[0].second > 0.0f);
    CHECK(SoftVolumeAt(volume, blobbed.goal) == Approx(volume.rest).epsilon(0.01));
    f32 moved = 0.0f;
    for (std::size_t p = 0; p < blobbed.goal.size(); ++p) {
        moved = std::max(moved, (blobbed.goal[p] - blobbed.carried[p]).length());
    }
    CHECK(moved > 0.05f * rig.points().size);
    // At no Amount it is the model.
    f.model().physics.soft[0].effects[0].amount = 0.0f;
    CHECK(!StepAt(f.document, kInvalidIndex, 0.0f).moved);

    // Inflate, round: the volume is still the one asked for, and no point
    // goes in under its own face.
    f.model().physics.soft[0].effects.clear();
    SoftEffect& inflate = AddEffect(f, SoftEffectKind::Inflate);
    inflate.amount = 0.3f;
    inflate.inflate->along = SoftAlong::Surface;
    inflate.inflate->round = 0.0f;
    const SoftStepResult creased = StepAt(f.document, kInvalidIndex, 0.0f);
    f.model().physics.soft[0].effects[0].inflate->round = 1.0f;
    const SoftStepResult rounded = StepAt(f.document, kInvalidIndex, 0.0f);
    CHECK(SoftVolumeAt(volume, creased.goal) == Approx(volume.rest * 1.3).epsilon(1e-3));
    CHECK(SoftVolumeAt(volume, rounded.goal) == Approx(volume.rest * 1.3).epsilon(1e-3));
    f32 apart = 0.0f;
    for (std::size_t p = 0; p < rounded.goal.size(); ++p) {
        apart = std::max(apart, (rounded.goal[p] - creased.goal[p]).length());
        CHECK((rounded.goal[p] - rounded.carried[p]).dot(rig.points().points[p].outward) > -1e-4f);
    }
    CHECK(apart > 1e-3f);
}

TEST_CASE("G-S2 Melt ends as a pool of the body's volume, on the floor, within the depth read", "[wem][soft][shapes]") {
    Figure f = MakeBody();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    REQUIRE(volume.encloses);
    std::vector<Vector3f> rest;
    rig.carry(Bind(f.model()), rest);
    const f64 before = SoftVolumeAt(volume, rest);
    SoftMelt settings;
    settings.uneven = 0.0f;
    const f32 floor = 0.0f;
    // Each point's place up its own column of the body: the wall's by its
    // height, the caps' at the ends.
    const SoftColumns columns = SoftMeltColumns(rig);
    REQUIRE(columns.share.size() == rest.size());
    for (std::size_t p = 0; p < rest.size(); ++p) {
        CHECK(columns.share[p] == Approx(rest[p].z / 6.0f).margin(0.06));
        CHECK(columns.thickness[p] == Approx(6.0f).margin(0.5));
    }

    // The pool: as wide as asked, and as deep as the volume makes it.
    const SoftPool pool = SoftMeltPool(rig, volume, columns, rest, floor, settings);
    CHECK(pool.height == Approx(6.0f));
    CHECK(pool.radius == Approx(2.5f * 1.0f).epsilon(0.01)); // The tube is a unit across.
    CHECK(pool.depth > 0.0f);
    CHECK(pool.depth < 0.5f * pool.height);
    // Wider, shallower: the size is set and the depth read.
    SoftMelt wide = settings;
    wide.puddle = 4.0f;
    CHECK(SoftMeltPool(rig, volume, columns, rest, floor, wide).depth == Approx(pool.depth * 2.5f * 2.5f / 16.0f).epsilon(0.02));

    // Nothing melted, nothing moved.
    std::vector<Vector3f> same = rest;
    SoftMeltTo(rig, volume, columns, same, floor, settings, 0.0f, 1);
    CHECK(std::memcmp(same.data(), rest.data(), rest.size() * sizeof(Vector3f)) == 0);

    for (const f32 progress : {0.25f, 0.5f, 0.75f, 1.0f}) {
        std::vector<Vector3f> melted = rest;
        SoftMeltTo(rig, volume, columns, melted, floor, settings, progress, 1);
        INFO("progress " << progress);
        // Keep volume, at every progress.
        CHECK(SoftVolumeAt(volume, melted) == Approx(before).epsilon(0.01));
        for (const Vector3f& at : melted) {
            CHECK(at.z >= floor - 1e-4f);
        }
    }
    std::vector<Vector3f> ended = rest;
    SoftMeltTo(rig, volume, columns, ended, floor, settings, 1.0f, 1);
    f32 top = 0.0f, out = 0.0f, lowest = 1e30f;
    for (const Vector3f& at : ended) {
        top = std::max(top, at.z - floor);
        lowest = std::min(lowest, at.z - floor);
        out = std::max(out, std::sqrt(at.x * at.x + at.y * at.y));
    }
    WARN("Melt: ends " << top / pool.height << " of its height deep; " << out / pool.radius
                       << " of the pool's radius out");
    // On the floor, as deep as was read, as wide as was set.
    CHECK(lowest == Approx(0.0f).margin(1e-4));
    CHECK(top == Approx(pool.depth).epsilon(0.01));
    CHECK(out <= pool.radius * 1.001f);
    CHECK(out > 0.95f * pool.radius);

    // Nothing folds: higher stays higher, at the same distance out. The pool's
    // rim is steep, so a point on it stands a hair up or down by its rounding:
    // half a hundredth of the body's height is given.
    const u32 segments = 16, rings = 19;
    const f32 hair = 0.005f * pool.height;
    for (u32 r = 0; r + 1 < rings; ++r) {
        CHECK(ended[(r + 1) * segments].z >= ended[r * segments].z - hair);
    }

    // From the top, the top is the first to go; from the bottom, the last.
    // And through the whole of it a higher ring is never under a lower one.
    for (const SoftMeltOrder order : {SoftMeltOrder::Top, SoftMeltOrder::Bottom, SoftMeltOrder::AllAtOnce}) {
        SoftMelt ordered = settings;
        ordered.order = order;
        for (const f32 progress : {0.1f, 0.3f, 0.5f, 0.7f, 0.9f}) {
            std::vector<Vector3f> part = rest;
            SoftMeltTo(rig, volume, columns, part, floor, ordered, progress, 1);
            for (u32 r = 0; r + 1 < rings; ++r) {
                CHECK(part[(r + 1) * segments].z >= part[r * segments].z - hair);
            }
        }
    }
    const auto sunk = [&](SoftMeltOrder order, u32 ring) {
        SoftMelt ordered = settings;
        ordered.order = order;
        std::vector<Vector3f> part = rest;
        SoftMeltTo(rig, volume, columns, part, floor, ordered, 0.3f, 1);
        // How much of its height a ring has lost.
        return 1.0f - part[ring * segments].z / rest[ring * segments].z;
    };
    CHECK(sunk(SoftMeltOrder::Top, rings - 1) > sunk(SoftMeltOrder::Top, 2) + 0.1f);
    CHECK(sunk(SoftMeltOrder::Bottom, 2) > sunk(SoftMeltOrder::Bottom, rings - 1) + 0.1f);

    // Backward, it rises out of its pool: the two ends change places.
    SoftMelt backward = settings;
    backward.backward = true;
    std::vector<Vector3f> risen = rest, pooled = rest;
    SoftMeltTo(rig, volume, columns, risen, floor, backward, 1.0f, 1);
    SoftMeltTo(rig, volume, columns, pooled, floor, backward, 0.0f, 1);
    CHECK(std::memcmp(risen.data(), rest.data(), rest.size() * sizeof(Vector3f)) == 0);
    f32 apart = 0.0f;
    for (std::size_t p = 0; p < pooled.size(); ++p) {
        apart = std::max(apart, (pooled[p] - ended[p]).length());
    }
    CHECK(apart < 1e-4f);

    // Uneven, by the seed: the same seed is the same melt.
    SoftMelt uneven = settings;
    uneven.uneven = 1.0f;
    std::vector<Vector3f> a = rest, b = rest, c = rest;
    SoftMeltTo(rig, volume, columns, a, floor, uneven, 0.5f, 3);
    SoftMeltTo(rig, volume, columns, b, floor, uneven, 0.5f, 3);
    SoftMeltTo(rig, volume, columns, c, floor, uneven, 0.5f, 4);
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(Vector3f)) == 0);
    CHECK(std::memcmp(a.data(), c.data(), a.size() * sizeof(Vector3f)) != 0);
}

TEST_CASE("G-S2 Melt through a step: its span is a progress, and a Decay clip holds the end", "[wem][soft][shapes]") {
    Figure f = MakeBody();
    const u32 id = f.model().physics.soft[0].id;
    SoftEffect& melt = AddEffect(f, SoftEffectKind::Melt);
    melt.everywhere = false;
    melt.melt->uneven = 0.0f;
    const u32 effect = melt.id;
    const u32 clip = 0;
    f.document.clips[clip].looping = false;
    const SoftSpan span{0.2f, 0.8f, 0.0f, 0.0f};
    SetSoftSpans(f.document, 0, clip, id, effect, std::span<const SoftSpan>(&span, 1));
    const auto tallest = [&](u32 in, f32 seconds) {
        f32 top = 0.0f;
        for (const Vector3f& at : StepAt(f.document, in, seconds).goal) {
            top = std::max(top, at.z);
        }
        return top;
    };
    // Before its span the body stands; through it, it sinks; after it, it is
    // held. The tube's flat top is one point and its ring, so its pool is a
    // cone: deeper than a body with a top to spread.
    CHECK(!StepAt(f.document, clip, 0.1f).moved);
    const f32 standing = tallest(clip, 0.1f), half = tallest(clip, 0.5f), down = tallest(clip, 0.8f);
    CHECK(half < 0.9f * standing);
    CHECK(down < half);
    CHECK(down < 0.6f * standing);
    CHECK(StepAt(f.document, clip, 1.0f).moved);
    CHECK(tallest(clip, 1.0f) < 0.6f * standing);

    // A clip that places no span is not melted; one named Decay holds the end.
    CHECK(!StepAt(f.document, 1, 0.3f).moved);
    CHECK(!StepAt(f.document, kInvalidIndex, 0.0f).moved);
    f.document.clips[1].name = "Decay Flesh";
    CHECK(tallest(1, 0.3f) < 0.6f * standing);
    f.model().physics.soft[0].effects[0].melt->staysInDecay = false;
    CHECK(!StepAt(f.document, 1, 0.3f).moved);

    // Backward, the body is its pool until the span starts and stands after
    // it; and in a clip that does not place it, it is not a pool at all.
    f.model().physics.soft[0].effects[0].melt->backward = true;
    f.model().physics.soft[0].effects[0].melt->staysInDecay = true;
    CHECK(tallest(clip, 0.1f) < 0.6f * standing);
    CHECK(tallest(clip, 1.0f) == Approx(standing).epsilon(0.05));
    CHECK(!StepAt(f.document, 1, 0.3f).moved);
    CHECK(!StepAt(f.document, kInvalidIndex, 0.0f).moved);
}

TEST_CASE("G-S2 a Ripple's crest stands at speed times time, and a train comes round whole", "[wem][soft][shapes]") {
    SoftRipple settings;
    settings.wavelength = 1.0f;
    settings.speed = 1.5f;
    settings.fadeDistance = 0.0f;
    settings.fadeTime = 0.0f;
    const f32 size = 4.0f;
    const SoftWave wave = SoftWaveOf(settings, 0.05f, size);
    CHECK(wave.height == Approx(0.2f));
    CHECK(wave.length == Approx(4.0f));
    CHECK(wave.speed == Approx(6.0f));
    for (const f32 since : {0.25f, 0.5f, 1.0f}) {
        f32 best = 0.0f, at = 0.0f;
        for (f32 d = 0.0f; d < 12.0f; d += 0.005f) {
            const f32 h = SoftSwellHeight(wave, d, since);
            if (h > best) {
                best = h;
                at = d;
            }
        }
        CHECK(at == Approx(wave.speed * since).margin(0.01));
        CHECK(best == Approx(wave.height).epsilon(1e-3));
    }
    // Nothing before it sets out, and nothing a length away from it.
    CHECK(SoftSwellHeight(wave, 1.0f, -0.1f) == 0.0f);
    CHECK(SoftSwellHeight(wave, wave.speed * 0.5f + 1.01f * wave.length, 0.5f) == 0.0f);
    // It fades with the time and with the distance.
    SoftRipple fading = settings;
    fading.fadeTime = 0.6f;
    fading.fadeDistance = 1.0f;
    const SoftWave faded = SoftWaveOf(fading, 0.05f, size);
    CHECK(SoftSwellHeight(faded, wave.speed * 0.6f, 0.6f) < 0.2f * wave.height);
    // Water: the swell and its crests move apart.
    SoftRipple water = settings;
    water.water = true;
    water.envelope = 0.5f;
    const SoftWave pond = SoftWaveOf(water, 0.05f, size);
    CHECK(pond.envelope == Approx(0.5f));
    CHECK(SoftSwellHeight(pond, 0.5f * wave.speed * 1.0f + 1.01f * wave.length, 1.0f) == 0.0f);
    CHECK(SoftSwellHeight(pond, 0.5f * wave.speed * 1.0f, 1.0f) != 0.0f);

    // A train: the same a whole swing on, and a crest moves outward.
    for (const f32 d : {0.0f, 1.3f, 5.0f}) {
        CHECK(SoftTrainHeight(wave, d, 0.3f) == Approx(SoftTrainHeight(wave, d, 1.3f)).margin(1e-5));
    }
    CHECK(SoftTrainHeight(wave, 0.25f * wave.length, 0.5f) == Approx(wave.height).epsilon(1e-4));
    CHECK(SoftTrainHeight(wave, 0.5f * wave.length, 0.75f) == Approx(wave.height).epsilon(1e-4));
}

TEST_CASE("G-S2 a Ripple runs from its place: over the skin, through the body, and by the bind", "[wem][soft][shapes]") {
    Figure f = MakeBody();
    const u32 id = f.model().physics.soft[0].id;
    const SoftRig rig(f.model(), id);
    SoftPlace place;
    place.mesh = 0;
    place.face = FaceNear(f.model().meshes[0], Vector3f{1, 0, 0.2f});
    place.weights = Vector3f{0.34f, 0.33f, 0.33f};
    const SoftPlaceAt at = ResolveSoftPlace(f.model(), rig, place);
    REQUIRE(at.valid);
    const std::vector<f32> skin = SoftRippleDistances(rig, at, SoftThrough::Skin);
    const std::vector<f32> body = SoftRippleDistances(rig, at, SoftThrough::Body);
    REQUIRE(skin.size() == rig.pointCount());
    f32 farthest = 0.0f;
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        REQUIRE(std::isfinite(skin[p]));
        REQUIRE(std::isfinite(body[p]));
        // Neither is shorter than the straight line, and the way through is
        // no longer than the way round, by a cell.
        const f32 straight = (rig.points().points[p].at - at.at).length();
        CHECK(skin[p] >= straight - 1e-3f);
        CHECK(body[p] >= straight - 2.0f * rig.points().grid.cell);
        CHECK(body[p] <= skin[p] + 3.0f * rig.points().grid.cell);
        farthest = std::max(farthest, skin[p]);
    }
    CHECK(farthest > 5.0f); // To the tube's far end.
    const std::vector<f32> lumps = SoftLumpDistances(rig, skin);
    REQUIRE(lumps.size() == rig.lumpCount());

    // Once, along the surface, in a clip that bends the body: the height at a
    // point is by its distance at the bind, whatever the pose.
    SoftEffect& ripple = AddEffect(f, SoftEffectKind::Ripple);
    ripple.place = place;
    ripple.amount = 0.05f;
    ripple.ripple->plays = SoftRipplePlays::Once;
    ripple.ripple->fadeDistance = 0.0f;
    ripple.ripple->fadeTime = 0.0f;
    const u32 clip = 0;
    const SoftStepper stepper(f.document, 0, id);
    const SoftPoseAt poseAt = [&](f32 seconds, std::vector<Matrix44f>& skinned) {
        skinned = SkinAt(f.document, clip, seconds);
    };
    SoftStepResult posed, bound;
    stepper.step(SkinAt(f.document, clip, 0.5f), SoftClock{clip, 0.5f}, poseAt, posed);
    stepper.step(Bind(f.model()), SoftClock{clip, 0.5f}, poseAt, bound);
    REQUIRE(posed.moved);
    const SoftWave wave = SoftWaveOf(*ripple.ripple, ripple.amount, rig.points().size);
    f32 off = 0.0f, tallest = 0.0f;
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        const f32 want = SoftSwellHeight(wave, skin[p], 0.5f);
        off = std::max(off, std::abs((posed.goal[p] - posed.carried[p]).length() - std::abs(want)));
        off = std::max(off, std::abs((bound.goal[p] - bound.carried[p]).length() - std::abs(want)));
        tallest = std::max(tallest, std::abs(want));
    }
    CHECK(tallest > 0.5f * wave.height);
    CHECK(off < 1e-4f);

    // Along the bones it is a wave of the lumps' scales, each by the height at
    // it over its own size: met to 1e-5.
    f.model().physics.soft[0].effects[0].ripple->along = SoftAlong::Bones;
    const SoftStepper along(f.document, 0, id);
    SoftStepResult scaled;
    along.step(Bind(f.model()), SoftClock{clip, 0.5f}, poseAt, scaled);
    REQUIRE(scaled.moved);
    const std::vector<f32> radii = SoftLumpRadii(rig);
    u32 swelled = 0;
    for (u32 k = 0; k < rig.lumpCount(); ++k) {
        const f32 want = rig.scales(k) && radii[k] > 0.0f ? 1.0f + SoftSwellHeight(wave, lumps[k], 0.5f) / radii[k] : 1.0f;
        CHECK(scaled.departures[k].scale.x == Approx(want).margin(1e-5));
        CHECK(scaled.departures[k].scale.z == Approx(want).margin(1e-5));
        swelled += std::abs(want - 1.0f) > 1e-3f ? 1 : 0;
    }
    CHECK(swelled > 0u);
}

TEST_CASE("G-S2 a looping Ripple's last frame is its first", "[wem][soft][shapes]") {
    Figure f = MakeBody();
    const u32 id = f.model().physics.soft[0].id;
    const SoftRig rig(f.model(), id);
    SoftEffect& ripple = AddEffect(f, SoftEffectKind::Ripple);
    ripple.place.mesh = 0;
    ripple.place.face = FaceNear(f.model().meshes[0], Vector3f{1, 0, 0.2f});
    ripple.place.weights = Vector3f{0.34f, 0.33f, 0.33f};
    ripple.amount = 0.05f;
    ripple.ripple->plays = SoftRipplePlays::Loops;
    // A rate no whole number of which fits the clip: it is rounded to one.
    ripple.ripple->speed = 1.37f;
    const u32 clip = 0;
    const f32 duration = f.document.clips[clip].duration;
    const SoftStepResult first = StepAt(f.document, clip, 0.0f), last = StepAt(f.document, clip, duration);
    const SoftStepResult mid = StepAt(f.document, clip, 0.37f * duration);
    REQUIRE(first.moved);
    f32 apart = 0.0f, moved = 0.0f;
    for (std::size_t p = 0; p < first.goal.size(); ++p) {
        apart = std::max(apart, (first.goal[p] - last.goal[p]).length());
        moved = std::max(moved, (mid.goal[p] - first.goal[p]).length());
    }
    CHECK(apart < 1e-4f);
    CHECK(moved > 1e-2f);
    // A place that is not on the body sends nothing.
    f.model().physics.soft[0].effects[0].place.mesh = kInvalidIndex;
    CHECK(!StepAt(f.document, clip, 0.3f).moved);
}
