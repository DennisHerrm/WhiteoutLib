// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

/// Workshop ▸ Soft Body, the fit (EDIT_MODE_SOFTBODY_DESIGN.md §7;
/// EDIT_MODE_SOFTBODY_PLAN.md §11, G-S1b's second half): what the lumps can
/// follow exactly, what they follow closely, and that the fast solve gives
/// what the reference gives.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <whiteout/models/wem/anim/animator.h>
#include <whiteout/models/wem/anim/detach.h>
#include <whiteout/models/wem/anim/key_reduce.h>
#include <whiteout/models/wem/physics/soft_fields.h>
#include <whiteout/models/wem/physics/soft_fit.h>
#include <whiteout/models/wem/physics/soft_step.h>
#include <whiteout/models/wem/physics/soft_run.h>
#include <whiteout/models/wem/skinning/deform.h>

#include "wem_soft_figures.h"

#include <algorithm>
#include <array>
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

std::vector<Matrix44f> PoseAt(const Document& document, u32 clip, f32 seconds) {
    const Animator animator(document, 0);
    Mix mix;
    mix.plays.push_back(Play{clip, seconds, 1.0f, false});
    Pose pose;
    animator.evaluate(mix, pose);
    return pose.skinning;
}

/// @p rig's points carried by @p skin, each then moved by @p field.
template <class Field>
std::vector<Vector3f> Goal(const SoftRig& rig, std::span<const Matrix44f> skin, Field&& field) {
    std::vector<Vector3f> goal;
    rig.carry(skin, goal);
    for (Vector3f& p : goal) {
        p = field(p);
    }
    return goal;
}

bool AtRest(const Transform& t, f32 within) {
    return t.translation.length() < within && std::abs(t.scale.x - 1.0f) < within &&
           std::abs(t.scale.y - 1.0f) < within && std::abs(t.scale.z - 1.0f) < within &&
           std::abs(std::abs(t.rotation.w) - 1.0f) < within;
}

} // namespace

TEST_CASE("G-S1b a goal that is the pose leaves every lump at rest", "[wem][soft][fit]") {
    for (const Skinned skinned : {Skinned::Rigid, Skinned::Two, Skinned::Four}) {
        Figure f = MakeTube(skinned);
        const SoftBody& body = SoftenAll(f, 8, true);
        const SoftRig rig(f.model(), body.id);
        REQUIRE(!rig.empty());
        REQUIRE(rig.lumpCount() == body.lumps.size());
        for (const std::vector<Matrix44f>& skin : {Bind(f.model()), PoseAt(f.document, 0, 0.5f)}) {
            const std::vector<Vector3f> goal = Goal(rig, skin, [](const Vector3f& p) { return p; });
            for (const SoftFitMethod method : {SoftFitMethod::Fast, SoftFitMethod::Dense}) {
                std::vector<Transform> departures;
                const SoftFitReport report = rig.fit(skin, goal, departures, method);
                CHECK(report.largestMove < 1e-5f);
                CHECK(report.worstMiss < 2e-5f);
                REQUIRE(departures.size() == body.lumps.size());
                for (const Transform& departure : departures) {
                    CHECK(AtRest(departure, 2e-5f));
                }
            }
        }
    }
}

TEST_CASE("G-S1b the slots of a point sum to one and carry it as its skin does", "[wem][soft][fit]") {
    Figure f = MakeTube(Skinned::Two);
    const SoftBody& body = Soften(f, -1.0f, 8, false);
    const SoftRig rig(f.model(), body.id);
    REQUIRE(!rig.empty());
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        f32 sum = 0.0f;
        for (const SoftSlot& slot : rig.slotsOf(p)) {
            sum += slot.weight;
            CHECK(slot.node < f.model().nodes.size());
        }
        CHECK(sum == Approx(1.0f).margin(1e-5));
        CHECK(!rig.ringOf(p).empty());
    }
    CHECK(!rig.triangles().empty());
    // With every lump at rest the points are where the model's skin has them.
    const std::vector<Matrix44f> skin = PoseAt(f.document, 0, 0.5f);
    std::vector<Vector3f> carried, posed;
    rig.carry(skin, carried);
    rig.pose(skin, {}, posed);
    const std::vector<Vector3f> own = skinning::DeformMesh(f.model().meshes[0], skin);
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        const u32 vertex = rig.points().points[p].vertex;
        CHECK((carried[p] - own[vertex]).length() < 2e-5f);
        CHECK((posed[p] - own[vertex]).length() < 2e-5f);
    }
}

TEST_CASE("G-S1b a stretch along a model axis is exact at the bind, and under a turn about it",
          "[wem][soft][fit]") {
    for (const Skinned skinned : {Skinned::Rigid, Skinned::Two, Skinned::Four}) {
        for (const u32 detail : {0u, 8u}) {
            Figure f = MakeTube(skinned);
            const SoftBody& body = SoftenAll(f, detail, true);
            const SoftRig rig(f.model(), body.id);
            // The bind, and Twist's middle: the first bone turned a radian about Z.
            for (const std::vector<Matrix44f>& skin : {Bind(f.model()), PoseAt(f.document, 1, 0.5f)}) {
                const std::vector<Vector3f> goal =
                    Goal(rig, skin, [](const Vector3f& p) { return Vector3f{p.x, p.y, p.z * 1.3f}; });
                for (const SoftFitMethod method : {SoftFitMethod::Fast, SoftFitMethod::Dense}) {
                    std::vector<Transform> departures;
                    const SoftFitReport report = rig.fit(skin, goal, departures, method);
                    INFO("skinned " << static_cast<int>(skinned) << " detail " << detail << " method "
                                    << static_cast<int>(method));
                    CHECK(report.largestMove == Approx(1.8f).margin(1e-3));
                    // Exact, to what single precision keeps of a body six long.
                    CHECK(report.worstMiss < 1e-4f);
                    CHECK(report.follows() < 1e-4f);
                    // One lump a bone, each with a ring or more of its own: every
                    // one takes the stretch. (A lump whose points lie flat across
                    // the axis has no height to stretch, and stays as it is.)
                    for (std::size_t k = 0; detail == 0 && skinned == Skinned::Rigid && k < departures.size(); ++k) {
                        CHECK(departures[k].scale.z == Approx(1.3f).margin(1e-3));
                        CHECK(departures[k].scale.x == Approx(1.0f).margin(1e-3));
                    }
                }
            }
        }
    }
}

TEST_CASE("G-S1b a move and a turn of the whole body are followed exactly", "[wem][soft][fit]") {
    Figure f = MakeTube(Skinned::Four);
    const SoftBody& body = SoftenAll(f, 8, true);
    const SoftRig rig(f.model(), body.id);
    Transform whole;
    whole.translation = Vector3f{0.4f, -0.2f, 0.3f};
    whole.rotation = Quaternion::from_axis_angle(Vector3f{0.6f, 0.0f, 0.8f}, 0.7f);
    const Matrix44f moved = ToMatrix(whole);
    // The bind, and Twist's middle, where the bones are turned and none is
    // scaled: each lump's own turn and move then say it.
    for (const std::vector<Matrix44f>& skin : {Bind(f.model()), PoseAt(f.document, 1, 0.5f)}) {
        const std::vector<Vector3f> goal =
            Goal(rig, skin, [&](const Vector3f& p) { return transform_point(p, moved); });
        for (const SoftFitMethod method : {SoftFitMethod::Fast, SoftFitMethod::Dense}) {
            std::vector<Transform> departures;
            const SoftFitReport report = rig.fit(skin, goal, departures, method);
            CHECK(report.largestMove > 0.5f);
            CHECK(report.worstMiss < 3e-4f);
            CHECK(report.follows() < 1e-3f);
        }
    }
    // Bend's middle scales a bone unevenly: a turn seen through that scale is
    // no turn and scale along the model's axes, and the lumps come close.
    const std::vector<Matrix44f> bent = PoseAt(f.document, 0, 0.5f);
    const std::vector<Vector3f> goal = Goal(rig, bent, [&](const Vector3f& p) { return transform_point(p, moved); });
    std::vector<Transform> departures;
    const SoftFitReport report = rig.fit(bent, goal, departures);
    INFO("follows " << report.follows());
    CHECK(report.follows() < 0.08f);
}

TEST_CASE("G-S1b the fast solve follows a shear and a wave as the reference does", "[wem][soft][fit]") {
    struct Case {
        Skinned skinned;
        u32 detail;
        bool copyAll;
    };
    for (const Case& c : {Case{Skinned::Rigid, 0, true}, Case{Skinned::Two, 0, false}, Case{Skinned::Two, 12, true},
                          Case{Skinned::Four, 6, true}}) {
        Figure f = MakeTube(c.skinned, 19, 16);
        const SoftBody& body = SoftenAll(f, c.detail, c.copyAll);
        const SoftRig rig(f.model(), body.id);
        const std::vector<Matrix44f> skin = Bind(f.model());
        const auto shear = [](const Vector3f& p) { return Vector3f{p.x + 0.3f * p.z, p.y, p.z}; };
        const auto wave = [](const Vector3f& p) {
            return Vector3f{p.x + 0.4f * std::sin(6.2831853f * p.z / 6.0f), p.y, p.z};
        };
        for (int which = 0; which < 2; ++which) {
            const std::vector<Vector3f> goal = which == 0 ? Goal(rig, skin, shear) : Goal(rig, skin, wave);
            std::vector<Transform> fast, dense;
            const SoftFitReport quick = rig.fit(skin, goal, fast, SoftFitMethod::Fast);
            const SoftFitReport reference = rig.fit(skin, goal, dense, SoftFitMethod::Dense);
            INFO("skinned " << static_cast<int>(c.skinned) << " detail " << c.detail << " field " << which
                            << ": fast " << quick.follows() << " mean " << quick.meanMiss << ", dense "
                            << reference.follows() << " mean " << reference.meanMiss);
            // The fast solve is within a point of the reference's miss, at its
            // worst point and on average.
            CHECK(quick.follows() <= reference.follows() + 0.01f);
            CHECK(quick.meanMiss <= reference.meanMiss + 0.01f * quick.largestMove);
            // What `fit` says is what `measure` finds.
            const SoftFitReport again = rig.measure(skin, goal, fast);
            CHECK(again.worstMiss == Approx(quick.worstMiss).margin(1e-6));
        }
    }
}

TEST_CASE("G-S1b more lumps follow a wave better", "[wem][soft][fit]") {
    f32 before = 1.0f;
    for (const u32 detail : {0u, 12u, 36u}) {
        Figure f = MakeTube(Skinned::Rigid, 25, 16);
        const SoftBody& body = SoftenAll(f, detail, true);
        const SoftRig rig(f.model(), body.id);
        const std::vector<Matrix44f> skin = Bind(f.model());
        const std::vector<Vector3f> goal = Goal(rig, skin, [](const Vector3f& p) {
            return Vector3f{p.x + 0.4f * std::sin(6.2831853f * p.z / 6.0f), p.y, p.z};
        });
        std::vector<Transform> departures;
        const SoftFitReport report = rig.fit(skin, goal, departures);
        INFO("detail " << detail << ": mean miss " << report.meanMiss);
        CHECK(report.meanMiss < before);
        before = report.meanMiss;
    }
}

TEST_CASE("G-S1b a rigid part's lump never scales, and a lump no point rides stays at rest", "[wem][soft][fit]") {
    Figure f = MakeTube(Skinned::Rigid);
    SoftBody& body = SoftenAll(f, 0, true);
    REQUIRE(body.lumps.size() == 3u);
    body.lumps[1].rigid = true;
    // A lump nothing was bound to: an effect's, say, before its points are.
    SoftLump spare = body.lumps[0];
    spare.kind = SoftLumpKind::Surface;
    spare.carrier = f.bones[0];
    body.lumps.push_back(spare);
    const SoftRig rig(f.model(), body.id);
    CHECK(rig.scales(0));
    CHECK(!rig.scales(1));
    CHECK(rig.moves(1));
    const std::vector<Matrix44f> skin = Bind(f.model());
    const std::vector<Vector3f> goal =
        Goal(rig, skin, [](const Vector3f& p) { return Vector3f{p.x * 1.2f, p.y * 1.2f, p.z * 1.3f}; });
    for (const SoftFitMethod method : {SoftFitMethod::Fast, SoftFitMethod::Dense}) {
        std::vector<Transform> departures;
        rig.fit(skin, goal, departures, method);
        REQUIRE(departures.size() == 4u);
        CHECK(departures[0].scale.z == Approx(1.3f).margin(1e-3));
        CHECK(departures[1].scale.x == Approx(1.0f).margin(1e-6));
        CHECK(departures[1].scale.z == Approx(1.0f).margin(1e-6));
        CHECK(AtRest(departures[3], 1e-6f));
    }
}

// ============================================================================
// G-S2: the fields, and the step that runs them.
// ============================================================================

namespace {

/// An effect of @p kind added to @p body, on in every clip.
SoftEffect& AddEffect(SoftBody& body, SoftEffectKind kind) {
    SoftEffect effect = DefaultSoftEffect(kind);
    effect.id = body.nextEffect++;
    effect.everywhere = true;
    body.effects.push_back(effect);
    return body.effects.back();
}

SoftStepResult StepAt(const Document& document, u32 body, u32 clip, f32 seconds) {
    const SoftStepper stepper(document, 0, body);
    const Animator animator(document, 0);
    const auto poseAt = [&](f32 at, std::vector<Matrix44f>& skin) {
        Mix mix;
        mix.plays.push_back(Play{clip, at, 1.0f, false});
        Pose pose;
        animator.evaluate(mix, pose);
        skin = pose.skinning;
    };
    std::vector<Matrix44f> skin;
    if (clip == kInvalidIndex) {
        skin = Bind(document.models[0]);
    } else {
        poseAt(seconds, skin);
    }
    SoftStepResult out;
    stepper.step(skin, SoftClock{clip, seconds}, clip == kInvalidIndex ? SoftPoseAt{} : SoftPoseAt{poseAt}, out);
    return out;
}

/// A flat square of side 4 in the XY plane, facing +Z, on one bone: an open patch.
Figure MakePatch(u32 cells = 40) {
    Figure f;
    f.document.profiles = {ProfileId::Wc3Reforged};
    f.document.defaultProfile = ProfileId::Wc3Reforged;
    Model& model = f.document.models.emplace_back();
    f.bones.push_back(AddNode(model.nodes, "root", kInvalidNode, {0, 0, 0}));
    geom::MeshBuilder builder = StartMesh("patch");
    std::vector<geom::VertexId> grid;
    for (u32 y = 0; y <= cells; ++y) {
        for (u32 x = 0; x <= cells; ++x) {
            grid.push_back(builder.addVertex(Vector3f{-2.0f + 4.0f * static_cast<f32>(x) / static_cast<f32>(cells),
                                                      -2.0f + 4.0f * static_cast<f32>(y) / static_cast<f32>(cells), 0.0f}));
            builder.addInfluence(grid.back(), f.bones[0], 1.0f);
        }
    }
    for (u32 y = 0; y < cells; ++y) {
        for (u32 x = 0; x < cells; ++x) {
            const u32 a = y * (cells + 1) + x, b = a + 1, c = a + cells + 1, d = c + 1;
            builder.addTriangle(grid[a], grid[b], grid[d]);
            builder.addTriangle(grid[a], grid[d], grid[c]);
        }
    }
    model.meshes.push_back(builder.build().mesh);
    f.size = 4.0f;
    return f;
}

/// The face of @p mesh whose middle is nearest @p at.
u32 FaceNear(const Mesh& mesh, const Vector3f& at) {
    const std::span<const Vector3f> positions =
        mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    const geom::FaceSet& faces = mesh.faceSet();
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

TEST_CASE("G-S2 an offset's volume is an exact cubic", "[wem][soft][fields]") {
    Figure f = MakeTube(Skinned::Two);
    const SoftBody& body = SoftenAll(f, 0, true);
    const SoftRig rig(f.model(), body.id);
    const SoftVolume volume = MakeSoftVolume(rig);
    CHECK(volume.shells == 1u);
    CHECK(volume.outer == 1u);
    CHECK(volume.fans.empty());
    CHECK(volume.encloses);
    // A twelve-sided prism of radius 1, six long.
    const f32 prism = 0.5f * 12.0f * std::sin(6.2831853f / 12.0f) * 6.0f;
    CHECK(volume.rest == Approx(prism).epsilon(1e-4));
    std::vector<Vector3f> at(rig.pointCount()), moves(rig.pointCount());
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        at[p] = rig.points().points[p].at;
        // Not a tidy direction: outward, and a lean that grows with height.
        moves[p] = rig.points().points[p].outward + Vector3f{0.1f * at[p].z, -0.05f, 0.02f * at[p].x};
    }
    const std::array<f64, 4> cubic = OffsetCubic(volume, at, moves);
    CHECK(cubic[0] == Approx(SoftVolumeAt(volume, at)).epsilon(1e-9));
    // A fifth sample, where four would fix the curve.
    for (const f32 d : {-0.4f, 0.13f, 0.37f, 1.1f, 2.0f}) {
        std::vector<Vector3f> moved(at.size());
        for (std::size_t p = 0; p < at.size(); ++p) {
            moved[p] = at[p] + moves[p] * d;
        }
        const f64 predicted = ((cubic[3] * d + cubic[2]) * d + cubic[1]) * d + cubic[0];
        CHECK(predicted == Approx(SoftVolumeAt(volume, moved)).epsilon(1e-5));
    }
    // And its root gives the volume asked for.
    const f64 offset = SolveOffset(cubic, cubic[0] * 1.3, 12.0);
    CHECK(offset > 0.0);
    CHECK(((cubic[3] * offset + cubic[2]) * offset + cubic[1]) * offset + cubic[0] == Approx(cubic[0] * 1.3).epsilon(1e-9));
    CHECK(SolveOffset(cubic, cubic[0], 12.0) == 0.0);
}

TEST_CASE("G-S2 of two layers the outer one's volume is read, and both move outward", "[wem][soft][fields]") {
    // A box of side 2 turned inside out, to show its far side, round a box of
    // side 1: the cat slime's two layers.
    Figure f;
    f.document.profiles = {ProfileId::Wc3Reforged};
    f.document.defaultProfile = ProfileId::Wc3Reforged;
    Model& model = f.document.models.emplace_back();
    f.bones.push_back(AddNode(model.nodes, "root", kInvalidNode, {0, 0, 0}));
    geom::MeshBuilder builder = StartMesh("layers");
    AddBox(builder, {-1, -1, -1}, 2.0f, {{0, 1.0f}}, /*insideOut=*/true);
    AddBox(builder, {-0.5f, -0.5f, -0.5f}, 1.0f, {{0, 1.0f}});
    model.meshes.push_back(builder.build().mesh);
    SoftBody& body = SoftenAll(f, 0, true);
    const SoftRig rig(f.model(), body.id);
    const SoftVolume volume = MakeSoftVolume(rig);
    CHECK(volume.shells == 2u);
    CHECK(volume.outer == 1u);
    CHECK(volume.rest == Approx(8.0f).epsilon(1e-5));
    CHECK(volume.triangles.size() == 12u);

    SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
    inflate.amount = 0.3f;
    inflate.inflate->smooth = 0;
    const SoftStepResult out = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(out.moved);
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        // Every point of either layer goes away from the middle.
        CHECK((out.goal[p] - out.carried[p]).dot(out.carried[p]) > 0.0f);
    }
    CHECK(SoftVolumeAt(volume, out.goal) == Approx(8.0 * 1.3).epsilon(1e-4));
}

TEST_CASE("G-S2 Inflate reaches its volume on the skinned result", "[wem][soft][fields]") {
    for (const Skinned skinned : {Skinned::Rigid, Skinned::Two}) {
        for (const SoftAlong along : {SoftAlong::Surface, SoftAlong::Bones}) {
            Figure f = MakeTube(skinned, 19, 16);
            SoftBody& body = SoftenAll(f, along == SoftAlong::Surface ? 12 : 0, true);
            SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
            inflate.amount = 0.3f;
            inflate.inflate->along = along;
            const SoftStepper stepper(f.document, 0, body.id);
            for (const u32 clip : {kInvalidIndex, 0u}) {
                const SoftStepResult out = StepAt(f.document, body.id, clip, 0.5f);
                REQUIRE(out.moved);
                const f64 before = SoftVolumeAt(stepper.volume(), out.carried);
                const f64 after = SoftVolumeAt(stepper.volume(), out.posed);
                INFO("skinned " << static_cast<int>(skinned) << " along " << static_cast<int>(along) << " clip " << clip
                                << ": " << after / before);
                // Along the bones the lumps say it themselves, and it is exact;
                // along the surface it is met on the skinned result.
                CHECK(after / before == Approx(1.3).epsilon(along == SoftAlong::Bones ? 1e-4 : 0.01));
                if (along == SoftAlong::Bones) {
                    // One factor for every lump, and nothing moved or turned.
                    for (const Transform& departure : out.departures) {
                        CHECK(departure.translation.length() == 0.0f);
                        CHECK(departure.scale.x == Approx(out.departures[0].scale.x).margin(1e-6));
                        CHECK(departure.scale.x == Approx(departure.scale.z).margin(1e-6));
                        CHECK(departure.scale.x > 1.0f);
                    }
                }
            }
        }
    }
}

TEST_CASE("G-S2 a Bulge peaks at its place and reaches over the surface", "[wem][soft][fields]") {
    Figure f = MakeTube(Skinned::Two, 25, 24);
    SoftBody& body = SoftenAll(f, 12, true);
    SoftEffect& bulge = AddEffect(body, SoftEffectKind::Bulge);
    bulge.amount = 0.4f;
    bulge.bulge->keepVolume = false;
    bulge.place.mesh = 0;
    bulge.place.face = FaceNear(f.model().meshes[0], Vector3f{1, 0, 3});
    bulge.place.weights = Vector3f{0.4f, 0.3f, 0.3f};
    bulge.place.radius = 1.0f;
    const SoftRig rig(f.model(), body.id);
    const SoftPlaceAt place = ResolveSoftPlace(f.model(), rig, bulge.place);
    REQUIRE(place.valid);
    CHECK((place.at - Vector3f{1, 0, 3}).length() < 0.3f);
    CHECK(place.normal.x > 0.9f);
    CHECK(place.bone == f.bones[1]);
    const SoftStepResult out = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(out.moved);
    f32 most = 0.0f;
    u32 peak = 0;
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        const f32 moved = (out.goal[p] - out.carried[p]).length();
        const Vector3f& at = rig.points().points[p].at;
        if (moved > most) {
            most = moved;
            peak = p;
        }
        // Past the radius, and on the far side, nothing is asked.
        if ((at - place.at).length() > 1.2f) {
            CHECK(moved == 0.0f);
        }
        if (at.x < -0.2f) {
            CHECK(moved == 0.0f);
        }
    }
    CHECK((rig.points().points[peak].at - place.at).length() < 0.35f);
    // Its height is the Amount, of the radius.
    CHECK(most == Approx(0.4f).margin(0.05));
    // Twelve lumps over the whole tube are few for a bump a sixth of it wide:
    // they show it, and miss its top by over half. A place's own lumps are
    // what mend that (§6.2).
    CHECK(out.fit.follows() < 0.8f);
    CHECK(out.fit.meanMiss < 0.05f * out.fit.largestMove);

    // A place wider than the tube is thick: its far side is inside a ball of
    // the radius, and out of its reach, which is over the surface (§8.2).
    bulge.place.radius = 2.5f;
    const SoftStepResult wide = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(wide.moved);
    u32 far = 0;
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        const Vector3f& at = rig.points().points[p].at;
        if (at.x < -0.5f && std::abs(at.z - 3.0f) < 1.0f) {
            ++far;
            CHECK((at - place.at).length() < 2.5f);
            CHECK((wide.goal[p] - wide.carried[p]).length() == 0.0f);
        }
    }
    CHECK(far > 0u);
    bulge.place.radius = 1.0f;

    // Said along the bones, each lump swells by the profile at its own pivot.
    bulge.bulge->along = SoftAlong::Bones;
    const SoftStepResult bones = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(bones.moved);
    for (std::size_t k = 0; k < body.lumps.size(); ++k) {
        const f32 u = (body.lumps[k].rest - place.at).length() / 1.0f;
        const f32 expected = 1.0f + 0.4f * (u < 1.0f ? (1.0f - u * u) * (1.0f - u * u) : 0.0f);
        CHECK(bones.departures[k].scale.x == Approx(expected).margin(1e-5));
        CHECK(bones.departures[k].translation.length() == 0.0f);
    }
}

TEST_CASE("G-S2 a Ring on a plane raises what it presses down", "[wem][soft][fields]") {
    Figure f = MakePatch();
    SoftBody& body = SoftenAll(f, 0, true);
    const SoftRig rig(f.model(), body.id);
    const SoftVolume volume = MakeSoftVolume(rig);
    // A flat patch encloses nothing.
    CHECK(!volume.encloses);
    SoftEffect& ring = AddEffect(body, SoftEffectKind::Bulge);
    ring.amount = -0.4f;
    ring.bulge->shape = SoftBulgeShape::Ring;
    ring.bulge->keepVolume = false;
    ring.place.mesh = 0;
    ring.place.face = FaceNear(f.model().meshes[0], Vector3f{0, 0, 0});
    ring.place.weights = Vector3f{0.34f, 0.33f, 0.33f};
    ring.place.radius = 0.8f;
    const SoftStepResult out = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(out.moved);
    f64 net = 0.0, gross = 0.0;
    f32 deepest = 0.0f, highest = 0.0f;
    for (u32 p = 0; p < rig.pointCount(); ++p) {
        const f32 lift = out.goal[p].z - out.carried[p].z;
        net += static_cast<f64>(lift) * rig.points().points[p].area;
        gross += static_cast<f64>(std::abs(lift)) * rig.points().points[p].area;
        deepest = std::min(deepest, lift);
        highest = std::max(highest, lift);
    }
    // A dent, and a rim round it; and as much raised as pressed.
    CHECK(deepest < -0.25f);
    CHECK(highest > 0.02f);
    CHECK(std::abs(net) < 0.03 * gross);
}

TEST_CASE("G-S2 Wobble keeps the volume and its last frame is its first", "[wem][soft][fields]") {
    for (const SoftWobblePattern pattern :
         {SoftWobblePattern::Jelly, SoftWobblePattern::Sway, SoftWobblePattern::Roll}) {
        Figure f = MakeTube(Skinned::Two, 19, 16);
        SoftBody& body = SoftenAll(f, 12, true);
        SoftEffect& wobble = AddEffect(body, SoftEffectKind::Wobble);
        wobble.amount = 0.1f;
        wobble.wobble->pattern = pattern;
        // 1.3 swings a second in a loop a second long: one whole swing.
        wobble.wobble->speed = 1.3f;
        wobble.wobble->axis = SoftAxis::Z;
        const SoftStepper stepper(f.document, 0, body.id);
        const SoftStepResult first = StepAt(f.document, body.id, 0, 0.0f);
        const SoftStepResult quarter = StepAt(f.document, body.id, 0, 0.25f);
        const SoftStepResult last = StepAt(f.document, body.id, 0, 1.0f);
        REQUIRE(quarter.moved);
        f32 swing = 0.0f;
        for (u32 p = 0; p < stepper.rig().pointCount(); ++p) {
            CHECK((first.goal[p] - last.goal[p]).length() < 2e-5f);
            CHECK((first.posed[p] - last.posed[p]).length() < 2e-4f);
            swing = std::max(swing, (quarter.goal[p] - quarter.carried[p]).length());
        }
        // A tenth of the region's size, at the top of the swing.
        INFO("pattern " << static_cast<int>(pattern) << " swing " << swing);
        CHECK(swing == Approx(0.6f).epsilon(0.15));
        const f64 rest = SoftVolumeAt(stepper.volume(), quarter.carried);
        CHECK(SoftVolumeAt(stepper.volume(), quarter.goal) == Approx(rest).epsilon(2e-3));
        CHECK(SoftVolumeAt(stepper.volume(), quarter.posed) == Approx(rest).epsilon(2e-3));
    }
    // Along a side the body is thin on, a Jelly's stretch stops at four times:
    // asked for by its Amount alone it would be eight thousand here.
    {
        Figure f = MakeTube(Skinned::Two, 19, 16);
        SoftBody& body = SoftenAll(f, 12, true);
        SoftEffect& wobble = AddEffect(body, SoftEffectKind::Wobble);
        wobble.amount = 3.0f;
        wobble.wobble->pattern = SoftWobblePattern::Jelly;
        wobble.wobble->speed = 1.0f;
        wobble.wobble->axis = SoftAxis::X;
        // At the bind, where the tube is two across.
        const SoftStepResult quarter = StepAt(f.document, body.id, kInvalidIndex, 0.25f);
        REQUIRE(quarter.moved);
        f32 widest = 0.0f;
        for (const Vector3f& at : quarter.goal) {
            REQUIRE(std::isfinite(at.x + at.y + at.z));
            widest = std::max(widest, std::abs(at.x));
        }
        // The tube is two across: four times that, from its near side.
        CHECK(widest > 4.0f);
        CHECK(widest <= 7.0f + 1e-3f);
    }
    CHECK(SoftWholeCycles(1.0f, 4.93f, true) == Approx(5.0f / 4.93f));
    CHECK(SoftWholeCycles(1.0f, 2.77f, true) == Approx(3.0f / 2.77f));
    CHECK(SoftWholeCycles(0.1f, 2.0f, true) == Approx(0.5f));
    CHECK(SoftWholeCycles(1.0f, 4.93f, false) == 1.0f);
}

TEST_CASE("G-S2 a carrier scaled to nothing leaves its lumps at rest", "[wem][soft][fit]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    SoftBody& body = SoftenAll(f, 6, true);
    SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
    inflate.amount = 0.3f;
    inflate.everywhere = true;
    inflate.inflate->along = SoftAlong::Surface;
    const SoftStepper stepper(f.document, 0, body.id);
    // The tip hidden as a part is hidden: scaled to nothing about its pivot.
    std::vector<Matrix44f> skin = Bind(f.model());
    const Vector3f pivot = SoftBonePivot(f.model().nodes, f.bones[2]);
    Matrix44f hidden = Matrix44f::identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            hidden.data[i][j] = 0.0f;
        }
    }
    hidden.data[3][0] = pivot.x, hidden.data[3][1] = pivot.y, hidden.data[3][2] = pivot.z;
    skin[f.bones[2]] = hidden;
    SoftStepResult out;
    stepper.step(skin, SoftClock{kInvalidIndex, 0.0f}, SoftPoseAt{}, out);
    REQUIRE(out.moved);
    const f32 size = stepper.rig().points().size;
    u32 hung = 0, moved = 0;
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        const Transform& departure = out.departures[k];
        REQUIRE(std::isfinite(departure.translation.x + departure.translation.y + departure.translation.z));
        if (body.lumps[k].family() == f.bones[2]) {
            // No way back through it: any departure is as good, and none is
            // keyed for a clip to ease into.
            CHECK(SoftAtRest(departure, size));
            ++hung;
        } else {
            moved += SoftAtRest(departure, size) ? 0 : 1;
        }
    }
    CHECK(hung > 0u);
    CHECK(moved > 0u);
}

TEST_CASE("G-S2 a carrier flattened along one axis leaves its lumps at rest", "[wem][soft][fit]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    SoftBody& body = SoftenAll(f, 6, true);
    SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
    inflate.amount = 0.3f;
    inflate.everywhere = true;
    inflate.inflate->along = SoftAlong::Surface;
    const SoftStepper stepper(f.document, 0, body.id);
    // The tip hidden by one axis, as a Reforged death hides a base ring: a
    // thousandth along it and the other two kept, about its pivot. Its volume
    // is far from nothing, and a fit through it is noise a thousand times over.
    std::vector<Matrix44f> skin = Bind(f.model());
    const Vector3f pivot = SoftBonePivot(f.model().nodes, f.bones[2]);
    const f32 scale[3] = {0.001f, 0.26f, 1.0f};
    const f32 about[3] = {pivot.x, pivot.y, pivot.z};
    Matrix44f flat = Matrix44f::identity();
    for (int i = 0; i < 3; ++i) {
        flat.data[i][i] = scale[i];
        flat.data[3][i] = about[i] * (1.0f - scale[i]);
    }
    skin[f.bones[2]] = flat;
    SoftStepResult out;
    stepper.step(skin, SoftClock{kInvalidIndex, 0.0f}, SoftPoseAt{}, out);
    REQUIRE(out.moved);
    const f32 size = stepper.rig().points().size;
    u32 hung = 0, moved = 0;
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        const Transform& departure = out.departures[k];
        REQUIRE(std::isfinite(departure.translation.x + departure.translation.y + departure.translation.z));
        CHECK(departure.translation.length() < size);
        if (body.lumps[k].family() == f.bones[2]) {
            CHECK(SoftAtRest(departure, size));
            ++hung;
        } else {
            moved += SoftAtRest(departure, size) ? 0 : 1;
        }
    }
    CHECK(hung > 0u);
    CHECK(moved > 0u);
    // Squashed to a third along one axis, a carrier is no hidden part: its lumps fit.
    flat = Matrix44f::identity();
    flat.data[0][0] = 0.35f;
    flat.data[3][0] = pivot.x * 0.65f;
    skin[f.bones[2]] = flat;
    stepper.step(skin, SoftClock{kInvalidIndex, 0.0f}, SoftPoseAt{}, out);
    u32 fitted = 0;
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        fitted += body.lumps[k].family() == f.bones[2] && !SoftAtRest(out.departures[k], size) ? 1 : 0;
    }
    CHECK(fitted > 0u);
}

TEST_CASE("G-S2 a keyed squash is a stretch the lumps carry exactly, with the volume kept", "[wem][soft][fields]") {
    Figure f = MakeTube(Skinned::Two);
    SoftBody& body = SoftenAll(f, 8, true);
    SoftEffect& squash = AddEffect(body, SoftEffectKind::SquashStretch);
    squash.squash->mode = SoftSquashMode::Keyed;
    squash.squash->axis = SoftAxis::Up;
    squash.squash->stretch = 0.3f;
    squash.squash->squash = 0.3f;
    squash.amount = 1.0f;
    const SoftStepper stepper(f.document, 0, body.id);
    const SoftStepResult out = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(out.moved);
    f32 top = 0.0f;
    for (const Vector3f& p : out.goal) {
        top = std::max(top, p.z);
    }
    // Stretched by 30 % from its base, and thinned by the root of that.
    CHECK(top == Approx(6.0f * 1.3f).margin(1e-4));
    CHECK(out.fit.follows() < 1e-4f);
    CHECK(SoftVolumeAt(stepper.volume(), out.posed) == Approx(stepper.volume().rest).epsilon(1e-4));
    // A full squash the other way.
    body.effects[0].amount = -1.0f;
    const SoftStepResult down = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    top = 0.0f;
    for (const Vector3f& p : down.goal) {
        top = std::max(top, p.z);
    }
    CHECK(top == Approx(6.0f * 0.7f).margin(1e-4));
}

TEST_CASE("G-S2 Auto squash reads each bone's own motion and none of a still pose", "[wem][soft][fields]") {
    Figure f = MakeTube(Skinned::Two);
    // A third clip: the whole tube thrown along +X and stopped.
    Clip dash;
    dash.name = "Dash";
    dash.model = 0;
    dash.duration = 1.0f;
    dash.containers.emplace_back();
    Key<Vector3f>(dash, f.model().animChannels.channels[2].id, {0.0f, 0.4f, 0.5f, 1.0f},
                  {Vector3f{0, 0, 0}, Vector3f{8, 0, 0}, Vector3f{8, 0, 0}, Vector3f{8, 0, 0}});
    f.document.clips.push_back(dash);
    SoftBody& body = SoftenAll(f, 8, true);
    SoftEffect& squash = AddEffect(body, SoftEffectKind::SquashStretch);
    squash.squash->mode = SoftSquashMode::Auto;
    const SoftStepper stepper(f.document, 0, body.id);
    // At speed: longer along the motion.
    const SoftStepResult fast = StepAt(f.document, body.id, 2, 0.2f);
    REQUIRE(fast.moved);
    f32 low = 1e9f, high = -1e9f, wasLow = 1e9f, wasHigh = -1e9f;
    for (u32 p = 0; p < stepper.rig().pointCount(); ++p) {
        low = std::min(low, fast.goal[p].x);
        high = std::max(high, fast.goal[p].x);
        wasLow = std::min(wasLow, fast.carried[p].x);
        wasHigh = std::max(wasHigh, fast.carried[p].x);
    }
    CHECK(high - low > (wasHigh - wasLow) * 1.1f);
    // Standing still, long after the stop: as it is.
    const SoftStepResult still = StepAt(f.document, body.id, 2, 0.8f);
    for (u32 p = 0; p < stepper.rig().pointCount(); ++p) {
        CHECK((still.goal[p] - still.carried[p]).length() < 1e-4f);
    }
}

TEST_CASE("G-S2 Wind-up leans back before a move and carries past its end", "[wem][soft][fields]") {
    Figure f = MakeTube(Skinned::Two);
    // The whole tube still, then thrown along +X between 0.5 s and 0.6 s, then still.
    const auto lunge = [&](const char* name, bool looping) {
        Clip clip;
        clip.name = name;
        clip.model = 0;
        clip.duration = 1.5f;
        clip.looping = looping;
        clip.containers.emplace_back();
        Key<Vector3f>(clip, f.model().animChannels.channels[2].id, {0.0f, 0.5f, 0.6f, 1.5f},
                      {Vector3f{0, 0, 0}, Vector3f{0, 0, 0}, Vector3f{2, 0, 0}, Vector3f{2, 0, 0}});
        f.document.clips.push_back(clip);
        return static_cast<u32>(f.document.clips.size() - 1);
    };
    const u32 once = lunge("Lunge", false);
    SoftBody& body = SoftenAll(f, 8, true);
    SoftEffect& squash = AddEffect(body, SoftEffectKind::SquashStretch);
    squash.squash->mode = SoftSquashMode::Auto;
    // Nothing of the stretch, the squash or the turn: the Wind-up alone.
    squash.squash->stretch = 0.0f;
    squash.squash->squash = 0.0f;
    squash.squash->turning = 0.0f;
    squash.squash->windUp = 1.0f;
    const SoftStepper stepper(f.document, 0, body.id);
    const u32 count = stepper.rig().pointCount();
    const f32 window = SoftWindUpWindow(squash.squash->smoothing, 1.5f);
    CHECK(window == Approx(0.16f));
    // A small band is asked for whole; a cut in the animation never for half the region.
    CHECK(SoftWindUpLean(Vector3f{0.01f, 0, 0}, 6.0f).x == Approx(0.01f).epsilon(1e-4));
    CHECK(SoftWindUpLean(Vector3f{0, 300.0f, 0}, 6.0f).y < 3.0f);
    CHECK(SoftWindUpLean(Vector3f{0, 300.0f, 0}, 6.0f).y > 2.9f);
    // The lean along the move, at the body's top and at the middle of its base.
    const f32 size = stepper.rig().points().size;
    const auto lean = [&](u32 clip, f32 seconds, f32& top, f32& base) {
        const SoftStepResult out = StepAt(f.document, body.id, clip, seconds);
        top = base = 0.0f;
        for (u32 p = 0; p < count; ++p) {
            const Vector3f& at = stepper.rig().points().points[p].at;
            const f32 moved = out.goal[p].x - out.carried[p].x;
            if ((at - Vector3f{0, 0, 6}).length() < 1e-4f) {
                top = moved;
            } else if (at.length() < 1e-4f) {
                base = moved;
            }
            // Along the move and nowhere else.
            CHECK(std::abs(out.goal[p].y - out.carried[p].y) < 1e-5f);
            CHECK(std::abs(out.goal[p].z - out.carried[p].z) < 1e-5f);
        }
    };
    f32 top = 0.0f, base = 0.0f;
    // Just before the move: back, by the band a body of this size is asked for.
    lean(once, 0.45f, top, base);
    // It stood at 0 a moment ago and stands at 0; a moment on it is at 2.
    const f32 asked = SoftWindUpLean(Vector3f{-1, 0, 0}, size).x * std::min(6.0f / size, 1.0f);
    CHECK(asked < -0.5f);
    CHECK(top == Approx(asked).margin(1e-3));
    // What is held still does not lean: the middle of the base is the anchor.
    CHECK(std::abs(base) < 1e-4f);
    // Just past its end: on, the other way.
    lean(once, 0.65f, top, base);
    CHECK(top > 0.5f);
    // Well away from it on either side: as the animation has it.
    lean(once, 0.2f, top, base);
    CHECK(std::abs(top) < 1e-5f);
    lean(once, 1.0f, top, base);
    CHECK(std::abs(top) < 1e-5f);
    // By its level, and none at none.
    body.effects[0].squash->windUp = 0.5f;
    f32 half = 0.0f;
    lean(once, 0.45f, half, base);
    CHECK(half == Approx(0.5f * asked).margin(1e-3));
    body.effects[0].squash->windUp = 0.0f;
    lean(once, 0.45f, top, base);
    CHECK(std::abs(top) < 1e-5f);
    body.effects[0].squash->windUp = 1.0f;

    // A move at the clip's very start. Played once, the clip starts as the
    // animation has it; a loop leans back at its end, for the start to come.
    const auto early = [&](const char* name, bool looping) {
        Clip clip;
        clip.name = name;
        clip.model = 0;
        clip.duration = 1.5f;
        clip.looping = looping;
        clip.containers.emplace_back();
        Key<Vector3f>(clip, f.model().animChannels.channels[2].id, {0.0f, 0.05f, 0.15f, 1.3f, 1.5f},
                      {Vector3f{0, 0, 0}, Vector3f{0, 0, 0}, Vector3f{2, 0, 0}, Vector3f{2, 0, 0}, Vector3f{0, 0, 0}});
        f.document.clips.push_back(clip);
        return static_cast<u32>(f.document.clips.size() - 1);
    };
    const u32 played = early("Start", false);
    const u32 looped = early("Loop", true);
    lean(played, 0.0f, top, base);
    CHECK(std::abs(top) < 1e-5f);
    lean(played, 1.5f, top, base);
    CHECK(std::abs(top) < 1e-5f);
    f32 first = 0.0f, last = 0.0f;
    lean(looped, 0.0f, first, base);
    lean(looped, 1.5f, last, base);
    CHECK(first < -0.2f);
    // The loop closes: its last frame is its first.
    CHECK(last == Approx(first).margin(1e-4));
}

TEST_CASE("G-S2 a Bulge from contact is pressed in by what the body meets, and its volume comes back round it",
          "[wem][soft][fields]") {
    Figure f = MakeTube(Skinned::Two, 25, 24);
    SoftBody& body = SoftenAll(f, 8, true);
    // A ball that reaches 0.3 into the tube's side, half way up.
    ClothCollider ball;
    ball.id = f.model().physics.allocateId();
    ball.transform.data[3][0] = 1.5f;
    ball.transform.data[3][2] = 3.0f;
    ball.radius = 0.8f;
    f.model().physics.colliders.push_back(ball);
    const Vector3f centre{1.5f, 0.0f, 3.0f};
    SoftEffect& bulge = AddEffect(body, SoftEffectKind::Bulge);
    bulge.bulge->from = SoftComesFrom::Contact;
    bulge.bulge->keepVolume = false;
    bulge.amount = 1.0f;
    const SoftStepper stepper(f.document, 0, body.id);
    const u32 count = stepper.rig().pointCount();
    // The body lists no collider: nothing is met.
    CHECK(!StepAt(f.document, body.id, kInvalidIndex, 0.0f).moved);
    body.colliders = {ball.id};
    const SoftStepResult pressed = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(pressed.moved);
    u32 inside = 0;
    for (u32 p = 0; p < count; ++p) {
        const f32 was = (pressed.carried[p] - centre).length();
        if (was < ball.radius) {
            // Out to its surface, straight away from its middle.
            ++inside;
            CHECK((pressed.goal[p] - centre).length() == Approx(ball.radius).margin(1e-4));
            CHECK(cross(pressed.goal[p] - centre, pressed.carried[p] - centre).length() < 1e-4f);
        } else {
            // With no volume to keep, nothing else is asked to move.
            CHECK((pressed.goal[p] - pressed.carried[p]).length() == 0.0f);
        }
    }
    CHECK(inside > 3u);
    const f64 rest = stepper.volume().rest;
    const f64 dented = SoftVolumeAt(stepper.volume(), pressed.goal);
    CHECK(dented < rest * (1.0 - 1e-3));

    // *Keep volume*: what the dent displaced comes back, as a rise round the contact.
    body.effects[0].bulge->keepVolume = true;
    const SoftStepResult kept = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    CHECK(SoftVolumeAt(stepper.volume(), kept.goal) == Approx(rest).epsilon(1e-4));
    const f32 band = 0.3f * stepper.rig().points().size;
    f32 beside = 0.0f, away = 0.0f;
    for (u32 p = 0; p < count; ++p) {
        const f32 from = (kept.carried[p] - centre).length() - ball.radius;
        const f32 rise = (kept.goal[p] - kept.carried[p]).dot(stepper.rig().points().points[p].outward);
        if (from < 0.0f) {
            // What is pressed stays on the ball.
            CHECK((kept.goal[p] - centre).length() == Approx(ball.radius).margin(1e-4));
        } else if (from < 0.3f * band) {
            beside = std::max(beside, rise);
        } else if (from > band) {
            away = std::max(away, std::abs(rise));
        }
    }
    CHECK(beside > 1e-3f);
    CHECK(away < 1e-6f);
    // A place's radius is how wide the rise is.
    body.effects[0].place.radius = 0.5f * band;
    {
        const SoftStepResult narrow = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
        f32 past = 0.0f;
        for (u32 p = 0; p < count; ++p) {
            if ((narrow.carried[p] - centre).length() - ball.radius > 0.5f * band) {
                past = std::max(past, (narrow.goal[p] - narrow.carried[p]).length());
            }
        }
        CHECK(past < 1e-6f);
        CHECK(SoftVolumeAt(stepper.volume(), narrow.goal) == Approx(rest).epsilon(1e-4));
    }
    body.effects[0].place.radius = 0.0f;

    // By its Amount: half way out; and none at none.
    body.effects[0].bulge->keepVolume = false;
    body.effects[0].amount = 0.5f;
    {
        const SoftStepResult half = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
        for (u32 p = 0; p < count; ++p) {
            const f32 was = (half.carried[p] - centre).length();
            if (was < ball.radius) {
                CHECK((half.goal[p] - centre).length() == Approx(0.5f * (was + ball.radius)).margin(1e-4));
            }
        }
    }
    body.effects[0].amount = 0.0f;
    CHECK(!StepAt(f.document, body.id, kInvalidIndex, 0.0f).moved);
    body.effects[0].amount = 1.0f;
    // A collider that touches nothing presses nothing.
    f.model().physics.colliders[0].transform.data[3][0] = 9.0f;
    CHECK(!StepAt(f.document, body.id, kInvalidIndex, 0.0f).moved);
    f.model().physics.colliders[0].transform.data[3][0] = 1.5f;
    // The lumps follow it as they can: the dent shows on what is posed.
    {
        const SoftStepResult out = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
        f32 deepest = 0.0f, shown = 0.0f;
        for (u32 p = 0; p < count; ++p) {
            deepest = std::max(deepest, ball.radius - (out.carried[p] - centre).length());
            shown = std::max(shown, (out.posed[p] - out.carried[p]).length());
        }
        WARN("a dent " << deepest << " deep moves the posed surface " << shown << " at most, on 8 surface lumps");
        CHECK(shown > 0.0f);
    }
}

TEST_CASE("G-S2 a place claims seven lumps of its own, bound by a pattern, and they carry its bump",
          "[wem][soft][fields][claims]") {
    Figure f = MakePatch();
    // One lump: the bone's copy. A bump is not its to show.
    SoftBody& body = SoftenAll(f, 0, true);
    const std::size_t before = body.lumps.size();
    REQUIRE(before == 1u);
    SoftEffect& bulge = AddEffect(body, SoftEffectKind::Bulge);
    bulge.bulge->along = SoftAlong::Surface;
    bulge.bulge->keepVolume = false;
    bulge.amount = 0.4f;
    // Not placed, it claims nothing.
    CHECK(!SoftEffectClaims(bulge));
    const Mesh& mesh = f.model().meshes[0];
    bulge.place.mesh = 0;
    bulge.place.face = FaceNear(mesh, Vector3f{0.3f, -0.2f, 0.0f});
    bulge.place.weights = Vector3f{1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
    bulge.place.radius = 1.0f;
    CHECK(SoftEffectClaims(bulge));
    {
        // Along the bones, from contact, or travelling, its place is not a disc to serve.
        SoftEffect other = bulge;
        other.bulge->along = SoftAlong::Bones;
        CHECK(!SoftEffectClaims(other));
        CHECK(SoftClaimsDiffer(bulge, other));
        other = bulge;
        other.bulge->from = SoftComesFrom::Contact;
        CHECK(!SoftEffectClaims(other));
        other = bulge;
        other.bulge->travels = true;
        CHECK(!SoftEffectClaims(other));
        // A setting that is not its place changes no claim; its radius does.
        other = bulge;
        other.amount = 0.9f;
        other.bulge->shape = SoftBulgeShape::Ring;
        CHECK(!SoftClaimsDiffer(bulge, other));
        other.place.radius = 0.5f;
        CHECK(SoftClaimsDiffer(bulge, other));
        // An Inflate claims when it swells toward a place, along the surface.
        SoftEffect inflate = DefaultSoftEffect(SoftEffectKind::Inflate);
        inflate.place = bulge.place;
        inflate.inflate->along = SoftAlong::Surface;
        CHECK(!SoftEffectClaims(inflate));
        inflate.inflate->towardPlace = true;
        CHECK(SoftEffectClaims(inflate));
        CHECK(!SoftEffectClaims(DefaultSoftEffect(SoftEffectKind::Wobble)));
    }
    // How far the lumps leave the points from where the bump asks them to be.
    const auto miss = [&] {
        const SoftStepResult out = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
        f32 worst = 0.0f;
        for (std::size_t p = 0; p < out.posed.size(); ++p) {
            worst = std::max(worst, (out.posed[p] - out.goal[p]).length());
        }
        return worst;
    };
    const f32 bare = miss();
    CHECK(bare > 0.2f);

    // The whole disc rides the one bone and has slots to spare: the seven pay.
    {
        const SoftPoints points = GatherSoftPoints(f.model(), body.id, 4);
        const SoftPlaceRoom room = SoftPlaceRoomOf(f.model(), points, bulge);
        CHECK(room.valid);
        CHECK(room.bone == f.bones[0]);
        CHECK(room.room == Approx(1.0f).margin(1e-4));
        CHECK(room.pays());
        // With one weight a point there is no slot for a lane: they would change nothing.
        const SoftPoints crowded = GatherSoftPoints(f.model(), body.id, 1);
        CHECK(SoftPlaceRoomOf(f.model(), crowded, bulge).room == 0.0f);
        CHECK(!SoftPlaceRoomOf(f.model(), crowded, bulge).pays());
        // A place off the body is none.
        SoftEffect off = bulge;
        off.place.face = 1u << 30;
        CHECK(!SoftPlaceRoomOf(f.model(), points, off).valid);
    }
    const SoftMakeReport report = MakeSoftLumps(f.model(), body, 4, false);
    CHECK(report.claimed == kSoftPlaceLumps);
    REQUIRE(body.lumps.size() == before + kSoftPlaceLumps);
    const std::span<const Vector3f> positions =
        mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    const f32 cell = 4.0f / 40.0f;
    const Vector3f centre = body.lumps[before].anchor;
    // A hub at the place and six spokes round it at six tenths of its radius,
    // each on a point of the body, each the effect's and hung from the bone.
    CHECK((centre - Vector3f{0.3f, -0.2f, 0.0f}).length() < 1.5f * cell);
    for (std::size_t k = before; k < body.lumps.size(); ++k) {
        const SoftLump& lump = body.lumps[k];
        CHECK(lump.kind == SoftLumpKind::Surface);
        CHECK(lump.effect == body.effects[0].id);
        CHECK(lump.carrier == f.bones[0]);
        CHECK(lump.rider == 0);
        if (k > before) {
            CHECK((lump.anchor - centre).length() == Approx(kSoftPlaceSpoke).margin(1.5f * cell));
            // A sixth of a turn from the one before it.
            const Vector3f& last = body.lumps[k == before + 1 ? before + 6 : k - 1].anchor;
            CHECK((lump.anchor - last).length() == Approx(kSoftPlaceSpoke).margin(2.5f * cell));
        }
        bool onBody = false;
        for (const Vector3f& p : positions) {
            onBody = onBody || (p - lump.anchor).length() < 1e-6f;
        }
        CHECK(onBody);
    }

    // The pattern: all of a point's weight on the seven inside the spokes'
    // ring, none at the rim or past it, in a straight line between; the hub
    // whole at the place; three of them at most; and where the pattern is
    // whole, their anchors land by the point.
    {
        SoftSkinner skinner(f.model(), body);
        const u32 first = skinner.lumpBase() + static_cast<u32>(before);
        u32 inside = 0, between = 0, atHub = 0;
        for (u32 v = 0; v < positions.size(); ++v) {
            const f32 rho = (positions[v] - centre).length();
            f32 sum = 0.0f, onSeven = 0.0f;
            u32 lanes = 0;
            Vector3f lands{0, 0, 0};
            for (const geom::Influence& influence : skinner.of(0, v)) {
                CHECK(influence.weight > 0.0f);
                sum += influence.weight;
                if (influence.bone >= first) {
                    onSeven += influence.weight;
                    ++lanes;
                    lands = lands + body.lumps[influence.bone - skinner.lumpBase()].anchor * influence.weight;
                    if (rho < 1e-4f) {
                        CHECK(influence.bone == first);
                    }
                }
            }
            CHECK(sum == Approx(1.0f).margin(1e-5));
            CHECK(lanes <= 3u);
            if (rho >= 1.0f) {
                CHECK(onSeven == 0.0f);
            } else if (rho <= kSoftPlaceSpoke) {
                ++inside;
                CHECK(onSeven == Approx(1.0f).margin(1e-4));
                CHECK((lands - positions[v]).length() < 0.2f);
                atHub += rho < 1e-4f ? 1 : 0;
            } else {
                ++between;
                CHECK(onSeven == Approx((1.0f - rho) / (1.0f - kSoftPlaceSpoke)).margin(1e-3));
            }
        }
        CHECK(inside > 50u);
        CHECK(between > 50u);
        CHECK(atHub == 1u);
    }
    // With every lump at rest the model is its own skin: a pattern is lanes like any other.
    {
        const SoftRig rig(f.model(), body.id);
        std::vector<Vector3f> posed;
        rig.pose(Bind(f.model()), {}, posed);
        for (u32 p = 0; p < rig.pointCount(); ++p) {
            CHECK((posed[p] - rig.points().points[p].at).length() < 1e-5f);
        }
    }
    // And they carry the bump.
    const f32 carried = miss();
    WARN("a bump 0.4 high on one bone misses by " << bare << "; with its seven lumps by " << carried);
    CHECK(carried < 0.35f * bare);

    // The place moved, its seven move, and the lumps before them keep their numbers.
    const SoftLump copy = body.lumps[0];
    body.effects[0].place.face = FaceNear(mesh, Vector3f{-1.0f, 0.8f, 0.0f});
    CHECK(MakeSoftLumps(f.model(), body, 4, false).claimed == kSoftPlaceLumps);
    REQUIRE(body.lumps.size() == before + kSoftPlaceLumps);
    CHECK(body.lumps[0].kind == copy.kind);
    CHECK(body.lumps[0].node == copy.node);
    CHECK((body.lumps[before].anchor - Vector3f{-1.0f, 0.8f, 0.0f}).length() < 1.5f * cell);
    // A second effect claims its own seven, after the first's.
    {
        SoftEffect& second = AddEffect(body, SoftEffectKind::Inflate);
        second.inflate->along = SoftAlong::Surface;
        second.inflate->towardPlace = true;
        second.place = body.effects[0].place;
        second.place.face = FaceNear(mesh, Vector3f{1.0f, -1.0f, 0.0f});
        CHECK(MakeSoftLumps(f.model(), body, 4, false).claimed == 2u * kSoftPlaceLumps);
        REQUIRE(body.lumps.size() == before + 2u * kSoftPlaceLumps);
        CHECK(body.lumps[before].effect == body.effects[0].id);
        CHECK(body.lumps[before + kSoftPlaceLumps].effect == body.effects[1].id);
        body.effects.pop_back();
    }
    // Under one weight a point, or a classic group's equal shares, there are none.
    CHECK(MakeSoftLumps(f.model(), body, 1, false).claimed == 0u);
    CHECK(body.lumps.size() == before);
    CHECK(MakeSoftLumps(f.model(), body, 4, true).claimed == 0u);
    CHECK(body.lumps.size() == before);
    // Said along the bones they go; and no lane is left that names one.
    CHECK(MakeSoftLumps(f.model(), body, 4, false).claimed == kSoftPlaceLumps);
    body.effects[0].bulge->along = SoftAlong::Bones;
    CHECK(MakeSoftLumps(f.model(), body, 4, false).claimed == 0u);
    CHECK(body.lumps.size() == before);
    {
        SoftSkinner skinner(f.model(), body);
        for (u32 v = 0; v < positions.size(); ++v) {
            for (const geom::Influence& influence : skinner.of(0, v)) {
                CHECK(influence.bone < skinner.lumpBase() + before);
            }
        }
    }

    // The far face of a part thinner than the place's radius is out of its
    // disc (§8.2): it faces the other way, and takes none of the seven.
    {
        Figure tube = MakeTube(Skinned::Rigid, 25, 24);
        SoftBody& round = SoftenAll(tube, 0, true);
        SoftEffect& poke = AddEffect(round, SoftEffectKind::Bulge);
        poke.bulge->along = SoftAlong::Surface;
        const Mesh& skin = tube.model().meshes[0];
        poke.place.mesh = 0;
        poke.place.face = FaceNear(skin, Vector3f{1.0f, 0.0f, 4.5f});
        poke.place.weights = Vector3f{1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};
        poke.place.radius = 2.3f; // The tube is two across.
        const std::size_t own = round.lumps.size();
        REQUIRE(MakeSoftLumps(tube.model(), round, 4, false).claimed == kSoftPlaceLumps);
        SoftSkinner skinner(tube.model(), round);
        const std::span<const Vector3f> at = skin.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
        u32 near = 0, far = 0;
        for (u32 v = 0; v < at.size(); ++v) {
            f32 onSeven = 0.0f;
            for (const geom::Influence& influence : skinner.of(0, v)) {
                onSeven += influence.bone >= skinner.lumpBase() + own ? influence.weight : 0.0f;
            }
            if (std::abs(at[v].z - 4.5f) < 0.3f && at[v].x > 0.9f) {
                ++near;
                CHECK(onSeven > 0.5f);
            } else if (std::abs(at[v].z - 4.5f) < 0.3f && at[v].x < -0.9f) {
                ++far;
                CHECK(onSeven == 0.0f);
            }
        }
        CHECK(near > 0u);
        CHECK(far > 0u);
    }
}

TEST_CASE("G-S2 every field at Amount 0 is the identity to the bit", "[wem][soft][fields]") {
    Figure f = MakeTube(Skinned::Two);
    SoftBody& body = SoftenAll(f, 8, true);
    for (const SoftEffectKind kind : {SoftEffectKind::Inflate, SoftEffectKind::Bulge, SoftEffectKind::SquashStretch,
                                      SoftEffectKind::Wobble}) {
        SoftEffect& effect = AddEffect(body, kind);
        effect.amount = 0.0f;
        effect.place.mesh = 0;
        effect.place.face = 5;
        if (effect.squash) {
            effect.squash->mode = SoftSquashMode::Keyed;
        }
    }
    // And one along the bones.
    SoftEffect& bones = AddEffect(body, SoftEffectKind::Inflate);
    bones.amount = 0.0f;
    bones.inflate->along = SoftAlong::Bones;
    for (const u32 clip : {kInvalidIndex, 0u}) {
        const SoftStepResult out = StepAt(f.document, body.id, clip, 0.5f);
        CHECK(!out.moved);
        REQUIRE(out.goal.size() == out.carried.size());
        CHECK(std::memcmp(out.goal.data(), out.carried.data(), out.goal.size() * sizeof(Vector3f)) == 0);
        const Transform rest;
        for (const Transform& departure : out.departures) {
            CHECK(std::memcmp(&departure, &rest, sizeof rest) == 0);
        }
    }
    // An effect switched off, or with no span in the clip and not on everywhere, asks nothing either.
    body.effects[0].amount = 0.3f;
    body.effects[0].everywhere = false;
    CHECK(!StepAt(f.document, body.id, 0, 0.5f).moved);
    body.effects[0].everywhere = true;
    CHECK(StepAt(f.document, body.id, 0, 0.5f).moved);
    body.effects[0].enabled = false;
    CHECK(!StepAt(f.document, body.id, 0, 0.5f).moved);
}

TEST_CASE("G-S2 a span is keys: its level, its progress and its marks", "[wem][soft][spans]") {
    Figure f = MakeTube(Skinned::Two);
    SoftBody& body = SoftenAll(f, 0, true);
    SoftEffect& bulge = AddEffect(body, SoftEffectKind::Bulge);
    bulge.everywhere = false;
    const u32 id = body.id, effect = bulge.id;
    CHECK(SoftAmountChannel(f.model(), id, effect) == kInvalidIndex);
    CHECK(SoftSpansOf(f.document, 0, 0, id, effect).empty());
    const std::size_t channels = f.model().animChannels.channels.size();

    const std::vector<SoftSpan> asked{{0.2f, 0.5f, 0.1f, 0.05f}, {0.7f, 0.9f, 0.0f, 0.0f}};
    SetSoftSpans(f.document, 0, 0, id, effect, asked);
    CHECK(f.model().animChannels.channels.size() == channels + 1);
    REQUIRE(SoftAmountChannel(f.model(), id, effect) != kInvalidIndex);
    const AnimChannel& made = f.model().animChannels.channels.back();
    CHECK(made.target.kind == TrackTarget::Kind::Physics);
    CHECK(made.target.channel == Channel::SoftAmount);
    CHECK(made.target.sub == SoftSub(id, effect));
    const std::vector<SoftSpan> read = SoftSpansOf(f.document, 0, 0, id, effect);
    REQUIRE(read.size() == 2u);
    for (std::size_t k = 0; k < 2; ++k) {
        CHECK(read[k].start == Approx(asked[k].start).margin(1e-6));
        CHECK(read[k].end == Approx(asked[k].end).margin(1e-6));
        CHECK(read[k].easeIn == Approx(asked[k].easeIn).margin(1e-6));
        CHECK(read[k].easeOut == Approx(asked[k].easeOut).margin(1e-6));
    }
    // The other clip keys nothing of it.
    CHECK(SoftSpansOf(f.document, 0, 1, id, effect).empty());

    const auto at = [&](f32 seconds) { return SoftTimingAt(f.document, 0, 0, body, bulge, seconds); };
    // The spans reach neither end of the clip, and both ends are keyed at
    // nothing: a detach, which a clip's first bake does and which keys an edge
    // the game's way (the last key wrapped round to the first), finds them
    // keyed and leaves the spans where they were placed.
    {
        Figure wrapped = f;
        DetachClip(wrapped.document, 0, 0);
        const std::vector<SoftSpan> after = SoftSpansOf(wrapped.document, 0, 0, id, effect);
        REQUIRE(after.size() == 2u);
        for (std::size_t k = 0; k < 2; ++k) {
            CHECK(after[k].start == Approx(asked[k].start).margin(1e-3));
            CHECK(after[k].end == Approx(asked[k].end).margin(1e-3));
        }
        const SoftBody& same = *wrapped.model().physics.softBody(id);
        CHECK(SoftTimingAt(wrapped.document, 0, 0, same, *same.effect(effect), 0.0f).level == 0.0f);
        CHECK(SoftTimingAt(wrapped.document, 0, 0, same, *same.effect(effect), 0.1f).level == 0.0f);
        CHECK(SoftTimingAt(wrapped.document, 0, 0, same, *same.effect(effect), 0.95f).level == 0.0f);
    }
    CHECK(at(0.1f).level == 0.0f);
    CHECK(!at(0.1f).within);
    CHECK(at(0.1f).sinceStart.empty());
    CHECK(at(0.25f).level == Approx(0.5f).margin(1e-4));
    CHECK(at(0.35f).level == 1.0f);
    CHECK(at(0.35f).within);
    CHECK(at(0.35f).progress == Approx(0.5f).margin(1e-4));
    CHECK(at(0.475f).level == Approx(0.5f).margin(1e-4));
    CHECK(at(0.6f).level == 0.0f);
    // Between spans the progress holds, and the first span's start is still a mark.
    CHECK(at(0.6f).progress == 1.0f);
    REQUIRE(at(0.6f).sinceStart.size() == 1u);
    CHECK(at(0.6f).sinceStart[0] == Approx(0.4f).margin(1e-5));
    CHECK(at(0.8f).level == Approx(1.0f).margin(1e-3));
    REQUIRE(at(0.8f).sinceStart.size() == 2u);
    CHECK(at(0.8f).sinceStart[0] == Approx(0.1f).margin(1e-5));

    // The whole clip at full: two keys, and the level is 1 from end to end.
    SetSoftSpans(f.document, 0, 0, id, effect, std::vector<SoftSpan>{{0.0f, 1.0f, 0.0f, 0.0f}});
    CHECK(f.model().animChannels.channels.size() == channels + 1);
    REQUIRE(SoftSpansOf(f.document, 0, 0, id, effect).size() == 1u);
    CHECK(SoftSpansOf(f.document, 0, 0, id, effect)[0] == SoftSpan{0.0f, 1.0f, 0.0f, 0.0f});
    CHECK(at(0.0f).level == 1.0f);
    CHECK(at(1.0f).level == 1.0f);
    // No span takes the keys away, and the clip then follows *Everywhere*.
    SetSoftSpans(f.document, 0, 0, id, effect, {});
    CHECK(SoftSpansOf(f.document, 0, 0, id, effect).empty());
    CHECK(at(0.5f).level == 0.0f);
    bulge.everywhere = true;
    CHECK(at(0.5f).level == 1.0f);
    CHECK(at(0.5f).within);
}

TEST_CASE("G-S2 a Bulge from Times is an impact, and one that Travels runs its line", "[wem][soft][fields]") {
    Figure f = MakeTube(Skinned::Two, 25, 16);
    SoftBody& body = SoftenAll(f, 12, true);
    SoftEffect& bulge = AddEffect(body, SoftEffectKind::Bulge);
    bulge.everywhere = false;
    bulge.amount = 0.4f;
    bulge.bulge->keepVolume = false;
    bulge.bulge->from = SoftComesFrom::Times;
    bulge.place.mesh = 0;
    bulge.place.face = FaceNear(f.model().meshes[0], Vector3f{1, 0, 3});
    bulge.place.radius = 1.0f;
    SetSoftSpans(f.document, 0, 0, body.id, bulge.id, std::vector<SoftSpan>{{0.5f, 0.6f, 0.0f, 0.0f}});
    const auto height = [&](f32 seconds) {
        const SoftStepResult out = StepAt(f.document, body.id, 0, seconds);
        f32 most = 0.0f;
        for (std::size_t p = 0; p < out.goal.size(); ++p) {
            most = std::max(most, (out.goal[p] - out.carried[p]).length());
        }
        return most;
    };
    // Nothing before the mark; up in 0.03 s; down over the next quarter second.
    CHECK(height(0.4f) == 0.0f);
    CHECK(height(0.53f) == Approx(0.4f).margin(0.05));
    CHECK(height(0.65f) < height(0.53f));
    CHECK(height(0.65f) > 0.05f);
    CHECK(height(0.9f) == 0.0f);

    // Travels: along the chain of bones, where the span's progress has it.
    bulge.bulge->from = SoftComesFrom::Spans;
    bulge.bulge->travels = true;
    bulge.bulge->chain = f.bones;
    SetSoftSpans(f.document, 0, 1, body.id, bulge.id, std::vector<SoftSpan>{{0.0f, 1.0f, 0.0f, 0.0f}});
    const SoftStepper stepper(f.document, 0, body.id);
    for (const f32 seconds : {0.25f, 0.75f}) {
        const SoftStepResult out = StepAt(f.document, body.id, 1, seconds);
        REQUIRE(out.moved);
        f32 most = 0.0f, where = 0.0f;
        for (u32 p = 0; p < stepper.rig().pointCount(); ++p) {
            const f32 moved = (out.goal[p] - out.carried[p]).length();
            if (moved > most) {
                most = moved;
                where = stepper.rig().points().points[p].at.z;
            }
        }
        // The chain is four long, from the first bone to the last.
        CHECK(where == Approx(4.0f * seconds).margin(0.3));
    }
}

TEST_CASE("G-S2 a Bulge that travels takes rows of lumps along its line, and they carry its swell",
          "[wem][soft][fields][claims]") {
    Figure f = MakeTube(Skinned::Rigid, 25, 16);
    // One lump a bone: the three copies. A swell a radius long is not theirs to show.
    SoftBody& body = SoftenAll(f, 0, true);
    const std::size_t before = body.lumps.size();
    REQUIRE(before == 3u);
    SoftEffect& bulge = AddEffect(body, SoftEffectKind::Bulge);
    bulge.everywhere = false;
    bulge.amount = 0.4f;
    bulge.bulge->along = SoftAlong::Surface;
    bulge.bulge->keepVolume = false;
    bulge.bulge->travels = true;
    bulge.place.radius = 1.0f;
    // With no line to run it claims nothing, of either kind.
    CHECK(!SoftEffectClaimsRow(bulge));
    CHECK(!SoftEffectClaims(bulge));
    bulge.bulge->chain = f.bones;
    CHECK(SoftEffectClaimsRow(bulge));
    CHECK(!SoftEffectClaims(bulge));
    {
        // Along the bones, from contact, or staying, it has no row.
        SoftEffect other = bulge;
        other.bulge->along = SoftAlong::Bones;
        CHECK(!SoftEffectClaimsRow(other));
        CHECK(SoftClaimsDiffer(bulge, other));
        other = bulge;
        other.bulge->from = SoftComesFrom::Contact;
        CHECK(!SoftEffectClaimsRow(other));
        other = bulge;
        other.bulge->travels = false;
        CHECK(!SoftEffectClaimsRow(other));
        CHECK(SoftClaimsDiffer(bulge, other));
        // Its height and its shape change no claim; its line and its radius do.
        other = bulge;
        other.amount = 0.9f;
        other.bulge->shape = SoftBulgeShape::Ring;
        CHECK(!SoftClaimsDiffer(bulge, other));
        other.place.radius = 0.5f;
        CHECK(SoftClaimsDiffer(bulge, other));
        other = bulge;
        other.bulge->chain = {f.bones[0], f.bones[1]};
        CHECK(SoftClaimsDiffer(bulge, other));
        other = bulge;
        other.bulge->chain.clear();
        other.bulge->line = {Vector3f{1, 0, 1}, Vector3f{1, 0, 5}};
        CHECK(SoftEffectClaimsRow(other));
        SoftEffect moved = other;
        CHECK(!SoftClaimsDiffer(other, moved));
        moved.bulge->line[1].z = 4.0f;
        CHECK(SoftClaimsDiffer(other, moved));
    }
    // It runs the chain across Twist. How far the lumps leave the points from
    // where the swell asks them to be, at the worst of three moments.
    SetSoftSpans(f.document, 0, 1, body.id, bulge.id, std::vector<SoftSpan>{{0.0f, 1.0f, 0.0f, 0.0f}});
    const auto miss = [&] {
        f32 worst = 0.0f;
        for (const f32 seconds : {0.3f, 0.5f, 0.7f}) {
            const SoftStepResult out = StepAt(f.document, body.id, 1, seconds);
            for (std::size_t p = 0; p < out.posed.size(); ++p) {
                worst = std::max(worst, (out.posed[p] - out.goal[p]).length());
            }
        }
        return worst;
    };
    const f32 bare = miss();
    CHECK(bare > 0.15f);

    const SoftMakeReport report = MakeSoftLumps(f.model(), body, 4, false);
    REQUIRE(body.lumps.size() > before);
    CHECK(report.claimed == body.lumps.size() - before);
    CHECK(report.claimed <= kSoftRowLumps);
    // Each bone's row: on the chain's axis, over the stretch of it the bone's
    // points lie by (the tube's bones carry z under 1, 1 to 3, and the rest;
    // the chain ends at 4), a step of half a radius or less apart.
    const f32 from[3] = {0.0f, 1.0f, 3.0f}, to[3] = {0.75f, 2.75f, 4.0f};
    u32 inRow[3] = {0, 0, 0};
    for (std::size_t k = before; k < body.lumps.size(); ++k) {
        const SoftLump& lump = body.lumps[k];
        CHECK(lump.kind == SoftLumpKind::Surface);
        CHECK(lump.effect == bulge.id);
        CHECK(lump.rider == 0);
        CHECK(std::abs(lump.anchor.x) + std::abs(lump.anchor.y) < 1e-5f);
        const auto bone = std::find(f.bones.begin(), f.bones.end(), lump.carrier);
        REQUIRE(bone != f.bones.end());
        const std::size_t b = static_cast<std::size_t>(bone - f.bones.begin());
        CHECK(lump.anchor.z >= from[b] - 1e-4f);
        CHECK(lump.anchor.z <= to[b] + 1e-4f);
        if (inRow[b] == 0) {
            CHECK(lump.anchor.z == Approx(from[b]).margin(1e-4));
        } else {
            const f32 step = lump.anchor.z - body.lumps[k - 1].anchor.z;
            CHECK(step > 0.3f);
            CHECK(step <= 0.5f + 1e-4f);
        }
        ++inRow[b];
    }
    for (std::size_t b = 0; b < 3; ++b) {
        CHECK(inRow[b] >= 3u);
    }
    CHECK(body.lumps.back().anchor.z == Approx(4.0f).margin(1e-4));

    // The pattern: all of a point is on its bone's row, on the two lumps it
    // lies between, and they land where it lies along the line.
    const Mesh& mesh = f.model().meshes[0];
    const std::span<const Vector3f> positions =
        mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    {
        SoftSkinner skinner(f.model(), body);
        const u32 first = skinner.lumpBase() + static_cast<u32>(before);
        u32 between = 0;
        for (u32 v = 0; v < positions.size(); ++v) {
            f32 sum = 0.0f, onRow = 0.0f, lands = 0.0f;
            u32 lanes = 0;
            for (const geom::Influence& influence : skinner.of(0, v)) {
                sum += influence.weight;
                if (influence.bone >= first) {
                    const SoftLump& lump = body.lumps[influence.bone - skinner.lumpBase()];
                    onRow += influence.weight;
                    lands += influence.weight * lump.anchor.z;
                    ++lanes;
                    // Its own bone's row, and no other's.
                    const std::size_t b = positions[v].z < 1.0f ? 0 : (positions[v].z < 3.0f ? 1 : 2);
                    CHECK(lump.carrier == f.bones[b]);
                }
            }
            CHECK(sum == Approx(1.0f).margin(1e-5));
            CHECK(onRow == Approx(1.0f).margin(1e-5));
            CHECK(lanes <= 2u);
            CHECK(lands == Approx(std::min(positions[v].z, 4.0f)).margin(1e-4));
            between += lanes == 2 ? 1 : 0;
        }
        CHECK(between > 50u);
    }
    // With every lump at rest the model is its own skin.
    {
        const SoftRig rig(f.model(), body.id);
        std::vector<Vector3f> posed;
        rig.pose(Bind(f.model()), {}, posed);
        for (u32 p = 0; p < rig.pointCount(); ++p) {
            CHECK((posed[p] - rig.points().points[p].at).length() < 1e-5f);
        }
    }
    // And they carry the swell.
    const f32 carried = miss();
    WARN("a swell 0.4 high down a chain of three bones misses by " << bare << "; with its rows of "
                                                                   << report.claimed << " lumps by " << carried);
    CHECK(carried < 0.5f * bare);

    // Made again as they stand: nothing moved, nothing changes, and the lumps
    // before them keep their numbers when the radius does.
    CHECK(!ClaimSoftPlaces(f.model(), body));
    body.effects[0].place.radius = 0.8f;
    CHECK(ClaimSoftPlaces(f.model(), body));
    CHECK(body.lumps[0].kind == SoftLumpKind::Copy);
    CHECK(body.lumps[2].node == f.bones[2]);
    CHECK(body.lumps.size() > before + report.claimed);
    // A swell that reaches no point claims none: three radii of 0.3 fall short of the skin.
    body.effects[0].place.radius = 0.3f;
    CHECK(ClaimSoftPlaces(f.model(), body));
    CHECK(body.lumps.size() == before);
    body.effects[0].place.radius = 1.0f;

    // With one weight a point there is one slot: the nearer lump of the row has all of it.
    CHECK(MakeSoftLumps(f.model(), body, 1, false).claimed == report.claimed);
    {
        SoftSkinner skinner(f.model(), body);
        for (u32 v = 0; v < positions.size(); ++v) {
            const std::span<const geom::Influence> skin = skinner.of(0, v);
            REQUIRE(skin.size() == 1u);
            REQUIRE(skin[0].bone >= skinner.lumpBase() + before);
            CHECK(skin[0].weight == Approx(1.0f).margin(1e-5));
            const f32 at = body.lumps[skin[0].bone - skinner.lumpBase()].anchor.z;
            CHECK(std::abs(at - std::min(positions[v].z, 4.0f)) <= 0.25f + 1e-4f);
        }
    }
    // Under a classic group's equal shares there are none.
    CHECK(MakeSoftLumps(f.model(), body, 4, true).claimed == 0u);
    CHECK(body.lumps.size() == before);

    // A line drawn over the surface, up the tube's side from 1 to 5: rows on
    // that line, for the two bones whose points lie by a stretch of it.
    body.effects[0].bulge->chain.clear();
    body.effects[0].bulge->line = {Vector3f{1, 0, 1}, Vector3f{1, 0, 5}};
    const u32 drawn = MakeSoftLumps(f.model(), body, 4, false).claimed;
    CHECK(drawn >= 8u);
    REQUIRE(body.lumps.size() == before + drawn);
    for (std::size_t k = before; k < body.lumps.size(); ++k) {
        const SoftLump& lump = body.lumps[k];
        CHECK((Vector3f{lump.anchor.x, lump.anchor.y, 0.0f} - Vector3f{1, 0, 0}).length() < 1e-5f);
        CHECK(lump.anchor.z >= 1.0f - 1e-4f);
        CHECK(lump.anchor.z <= 5.0f + 1e-4f);
        // The first bone's points all lie by the line's start: it has no stretch.
        CHECK(lump.carrier != f.bones[0]);
    }
    // No line, no rows.
    body.effects[0].bulge->line.clear();
    CHECK(MakeSoftLumps(f.model(), body, 4, false).claimed == 0u);
    CHECK(body.lumps.size() == before);

    // On a tube whose rings are a whole radius apart, a lump half way between
    // two of them would carry nothing: every lump of a row has a point on it.
    {
        Figure sparse = MakeTube(Skinned::Rigid, 7, 16);
        SoftBody& coarse = SoftenAll(sparse, 0, true);
        SoftEffect& surge = AddEffect(coarse, SoftEffectKind::Bulge);
        surge.bulge->along = SoftAlong::Surface;
        surge.bulge->travels = true;
        surge.bulge->chain = sparse.bones;
        surge.place.radius = 1.0f;
        const std::size_t own = coarse.lumps.size();
        REQUIRE(MakeSoftLumps(sparse.model(), coarse, 4, false).claimed > 0u);
        SoftSkinner skinner(sparse.model(), coarse);
        std::vector<f32> leans(coarse.lumps.size(), 0.0f);
        const u32 vertices = sparse.model().meshes[0].vertexCount();
        for (u32 v = 0; v < vertices; ++v) {
            for (const geom::Influence& influence : skinner.of(0, v)) {
                if (influence.bone >= skinner.lumpBase()) {
                    leans[influence.bone - skinner.lumpBase()] += influence.weight;
                }
            }
        }
        for (std::size_t k = own; k < coarse.lumps.size(); ++k) {
            CHECK(leans[k] > 0.0f);
        }
    }
}

TEST_CASE("G-S2 Reduce Keys leaves a span's keys as they were placed", "[wem][soft][spans]") {
    // A span at full for a whole clip is two equal keys: a constant track,
    // which the reducers would bring down to one, and with it the span's end.
    Figure f = MakeTube(Skinned::Two);
    SoftBody& body = SoftenAll(f, 0, true);
    const u32 id = body.id, effect = AddEffect(body, SoftEffectKind::Inflate).id;
    const f32 duration = f.document.clips[0].duration;
    const std::vector<SoftSpan> whole{{0.0f, duration, 0.0f, 0.0f}};
    const std::vector<SoftSpan> eased{{0.2f, 0.6f, 0.1f, 0.1f}};
    SetSoftSpans(f.document, 0, 0, id, effect, whole);
    SetSoftSpans(f.document, 0, 1, id, effect, eased);
    const auto keys = [&](u32 clip) {
        const u32 channel = SoftAmountChannel(f.model(), id, effect);
        for (const SubTrackContainer& container : f.document.clips[clip].containers) {
            for (const SubTrack& track : container.subTracks) {
                if (track.channel == channel) {
                    return track.times;
                }
            }
        }
        return std::vector<f32>{};
    };
    const std::vector<f32> wholeKeys = keys(0), easedKeys = keys(1);
    REQUIRE(wholeKeys.size() == 2u);
    // Its four, and one at nothing on each edge of the clip it reaches neither of.
    REQUIRE(easedKeys.size() == 6u);
    CHECK(easedKeys.front() == 0.0f);
    CHECK(easedKeys.back() == duration);
    ReduceKeysExactly(f.document, 0);
    CHECK(keys(0) == wholeKeys);
    CHECK(keys(1) == easedKeys);
    ReduceKeys(f.document, 0);
    CHECK(keys(0) == wholeKeys);
    CHECK(keys(1) == easedKeys);
    CHECK(SoftSpansOf(f.document, 0, 0, id, effect) == whole);
    REQUIRE(SoftSpansOf(f.document, 0, 1, id, effect).size() == 1u);
    CHECK(SoftSpansOf(f.document, 0, 1, id, effect)[0].start == Approx(0.2f).margin(1e-6));
    CHECK(SoftSpansOf(f.document, 0, 1, id, effect)[0].end == Approx(0.6f).margin(1e-6));
}

TEST_CASE("G-S2 the volume the fields ask for is met on the skinned result, where the fit misses",
          "[wem][soft][fields]") {
    // A bump twelve lumps cannot draw. The field keeps the volume on the goal;
    // the fit shows half the bump, so the skinned result would hold less, and
    // the lumps' last even scale is what brings it back (§7).
    Figure f = MakeTube(Skinned::Two, 25, 24);
    SoftBody& body = SoftenAll(f, 12, true);
    SoftEffect& bulge = AddEffect(body, SoftEffectKind::Bulge);
    bulge.amount = 0.8f;
    bulge.bulge->keepVolume = true;
    bulge.place.mesh = 0;
    bulge.place.face = FaceNear(f.model().meshes[0], Vector3f{1, 0, 3});
    bulge.place.weights = Vector3f{0.4f, 0.3f, 0.3f};
    bulge.place.radius = 1.0f;
    const SoftStepper stepper(f.document, 0, body.id);
    const SoftStepResult out = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(out.moved);
    const f64 rest = SoftVolumeAt(stepper.volume(), out.carried);
    const f64 goal = SoftVolumeAt(stepper.volume(), out.goal);
    const f64 posed = SoftVolumeAt(stepper.volume(), out.posed);
    INFO("goal " << goal / rest << " posed " << posed / rest << " follows " << out.fit.follows());
    CHECK(goal / rest == Approx(1.0).epsilon(1e-4));
    CHECK(out.fit.worstMiss > 0.2f * out.fit.largestMove);
    CHECK(posed / rest == Approx(1.0).epsilon(1e-3));

    // And an Inflate that swells most at a place, on the same few lumps.
    body.effects.clear();
    SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
    inflate.amount = 0.2f;
    inflate.inflate->along = SoftAlong::Surface;
    inflate.inflate->towardPlace = true;
    inflate.place = SoftPlace{0, FaceNear(f.model().meshes[0], Vector3f{1, 0, 3}), Vector3f{0.4f, 0.3f, 0.3f}, 1.5f};
    const SoftStepResult swollen = StepAt(f.document, body.id, kInvalidIndex, 0.0f);
    REQUIRE(swollen.moved);
    const f64 asked = SoftVolumeAt(stepper.volume(), swollen.goal);
    const f64 shown = SoftVolumeAt(stepper.volume(), swollen.posed);
    INFO("asked " << asked / rest << " shown " << shown / rest << " follows " << swollen.fit.follows());
    CHECK(asked / rest == Approx(1.2).epsilon(1e-3));
    CHECK(shown / asked == Approx(1.0).epsilon(1e-3));
}
