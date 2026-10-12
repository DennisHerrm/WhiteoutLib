// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

/// Workshop ▸ Soft Body, the library's ground (EDIT_MODE_SOFTBODY_DESIGN.md §5,
/// §6; EDIT_MODE_SOFTBODY_PLAN.md §8, §9): a region's counts, on figures whose
/// answers can be worked out by hand.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <whiteout/models/wem/anim/animator.h>
#include <whiteout/models/wem/converters.h>
#include <whiteout/models/wem/fbx_converter.h>
#include <whiteout/models/wem/geometry/builder.h>
#include <whiteout/models/wem/nodes/node.h>
#include <whiteout/models/wem/obj_converter.h>
#include <whiteout/models/wem/parser.h>
#include <whiteout/models/wem/physics/crossing.h>
#include <whiteout/models/wem/physics/references.h>
#include <whiteout/models/wem/physics/soft_bake.h>
#include <whiteout/models/wem/physics/soft_fit.h>
#include <whiteout/models/wem/physics/soft_lumps.h>
#include <whiteout/models/wem/physics/soft_region.h>
#include <whiteout/models/wem/physics/soft_run.h>
#include <whiteout/models/wem/reflect_bytes.h>
#include <whiteout/models/wem/skinning/deform.h>
#include <whiteout/models/wem/skinning/quantize.h>
#include <whiteout/models/wem/writer.h>

#include "whiteout/models/wem/chunk_tags.h"
#include "wem_soft_figures.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace whiteout;
using namespace whiteout::models::wem;
using Catch::Approx;
using namespace softfig;

namespace {

/// The figure the counts are checked on. Mesh 0 is the body: a box on one
/// bone, a box of side 2 on two, a box on four (given as five: the fifth is
/// light and stands on c1, so a fold to four gives it to c1), and one more
/// turned inside out. Mesh 1 is a rider the region leaves out.
struct CountFigure {
    Model model;
    u32 a, b1, b2, c1, c2, c3, c4, d, e, fifth, helper;
};

CountFigure MakeCountFigure() {
    CountFigure f;
    NodeTree& tree = f.model.nodes;
    f.a = AddNode(tree, "a", kInvalidNode, {0, 0, 0});
    f.b1 = AddNode(tree, "b1", f.a, {10, 0, 0});
    f.b2 = AddNode(tree, "b2", f.b1, {11, 0, 0});
    f.c1 = AddNode(tree, "c1", kInvalidNode, {20, 0, 0});
    f.c2 = AddNode(tree, "c2", kInvalidNode, {20, 1, 0});
    f.c3 = AddNode(tree, "c3", kInvalidNode, {21, 0, 0});
    // A helper between c3 and c4: "below" walks every node, not bones alone.
    f.helper = AddNode(tree, "helper", f.c3, {21, 0.5f, 0}, NodeKind::Helper);
    f.c4 = AddNode(tree, "c4", f.helper, {21, 1, 0});
    f.d = AddNode(tree, "d", f.c2, {20, 2, 0});
    f.e = AddNode(tree, "e", kInvalidNode, {40, 0, 0});
    f.fifth = AddNode(tree, "fifth", f.c1, {20, 0, 0});

    geom::MeshBuilder body = StartMesh("body");
    AddBox(body, {0, 0, 0}, 1.0f, {{f.e, 1.0f}});
    AddBox(body, {10, 0, 0}, 2.0f, {{f.b1, 0.75f}, {f.b2, 0.25f}});
    AddBox(body, {20, 0, 0}, 1.0f, {{f.c1, 0.24f}, {f.c2, 0.24f}, {f.c3, 0.24f}, {f.c4, 0.24f}, {f.fifth, 0.04f}});
    AddBox(body, {30, 0, 0}, 1.0f, {{f.e, 1.0f}}, /*insideOut=*/true);
    f.model.meshes.push_back(body.build().mesh);

    geom::MeshBuilder rider = StartMesh("rider");
    AddBox(rider, {20, 5, 0}, 1.0f, {{f.c1, 0.5f}, {f.d, 0.5f}});
    f.model.meshes.push_back(rider.build().mesh);
    return f;
}

const SoftBoneCount* BoneOf(const SoftCount& count, u32 node) {
    for (const SoftBoneCount& bone : count.bones) {
        if (bone.node == node) {
            return &bone;
        }
    }
    return nullptr;
}

} // namespace

TEST_CASE("wem soft a region is counted by the bones its points ride", "[wem][soft][region]") {
    const CountFigure f = MakeCountFigure();
    const SoftMeshPoints body[] = {SoftPointsOfMesh(f.model, 0)};
    REQUIRE(body[0].vertices.size() == 32u);
    const SoftCount count = CountSoftRegion(f.model, body, 4);

    CHECK(count.points == 32u);
    CHECK(count.faces == 48u);
    // 6 + 24 + 6 + 6.
    CHECK(count.area == Approx(42.0f).epsilon(1e-5));
    CHECK(count.byBones[0] == Approx(12.0 / 42.0).epsilon(1e-5));
    CHECK(count.byBones[1] == Approx(24.0 / 42.0).epsilon(1e-5));
    CHECK(count.byBones[2] == Approx(0.0).margin(1e-6));
    CHECK(count.byBones[3] == Approx(6.0 / 42.0).epsilon(1e-5));

    // The fifth influence was folded into c1, and `a` carries no skin at all.
    REQUIRE(count.bones.size() == 7u);
    CHECK(BoneOf(count, f.fifth) == nullptr);
    CHECK(BoneOf(count, f.a) == nullptr);
    REQUIRE(BoneOf(count, f.c1) != nullptr);
    CHECK(BoneOf(count, f.c1)->share == Approx(0.28 * 6.0 / 42.0).epsilon(1e-4));
    // Largest share first.
    CHECK(count.bones[0].node == f.b1);
    CHECK(count.bones[0].share == Approx(18.0 / 42.0).epsilon(1e-5));
    CHECK(count.bones[1].node == f.e);
    CHECK(count.bones[1].share == Approx(12.0 / 42.0).epsilon(1e-5));
    CHECK(count.bones[2].node == f.b2);
    CHECK(count.largestShare(2) == Approx(30.0 / 42.0).epsilon(1e-5));
    f32 total = 0.0f;
    for (const SoftBoneCount& bone : count.bones) {
        total += bone.share;
    }
    CHECK(total == Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("wem soft a region bone is used as it is or copied by what else rides it", "[wem][soft][region]") {
    const CountFigure f = MakeCountFigure();
    const SoftMeshPoints body[] = {SoftPointsOfMesh(f.model, 0)};
    const SoftCount count = CountSoftRegion(f.model, body, 4);
    const auto bone = [&](u32 node) {
        const SoftBoneCount* found = BoneOf(count, node);
        REQUIRE(found != nullptr);
        return *found;
    };
    // An end of the skeleton that is the region's alone.
    CHECK(bone(f.e).asItIs());
    CHECK(bone(f.b2).asItIs());
    CHECK(bone(f.c4).asItIs());
    // A bone that carries skin hangs below it.
    CHECK(bone(f.b1).skinBelow);
    CHECK(!bone(f.b1).outside);
    CHECK(!bone(f.b1).asItIs());
    // ... through a helper too.
    CHECK(bone(f.c3).skinBelow);
    CHECK(!bone(f.c3).asItIs());
    // The rider rides it.
    CHECK(bone(f.c1).outside);
    CHECK(!bone(f.c1).asItIs());
    // The rider rides a bone below it.
    CHECK(bone(f.c2).outsideBelow);
    CHECK(!bone(f.c2).outside);
    CHECK(!bone(f.c2).asItIs());

    CHECK(count.asTheyAre() == 3u);
    CHECK(count.copied() == 4u);
    // e, c1, c2 and c3 hang from the model; b1 hangs from a, b2 from b1, c4 from the helper.
    CHECK(bone(f.e).parentless);
    CHECK(!bone(f.b1).parentless);
    CHECK(!bone(f.c4).parentless);
    CHECK(count.parentless() == 4u);

    // With the rider in the region nothing outside rides anything.
    const SoftMeshPoints both[] = {SoftPointsOfMesh(f.model, 0), SoftPointsOfMesh(f.model, 1)};
    const SoftCount whole = CountSoftRegion(f.model, both, 4);
    for (const SoftBoneCount& each : whole.bones) {
        CHECK(!each.outside);
        CHECK(!each.outsideBelow);
    }
    // d is ridden now, and is an end; c2 has it below.
    REQUIRE(BoneOf(whole, f.d) != nullptr);
    CHECK(BoneOf(whole, f.d)->asItIs());
    CHECK(BoneOf(whole, f.c2)->skinBelow);
}

TEST_CASE("wem soft the count is for the weights a point a file holds", "[wem][soft][region]") {
    const CountFigure f = MakeCountFigure();
    const SoftMeshPoints body[] = {SoftPointsOfMesh(f.model, 0)};
    // Three weights a point: the four-bone box folds to three.
    const SoftCount three = CountSoftRegion(f.model, body, 3);
    CHECK(three.byBones[3] == Approx(0.0).margin(1e-6));
    CHECK(three.byBones[2] == Approx(6.0 / 42.0).epsilon(1e-5));
    // Five: the light fifth influence stays, and its bone is a region bone
    // that hangs below c1.
    const SoftCount five = CountSoftRegion(f.model, body, 5);
    CHECK(BoneOf(five, f.fifth) != nullptr);
    CHECK(five.bones.size() == 8u);
    REQUIRE(BoneOf(five, f.c1) != nullptr);
    CHECK(BoneOf(five, f.c1)->skinBelow);

    // A point's skin is normalised and heaviest first.
    SoftSkinSource source(f.model, 4);
    const std::span<const geom::Influence> skin = source.of(0, 16);
    REQUIRE(skin.size() == 4u);
    f32 total = 0.0f;
    for (const geom::Influence& influence : skin) {
        total += influence.weight;
    }
    CHECK(total == Approx(1.0f).epsilon(1e-5));
    CHECK(std::is_sorted(skin.begin(), skin.end(),
                         [](const geom::Influence& x, const geom::Influence& y) { return x.weight > y.weight; }));
}

TEST_CASE("wem soft a region's shells carry their signed volume", "[wem][soft][region]") {
    const CountFigure f = MakeCountFigure();
    const SoftMeshPoints body[] = {SoftPointsOfMesh(f.model, 0)};
    const SoftCount count = CountSoftRegion(f.model, body, 4);
    REQUIRE(count.shells.size() == 4u);
    for (const SoftShell& shell : count.shells) {
        CHECK(shell.mesh == 0u);
        CHECK(shell.faces == 12u);
        CHECK(shell.closed);
    }
    CHECK(count.shells[0].volume == Approx(1.0f).epsilon(1e-4));
    CHECK(count.shells[1].volume == Approx(8.0f).epsilon(1e-4));
    CHECK(count.shells[2].volume == Approx(1.0f).epsilon(1e-4));
    // Turned inside out: the same solid, counted against.
    CHECK(count.shells[3].volume == Approx(-1.0f).epsilon(1e-4));
    CHECK(count.shells[1].area == Approx(24.0f).epsilon(1e-5));

    // Two sides of a box are an open shell: no other face has all its corners
    // among their six.
    std::vector<u32> faces = {0, 1, 4, 5};
    const SoftMeshPoints part[] = {SoftPointsOfFaces(f.model, 0, faces)};
    REQUIRE(part[0].vertices.size() == 6u);
    const SoftCount open = CountSoftRegion(f.model, part, 4);
    REQUIRE(open.shells.size() == 1u);
    CHECK(!open.shells[0].closed);
    CHECK(open.faces == 4u);
    CHECK(open.points == 6u);
    CHECK(open.area == Approx(2.0f).epsilon(1e-5));
}

TEST_CASE("wem soft a region's points are those its faces draw", "[wem][soft][region]") {
    const CountFigure f = MakeCountFigure();
    const std::vector<u32> faces = {12, 13};
    const SoftMeshPoints points = SoftPointsOfFaces(f.model, 0, faces);
    CHECK(points.mesh == 0u);
    CHECK(points.vertices.size() == 4u);
    CHECK(std::is_sorted(points.vertices.begin(), points.vertices.end()));
    // A face or a mesh that is not there gives nothing.
    const std::vector<u32> missing = {4000};
    CHECK(SoftPointsOfFaces(f.model, 0, missing).vertices.empty());
    CHECK(SoftPointsOfMesh(f.model, 9).vertices.empty());
    CHECK(CountSoftRegion(f.model, {}, 4).bones.empty());
}

// ============================================================================
// S1: the records, the lumps, the lanes, the skin a file holds
// ============================================================================

namespace {

/// A soft point's skin as a file for @p weights holds the SOURCE: what the
/// expansion has to give back.
std::vector<geom::Influence> AsFileHolds(const Model& model, u32 vertex, SoftWeights weights, u32 cap, u32 mesh = 0) {
    SoftSkinSource source(model, cap);
    const std::span<const geom::Influence> skin = source.of(mesh, vertex);
    std::vector<geom::Influence> out(skin.begin(), skin.end());
    if (weights == SoftWeights::Bytes) {
        std::array<f32, 4> floats{};
        for (std::size_t i = 0; i < out.size() && i < 4; ++i) {
            floats[i] = out[i].weight;
        }
        const std::array<u8, 4> bytes = skinning::QuantizeWeights(floats, std::min<std::size_t>(out.size(), 4));
        for (std::size_t i = 0; i < out.size() && i < 4; ++i) {
            out[i].weight = static_cast<f32>(bytes[i]) / 255.0f;
        }
    } else if (weights == SoftWeights::EqualShares) {
        const std::vector<std::vector<geom::Influence>> one(1, out);
        const skinning::ClassicSkin snapped = skinning::QuantizeClassic(one, {});
        out.clear();
        for (const u32 node : snapped.groups[snapped.groupOf[0]]) {
            out.push_back({node, 1.0f / static_cast<f32>(snapped.groups[snapped.groupOf[0]].size())});
        }
    }
    return out;
}

u64 KeysOf(const Document& document) {
    u64 keys = 0;
    for (const Clip& clip : document.clips) {
        for (const SubTrackContainer& container : clip.containers) {
            for (const SubTrack& track : container.subTracks) {
                keys += track.times.size();
            }
        }
    }
    return keys;
}

/// G-S1: after the expansion, with nothing baked, every clip puts every point
/// where the source's skin did, and nothing else changed.
void CheckRestIsExact(const Figure& f, ProfileId target) {
    const Model& source = f.model();
    const SoftWeights weights = SoftWeightsFor(target);
    const u32 cap = Profile(target).maxBoneInfluences;
    Document expanded = f.document;
    Diagnostics diagnostics;
    const SoftExpansion made = ExpandSoftToBones(expanded, target, diagnostics);
    const Model& out = expanded.models[0];
    const SoftBody& body = out.physics.soft[0];

    // A bone a copy and a surface lump, and no key anywhere.
    u32 wanted = 0;
    for (const SoftLump& lump : body.lumps) {
        wanted += lump.kind == SoftLumpKind::Bone ? 0 : 1;
    }
    CHECK(made.bones == wanted);
    CHECK(out.nodes.size() == source.nodes.size() + wanted);
    CHECK(KeysOf(expanded) == KeysOf(f.document));
    REQUIRE(made.boneOf[0][0].size() == body.lumps.size());

    // Each made bone's family, to add a family's weights up by.
    std::vector<u32> familyOf(out.nodes.size());
    for (u32 n = 0; n < familyOf.size(); ++n) {
        familyOf[n] = n;
    }
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        familyOf[made.boneOf[0][0][k]] = body.lumps[k].family();
    }

    const Mesh& before = source.meshes[0];
    const Mesh& after = out.meshes[0];
    const std::span<const u32> bodyOf = before.attributes.get<u32>(geom::names::kSoftBody, geom::Domain::Vertex);
    const u32 vertices = before.vertexCount();
    std::vector<std::vector<geom::Influence>> reference(vertices);
    u32 most = 0;
    f32 worstFamily = 0.0f;
    for (u32 v = 0; v < vertices; ++v) {
        const bool soft = v < bodyOf.size() && bodyOf[v] != 0;
        const std::span<const geom::Influence> own = before.skin.forVertex(v);
        reference[v] = soft ? AsFileHolds(source, v, weights, cap) : std::vector<geom::Influence>(own.begin(), own.end());
        if (!soft) {
            continue;
        }
        const std::span<const geom::Influence> skin = after.skin.forVertex(v);
        most = std::max<u32>(most, static_cast<u32>(skin.size()));
        if (skin.size() > cap) {
            std::string text = "vertex " + std::to_string(v) + ":";
            for (const geom::Influence& influence : skin) {
                text += " " + out.nodes.nodes[influence.bone].name + "=" + std::to_string(influence.weight);
            }
            text += " | source:";
            for (const geom::Influence& influence : reference[v]) {
                text += " " + source.nodes.nodes[influence.bone].name + "=" + std::to_string(influence.weight);
            }
            std::printf("%s\n", text.c_str());
        }
        // Each family's weights sum to the bone's own.
        for (const geom::Influence& bone : reference[v]) {
            f32 sum = 0.0f;
            for (const geom::Influence& influence : skin) {
                sum += familyOf[influence.bone] == bone.bone ? influence.weight : 0.0f;
            }
            worstFamily = std::max(worstFamily, std::abs(sum - bone.weight));
        }
        if (weights == SoftWeights::Bytes) {
            for (const geom::Influence& influence : skin) {
                const f32 byte = influence.weight * 255.0f;
                CHECK(std::abs(byte - std::round(byte)) < 1e-3f);
            }
        }
    }
    CHECK(most <= cap);
    CHECK(worstFamily < 1e-6f);

    const Animator animatorBefore(f.document, 0);
    const Animator animatorAfter(expanded, 0);
    const std::span<const Vector3f> rest =
        before.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    f32 worst = 0.0f;
    for (u32 clip = 0; clip < f.document.clips.size(); ++clip) {
        for (u32 frame = 0; frame <= 30; ++frame) {
            Mix mix;
            mix.plays.push_back(Play{clip, static_cast<f32>(frame) / 30.0f, 1.0f, false});
            Pose poseBefore, poseAfter;
            animatorBefore.evaluate(mix, poseBefore);
            animatorAfter.evaluate(mix, poseAfter);
            const std::vector<Vector3f> moved = skinning::DeformMesh(after, poseAfter.skinning);
            for (u32 v = 0; v < vertices; ++v) {
                const Vector3f was = skinning::DeformPosition(rest[v], reference[v], poseBefore.skinning);
                worst = std::max(worst, (moved[v] - was).length());
            }
        }
    }
    CHECK(worst <= 1e-5f * f.size);

    // What no file holds goes, and what is left is a sound document.
    CHECK(DropSoftForProfile(expanded, target, diagnostics) == 1u);
    CHECK(expanded.models[0].physics.soft.empty());
    CHECK(!expanded.models[0].meshes[0].attributes.has(geom::names::kSoftBody, geom::Domain::Vertex));
    Diagnostics check;
    CheckPhysics(expanded.models[0], check);
    CHECK(!check.hasErrors());
}

} // namespace

TEST_CASE("wem soft an effect starts where it shows on a model that plays in place", "[wem][soft]") {
    // A game's walks and attacks move no root: what a body does of its own
    // accord has to come from the swing, the trail and the lean.
    const SoftEffect jiggle = DefaultSoftEffect(SoftEffectKind::Jiggle);
    CHECK(jiggle.jiggle->preset == SoftJigglePreset::Jelly);
    CHECK(jiggle.jiggle->bounce > SoftJiggle{}.bounce);
    const SoftEffect squash = DefaultSoftEffect(SoftEffectKind::SquashStretch);
    CHECK(squash.squash->stretch > SoftSquash{}.stretch);
    CHECK(squash.squash->drag > 0.0f);
    CHECK(squash.squash->windUp > 0.0f);
    // Wave after wave: one alone has passed before it is looked at.
    const SoftEffect ripple = DefaultSoftEffect(SoftEffectKind::Ripple);
    CHECK(ripple.ripple->plays == SoftRipplePlays::Loops);
    // And none starts so far that a creature's thin parts are lost to it: a
    // swing a fist keeps its fingers through, two crests a body and no higher
    // than a wrist takes, a wobble under a tenth of the size.
    CHECK(jiggle.jiggle->limit < 0.1f);
    CHECK(ripple.ripple->wavelength <= 0.5f);
    CHECK(ripple.amount <= 0.03f);
    CHECK(DefaultSoftEffect(SoftEffectKind::Wobble).amount < 0.1f);
    CHECK(SoftJiggle{}.limit == 0.25f);
    CHECK(SoftRipple{}.wavelength == 1.0f);
    // A setting reads by its name wherever a record is read as text.
    CHECK(std::string(ToString(SoftWobblePattern::Random)) == "random");
    CHECK(std::string(ToString(SoftAlong::Bones)) == "bones");
    CHECK(std::string(ToString(SoftMeltOrder::AllAtOnce)) == "all_at_once");
    CHECK(std::string(ToString(SoftAxis::Motion)) == "motion");
    CHECK(std::string(ToString(SoftThrough::Count)) == "unknown");
    // And the settings a record is read into are as they were.
    CHECK(SoftRipple{}.plays == SoftRipplePlays::Once);
    CHECK(SoftJiggle{}.preset == SoftJigglePreset::Fat);
}

TEST_CASE("wem soft the records survive a .wem, and an older file reads with none", "[wem][soft][io]") {
    Figure f = MakeTube(Skinned::Two);
    f.model().materialSlots = {"skin", "bones", "eyes", "trim"};
    SoftBody& body = Soften(f, 1.0f, 8, false);
    body.inside = {SoftPart{0, kInvalidIndex}};
    body.riders = {SoftPart{0, 3}};
    body.blend = 0.2f;
    body.colliders = {};
    for (u32 kind = 0; kind < static_cast<u32>(SoftEffectKind::Count); ++kind) {
        SoftEffect effect = DefaultSoftEffect(static_cast<SoftEffectKind>(kind));
        effect.id = body.nextEffect++;
        effect.name = ToString(effect.kind);
        effect.seed = 40 + kind;
        effect.place = SoftPlace{0, 5, Vector3f{0.2f, 0.3f, 0.5f}, 1.5f};
        CHECK(effect.settingsMatchKind());
        body.effects.push_back(effect);
    }
    body.effects[1].bulge->chain = {f.bones[1], f.bones[2]};
    body.effects[1].bulge->line = {Vector3f{0, 0, 1}, Vector3f{0, 0, 5}};
    body.effects[1].bulge->event = "Hit";
    body.effects[2].jiggle->kickIds = {body.nextEffect++};
    body.effects[2].jiggle->kickMesh = {0};
    body.effects[2].jiggle->kickFace = {7};
    body.effects[2].jiggle->kickWeights = {Vector3f{1, 0, 0}};
    body.effects[2].jiggle->kickDirection = {Vector3f{0, 0, -1}};
    body.effects[2].jiggle->kickStrength = {3.0f};
    body.effects[5].drip->siteMesh = {0, 0};
    body.effects[5].drip->siteFace = {3, 9};
    body.effects[5].drip->siteWeights = {Vector3f{1, 0, 0}, Vector3f{0, 1, 0}};
    body.effects[5].drip->siteThickness = {0.0f, 0.05f};
    Diagnostics sound;
    CheckPhysics(f.model(), sound);
    CHECK(!sound.hasErrors());
    // Its id is one of the set's, so a channel can name it.
    CHECK(f.model().physics.hasId(body.id));
    CHECK(f.model().physics.softBody(body.id) == &body);
    CHECK(!f.model().physics.empty());

    const std::vector<u8> bytes = Writer().write(f.document);
    const std::optional<Document> read = Parser().parse(std::span<const u8>(bytes.data(), bytes.size()));
    REQUIRE(read.has_value());
    REQUIRE(read->models[0].physics.soft.size() == 1u);
    CHECK(ReflectBytes(read->models[0].physics) == ReflectBytes(f.model().physics));
    CHECK(read->models[0].physics.soft[0].lumps.size() == body.lumps.size());
    // The layers go with the mesh.
    CHECK(ReflectBytes(read->models[0].meshes[0]) == ReflectBytes(f.model().meshes[0]));

    // The v8 file an older writer made: the model's last field, the empty
    // reference to its soft bodies, cut, the fill slid in behind it, and the
    // chunk stamped v8.
    Figure plain = MakeTube(Skinned::Two);
    std::vector<u8> old = Writer().write(plain.document);
    WEMHeader header{};
    std::memcpy(&header, old.data(), sizeof(header));
    bool cut = false;
    for (u32 i = 0; i < header.indexCount; ++i) {
        IndexEntry entry{};
        const std::size_t at = header.indexOffset + i * sizeof(IndexEntry);
        std::memcpy(&entry, old.data() + at, sizeof(entry));
        if (entry.tag != ChunkTagTraits<Model>::value) {
            continue;
        }
        REQUIRE(entry.count == 1u);
        REQUIRE(entry.version == 9u);
        u32 end = header.indexOffset > entry.offset ? header.indexOffset : static_cast<u32>(old.size());
        for (u32 j = 0; j < header.indexCount; ++j) {
            IndexEntry other{};
            std::memcpy(&other, old.data() + header.indexOffset + j * sizeof(IndexEntry), sizeof(other));
            if (other.offset > entry.offset && other.offset < end) {
                end = other.offset;
            }
        }
        u32 last = end;
        while (last > entry.offset && old[last - 1] == 0xAA) {
            --last;
        }
        REQUIRE(last >= entry.offset + sizeof(Reference));
        Reference none{};
        std::memcpy(&none, old.data() + last - sizeof(Reference), sizeof(none));
        REQUIRE(none.entries == 0u);
        std::memmove(old.data() + last - sizeof(Reference), old.data() + last, end - last);
        std::memset(old.data() + end - sizeof(Reference), 0xAA, sizeof(Reference));
        entry.version = 8;
        std::memcpy(old.data() + at, &entry, sizeof(entry));
        cut = true;
    }
    REQUIRE(cut);
    const std::optional<Document> older = Parser().parse(std::span<const u8>(old.data(), old.size()));
    REQUIRE(older.has_value());
    REQUIRE(older->models.size() == 1u);
    CHECK(older->models[0].physics.soft.empty());
    CHECK(older->models[0].nodes.size() == plain.model().nodes.size());
    CHECK(ReflectBytes(older->models[0].meshes[0]) == ReflectBytes(plain.model().meshes[0]));
    CHECK(older->clips.size() == 2u);
}

TEST_CASE("wem soft the layers are reserved and the channels are the physics set's", "[wem][soft][io]") {
    using geom::AttrType;
    using geom::Domain;
    CHECK(geom::LookupReserved(geom::names::kSoftBody).type == AttrType::U32);
    CHECK(geom::LookupReserved(geom::names::kSoftSoftness).type == AttrType::F32);
    CHECK(geom::LookupReserved(geom::names::kSoftPainted).type == AttrType::Bool);
    CHECK(geom::LookupReserved(geom::names::kSoftBindLump).type == AttrType::U32x4);
    CHECK(geom::LookupReserved(geom::names::kSoftBindWeight).type == AttrType::F32x4);
    CHECK(geom::LookupReserved(geom::names::kSoftBindLump).domain == Domain::Vertex);
    // Lump numbers, not vertex ids: a vertex renumbering must leave them alone.
    geom::AttrLayer layer;
    layer.name = geom::names::kSoftBindLump;
    layer.type = AttrType::U32x4;
    CHECK(!geom::IsVertexReferenceLayer(layer));

    CHECK(SoftBodyOfSub(SoftSub(7, 3)) == 7u);
    CHECK(SoftIndexOfSub(SoftSub(7, 3)) == 3u);
    CHECK(DefaultValueType(Channel::SoftAmount) == AttrType::F32);
    CHECK(DefaultValueType(Channel::SoftLumpRotation) == AttrType::Quat);
    CHECK(DefaultValueType(Channel::SoftLumpScale) == AttrType::F32x3);
    CHECK(std::string(ToString(Channel::SoftLumpTranslation)) == "softLumpTranslation");

    // A soft channel names a body, and an effect or a lump of it.
    Figure f = MakeTube(Skinned::Two);
    SoftBody& body = Soften(f, 1.0f, 0, false);
    SoftEffect effect = DefaultSoftEffect(SoftEffectKind::Inflate);
    effect.id = body.nextEffect++;
    body.effects.push_back(effect);
    const auto declare = [&](Channel channel, u32 sub) {
        AnimChannel made;
        made.id = f.model().animChannels.nextFreeId();
        made.target.kind = TrackTarget::Kind::Physics;
        made.target.sub = sub;
        made.target.channel = channel;
        made.valueType = DefaultValueType(channel);
        return f.model().animChannels.add(made);
    };
    const auto errors = [&] {
        Diagnostics out;
        CheckPhysics(f.model(), out);
        return out.hasErrors();
    };
    declare(Channel::SoftAmount, SoftSub(body.id, effect.id));
    declare(Channel::SoftLumpScale, SoftSub(body.id, static_cast<u32>(body.lumps.size())));
    CHECK(!errors());
    const u32 past = declare(Channel::SoftLumpTranslation, SoftSub(body.id, static_cast<u32>(body.lumps.size()) + 1));
    CHECK(errors());
    f.model().animChannels.channels.back().target.sub = SoftSub(body.id, 1);
    CHECK(!errors());
    (void)past;
    declare(Channel::SoftAmount, SoftSub(body.id, 99));
    CHECK(errors());
    // The body gone, its channels are invalidated with it.
    Diagnostics out;
    const u32 gone[] = {body.id};
    f.model().physics.soft.clear();
    InvalidatePhysicsChannels(f.model().animChannels, gone, out);
    for (const AnimChannel& channel : f.model().animChannels.channels) {
        if (channel.target.kind == TrackTarget::Kind::Physics) {
            CHECK(channel.target.sub == 0u);
        }
    }
    CHECK(!errors());
}

TEST_CASE("wem soft every bone a region rides has a lump of its own", "[wem][soft][lumps]") {
    Figure f = MakeTube(Skinned::Two);
    const SoftBody& body = Soften(f, 1.0f, 0, false);
    // The region starts at z = 1: b0 is ridden from below it too, b1 has b2
    // below it, and b2 is an end of the skeleton that is the region's alone.
    REQUIRE(body.lumps.size() == 3u);
    CHECK(body.lumps[0].kind == SoftLumpKind::Copy);
    CHECK(body.lumps[0].node == f.bones[0]);
    CHECK(body.lumps[0].carrier == f.bones[0]);
    CHECK(body.lumps[1].kind == SoftLumpKind::Copy);
    CHECK(body.lumps[2].kind == SoftLumpKind::Bone);
    CHECK(body.lumps[2].node == f.bones[2]);
    CHECK(body.lumps[2].carrier == f.bones[1]);
    CHECK(body.lumps[2].family() == f.bones[2]);
    CHECK(body.lumps[1].rest.z == Approx(2.0f));
    CHECK(body.influences == 4u);

    Figure all = MakeTube(Skinned::Two);
    const SoftBody& copied = Soften(all, 1.0f, 0, true);
    for (const SoftLump& lump : copied.lumps) {
        CHECK(lump.kind == SoftLumpKind::Copy);
    }

    // Used as chains, a bone with skinned bones below it is itself: only a
    // bone something else rides is still copied.
    Figure chain = MakeTube(Skinned::Two);
    Model& model = chain.model();
    SoftBody made;
    made.id = model.physics.allocateId();
    made.meshes = {0};
    made.chainsAsTheyAre = true;
    const std::span<u32> bodyOf = model.meshes[0].attributes.getOrCreate<u32>(
        geom::names::kSoftBody, geom::Domain::Vertex, geom::AttrType::U32);
    const std::span<f32> softness = model.meshes[0].attributes.getOrCreate<f32>(
        geom::names::kSoftSoftness, geom::Domain::Vertex, geom::AttrType::F32);
    std::fill(bodyOf.begin(), bodyOf.end(), made.id);
    std::fill(softness.begin(), softness.end(), 1.0f);
    model.physics.soft.push_back(made);
    MakeSoftLumps(model, model.physics.soft.back(), 4, false);
    for (const SoftLump& lump : model.physics.soft.back().lumps) {
        CHECK(lump.kind == SoftLumpKind::Bone);
    }
}

TEST_CASE("wem soft a bone a stage places or reads is copied", "[wem][soft][lumps]") {
    // The tube's last bone is an end of the skeleton and the region's alone:
    // used as it is, until a stage stands between its keys and what plays.
    const auto kindOf = [](StageKind kind, bool reads) {
        Figure f = MakeTube(Skinned::Two);
        Model& model = f.model();
        const u32 helper = AddNode(model.nodes, "helper", kInvalidNode, {4, 0, 0}, NodeKind::Helper);
        PoseStage stage;
        stage.id = 1;
        stage.kind = kind;
        // Off: a switch must not remake a body.
        stage.enabled = false;
        stage.driven = {reads ? helper : f.bones[2]};
        stage.sources.push_back(StageSource{1, reads ? f.bones[2] : helper, 1.0f});
        model.poseStages.push_back(stage);
        CHECK(SoftBoneIsStaged(model, f.bones[2]) == (reads || kind != StageKind::Link));
        const SoftBody& body = Soften(f, 1.0f, 0, false);
        REQUIRE(body.lumps.size() == 3u);
        return body.lumps[2];
    };
    // Placed: the stage writes over what a bake keyed on the bone.
    for (const StageKind kind : {StageKind::Position, StageKind::Orientation, StageKind::LookAt, StageKind::Spring}) {
        const SoftLump lump = kindOf(kind, false);
        CHECK(lump.kind == SoftLumpKind::Copy);
        CHECK(lump.carrier == lump.node);
    }
    // A Link carries its own bone with its keys.
    CHECK(kindOf(StageKind::Link, false).kind == SoftLumpKind::Bone);
    // Read: what the stage moves would take the bone's departure with it.
    CHECK(kindOf(StageKind::Link, true).kind == SoftLumpKind::Copy);
    CHECK(kindOf(StageKind::Orientation, true).kind == SoftLumpKind::Copy);

    // An IK chain's joints, its goal and a Look At's up node.
    Figure f = MakeTube(Skinned::Two);
    Model& model = f.model();
    PoseStage chain;
    chain.id = 1;
    chain.kind = StageKind::ChainIk;
    chain.driven = {f.bones[0], f.bones[1]};
    chain.targets = {f.bones[2]};
    model.poseStages.push_back(chain);
    CHECK(SoftBoneIsStaged(model, f.bones[0]));
    CHECK(SoftBoneIsStaged(model, f.bones[1]));
    CHECK(SoftBoneIsStaged(model, f.bones[2]));
    model.poseStages.clear();
    PoseStage look;
    look.id = 1;
    look.kind = StageKind::LookAt;
    look.driven = {f.bones[0]};
    look.upNode = f.bones[2];
    model.poseStages.push_back(look);
    CHECK(SoftBoneIsStaged(model, f.bones[2]));
    CHECK(!SoftBoneIsStaged(model, f.bones[1]));

    // Used as chains too: the chain's rule gives way to it.
    const SoftMeshPoints region[] = {SoftPointsOfMesh(model, 0)};
    for (const SoftBoneCount& bone : CountSoftRegion(model, region, 4).bones) {
        CHECK(bone.staged == (bone.node != f.bones[1]));
        CHECK(!(bone.staged && bone.asItIs()));
    }
}

TEST_CASE("wem soft surface lumps go to the largest bones, three each, in one order", "[wem][soft][lumps]") {
    Figure f = MakeTube(Skinned::Two, 25, 16);
    const SoftBody& body = Soften(f, 1.0f, 7, true);
    u32 surface = 0;
    std::map<u32, u32> perBone;
    for (const SoftLump& lump : body.lumps) {
        if (lump.kind == SoftLumpKind::Surface) {
            ++surface;
            ++perBone[lump.carrier];
            CHECK(lump.node == kInvalidNode);
            CHECK(lump.depth > 0.0f);
            // On the tube's wall or its top cap.
            const f32 radius = std::sqrt(lump.anchor.x * lump.anchor.x + lump.anchor.y * lump.anchor.y);
            CHECK((radius == Approx(1.0f).margin(1e-4) || lump.anchor.z == Approx(6.0f).margin(1e-4)));
        }
    }
    CHECK(surface == 7u);
    // Seven are three, three, and one more to the largest.
    REQUIRE(perBone.size() == 2u);
    std::vector<u32> counts;
    for (const auto& [bone, count] : perBone) {
        counts.push_back(count);
    }
    std::sort(counts.begin(), counts.end());
    CHECK(counts == std::vector<u32>{3, 4});
    // The same points give the same lumps.
    Figure again = MakeTube(Skinned::Two, 25, 16);
    const SoftBody& twin = Soften(again, 1.0f, 7, true);
    CHECK(ReflectBytes(twin) == ReflectBytes(body));
    // Lumps of one bone keep apart: no two within a tenth of the tube.
    for (std::size_t a = 0; a < body.lumps.size(); ++a) {
        for (std::size_t b = a + 1; b < body.lumps.size(); ++b) {
            if (body.lumps[a].kind == SoftLumpKind::Surface && body.lumps[b].kind == SoftLumpKind::Surface &&
                body.lumps[a].carrier == body.lumps[b].carrier) {
                CHECK((body.lumps[a].anchor - body.lumps[b].anchor).length() > 0.6f);
            }
        }
    }
}

TEST_CASE("wem soft a point's softness is handed over heaviest bone first", "[wem][soft][lanes]") {
    const geom::Influence two[] = {{4, 0.7f}, {9, 0.3f}};
    // Below the heavier bone's weight only it gives way.
    std::array<f32, 8> shares = SoftShares(two, 0.35f, 4);
    CHECK(shares[0] == Approx(0.5f));
    CHECK(shares[1] == Approx(0.0f));
    shares = SoftShares(two, 0.85f, 4);
    CHECK(shares[0] == Approx(1.0f));
    CHECK(shares[1] == Approx(0.5f));
    CHECK(SoftShares(two, 1.0f, 4)[1] == Approx(1.0f));
    CHECK(SoftShares(two, 0.0f, 4)[0] == 0.0f);
    // No slot to spare: all stiff below a half, all soft from it.
    const geom::Influence four[] = {{1, 0.4f}, {2, 0.3f}, {3, 0.2f}, {4, 0.1f}};
    CHECK(SoftShares(four, 0.49f, 4)[0] == 0.0f);
    CHECK(SoftShares(four, 0.5f, 4)[3] == 1.0f);
    CHECK(SoftShares(two, 0.4f, 2)[0] == 0.0f);

    // On a figure whose softness rises along it, what a point hands its lumps
    // is its softness, so it changes between neighbours no faster than the
    // softness does: no bone of a point is left stiff while another is soft.
    Figure f = MakeTube(Skinned::Two, 25, 16);
    const SoftBody& body = Soften(f, -1.0f, 0, true);
    SoftSkinner skinner(f.model(), body);
    const std::span<const f32> softness =
        f.model().meshes[0].attributes.get<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex);
    u32 blended = 0;
    for (u32 v = 0; v < softness.size(); ++v) {
        if (!skinner.holds(0, v)) {
            continue;
        }
        f32 handed = 0.0f, total = 0.0f;
        for (const geom::Influence& influence : skinner.of(0, v)) {
            handed += influence.bone >= skinner.lumpBase() ? influence.weight : 0.0f;
            total += influence.weight;
        }
        CHECK(total == Approx(1.0f).margin(1e-5));
        CHECK(handed == Approx(softness[v]).margin(1e-5));
        blended += softness[v] > 0.0f && softness[v] < 1.0f ? 1 : 0;
    }
    CHECK(blended > 50u);
}

TEST_CASE("wem soft solved lanes give a point back from its lumps' anchors", "[wem][soft][lanes]") {
    // A flat sheet on one bone, with a lump placed by hand at each corner: every
    // point lies in a triangle of three of them, so three lanes land on it.
    Figure f;
    f.document.profiles = {ProfileId::Wc3Reforged};
    f.document.defaultProfile = ProfileId::Wc3Reforged;
    Model& model = f.document.models.emplace_back();
    const u32 bone = AddNode(model.nodes, "sheet", kInvalidNode, {0, 0, -3});
    geom::MeshBuilder builder = StartMesh("sheet");
    const u32 side = 9;
    std::vector<geom::VertexId> grid;
    for (u32 y = 0; y < side; ++y) {
        for (u32 x = 0; x < side; ++x) {
            grid.push_back(builder.addVertex(Vector3f{static_cast<f32>(x), static_cast<f32>(y), 0.0f}));
            builder.addInfluence(grid.back(), bone, 1.0f);
        }
    }
    for (u32 y = 0; y + 1 < side; ++y) {
        for (u32 x = 0; x + 1 < side; ++x) {
            builder.addTriangle(grid[y * side + x], grid[y * side + x + 1], grid[(y + 1) * side + x + 1]);
            builder.addTriangle(grid[y * side + x], grid[(y + 1) * side + x + 1], grid[(y + 1) * side + x]);
        }
    }
    model.meshes.push_back(builder.build().mesh);
    Mesh& mesh = model.meshes[0];
    SoftBody made;
    made.id = model.physics.allocateId();
    made.meshes = {0};
    const std::span<u32> bodyOf =
        mesh.attributes.getOrCreate<u32>(geom::names::kSoftBody, geom::Domain::Vertex, geom::AttrType::U32);
    const std::span<f32> softness =
        mesh.attributes.getOrCreate<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex, geom::AttrType::F32);
    std::fill(bodyOf.begin(), bodyOf.end(), made.id);
    std::fill(softness.begin(), softness.end(), 1.0f);
    SoftLump own;
    own.kind = SoftLumpKind::Bone;
    own.node = bone;
    own.anchor = own.rest = Vector3f{0, 0, -3};
    made.lumps.push_back(own);
    for (const Vector3f& corner : {Vector3f{0, 0, 0}, Vector3f{8, 0, 0}, Vector3f{8, 8, 0}, Vector3f{0, 8, 0}}) {
        SoftLump lump;
        lump.kind = SoftLumpKind::Surface;
        lump.carrier = bone;
        lump.anchor = lump.rest = corner;
        made.lumps.push_back(lump);
    }
    model.physics.soft.push_back(made);
    SoftBody& body = model.physics.soft.back();
    const SoftBindReport report = BindLumps(model, body, 4, false);
    CHECK(report.points == side * side);
    CHECK(report.byBones[0] == side * side);
    CHECK(report.worstReturn <= 1e-5f * 8.0f);

    const auto lumps = mesh.attributes.get<std::array<u32, 4>>(geom::names::kSoftBindLump, geom::Domain::Vertex);
    const auto weights = mesh.attributes.get<std::array<f32, 4>>(geom::names::kSoftBindWeight, geom::Domain::Vertex);
    const std::span<const Vector3f> positions =
        mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    for (u32 v = 0; v < positions.size(); ++v) {
        Vector3f lands{0, 0, 0};
        f32 total = 0.0f;
        u32 used = 0;
        for (std::size_t k = 0; k < 4; ++k) {
            if (lumps[v][k] != 0) {
                REQUIRE(lumps[v][k] <= body.lumps.size());
                lands = lands + body.lumps[lumps[v][k] - 1].anchor * weights[v][k];
                total += weights[v][k];
                ++used;
            }
        }
        // One bone leaves three lanes, and all of them go to the surface here:
        // the bone's own lump is three units under the sheet.
        CHECK(used <= 3u);
        CHECK(total == Approx(1.0f).margin(1e-5));
        CHECK((lands - positions[v]).length() <= 1e-5f * 8.0f);
    }
    // A surface lump's pivot is the middle of what it carries: inside the sheet.
    for (const SoftLump& lump : body.lumps) {
        if (lump.kind == SoftLumpKind::Surface) {
            CHECK(lump.rest.x > 0.0f);
            CHECK(lump.rest.x < 8.0f);
            CHECK(lump.rest.z == Approx(0.0f).margin(1e-5));
        }
    }
}

TEST_CASE("wem soft at rest the deform is the model's own skin", "[wem][soft][deform]") {
    for (const Skinned skinned : {Skinned::Rigid, Skinned::Two, Skinned::Four}) {
        for (const f32 soft : {0.3f, 1.0f, -1.0f}) {
            for (const u32 detail : {0u, 8u}) {
                Figure f = MakeTube(skinned);
                Soften(f, soft, detail, detail != 0);
                const SoftDeformer deformer(f.model());
                REQUIRE(deformer.soft(0));
                const Animator animator(f.document, 0);
                f32 worst = 0.0f;
                for (u32 frame = 0; frame <= 10; ++frame) {
                    Mix mix;
                    mix.plays.push_back(Play{0, static_cast<f32>(frame) / 10.0f, 1.0f, false});
                    Pose pose;
                    animator.evaluate(mix, pose);
                    const std::vector<Matrix44f> palette = deformer.palette(pose.skinning, {});
                    const std::vector<Vector3f> moved = deformer.positions(0, palette);
                    const std::vector<Vector3f> own = skinning::DeformMesh(f.model().meshes[0], pose.skinning);
                    REQUIRE(moved.size() == own.size());
                    for (std::size_t v = 0; v < own.size(); ++v) {
                        worst = std::max(worst, (moved[v] - own[v]).length());
                    }
                }
                CHECK(worst <= 1e-5f * f.size);
            }
        }
    }
}

TEST_CASE("wem soft a lump's departure moves the points it carries, by their softness", "[wem][soft][deform]") {
    Figure f = MakeTube(Skinned::Rigid);
    const SoftBody& body = Soften(f, 1.0f, 0, true);
    const SoftDeformer deformer(f.model());
    std::vector<Matrix44f> skin(f.model().nodes.size(), Matrix44f::identity());
    // The copy of the top bone moves up by one: its points follow whole.
    std::vector<Transform> departures(body.lumps.size());
    u32 topCopy = kInvalidIndex;
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        if (body.lumps[k].node == f.bones[2]) {
            topCopy = k;
        }
    }
    REQUIRE(topCopy != kInvalidIndex);
    departures[topCopy].translation = Vector3f{0, 0, 1};
    const std::vector<Vector3f> moved = deformer.positions(0, deformer.palette(skin, departures));
    const std::span<const Vector3f> rest =
        f.model().meshes[0].attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    for (u32 v = 0; v < rest.size(); ++v) {
        const f32 lifted = rest[v].z >= 3.0f ? 1.0f : 0.0f;
        CHECK((moved[v] - rest[v]).length() == Approx(lifted).margin(1e-5));
    }
    // A scale along the model's axes about the lump's pivot.
    departures[topCopy] = Transform{};
    departures[topCopy].scale = Vector3f{1, 1, 2};
    const std::vector<Vector3f> stretched = deformer.positions(0, deformer.palette(skin, departures));
    for (u32 v = 0; v < rest.size(); ++v) {
        if (rest[v].z >= 3.0f) {
            CHECK(stretched[v].z == Approx(4.0f + 2.0f * (rest[v].z - 4.0f)).margin(1e-4));
            CHECK(stretched[v].x == Approx(rest[v].x).margin(1e-5));
        }
    }
    // Half soft, half the way.
    Figure half = MakeTube(Skinned::Rigid);
    const SoftBody& halfBody = Soften(half, 0.5f, 0, true);
    const SoftDeformer halfDeformer(half.model());
    std::vector<Transform> lift(halfBody.lumps.size());
    for (Transform& each : lift) {
        each.translation = Vector3f{0, 0, 1};
    }
    const std::vector<Vector3f> halfway = halfDeformer.positions(0, halfDeformer.palette(skin, lift));
    for (u32 v = 0; v < rest.size(); ++v) {
        CHECK((halfway[v] - rest[v]).length() == Approx(rest[v].z >= 1.0f - 1e-4f ? 0.5f : 0.0f).margin(1e-5));
    }
}

TEST_CASE("wem soft a bone a point is painted down on is copied, so the point stays with its animation",
          "[wem][soft][lumps][paint]") {
    // The top bone carries the tube from z = 3 up, and is the region's alone.
    Figure f = MakeTube(Skinned::Rigid);
    SoftBody& body = Soften(f, 1.0f, 0, false);
    const auto top = [&]() {
        u32 found = kInvalidIndex;
        for (u32 k = 0; k < body.lumps.size(); ++k) {
            if (body.lumps[k].kind != SoftLumpKind::Surface && body.lumps[k].node == f.bones[2]) {
                found = k;
            }
        }
        REQUIRE(found != kInvalidIndex);
        return found;
    };
    REQUIRE(body.lumps[top()].kind == SoftLumpKind::Bone);
    const std::vector<SoftLump> made = body.lumps;
    // Nothing painted: nothing to settle.
    CHECK(!SettleSoftLumps(f.model(), body, 4, false));

    // The ring at z = 5 painted down. Used as it is, the bone would take the
    // ring wherever an effect sends the rest of what it carries.
    Mesh& mesh = f.model().meshes[0];
    const std::span<const Vector3f> rest =
        mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    const auto paint = [&](f32 value) {
        const std::span<f32> softness = mesh.attributes.get<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex);
        for (u32 v = 0; v < rest.size(); ++v) {
            if (std::abs(rest[v].z - 5.0f) < 1e-3f) {
                softness[v] = value;
            }
        }
    };
    paint(0.0f);
    CHECK(SettleSoftLumps(f.model(), body, 4, false));
    REQUIRE(body.lumps.size() == made.size());
    CHECK(body.lumps[top()].kind == SoftLumpKind::Copy);
    CHECK(body.lumps[top()].carrier == f.bones[2]);
    {
        const SoftDeformer deformer(f.model());
        const std::vector<Matrix44f> skin(f.model().nodes.size(), Matrix44f::identity());
        std::vector<Transform> departures(body.lumps.size());
        departures[top()].translation = Vector3f{0, 0, 1};
        const std::vector<Vector3f> moved = deformer.positions(0, deformer.palette(skin, departures));
        for (u32 v = 0; v < rest.size(); ++v) {
            const bool painted = std::abs(rest[v].z - 5.0f) < 1e-3f;
            const f32 lifted = rest[v].z >= 3.0f && !painted ? 1.0f : 0.0f;
            CHECK((moved[v] - rest[v]).length() == Approx(lifted).margin(1e-5));
        }
    }

    // Painted up again, the bone is the soft part's alone once more, and the
    // body is the one that was made.
    paint(1.0f);
    CHECK(SettleSoftLumps(f.model(), body, 4, false));
    CHECK(SameReflected(body.lumps, made));
}

TEST_CASE("G-S1 an expanded soft body plays every clip as its source did", "[wem][soft][expand]") {
    for (const Skinned skinned : {Skinned::Rigid, Skinned::Two, Skinned::Four}) {
        for (const f32 soft : {0.3f, 1.0f}) {
            for (const bool copyAll : {false, true}) {
                for (const u32 detail : {0u, 8u}) {
                    Figure f = MakeTube(skinned);
                    Soften(f, soft, detail, copyAll);
                    CheckRestIsExact(f, ProfileId::Wc3Reforged);
                }
            }
        }
    }
}

TEST_CASE("G-S1 the rest is exact under three weights a point and in no game's weights", "[wem][soft][expand]") {
    for (const Skinned skinned : {Skinned::Rigid, Skinned::Two}) {
        for (const u32 detail : {0u, 8u}) {
            Figure f = MakeTube(skinned);
            Soften(f, 1.0f, detail, false);
            // Solved for four weights a point, written for three: bound again.
            CheckRestIsExact(f, ProfileId::Diablo3);
            Figure blended = MakeTube(skinned);
            Soften(blended, -1.0f, detail, true);
            CheckRestIsExact(blended, ProfileId::Generic);
        }
    }
}

TEST_CASE("G-S1 the rest is exact in classic groups, past 256 of them too", "[wem][soft][expand]") {
    for (const Skinned skinned : {Skinned::Rigid, Skinned::Two}) {
        for (const f32 soft : {0.5f, 1.0f}) {
            Figure f = MakeTube(skinned, 13, 12, ProfileId::Wc3Classic);
            Soften(f, soft, 0, true, 8, true);
            CheckRestIsExact(f, ProfileId::Wc3Classic);
        }
    }
    // One lump a bone: the groups are the source's, bone for bone.
    {
        Figure f = MakeTube(Skinned::Two, 13, 12, ProfileId::Wc3Classic);
        const SoftBody& body = Soften(f, 1.0f, 0, true, 8, true);
        SoftSkinner skinner(f.model(), body, SoftWeights::EqualShares);
        for (u32 v = 0; v < f.model().meshes[0].vertexCount(); ++v) {
            if (!skinner.holds(0, v)) {
                continue;
            }
            std::vector<u32> back;
            for (const geom::Influence& influence : skinner.of(0, v)) {
                REQUIRE(influence.bone >= skinner.lumpBase());
                back.push_back(body.lumps[influence.bone - skinner.lumpBase()].family());
            }
            std::sort(back.begin(), back.end());
            std::vector<u32> source;
            for (const geom::Influence& influence : AsFileHolds(f.model(), v, SoftWeights::EqualShares, 8)) {
                source.push_back(influence.bone);
            }
            std::sort(source.begin(), source.end());
            CHECK(back == source);
        }
    }
    // A geoset pushed past 256 groups: a thousand points on one bone, and 96
    // lumps over them. The expansion brings the groups back under the limit
    // inside each point's families, and the rest holds.
    Figure crowded = MakeTube(Skinned::Rigid, 41, 24, ProfileId::Wc3Classic);
    const SoftBody& body = Soften(crowded, 1.0f, 96, true, 8, true);
    SoftSkinner skinner(crowded.model(), body, SoftWeights::EqualShares);
    std::set<std::vector<u32>> before;
    for (u32 v = 0; v < crowded.model().meshes[0].vertexCount(); ++v) {
        std::vector<u32> group;
        for (const geom::Influence& influence : skinner.of(0, v)) {
            group.push_back(influence.bone);
        }
        std::sort(group.begin(), group.end());
        before.insert(group);
    }
    REQUIRE(before.size() > 256u);
    Document expanded = crowded.document;
    Diagnostics diagnostics;
    const SoftExpansion made = ExpandSoftToBones(expanded, ProfileId::Wc3Classic, diagnostics);
    CHECK(made.reduced > 0u);
    // And the export says so: those points lost their lumps of the surface.
    bool said = false;
    for (const Diagnostic& d : diagnostics.all()) {
        said = said || (d.severity == Severity::Warning && d.message.find("256 groups") != std::string::npos);
    }
    CHECK(said);
    std::set<std::vector<u32>> after;
    const Mesh& mesh = expanded.models[0].meshes[0];
    for (u32 v = 0; v < mesh.vertexCount(); ++v) {
        std::vector<u32> group;
        for (const geom::Influence& influence : mesh.skin.forVertex(v)) {
            group.push_back(influence.bone);
        }
        std::sort(group.begin(), group.end());
        after.insert(group);
    }
    CHECK(after.size() <= 256u);
    CheckRestIsExact(crowded, ProfileId::Wc3Classic);
}

TEST_CASE("wem soft a body follows the nodes, meshes and materials it names", "[wem][soft][references]") {
    Figure f = MakeTube(Skinned::Two);
    SoftBody& body = Soften(f, 1.0f, 4, false);
    body.riders = {SoftPart{0, 2}};
    SoftEffect drip = DefaultSoftEffect(SoftEffectKind::Drip);
    drip.id = body.nextEffect++;
    drip.drip->material = 3;
    body.effects.push_back(drip);
    SoftEffect bulge = DefaultSoftEffect(SoftEffectKind::Bulge);
    bulge.id = body.nextEffect++;
    bulge.bulge->chain = {f.bones[1], f.bones[2]};
    bulge.place.radius = 2.0f;
    body.effects.push_back(bulge);
    PhysicsSet& physics = f.model().physics;

    // A material slot goes: the ones above it move down, and its users take
    // the replacement. A part that rides follows where its faces can still be
    // told apart: no face of its mesh draws with the replacement.
    REQUIRE(!f.model().meshes[0].sections.empty());
    f.model().meshes[0].sections[0].materialSlot = 1;
    CHECK(RemapSoftSlots(f.model(), 2, 0) == 0u);
    CHECK(body.riders[0].material == 0u);
    CHECK(body.effects[0].drip->material == 2u);
    // Where one does, the part would grow to those faces: it is let go, its
    // lump left hanging from nothing, and the part after it keeps its lump.
    {
        Figure grown = f;
        SoftBody& held = grown.model().physics.soft[0];
        grown.model().meshes[0].sections[0].materialSlot = 0;
        held.riders = {SoftPart{0, 1}, SoftPart{0, 4}};
        SoftLump first, second;
        first.rider = 1;
        first.carrier = grown.bones[0];
        second.rider = 2;
        second.carrier = grown.bones[0];
        held.lumps.push_back(first);
        held.lumps.push_back(second);
        CHECK(RemapSoftSlots(grown.model(), 1, 0) == 1u);
        REQUIRE(held.riders.size() == 1u);
        CHECK(held.riders[0].material == 3u);
        CHECK(held.lumps[held.lumps.size() - 2].rider == 0u);
        CHECK(held.lumps[held.lumps.size() - 2].carrier == kInvalidNode);
        CHECK(held.lumps.back().rider == 1u);
        CHECK(held.lumps.back().carrier == grown.bones[0]);
    }
    // The other way round as well: a part that is the replacement's own faces
    // would take those of the slot that goes, where its mesh draws with it.
    {
        Figure joined = f;
        SoftBody& held = joined.model().physics.soft[0];
        joined.model().meshes[0].sections[0].materialSlot = 1;
        held.riders = {SoftPart{0, 0}};
        CHECK(RemapSoftSlots(joined.model(), 1, 0) == 1u);
        CHECK(held.riders.empty());
        // On a mesh the slot draws nothing of, it is only renumbered.
        joined.model().meshes[0].sections[0].materialSlot = 3;
        held.riders = {SoftPart{0, 2}};
        CHECK(RemapSoftSlots(joined.model(), 1, 2) == 0u);
        REQUIRE(held.riders.size() == 1u);
        CHECK(held.riders[0].material == 1u);
    }
    // Meshes merged: the one that is left draws as one section, so a part the
    // body kept clear of is the whole of it, once, and matches its faces.
    {
        Figure merged = f;
        SoftBody& held = merged.model().physics.soft[0];
        held.riders.clear();
        held.inside = {SoftPart{1, 4}, SoftPart{2, 5}, SoftPart{3, kInvalidIndex}};
        const std::vector<u8> in = {0, 1, 1, 0};
        const std::vector<u32> base = {0, 0, 7, 0};
        CHECK(SoftFollowMergedMeshes(merged.model().physics, in, 1, base) == 0u);
        REQUIRE(held.inside.size() == 2u);
        CHECK(held.inside[0].mesh == 1u);
        CHECK(held.inside[0].material == kInvalidIndex);
        CHECK(held.inside[1].mesh == 3u);
    }

    // Twice the size: the lumps and a place's radius are lengths, and so is
    // where a Drip's strands were built and how wide.
    const Vector3f anchor = body.lumps[1].anchor;
    const f32 depth = body.lumps.back().depth;
    body.effects[0].drip->strandAt = {Vector3f{1.0f, 2.0f, 3.0f}};
    body.effects[0].drip->strandWidth = {0.25f};
    RescalePhysics(physics, 2.0f);
    CHECK((body.lumps[1].anchor - anchor * 2.0f).length() < 1e-5f);
    CHECK(body.lumps.back().depth == Approx(2.0f * depth));
    CHECK(body.effects[1].place.radius == Approx(4.0f));
    CHECK((body.effects[0].drip->strandAt[0] - Vector3f{2.0f, 4.0f, 6.0f}).length() < 1e-6f);
    CHECK(body.effects[0].drip->strandWidth[0] == Approx(0.5f));
    RescalePhysics(physics, 0.5f);
    body.effects[0].drip->strandAt.clear();
    body.effects[0].drip->strandWidth.clear();

    // The top bone goes, its skin to its parent: its own lump stays in the
    // list as a surface lump of the parent, and the chain loses it.
    const std::size_t lumps = body.lumps.size();
    std::vector<u8> doomed(f.model().nodes.size(), 0);
    std::vector<u32> heir(f.model().nodes.size(), kInvalidNode);
    doomed[f.bones[2]] = 1;
    heir[f.bones[2]] = f.bones[1];
    SoftFollowRemovedNodes(physics, doomed, heir);
    std::vector<u32> remap = {0, 1, kInvalidNode};
    RemapPhysicsNodes(physics, remap);
    REQUIRE(body.lumps.size() == lumps);
    for (const SoftLump& lump : body.lumps) {
        CHECK(lump.carrier != f.bones[2]);
        if (lump.kind != SoftLumpKind::Surface) {
            CHECK(lump.node < 2u);
        }
    }
    CHECK(body.effects[1].bulge->chain == std::vector<u32>{f.bones[1]});

    // Its mesh goes, and the body with it.
    const u32 id = body.id;
    const std::vector<u32> none = {kInvalidIndex};
    const std::vector<u32> gone = RemapSoftMeshes(physics, none);
    CHECK(gone == std::vector<u32>{id});
    CHECK(physics.soft.empty());
    CHECK(!CarriesSoft(f.document));
}

TEST_CASE("wem soft a document that keys a span converts as one that does not", "[wem][soft][convert]") {
    // No writer knows a soft channel: each has to pass over it.
    u32 exercised = 0;
    for (const ProfileId profile : {ProfileId::Wc3Reforged, ProfileId::Generic}) {
        Figure plain = MakeTube(Skinned::Two, 13, 12, profile);
        Figure keyed = MakeTube(Skinned::Two, 13, 12, profile);
        Model& model = keyed.model();
        AnimChannel span;
        span.id = model.animChannels.nextFreeId();
        span.target.kind = TrackTarget::Kind::Physics;
        span.target.sub = SoftSub(1, 1);
        span.target.channel = Channel::SoftAmount;
        span.valueType = geom::AttrType::F32;
        const u32 channel = model.animChannels.add(span);
        Key<f32>(keyed.document.clips[0], channel, {0.0f, 0.5f, 1.0f}, {0.0f, 1.0f, 0.0f});
        AnimChannel lump = span;
        lump.id = model.animChannels.nextFreeId();
        lump.target.channel = Channel::SoftLumpTranslation;
        lump.valueType = geom::AttrType::F32x3;
        Key<Vector3f>(keyed.document.clips[0], model.animChannels.add(lump), {0.0f, 1.0f},
                      {Vector3f{0, 0, 0}, Vector3f{0, 0, 1}});
        const MdxConverter mdx;
        const GltfConverter gltf;
        const FbxConverter fbx;
        const ObjConverter obj;
        for (const FormatConverter* converter : {static_cast<const FormatConverter*>(&mdx),
                                                 static_cast<const FormatConverter*>(&gltf),
                                                 static_cast<const FormatConverter*>(&fbx),
                                                 static_cast<const FormatConverter*>(&obj)}) {
            const Result<std::vector<u8>> without = converter->exportToBytes(plain.document, profile);
            if (!without.ok() || !without.value) {
                continue;
            }
            const Result<std::vector<u8>> with = converter->exportToBytes(keyed.document, profile);
            REQUIRE(with.ok());
            CHECK(*with.value == *without.value);
            ++exercised;
        }
    }
    CHECK(exercised >= 2u);
}

TEST_CASE("G-S1 a tree that lists a bone before its parent expands the same", "[wem][soft][expand]") {
    // An MDX-imported tree may: the chain's bones are listed tip first here.
    for (const u32 detail : {0u, 8u}) {
        Figure f = MakeTube(Skinned::Two);
        Model& model = f.model();
        const u32 count = model.nodes.size();
        std::vector<Node> turned(count);
        for (u32 n = 0; n < count; ++n) {
            turned[count - 1 - n] = model.nodes.nodes[n];
            if (model.nodes.nodes[n].parent != kInvalidNode) {
                turned[count - 1 - n].parent = count - 1 - model.nodes.nodes[n].parent;
            }
        }
        model.nodes.nodes = turned;
        model.nodes.invalidateHierarchy();
        REQUIRE(!model.nodes.parentsPrecedeChildren());
        for (geom::Influence& influence : model.meshes[0].skin.influences) {
            influence.bone = count - 1 - influence.bone;
        }
        for (AnimChannel& channel : model.animChannels.channels) {
            channel.target.node = count - 1 - channel.target.node;
        }
        std::reverse(f.bones.begin(), f.bones.end());
        const SoftBody& body = Soften(f, 1.0f, detail, false);
        // The same lumps as the tree in order gives: two copies and the tip,
        // which is copied too where a lump of the surface hangs from it.
        u32 copies = 0;
        bool tipCarries = false;
        for (const SoftLump& lump : body.lumps) {
            copies += lump.kind == SoftLumpKind::Copy ? 1 : 0;
            if (lump.kind == SoftLumpKind::Bone) {
                CHECK(model.nodes.nodes[lump.node].name == "b2");
            }
            tipCarries = tipCarries || (lump.kind == SoftLumpKind::Surface && lump.carrier < model.nodes.size() &&
                                        model.nodes.nodes[lump.carrier].name == "b2");
        }
        CHECK(tipCarries == (detail != 0));
        CHECK(copies == (tipCarries ? 3u : 2u));
        CheckRestIsExact(f, ProfileId::Wc3Reforged);
    }
}

TEST_CASE("wem soft the edge blend rises from the region's border over the surface", "[wem][soft][region]") {
    Figure f = MakeTube(Skinned::Two, 25, 16);
    const Model& model = f.model();
    const std::span<const Vector3f> positions =
        model.meshes[0].attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    // The part above z = 1: its border is the ring at z = 1, and it is 5 tall.
    SoftMeshPoints part;
    part.mesh = 0;
    for (u32 v = 0; v < positions.size(); ++v) {
        if (positions[v].z >= 1.0f - 1e-4f) {
            part.vertices.push_back(v);
        }
    }
    const SoftMeshPoints region[] = {part};
    CHECK(SoftRegionSize(model, region) == Approx(5.0f).margin(1e-4));
    const std::vector<std::vector<f32>> blended = SoftBlendSoftness(model, region, 0.25f);
    REQUIRE(blended.size() == 1u);
    REQUIRE(blended[0].size() == part.vertices.size());
    for (std::size_t k = 0; k < part.vertices.size(); ++k) {
        const Vector3f& p = positions[part.vertices[k]];
        const f32 radius = std::sqrt(p.x * p.x + p.y * p.y);
        if (radius > 0.99f) {
            // Straight up the wall, so the distance over the surface is the height.
            CHECK(blended[0][k] == Approx(std::clamp((p.z - 1.0f) / 1.25f, 0.0f, 1.0f)).margin(0.02));
        } else {
            CHECK(blended[0][k] == 1.0f);
        }
    }
    // No blend: stiff on the border alone.
    const std::vector<std::vector<f32>> hard = SoftBlendSoftness(model, region, 0.0f);
    for (std::size_t k = 0; k < part.vertices.size(); ++k) {
        CHECK(hard[0][k] == (std::abs(positions[part.vertices[k]].z - 1.0f) < 1e-4f ? 0.0f : 1.0f));
    }
    // A whole mesh has no border, and is soft everywhere.
    const SoftMeshPoints whole[] = {SoftPointsOfMesh(model, 0)};
    for (const f32 softness : SoftBlendSoftness(model, whole, 0.25f)[0]) {
        CHECK(softness == 1.0f);
    }
}

TEST_CASE("wem soft folding small weights is planned before it is written", "[wem][soft][region]") {
    Figure f = MakeTube(Skinned::Four);
    Model& model = f.model();
    const SoftMeshPoints region[] = {SoftPointsOfMesh(model, 0)};
    const SoftFoldPlan plan = PlanSoftFold(model, region, 0.10f, 4);
    CHECK(plan.before[3] == model.meshes[0].vertexCount());
    REQUIRE(!plan.changes.empty());
    CHECK(plan.after[3] < plan.before[3]);
    CHECK(plan.after[0] + plan.after[1] + plan.after[2] + plan.after[3] == model.meshes[0].vertexCount());
    for (const SoftFoldPlan::Change& change : plan.changes) {
        f32 total = 0.0f;
        for (const geom::Influence& influence : change.skin) {
            CHECK(influence.weight >= 0.10f - 1e-4f);
            total += influence.weight;
        }
        CHECK(total == Approx(1.0f).margin(1e-5));
    }
    // At the bind a fold moves nothing; in a pose it does, and says how far.
    std::vector<Matrix44f> rest(model.nodes.size(), Matrix44f::identity());
    CHECK(SoftFoldShift(model, plan, rest) == Approx(0.0f).margin(1e-6));
    Mix mix;
    mix.plays.push_back(Play{0, 0.5f, 1.0f, false});
    Pose pose;
    Animator(f.document, 0).evaluate(mix, pose);
    const f32 shift = SoftFoldShift(model, plan, pose.skinning);
    CHECK(shift > 1e-3f);
    CHECK(shift < 1.0f);
    // Nothing was written yet; written, there is nothing left to fold.
    CHECK(PlanSoftFold(model, region, 0.10f, 4).changes.size() == plan.changes.size());
    ApplySoftFold(model, plan);
    CHECK(PlanSoftFold(model, region, 0.10f, 4).changes.empty());
}

TEST_CASE("wem soft a body keeps clear of what lies inside it", "[wem][soft][region]") {
    Figure f = MakeTube(Skinned::Rigid, 25, 16);
    Model& model = f.model();
    // A box in the middle of the tube, rigid on b1, and one well outside it.
    geom::MeshBuilder inner = StartMesh("bones");
    AddBox(inner, {-0.25f, -0.25f, 2.0f}, 0.5f, {{f.bones[1], 1.0f}});
    model.meshes.push_back(inner.build().mesh);
    geom::MeshBuilder outer = StartMesh("far");
    AddBox(outer, {5.0f, 5.0f, 2.0f}, 0.5f, {{f.bones[1], 1.0f}});
    model.meshes.push_back(outer.build().mesh);
    model.materialSlots = {"skin"};

    SoftBody& body = Soften(f, 1.0f, 6, true);
    const SoftPoints points = GatherSoftPoints(model, body.id, 4);
    const std::vector<SoftInside> inside = SoftPartsInside(model, points);
    REQUIRE(inside.size() == 1u);
    CHECK(inside[0].part.mesh == 1u);
    CHECK(inside[0].points == 8u);
    // Nothing to keep clear of yet.
    for (const SoftLump& lump : body.lumps) {
        CHECK(lump.clear == 0.0f);
    }
    body.inside = {inside[0].part};
    MeasureSoftClear(model, body);
    u32 measured = 0;
    for (const SoftLump& lump : body.lumps) {
        if (lump.clear > 0.0f) {
            ++measured;
            // From the wall, a unit out, to the box's corners, 0.35 out, and no nearer.
            CHECK(lump.clear >= 0.6f);
            CHECK(lump.clear < 6.0f);
        }
    }
    CHECK(measured >= 3u);
    // The lumps nearest the box are the ones that carry the wall round it.
    f32 nearest = 1e9f;
    for (const SoftLump& lump : body.lumps) {
        if (lump.clear > 0.0f) {
            nearest = std::min(nearest, lump.clear);
        }
    }
    CHECK(nearest == Approx(0.65f).margin(0.1));
}

namespace {

/// The tube's upper part a soft body, with two parts that ride it rigidly: a
/// saddle against the wall on b1 (mesh 1), and a strap that rides b1 and b2 by
/// halves (mesh 2). Their softness layers are left at 0.
SoftBody& MakeRiders(Figure& f, u32 cap, bool equalShares) {
    Model& model = f.model();
    geom::MeshBuilder saddle = StartMesh("saddle");
    AddBox(saddle, {1.0f, -0.25f, 1.75f}, 0.5f, {{f.bones[1], 1.0f}});
    model.meshes.push_back(saddle.build().mesh);
    geom::MeshBuilder strap = StartMesh("strap");
    AddBox(strap, {-1.5f, -0.25f, 2.75f}, 0.5f, {{f.bones[1], 0.5f}, {f.bones[2], 0.5f}});
    model.meshes.push_back(strap.build().mesh);
    SoftBody& body = Soften(f, 1.0f, 6, true, cap, equalShares);
    for (const u32 m : {1u, 2u}) {
        Mesh& mesh = model.meshes[m];
        const std::span<u32> bodyOf =
            mesh.attributes.getOrCreate<u32>(geom::names::kSoftBody, geom::Domain::Vertex, geom::AttrType::U32);
        std::fill(bodyOf.begin(), bodyOf.end(), body.id);
        mesh.attributes.getOrCreate<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex, geom::AttrType::F32);
    }
    body.meshes = {0, 1, 2};
    body.riders = {SoftPart{1, kInvalidIndex}, SoftPart{2, kInvalidIndex}};
    return body;
}

/// G-S1 for the riders: expanded with nothing baked, every clip puts each part
/// where its own skin did, and every weight of it is on a bone the body made.
void CheckRidersAtRest(const Figure& f, ProfileId target) {
    const SoftWeights weights = SoftWeightsFor(target);
    const u32 cap = Profile(target).maxBoneInfluences;
    Document expanded = f.document;
    Diagnostics diagnostics;
    ExpandSoftToBones(expanded, target, diagnostics);
    const Animator animatorBefore(f.document, 0);
    const Animator animatorAfter(expanded, 0);
    f32 worst = 0.0f;
    for (const u32 m : {1u, 2u}) {
        const Mesh& before = f.model().meshes[m];
        const Mesh& after = expanded.models[0].meshes[m];
        const std::span<const Vector3f> rest =
            before.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
        std::vector<std::vector<geom::Influence>> reference(rest.size());
        for (u32 v = 0; v < rest.size(); ++v) {
            reference[v] = AsFileHolds(f.model(), v, weights, cap, m);
            const std::span<const geom::Influence> skin = after.skin.forVertex(v);
            CHECK(skin.size() <= cap);
            CHECK(skin.size() == reference[v].size());
            for (const geom::Influence& influence : skin) {
                CHECK(influence.bone >= f.model().nodes.size());
            }
        }
        for (u32 clip = 0; clip < f.document.clips.size(); ++clip) {
            for (u32 frame = 0; frame <= 30; ++frame) {
                Mix mix;
                mix.plays.push_back(Play{clip, static_cast<f32>(frame) / 30.0f, 1.0f, false});
                Pose poseBefore, poseAfter;
                animatorBefore.evaluate(mix, poseBefore);
                animatorAfter.evaluate(mix, poseAfter);
                const std::vector<Vector3f> moved = skinning::DeformMesh(after, poseAfter.skinning);
                for (u32 v = 0; v < rest.size(); ++v) {
                    const Vector3f was = skinning::DeformPosition(rest[v], reference[v], poseBefore.skinning);
                    worst = std::max(worst, (moved[v] - was).length());
                }
            }
        }
    }
    CHECK(worst <= 1e-5f * f.size);
}

} // namespace

TEST_CASE("wem soft a part that rides rigidly is a lump a bone, and never scales", "[wem][soft][riders]") {
    Figure f = MakeTube(Skinned::Rigid, 25, 16);
    Model& model = f.model();
    SoftBody& body = MakeRiders(f, 4, false);
    const SoftMakeReport report = MakeSoftLumps(model, body, 4, false);
    CHECK(report.riders == 3u);
    CHECK(report.surface.made == 6u);

    // One lump for the saddle, on b1; two for the strap, on b1 and b2. None of
    // the surface's own lumps went to a rider's point.
    std::vector<u32> saddle, strap;
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        const SoftLump& lump = body.lumps[k];
        if (lump.rider != 0) {
            CHECK(lump.rigid);
            CHECK(lump.kind == SoftLumpKind::Surface);
            (lump.rider == 1 ? saddle : strap).push_back(k);
        } else {
            CHECK(!lump.rigid);
            if (lump.kind == SoftLumpKind::Surface) {
                CHECK(std::hypot(lump.anchor.x, lump.anchor.y) <= 1.0f + 1e-4f);
            }
        }
    }
    REQUIRE(saddle.size() == 1u);
    REQUIRE(strap.size() == 2u);
    CHECK(body.lumps[saddle[0]].carrier == f.bones[1]);
    CHECK(body.lumps[saddle[0]].rest.x == Approx(1.25f).margin(0.1));
    CHECK(body.lumps[saddle[0]].rest.z == Approx(2.0f).margin(0.1));
    CHECK(body.lumps[strap[0]].carrier == f.bones[1]);
    CHECK(body.lumps[strap[1]].carrier == f.bones[2]);

    // A rider's point is its part's, and wholly soft whatever its layer holds.
    const SoftPoints points = GatherSoftPoints(model, body.id, 4);
    u32 riding = 0;
    for (const SoftPoint& point : points.points) {
        if (point.mesh == 0) {
            CHECK(point.rider == 0);
            continue;
        }
        ++riding;
        CHECK(point.rider == point.mesh);
        CHECK(point.softness == 1.0f);
    }
    CHECK(riding == 16u);
    // However many lumps the surface is given, none goes to a rider's point:
    // spread over them too, the farthest of a bone's points are the saddle's.
    {
        std::vector<SoftLump> many;
        MakeSurfaceLumps(model, points, 96, many);
        CHECK(many.size() > 40u);
        for (const SoftLump& lump : many) {
            CHECK(std::hypot(lump.anchor.x, lump.anchor.y) <= 1.0f + 1e-4f);
        }
    }

    // Each of a part's bones' weights is the part's lump of that bone, whole.
    SoftSkinner skinner(model, body);
    for (u32 v = 0; v < 8; ++v) {
        const std::span<const geom::Influence> one = skinner.of(1, v);
        REQUIRE(one.size() == 1u);
        CHECK(one[0].bone == skinner.lumpBase() + saddle[0]);
        CHECK(one[0].weight == Approx(1.0f));
        const std::span<const geom::Influence> two = skinner.of(2, v);
        REQUIRE(two.size() == 2u);
        for (const geom::Influence& influence : two) {
            CHECK((influence.bone == skinner.lumpBase() + strap[0] || influence.bone == skinner.lumpBase() + strap[1]));
            CHECK(influence.weight == Approx(0.5f));
        }
        CHECK(two[0].bone != two[1].bone);
    }
    // And a rider's lump carries its own part alone: no point of the wall has a lane to one.
    for (u32 v = 0; v < model.meshes[0].vertexCount(); ++v) {
        for (const geom::Influence& influence : skinner.of(0, v)) {
            const bool riders = influence.bone >= skinner.lumpBase() &&
                                body.lumps[influence.bone - skinner.lumpBase()].rider != 0;
            CHECK(!riders);
        }
    }

    // The body swollen by three tenths away from its axis: the saddle goes out
    // with the wall, and is the size it was.
    const SoftRig rig(model, body.id);
    for (const u32 k : {saddle[0], strap[0], strap[1]}) {
        CHECK(!rig.scales(k));
        CHECK(rig.moves(k));
    }
    const std::vector<Matrix44f> skin(model.nodes.size(), Matrix44f::identity());
    std::vector<Vector3f> rest;
    rig.carry(skin, rest);
    std::vector<Vector3f> goal = rest;
    for (Vector3f& p : goal) {
        p.x *= 1.3f;
        p.y *= 1.3f;
    }
    std::vector<Transform> departures;
    rig.fit(skin, goal, departures);
    REQUIRE(departures.size() == body.lumps.size());
    const Transform& moved = departures[saddle[0]];
    CHECK(moved.scale.x == Approx(1.0f).margin(1e-6));
    CHECK(moved.scale.y == Approx(1.0f).margin(1e-6));
    CHECK(moved.scale.z == Approx(1.0f).margin(1e-6));
    CHECK(moved.translation.x == Approx(0.375f).margin(0.05));
    std::vector<Vector3f> posed;
    rig.pose(skin, departures, posed);
    std::vector<u32> corners;
    for (u32 p = 0; p < points.points.size(); ++p) {
        if (points.points[p].mesh == 1) {
            corners.push_back(p);
        }
    }
    REQUIRE(corners.size() == 8u);
    for (const u32 a : corners) {
        for (const u32 b : corners) {
            CHECK((posed[a] - posed[b]).length() == Approx((rest[a] - rest[b]).length()).margin(1e-4));
        }
    }
    // And the wall beside it did swell.
    f32 widest = 0.0f;
    for (u32 p = 0; p < points.points.size(); ++p) {
        if (points.points[p].mesh == 0) {
            widest = std::max(widest, std::hypot(posed[p].x, posed[p].y));
        }
    }
    CHECK(widest > 1.2f);

    CheckRidersAtRest(f, ProfileId::Wc3Reforged);
}

TEST_CASE("G-S1 the rest is exact for a part that rides rigidly, in classic groups too", "[wem][soft][riders]") {
    Figure f = MakeTube(Skinned::Rigid, 25, 16, ProfileId::Wc3Classic);
    SoftBody& body = MakeRiders(f, 8, true);
    MakeSoftLumps(f.model(), body, 8, true);
    CheckRidersAtRest(f, ProfileId::Wc3Classic);
}

TEST_CASE("wem soft a place follows its spot when its mesh's faces are numbered again", "[wem][soft][references]") {
    // A strip of six triangles in a plane, and the same strip from its third on.
    const auto strip = [](u32 from) {
        geom::MeshBuilder builder;
        builder.addSection(MeshSection{});
        std::vector<geom::VertexId> corners;
        for (u32 i = 0; i < 8; ++i) {
            corners.push_back(builder.addVertex(Vector3f{static_cast<f32>(i / 2), static_cast<f32>(i % 2), 0.0f}));
        }
        for (u32 t = from; t < 6; ++t) {
            builder.addTriangle(corners[t], corners[t + 1], corners[t + 2]);
        }
        return builder.build().mesh;
    };
    const Mesh before = strip(0), after = strip(2);
    REQUIRE(before.faceCount() == 6u);
    REQUIRE(after.faceCount() == 4u);
    const auto pointOn = [](const Mesh& mesh, u32 face, const Vector3f& weights) {
        const geom::FaceSet& faces = mesh.faceSet();
        std::size_t corner = 0;
        for (u32 f = 0; f < face; ++f) {
            corner += faces.faceValence[f];
        }
        const auto positions = mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
        return positions[faces.cornerVertex[corner]] * weights.x + positions[faces.cornerVertex[corner + 1]] * weights.y +
               positions[faces.cornerVertex[corner + 2]] * weights.z;
    };
    PhysicsSet physics;
    SoftBody body;
    body.id = 1;
    // An effect's place on a face that stays, a kick's on one that goes, a
    // site's on another mesh, and an effect with no place at all.
    const Vector3f weights{0.2f, 0.3f, 0.5f};
    SoftEffect bulge = DefaultSoftEffect(SoftEffectKind::Bulge);
    bulge.id = body.nextEffect++;
    bulge.place.mesh = 0;
    bulge.place.face = 4;
    bulge.place.weights = weights;
    body.effects.push_back(bulge);
    SoftEffect jiggle = DefaultSoftEffect(SoftEffectKind::Jiggle);
    jiggle.id = body.nextEffect++;
    jiggle.jiggle->kickIds = {body.nextEffect++};
    jiggle.jiggle->kickMesh = {0};
    jiggle.jiggle->kickFace = {0};
    jiggle.jiggle->kickWeights = {Vector3f{1, 0, 0}};
    jiggle.jiggle->kickDirection = {Vector3f{0, 0, 0}};
    jiggle.jiggle->kickStrength = {1.0f};
    body.effects.push_back(jiggle);
    SoftEffect drip = DefaultSoftEffect(SoftEffectKind::Drip);
    drip.id = body.nextEffect++;
    drip.drip->siteMesh = {1};
    drip.drip->siteFace = {5};
    drip.drip->siteWeights = {weights};
    drip.drip->siteThickness = {0.0f};
    body.effects.push_back(drip);
    body.effects.push_back(DefaultSoftEffect(SoftEffectKind::Inflate));
    physics.soft.push_back(body);
    const Vector3f stood = pointOn(before, 4, weights);

    CHECK(SoftFollowRenumberedFaces(physics, 0, before, after) == 2u);
    const SoftBody& now = physics.soft[0];
    // The same spot, on the face that has it now: two numbers down.
    CHECK(now.effects[0].place.face == 2u);
    CHECK((pointOn(after, now.effects[0].place.face, now.effects[0].place.weights) - stood).length() < 1e-5f);
    // Its face gone, the kick stands on the nearest that is left: the corner
    // at (1, 0) is the strip's nearest point to where it stood, at (0, 0).
    const SoftJiggle& kicks = *now.effects[1].jiggle;
    REQUIRE(kicks.kickFace[0] < after.faceCount());
    CHECK((pointOn(after, kicks.kickFace[0], kicks.kickWeights[0]) - Vector3f{1, 0, 0}).length() < 1e-5f);
    // Another mesh's, and no place, are left as they are.
    CHECK(now.effects[2].drip->siteFace == std::vector<u32>{5});
    CHECK(now.effects[3].place.mesh == kInvalidIndex);
    // Asked again of the mesh as it stands, nothing moves.
    CHECK(SoftFollowRenumberedFaces(physics, 0, after, after) == 0u);
    // A mesh left with no face leaves a place nowhere to stand: it has none.
    Mesh bare = after;
    geom::MeshBuilder none;
    none.addSection(MeshSection{});
    bare = none.build().mesh;
    CHECK(SoftFollowRenumberedFaces(physics, 0, after, bare) == 2u);
    CHECK(physics.soft[0].effects[0].place.mesh == kInvalidIndex);
    CHECK(physics.soft[0].effects[1].jiggle->kickMesh[0] == kInvalidIndex);
}
