// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

// The soft bodies' part of a clip's bake, and what a file makes of it
// (EDIT_MODE_SOFTBODY_DESIGN.md §11, §12; the plan's G-S4 and G-S5, the
// library's half).

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <whiteout/models/wem/anim/animator.h>
#include <whiteout/models/wem/anim/crossing.h>
#include <whiteout/models/wem/converters.h>
#include <whiteout/models/wem/fbx_converter.h>
#include <whiteout/models/wem/parser.h>
#include <whiteout/models/wem/physics/bake.h>
#include <whiteout/models/wem/physics/crossing.h>
#include <whiteout/models/wem/physics/soft_bake.h>
#include <whiteout/models/wem/physics/soft_run.h>
#include <whiteout/models/wem/physics/soft_step.h>
#include <whiteout/models/wem/reflect_bytes.h>
#include <whiteout/models/wem/retarget.h>
#include <whiteout/models/wem/skinning/deform.h>
#include <whiteout/models/wem/writer.h>

#include "wem_soft_figures.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace softfig;
using Catch::Approx;

namespace {

/// The host's physics for a model with nothing to simulate.
BakeHooks NoPhysics() {
    BakeHooks hooks;
    hooks.reset = [](const Document&, u32, std::span<const u8>) {};
    hooks.step = [](const Document&, u32, const BakeStep&, Pose&) {};
    return hooks;
}

SoftEffect& AddEffect(SoftBody& body, SoftEffectKind kind) {
    SoftEffect effect = DefaultSoftEffect(kind);
    effect.id = body.nextEffect++;
    body.effects.push_back(effect);
    return body.effects.back();
}

BakeReport Bake(Document& document, u32 clip) {
    const BakeRun run = SimulateClip(document, clip, NoPhysics());
    INFO(run.error);
    REQUIRE(run.complete);
    return BakeClip(document, run);
}

std::vector<Matrix44f> SkinAt(const Document& document, u32 clip, f32 seconds) {
    const Animator animator(document, 0);
    Mix mix;
    mix.plays.push_back(Play{clip, seconds, 1.0f, false});
    Pose pose;
    animator.evaluate(mix, pose);
    return pose.skinning;
}

/// Where the editor's deform has mesh 0's vertices at @p seconds of @p clip:
/// the pose, with each lump's departure as the clip's baked tracks hold it.
std::vector<Vector3f> Drawn(const Document& document, u32 clip, f32 seconds) {
    const SoftDeformer deformer(document.models[0]);
    const std::vector<Matrix44f> skin = SkinAt(document, clip, seconds);
    return deformer.positions(0, deformer.palette(skin, SoftDeparturesAt(document, 0, clip, seconds)));
}

/// Where a step of body @p body puts its points at @p seconds of @p clip, by
/// the mesh's vertices: the body stepped live on @p document's own pose.
std::vector<Vector3f> Stepped(const Document& document, u32 body, u32 clip, f32 seconds) {
    const SoftStepper stepper(document, 0, body);
    const auto poseAt = [&](f32 at, std::vector<Matrix44f>& skin) { skin = SkinAt(document, clip, at); };
    SoftStepResult out;
    stepper.step(SkinAt(document, clip, seconds), SoftClock{clip, seconds}, poseAt, out);
    const SoftDeformer deformer(document.models[0]);
    return deformer.positions(0, deformer.palette(SkinAt(document, clip, seconds), out.departures));
}

f32 Farthest(const std::vector<Vector3f>& a, const std::vector<Vector3f>& b) {
    REQUIRE(a.size() == b.size());
    f32 far = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        far = std::max(far, (a[i] - b[i]).length());
    }
    return far;
}

bool IsLump(const Model& model, const SubTrack& track) {
    const AnimChannel* channel = model.animChannels.find(track.channel);
    return channel != nullptr && channel->target.kind == TrackTarget::Kind::Physics &&
           (channel->target.channel == Channel::SoftLumpTranslation ||
            channel->target.channel == Channel::SoftLumpRotation || channel->target.channel == Channel::SoftLumpScale);
}

} // namespace

TEST_CASE("G-S4 a bake writes the lumps' tracks inside the spans, and Unbake is exact", "[wem][soft][bake]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = SoftenAll(f, 6, true).id;
    {
        SoftBody& body = *f.model().physics.softBody(id);
        // Every lump a copy or on the surface: no key of the animation is the bake's.
        for (const SoftLump& lump : body.lumps) {
            REQUIRE(lump.kind != SoftLumpKind::Bone);
        }
        SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
        inflate.amount = 0.3f;
        inflate.inflate->along = SoftAlong::Surface;
        const SoftSpan span{0.25f, 0.75f, 0.1f, 0.1f};
        SetSoftSpans(f.document, 0, 0, id, inflate.id, {&span, 1});
    }
    f.document.clips[0].physics.emplace();
    const Document source = f.document;
    const std::vector<u8> before = ReflectBytes(source);
    const u64 inputs = BakeInputs(source, 0);

    Document baked = source;
    const BakeReport report = Bake(baked, 0);
    INFO(report.error);
    REQUIRE(report.ok);
    REQUIRE(report.soft.size() == 1u);
    CHECK(report.soft[0].body == id);
    CHECK(report.soft[0].lumps > 0u);
    CHECK(report.soft[0].keys > 0u);
    // No node was written: the soft body's tracks are its own.
    CHECK(report.nodes == 0u);
    const Clip& clip = baked.clips[0];
    REQUIRE(clip.physics.has_value());
    REQUIRE(clip.physics->baked.has_value());
    const PhysicsBake& record = *clip.physics->baked;
    CHECK(record.softIds == std::vector<u32>{id});
    CHECK(record.softLumps == std::vector<u32>{report.soft[0].lumps});
    CHECK(record.softKeys == std::vector<u32>{report.soft[0].keys});
    CHECK(record.softChannels.empty());
    CHECK(record.inputs == inputs);

    // Every track the clip had is as it was, and every track it gained is a lump's.
    const f32 dt = 1.0f / 60.0f;
    u32 lumpTracks = 0, lumpKeys = 0;
    for (const SubTrack& track : clip.containers.front().subTracks) {
        if (!IsLump(baked.models[0], track)) {
            const SubTrack* was = source.clips[0].containers.front().find(track.channel);
            REQUIRE(was != nullptr);
            CHECK((ReflectBytes(track) == ReflectBytes(*was)));
            continue;
        }
        ++lumpTracks;
        lumpKeys += static_cast<u32>(track.times.size());
        // Inside the span, a step either side at most; and at rest at both edges.
        REQUIRE(!track.times.empty());
        CHECK(track.times.front() >= 0.25f - dt - 1e-4f);
        CHECK(track.times.back() <= 0.75f + dt + 1e-4f);
        const AnimChannel* channel = baked.models[0].animChannels.find(track.channel);
        const std::size_t size = channel->valueType == geom::AttrType::Quat ? sizeof(Quaternion) : sizeof(Vector3f);
        for (const std::size_t k : {std::size_t{0}, track.times.size() - 1}) {
            if (channel->target.channel == Channel::SoftLumpTranslation) {
                Vector3f value;
                std::memcpy(&value, track.values.data() + k * size, size);
                CHECK(value.length() == 0.0f);
            } else if (channel->target.channel == Channel::SoftLumpScale) {
                Vector3f value;
                std::memcpy(&value, track.values.data() + k * size, size);
                CHECK((value - Vector3f{1, 1, 1}).length() == 0.0f);
            }
        }
    }
    CHECK(lumpTracks > 0u);
    CHECK(lumpKeys == report.soft[0].keys);
    // Thinned: far fewer than a key every step of the span.
    CHECK(lumpKeys < lumpTracks * 31u);
    CHECK((ReflectBytes(baked.clips[1]) == ReflectBytes(source.clips[1])));
    CHECK(ClipKeysSoftLumps(baked, 0));
    CHECK(!ClipKeysSoftLumps(baked, 1));
    CHECK(!ClipKeysSoftLumps(source, 0));

    // What the tracks play is what a step gives, within the clip's accuracy
    // (a thousandth of the model's height); and nothing outside the span.
    const f32 accuracy = 0.001f * 6.0f;
    CHECK(report.soft[0].within <= accuracy + 1e-5f);
    f32 swell = 0.0f;
    for (const f32 seconds : {0.3f, 0.4f, 0.5f, 0.65f, 0.72f}) {
        const std::vector<Vector3f> drawn = Drawn(baked, 0, seconds);
        CHECK(Farthest(drawn, Stepped(source, id, 0, seconds)) <= accuracy * 1.5f);
        swell = std::max(swell, Farthest(drawn, Drawn(source, 0, seconds)));
    }
    CHECK(swell > 0.05f);
    for (const f32 seconds : {0.0f, 0.1f, 0.2f, 0.85f, 1.0f}) {
        CHECK(Farthest(Drawn(baked, 0, seconds), Drawn(source, 0, seconds)) == 0.0f);
    }
    // The other clip keys nothing of it: every lump at rest there.
    for (const Transform& departure : SoftDeparturesAt(baked, 0, 1, 0.5f)) {
        CHECK(SoftAtRest(departure, 6.0f));
    }

    // In date; a setting changed, or a softness stroke, and it is out of date.
    CHECK(OutOfDateBakes(baked).empty());
    {
        Document changed = baked;
        changed.models[0].physics.softBody(id)->effects[0].amount = 0.4f;
        CHECK(OutOfDateBakes(changed) == std::vector<u32>{0});
        Document painted = baked;
        painted.models[0]
            .meshes[0]
            .attributes.getOrCreate<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex, geom::AttrType::F32)[5] = 0.5f;
        CHECK(OutOfDateBakes(painted) == std::vector<u32>{0});
        // A bound is measured from what the document holds, the bake's own
        // keys among it, and a save measures every one again: none is in the
        // key, so a clip is not out of date for having been saved.
        Document measured = baked;
        measured.bounds.maximum.x += 3.0f;
        measured.models[0].bounds.maximum.x += 3.0f;
        measured.models[0].meshes[0].bounds.maximum.x += 3.0f;
        for (MeshSection& section : measured.models[0].meshes[0].sections) {
            section.bounds.maximum.x += 3.0f;
        }
        for (Clip& clip : measured.clips) {
            clip.bounds.maximum.x += 3.0f;
        }
        CHECK(BakeInputs(measured, 0) == BakeInputs(baked, 0));
        CHECK(BakeInDate(measured, 0));
        CHECK(OutOfDateBakes(measured).empty());
        CHECK(!BakeInDate(painted, 0));
        // The other clip was not baked: there is nothing of it to be in date.
        CHECK(!BakeInDate(baked, 1));
    }

    // A second bake of it replaces the first: the same tracks, the same channels.
    {
        Document again = baked;
        REQUIRE(Bake(again, 0).ok);
        CHECK((ReflectBytes(again.clips[0]) == ReflectBytes(baked.clips[0])));
        CHECK(again.models[0].animChannels.channels.size() == baked.models[0].animChannels.channels.size());
    }

    // Unbake: the document as it was, to the byte.
    UnbakeClip(baked, 0);
    CHECK((ReflectBytes(baked) == before));
}

TEST_CASE("G-S4 a bake of a clip that was never set up leaves it so, and its Unbake leaves no settings",
          "[wem][soft][bake]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = SoftenAll(f, 6, true).id;
    {
        SoftEffect& inflate = AddEffect(*f.model().physics.softBody(id), SoftEffectKind::Inflate);
        inflate.amount = 0.3f;
        inflate.everywhere = true;
    }
    // On in every clip, with no span and no settings: nothing set the clip up.
    REQUIRE(!f.document.clips[0].physics.has_value());
    REQUIRE(!IsSetUp(f.document.clips[0]));
    const Document source = f.document;
    const std::vector<u8> before = ReflectBytes(source);
    const u64 inputs = BakeInputs(source, 0);
    Document baked = source;
    REQUIRE(Bake(baked, 0).ok);
    // The record needs settings to stand in. They are the bake's, not the
    // clip's: it carries no dropped rig over from what it comes from, as before.
    REQUIRE(baked.clips[0].physics.has_value());
    REQUIRE(baked.clips[0].physics->baked.has_value());
    CHECK(baked.clips[0].physics->baked->madeSettings);
    CHECK(!IsSetUp(baked.clips[0]));
    CHECK(baked.clips[0].physics->baked->inputs == inputs);
    CHECK(OutOfDateBakes(baked).empty());
    // Baked again it reads the same clip, and writes the same keys.
    {
        Document again = baked;
        REQUIRE(Bake(again, 0).ok);
        CHECK((ReflectBytes(again.clips[0]) == ReflectBytes(baked.clips[0])));
    }
    // Set up by hand it is another clip to the bake, and keeps its settings.
    {
        Document set = baked;
        SetUpClip(set.clips[0]);
        CHECK(IsSetUp(set.clips[0]));
        CHECK(OutOfDateBakes(set) == std::vector<u32>{0});
        UnbakeClip(set, 0);
        CHECK(set.clips[0].physics.has_value());
    }
    UnbakeClip(baked, 0);
    CHECK(!baked.clips[0].physics.has_value());
    CHECK((ReflectBytes(baked) == before));
}

TEST_CASE("G-S4 a bone used as it is is keyed in place, scale and all, and Unbake gives its keys back",
          "[wem][soft][bake]") {
    Figure f = MakeTube(Skinned::Rigid, 19, 16);
    const u32 id = SoftenAll(f, 0, false).id;
    u32 own = kInvalidNode;
    {
        SoftBody& body = *f.model().physics.softBody(id);
        for (const SoftLump& lump : body.lumps) {
            if (lump.kind == SoftLumpKind::Bone) {
                own = lump.node;
            }
        }
        // The end of the chain is the body's alone, and is used as it is.
        REQUIRE(own == f.bones[2]);
        SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
        inflate.amount = 0.3f;
        inflate.inflate->along = SoftAlong::Bones;
        const SoftSpan span{0.25f, 0.75f, 0.1f, 0.1f};
        SetSoftSpans(f.document, 0, 0, id, inflate.id, {&span, 1});
    }
    f.document.clips[0].physics.emplace();
    const Document source = f.document;
    const std::vector<u8> before = ReflectBytes(source);
    Document baked = source;
    const BakeReport report = Bake(baked, 0);
    INFO(report.error);
    REQUIRE(report.ok);
    REQUIRE(report.soft.size() == 1u);
    const PhysicsBake& record = *baked.clips[0].physics->baked;
    // The bone's scale track is the bake's now: the clip keyed one, and its
    // source is kept.
    REQUIRE(!record.softChannels.empty());
    u32 scale = kInvalidIndex;
    for (const u32 channel : record.softChannels) {
        const AnimChannel* entry = baked.models[0].animChannels.find(channel);
        REQUIRE(entry != nullptr);
        CHECK(entry->target.kind == TrackTarget::Kind::Node);
        CHECK(entry->target.node == own);
        if (entry->target.channel == Channel::Scale) {
            scale = channel;
        }
    }
    REQUIRE(scale != kInvalidIndex);
    const SubTrack* was = source.clips[0].containers.front().find(scale);
    const SubTrack* now = baked.clips[0].containers.front().find(scale);
    REQUIRE(was != nullptr);
    REQUIRE(now != nullptr);
    CHECK((ReflectBytes(*was) != ReflectBytes(*now)));
    bool kept = false;
    for (const SubTrackContainer& container : record.source) {
        if (const SubTrack* held = container.find(scale)) {
            kept = ReflectBytes(*held) == ReflectBytes(*was);
        }
    }
    CHECK(kept);
    // Outside the span the bone plays its hand keys; inside, those times the swell.
    const auto scaleAt = [&](const Document& document, f32 seconds) {
        const Animator animator(document, 0);
        Mix mix;
        mix.plays.push_back(Play{0, seconds, 1.0f, false});
        Pose pose;
        animator.sample(mix, pose, true);
        return pose.local[own].scale;
    };
    CHECK((scaleAt(baked, 0.1f) - scaleAt(source, 0.1f)).length() < 1e-4f);
    CHECK((scaleAt(baked, 0.9f) - scaleAt(source, 0.9f)).length() < 1e-4f);
    CHECK(scaleAt(baked, 0.5f).x > scaleAt(source, 0.5f).x * 1.02f);

    // The baked pose with the lumps' tracks draws what a step on the source
    // draws: the bone's own keys carry its lump, and its track carries none.
    const f32 accuracy = 0.001f * 6.0f;
    for (const f32 seconds : {0.3f, 0.5f, 0.7f}) {
        CHECK(Farthest(Drawn(baked, 0, seconds), Stepped(source, id, 0, seconds)) <= accuracy * 3.0f);
    }
    // And the pose the bodies are stepped on is the source's, bake or no bake:
    // a step on a baked clip does not swell what is already swollen.
    {
        const Animator animator(baked, 0);
        const Animator plain(source, 0);
        for (const f32 seconds : {0.0f, 0.3f, 0.5f, 0.7f}) {
            Pose from, expected;
            SoftSourcePose(baked, animator, 0, seconds, from);
            SoftSourcePose(source, plain, 0, seconds, expected);
            REQUIRE(from.skinning.size() == expected.skinning.size());
            for (std::size_t n = 0; n < from.skinning.size(); ++n) {
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        CHECK(from.skinning[n].data[r][c] == Approx(expected.skinning[n].data[r][c]).margin(2e-3));
                    }
                }
            }
        }
    }
    UnbakeClip(baked, 0);
    CHECK((ReflectBytes(baked) == before));
}

TEST_CASE("G-S5 a file's bones play what the editor draws: the expansion keys them from the lumps' tracks",
          "[wem][soft][bake]") {
    for (const bool copies : {true, false}) {
        Figure f = MakeTube(Skinned::Two, 19, 16);
        const u32 id = SoftenAll(f, 6, copies).id;
        {
            SoftBody& body = *f.model().physics.softBody(id);
            // §6.5, a lump has no child bone: no lump of the surface hangs from
            // a bone used as it is, which a bake keys in place.
            for (const SoftLump& lump : body.lumps) {
                for (const SoftLump& other : body.lumps) {
                    CHECK(!(lump.kind == SoftLumpKind::Surface && other.kind == SoftLumpKind::Bone &&
                            other.node == lump.carrier));
                }
            }
            SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
            inflate.amount = 0.3f;
            inflate.inflate->along = SoftAlong::Surface;
            const SoftSpan span{0.2f, 0.8f, 0.1f, 0.1f};
            SetSoftSpans(f.document, 0, 0, id, inflate.id, {&span, 1});
        }
        f.document.clips[0].physics.emplace();
        const Document source = f.document;
        REQUIRE(Bake(f.document, 0).ok);
        const Document baked = f.document;
        // So the baked clip draws what the body was stepped to, copies or not:
        // no lump takes its carrier's departure on top of its own.
        for (const f32 seconds : {0.3f, 0.5f, 0.7f}) {
            INFO("copies " << copies << " at " << seconds);
            CHECK(Farthest(Drawn(baked, 0, seconds), Stepped(source, id, 0, seconds)) <= 0.001f * 6.0f * 3.0f);
        }

        Document file = baked;
        Diagnostics diagnostics;
        const SoftExpansion expansion = ExpandSoftToBones(file, ProfileId::Generic, diagnostics);
        CHECK(expansion.bones > 0u);
        CHECK(expansion.keys > 0u);
        // Skinned by its bones alone, as a game skins it, the file's mesh is
        // where the editor's deform had it: floats, so to rounding.
        for (const f32 seconds : {0.0f, 0.3f, 0.5f, 0.7f, 1.0f}) {
            const std::vector<Vector3f> drawn = Drawn(baked, 0, seconds);
            const std::vector<Vector3f> played = skinning::DeformMesh(file.models[0].meshes[0], SkinAt(file, 0, seconds));
            INFO("copies " << copies << " at " << seconds);
            CHECK(Farthest(drawn, played) < 2e-3f);
        }
        // The clip that was not baked plays as it did, with no key on a made bone.
        for (const f32 seconds : {0.0f, 0.5f}) {
            const std::vector<Vector3f> played = skinning::DeformMesh(file.models[0].meshes[0], SkinAt(file, 1, seconds));
            CHECK(Farthest(played, skinning::DeformMesh(baked.models[0].meshes[0], SkinAt(baked, 1, seconds))) < 1e-4f);
        }
        const u32 nodesBefore = baked.models[0].nodes.size();
        for (const SubTrack& track : file.clips[1].containers.front().subTracks) {
            const AnimChannel* channel = file.models[0].animChannels.find(track.channel);
            REQUIRE(channel != nullptr);
            if (channel->target.kind == TrackTarget::Kind::Node) {
                CHECK(channel->target.node < nodesBefore);
            }
        }
    }
}

namespace {

/// The tube's rig said as an explicit one whose rest chain does not compose to
/// its bind, as an `.m3`'s does not. Each bone's bind is its pivot, turned a
/// quarter about Y (square to the model, so a scale along the model's axes is
/// one along the bone's), and stored as an inverse; at rest the bone stands a
/// little off it, turned further.
void MakeExplicit(Figure& f) {
    NodeTree& tree = f.model().nodes;
    tree.rig = RigConvention::ExplicitBind;
    tree.poseSchema = {PoseSchema{"bindInverse", PoseSpace::Model, true}};
    tree.authoritativePose = 0;
    std::vector<Transform> rest(tree.size());
    for (u32 n = 0; n < tree.size(); ++n) {
        Node& node = tree.nodes[n];
        Transform bind;
        bind.translation = node.pivot;
        bind.rotation = Quaternion::from_axis_angle(Vector3f{0, 1, 0}, 1.5707963f);
        node.poses = {Inverse(bind)};
        rest[n] = bind;
        rest[n].translation = bind.translation + Vector3f{0.15f, -0.1f, 0.05f * static_cast<f32>(n + 1)};
        rest[n].rotation = (Quaternion::from_axis_angle(Vector3f{1, 0, 0}, 1.2f) * bind.rotation).normalized();
        const Matrix44f above = node.parent < tree.size() ? ToMatrix(rest[node.parent]) : Matrix44f::identity();
        node.local = FromMatrix(ToMatrix(rest[n]) * Matrix44f::inverse(above));
    }
}

} // namespace

TEST_CASE("G-S5 on an explicit rig whose rest is not its bind, a file's bones play what the editor draws",
          "[wem][soft][bake]") {
    // Lumps on the surface, whose bones are all copied; then none, where the
    // tube's last bone is used as it is and keyed in place.
    for (const u32 detail : {6u, 0u}) {
        Figure f = MakeTube(Skinned::Two, 19, 16);
        MakeExplicit(f);
        {
            // The rig as made: its rest chain is not the skin's bind.
            const NodeTree& tree = f.model().nodes;
            const Matrix44f atRest = tree.inverseBindMatrix(f.bones[2]) * ToMatrix(tree.worldBind(f.bones[2]));
            REQUIRE(std::abs(atRest.data[3][0]) + std::abs(atRest.data[3][1]) + std::abs(atRest.data[3][2]) > 0.05f);
        }
        const u32 id = SoftenAll(f, detail, false).id;
        {
            SoftBody& body = *f.model().physics.softBody(id);
            CHECK(std::any_of(body.lumps.begin(), body.lumps.end(), [](const SoftLump& lump) {
                      return lump.kind == SoftLumpKind::Bone;
                  }) == (detail == 0u));
            SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
            inflate.amount = 0.3f;
            inflate.inflate->along = SoftAlong::Surface;
            const SoftSpan span{0.2f, 0.8f, 0.1f, 0.1f};
            SetSoftSpans(f.document, 0, 0, id, inflate.id, {&span, 1});
        }
        f.document.clips[0].physics.emplace();
        const Document source = f.document;
        REQUIRE(Bake(f.document, 0).ok);
        const Document baked = f.document;
        // A bone keyed in place takes its lump's departure in the frame the skin
        // binds it in: the baked clip draws what the body was stepped to.
        for (const f32 seconds : {0.3f, 0.5f, 0.7f}) {
            INFO("detail " << detail << " at " << seconds);
            CHECK(Farthest(Drawn(baked, 0, seconds), Stepped(source, id, 0, seconds)) <= 0.001f * 6.0f * 3.0f);
        }
        // The bone itself: where its keys now put a point by its pivot is where
        // its lump's departure put it, to a small part of how far that is.
        const SoftBody& stepped = *source.models[0].physics.softBody(id);
        const SoftStepper stepper(source, 0, id);
        const SoftDeformer deformer(source.models[0]);
        for (const f32 seconds : {0.3f, 0.5f, 0.7f}) {
            const auto poseAt = [&](f32 at, std::vector<Matrix44f>& skin) { skin = SkinAt(source, 0, at); };
            const std::vector<Matrix44f> from = SkinAt(source, 0, seconds);
            SoftStepResult out;
            stepper.step(from, SoftClock{0, seconds}, poseAt, out);
            const std::vector<Matrix44f> wanted = deformer.palette(from, out.departures);
            const std::vector<Matrix44f> keyed = SkinAt(baked, 0, seconds);
            for (const SoftLump& lump : stepped.lumps) {
                if (lump.kind != SoftLumpKind::Bone) {
                    continue;
                }
                f32 off = 0.0f, departs = 0.0f;
                for (const Vector3f& arm : {Vector3f{1, 0, 0}, Vector3f{0, 1, 0}, Vector3f{0, 0, 1}}) {
                    const Vector3f point = lump.rest + arm;
                    const Vector3f there = transform_point(point, wanted[lump.node]);
                    off = std::max(off, (there - transform_point(point, keyed[lump.node])).length());
                    departs = std::max(departs, (there - transform_point(point, from[lump.node])).length());
                }
                INFO("detail " << detail << " at " << seconds << ": departs " << departs << ", off " << off);
                CHECK(departs > 0.02f);
                CHECK(off < 0.1f * departs);
            }
        }

        Document file = baked;
        Diagnostics diagnostics;
        const SoftExpansion expansion = ExpandSoftToBones(file, ProfileId::Generic, diagnostics);
        CHECK(expansion.bones > 0u);
        CHECK(expansion.keys > 0u);
        for (const f32 seconds : {0.0f, 0.3f, 0.5f, 0.7f, 1.0f}) {
            const std::vector<Vector3f> drawn = Drawn(baked, 0, seconds);
            const std::vector<Vector3f> played = skinning::DeformMesh(file.models[0].meshes[0], SkinAt(file, 0, seconds));
            INFO("detail " << detail << " at " << seconds);
            CHECK(Farthest(drawn, played) < 2e-3f);
        }
        // A made bone at rest moves a point as its family's bone does: the clip
        // that was not baked plays as it did.
        for (const f32 seconds : {0.0f, 0.5f}) {
            const std::vector<Vector3f> played = skinning::DeformMesh(file.models[0].meshes[0], SkinAt(file, 1, seconds));
            INFO("detail " << detail << " at " << seconds);
            CHECK(Farthest(played, skinning::DeformMesh(baked.models[0].meshes[0], SkinAt(baked, 1, seconds))) < 1e-4f);
        }
    }
}

TEST_CASE("G-S5 a baked Melt, Blobify, Ripple and Jiggle play from a file's bones as the editor drew them",
          "[wem][soft][bake]") {
    for (const SoftEffectKind kind :
         {SoftEffectKind::Melt, SoftEffectKind::Blobify, SoftEffectKind::Ripple, SoftEffectKind::Jiggle}) {
        Figure f = MakeTube(Skinned::Two, 19, 16);
        const u32 id = SoftenAll(f, 8, true).id;
        {
            SoftBody& body = *f.model().physics.softBody(id);
            SoftEffect& effect = AddEffect(body, kind);
            if (kind == SoftEffectKind::Melt) {
                // The largest deformation of the nine: down to its pool.
                const SoftSpan span{0.1f, 0.9f, 0.0f, 0.0f};
                SetSoftSpans(f.document, 0, 0, id, effect.id, {&span, 1});
            } else {
                effect.everywhere = true;
            }
            if (kind == SoftEffectKind::Ripple) {
                effect.amount = 0.05f;
                effect.place.mesh = 0;
                effect.place.face = 0;
                effect.place.weights = Vector3f{0.34f, 0.33f, 0.33f};
                effect.ripple->plays = SoftRipplePlays::Loops;
            }
        }
        f.document.clips[0].physics.emplace();
        const BakeReport report = Bake(f.document, 0);
        INFO(ToString(kind) << ": " << report.error);
        REQUIRE(report.ok);
        REQUIRE(report.soft.size() == 1u);
        CHECK(report.soft[0].keys > 0u);
        const Document baked = f.document;
        Document file = baked;
        Diagnostics diagnostics;
        const SoftExpansion expansion = ExpandSoftToBones(file, ProfileId::Generic, diagnostics);
        CHECK(expansion.keys > 0u);
        f32 moved = 0.0f, apart = 0.0f;
        for (const f32 seconds : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
            const std::vector<Vector3f> drawn = Drawn(baked, 0, seconds);
            const std::vector<Vector3f> played = skinning::DeformMesh(file.models[0].meshes[0], SkinAt(file, 0, seconds));
            const std::vector<Vector3f> plain = skinning::DeformMesh(baked.models[0].meshes[0], SkinAt(baked, 0, seconds));
            apart = std::max(apart, Farthest(drawn, played));
            moved = std::max(moved, Farthest(drawn, plain));
        }
        // What the effect moved is in the file, to rounding.
        CHECK(moved > 0.02f);
        CHECK(apart < 2e-3f + 1e-3f * moved);
    }
}

namespace {

using Points = std::vector<std::pair<Vector3f, Vector3f>>;

/// Every vertex of @p document's first model at @p seconds of the clip named
/// @p name: its rest position, and where its skin alone puts it, as a game
/// skins a file.
Points SkinnedPoints(const Document& document, const std::string& name, f32 seconds) {
    Points out;
    u32 clip = kInvalidIndex;
    for (u32 c = 0; c < document.clips.size(); ++c) {
        if (document.clips[c].name == name && document.clips[c].model == 0) {
            clip = c;
        }
    }
    if (clip == kInvalidIndex) {
        return out;
    }
    Mix mix;
    mix.plays.push_back(Play{clip, seconds, 1.0f, false});
    mix.globals = false;
    Pose pose;
    Animator(document, 0).evaluate(mix, pose);
    for (const Mesh& mesh : document.models[0].meshes) {
        const auto positions = mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
        for (u32 v = 0; v < positions.size(); ++v) {
            Vector3f at{0, 0, 0};
            f32 total = 0.0f;
            if (v < mesh.skin.vertexCount()) {
                for (const geom::Influence& influence : mesh.skin.forVertex(v)) {
                    if (influence.bone < pose.skinning.size()) {
                        at = at + transform_point(positions[v], pose.skinning[influence.bone]) * influence.weight;
                        total += influence.weight;
                    }
                }
            }
            out.emplace_back(positions[v], total > 0.0f ? at * (1.0f / total) : positions[v]);
        }
    }
    return out;
}

/// The farthest any point of @p skinned standing at a rest in @p targets is
/// from that target, and how many targets it met.
f32 Missed(const Points& skinned, const Points& targets, u32* matched) {
    f32 worst = 0.0f;
    *matched = 0;
    for (const auto& [want, to] : targets) {
        f32 nearest = 1e30f;
        for (const auto& [rest, at] : skinned) {
            if ((rest - want).length() < 1e-3f) {
                nearest = std::min(nearest, (at - to).length());
            }
        }
        if (nearest < 1e30f) {
            worst = std::max(worst, nearest);
            ++*matched;
        }
    }
    return worst;
}

} // namespace

TEST_CASE("G-S5 a baked Inflate and a baked Jiggle go out through each format and play as the editor drew them",
          "[wem][soft][bake][export]") {
    for (const SoftEffectKind kind : {SoftEffectKind::Inflate, SoftEffectKind::Jiggle}) {
        INFO(ToString(kind));
        Figure f = MakeTube(Skinned::Two, 19, 16);
        // One clip: a format that names its animations from a list keeps it.
        f.document.clips.pop_back();
        const u32 id = SoftenAll(f, 6, false).id;
        {
            SoftBody& body = *f.model().physics.softBody(id);
            SoftEffect& effect = AddEffect(body, kind);
            if (kind == SoftEffectKind::Inflate) {
                effect.amount = 0.3f;
                effect.inflate->along = SoftAlong::Surface;
                const SoftSpan span{0.2f, 0.8f, 0.1f, 0.1f};
                SetSoftSpans(f.document, 0, 0, id, effect.id, {&span, 1});
            } else {
                // A swing through the whole clip: every lump keyed at every bend.
                effect.everywhere = true;
            }
        }
        f.document.clips[0].physics.emplace();
        const Document unbaked = f.document;
        REQUIRE(Bake(f.document, 0).ok);
        const Document& baked = f.document;

        // What the editor draws at a few times, by rest position; and how far the
        // swell itself moves a point, which the bounds are said against.
        const auto rest = baked.models[0].meshes[0].attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
        std::vector<std::pair<f32, Points>> wanted;
        f32 swell = 0.0f;
        for (const f32 seconds : {0.3f, 0.5f, 0.7f}) {
            const std::vector<Vector3f> drawn = Drawn(baked, 0, seconds);
            const std::vector<Vector3f> plain = Drawn(unbaked, 0, seconds);
            Points targets;
            for (std::size_t v = 0; v < drawn.size(); ++v) {
                targets.emplace_back(rest[v], drawn[v]);
                swell = std::max(swell, (drawn[v] - plain[v]).length());
            }
            wanted.emplace_back(seconds, std::move(targets));
        }
        REQUIRE(swell > 0.05f);

        const auto expanded = [&](ProfileId profile) {
            Document out = baked;
            out.declare(profile);
            Diagnostics diagnostics;
            const SoftExpansion expansion = ExpandSoftToBones(out, profile, diagnostics);
            CHECK(expansion.keys > 0u);
            DropSoftForProfile(out, profile, diagnostics);
            CHECK(out.models[0].physics.soft.empty());
            return out;
        };
        const auto misses = [&](const Document& document) {
            f32 worst = 0.0f;
            for (const auto& [seconds, targets] : wanted) {
                u32 matched = 0;
                worst = std::max(worst, Missed(SkinnedPoints(document, "Bend", seconds), targets, &matched));
                CHECK(matched >= targets.size());
            }
            return worst;
        };
        const auto through = [&](const char* what, const Result<Document>& back, f32 bound) {
            INFO(what);
            for (const Diagnostic& d : back.diagnostics.all()) {
                if (d.severity == Severity::Error) {
                    WARN(what << ": " << d.message);
                }
            }
            REQUIRE(back.ok());
            Document read = *back.value;
            if (read.clips.size() == 1) {
                read.clips[0].name = "Bend";
            }
            const f32 missed = misses(read);
            WARN(what << ": missed " << missed << " of a swell of " << swell << " (bound " << bound << ")");
            CHECK(missed < bound);
        };

        // Warcraft III, Reforged: four byte weights a point.
        {
            const Document out = expanded(ProfileId::Wc3Reforged);
            CHECK(misses(out) < swell * 0.03f);
            const MdxConverter mdx;
            auto written = mdx.toMdx(out, ProfileId::Wc3Reforged, 1000);
            REQUIRE(written.ok());
            // Measured 1.5 % of the swell, here and in .m2: byte weights.
            through(".mdx v1000", mdx.fromMdx(*written.value), swell * 0.03f);
        }
        // Classic: equal shares in a group, so softness goes in steps, and with
        // lumps on the surface the worst point is off by more than the swell
        // itself (1.2 and 1.3 of it here): why the proposal there is *Detail*
        // 0 (§6.4). The file plays what the expansion made of it.
        {
            const Document out = expanded(ProfileId::Wc3Classic);
            const f32 own = misses(out);
            WARN("classic shares miss " << own << " of a swell of " << swell);
            CHECK(own < swell * 1.5f);
            const MdxConverter mdx;
            auto written = mdx.toMdx(out, ProfileId::Wc3Classic, 800);
            for (const Diagnostic& d : written.diagnostics.all()) {
                if (d.severity == Severity::Error) {
                    WARN(d.message);
                }
            }
            REQUIRE(written.ok());
            through(".mdx v800", mdx.fromMdx(*written.value), own + swell * 0.02f);
        }
        // World of Warcraft, by the library's writer (the viewer has no .m2 export).
        {
            const Document out = expanded(ProfileId::Wow);
            const M2Converter m2;
            auto written = m2.toM2(out, ProfileId::Wow);
            for (const Diagnostic& d : written.diagnostics.all()) {
                if (d.severity == Severity::Error) {
                    WARN(d.message);
                }
            }
            REQUIRE(written.ok());
            through(".m2", m2.fromM2(*written.value), swell * 0.03f);
        }
        // glTF: floats.
        {
            const Document out = expanded(ProfileId::Generic);
            const GltfConverter gltf;
            auto written = gltf.toGltf(out, ProfileId::Generic);
            for (const Diagnostic& d : written.diagnostics.all()) {
                if (d.severity == Severity::Error) {
                    WARN(d.message);
                }
            }
            REQUIRE(written.ok());
            // Floats: a thousandth of the swell at most, here and in FBX.
            through("glTF", gltf.fromGltf(*written.value), swell * 0.005f);
        }
        // FBX, through its bytes.
        {
            const Document out = expanded(ProfileId::Generic);
            const FbxConverter fbx;
            auto written = fbx.exportToBytes(out, ProfileId::Generic);
            for (const Diagnostic& d : written.diagnostics.all()) {
                if (d.severity == Severity::Error) {
                    WARN(d.message);
                }
            }
            REQUIRE(written.ok());
            through("FBX", fbx.importFromBytes(*written.value), swell * 0.005f);
        }
        // StarCraft II: an explicit rig, so the staging's retarget goes first, as
        // an export's does, and then its crossing: the game lerps a turn's
        // four numbers raw between two keys, so a clip keyed for Warcraft III
        // is restated until that plays what the editor showed.
        {
            Document out = expanded(ProfileId::Sc2);
            REQUIRE(RetargetSkeleton(out, ProfileId::Sc2).ok);
            CHECK(misses(out) < swell * 0.03f);
            Diagnostics crossing;
            CHECK(ResampleForTarget(out, ProfileId::Sc2, GameOf(baked.defaultProfile), crossing) == 1u);
            const M3Converter m3;
            auto written = m3.toM3(out, ProfileId::Sc2);
            for (const Diagnostic& d : written.diagnostics.all()) {
                if (d.severity == Severity::Error) {
                    WARN(d.message);
                }
            }
            REQUIRE(written.ok());
            // Byte weights, as .mdx and .m2: 1.5 % of the swell.
            through(".m3", m3.fromM3(*written.value), swell * 0.03f);
        }
    }
}

TEST_CASE("G-S5 two baked clips each keep their lumps' motion: one track a bone, keyed by both",
          "[wem][soft][bake][export]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    REQUIRE(f.document.clips.size() == 2u);
    const u32 id = SoftenAll(f, 6, true).id;
    {
        SoftBody& body = *f.model().physics.softBody(id);
        SoftEffect& inflate = AddEffect(body, SoftEffectKind::Inflate);
        inflate.amount = 0.3f;
        inflate.inflate->along = SoftAlong::Surface;
        const SoftSpan span{0.2f, 0.8f, 0.1f, 0.1f};
        SetSoftSpans(f.document, 0, 0, id, inflate.id, {&span, 1});
        SetSoftSpans(f.document, 0, 1, id, inflate.id, {&span, 1});
    }
    const Document unbaked = f.document;
    for (const u32 clip : {0u, 1u}) {
        f.document.clips[clip].physics.emplace();
        REQUIRE(Bake(f.document, clip).ok);
    }
    const Document& baked = f.document;
    Document file = baked;
    Diagnostics diagnostics;
    const SoftExpansion expansion = ExpandSoftToBones(file, ProfileId::Wc3Reforged, diagnostics);
    CHECK(expansion.keys > 0u);
    DropSoftForProfile(file, ProfileId::Wc3Reforged, diagnostics);
    // One channel a node and a kind of key: a file's node has one track of
    // each, and a second channel for the second clip would write over the first.
    std::map<std::pair<u32, u32>, u32> channels;
    for (const AnimChannel& channel : file.models[0].animChannels.channels) {
        if (channel.target.kind == TrackTarget::Kind::Node) {
            ++channels[{channel.target.node, static_cast<u32>(channel.target.channel)}];
        }
    }
    for (const auto& [key, count] : channels) {
        INFO("node " << key.first << " channel " << key.second);
        CHECK(count == 1u);
    }
    // Through a file and back, each clip plays its own swell.
    const MdxConverter mdx;
    auto written = mdx.toMdx(file, ProfileId::Wc3Reforged, 1000);
    REQUIRE(written.ok());
    const Result<Document> back = mdx.fromMdx(*written.value);
    REQUIRE(back.ok());
    const auto rest = baked.models[0].meshes[0].attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    for (const u32 clip : {0u, 1u}) {
        INFO(baked.clips[clip].name);
        const std::vector<Vector3f> drawn = Drawn(baked, clip, 0.5f);
        const std::vector<Vector3f> plain = Drawn(unbaked, clip, 0.5f);
        Points targets;
        f32 swell = 0.0f;
        for (std::size_t v = 0; v < drawn.size(); ++v) {
            targets.emplace_back(rest[v], drawn[v]);
            swell = std::max(swell, (drawn[v] - plain[v]).length());
        }
        REQUIRE(swell > 0.05f);
        u32 matched = 0;
        const f32 missed = Missed(SkinnedPoints(*back.value, baked.clips[clip].name, 0.5f), targets, &matched);
        CHECK(matched >= targets.size());
        CHECK(missed < swell * 0.15f);
    }
}

namespace {

/// A host whose every simulated body swings its node about X as a damped
/// pendulum: deterministic, and engine-free.
struct Swing {
    std::map<u32, std::pair<f32, f32>> state; ///< By body index: angle, speed.

    BakeHooks Hooks() {
        BakeHooks hooks;
        hooks.reset = [this](const Document&, u32, std::span<const u8>) { state.clear(); };
        hooks.step = [this](const Document& document, u32 model, const BakeStep& step, Pose& pose) {
            const PhysicsSet& physics = document.models[model].physics;
            for (u32 b = 0; b < physics.bodies.size() && b < step.switches.size(); ++b) {
                if (!step.switches[b].active) {
                    state.erase(b);
                    continue;
                }
                auto [it, fresh] = state.try_emplace(b, 0.0f, 6.0f);
                auto& [angle, speed] = it->second;
                speed += (-30.0f * std::sin(angle) - 0.8f * speed) * step.dt;
                angle += speed * step.dt;
                const u32 node = physics.bodies[b].node;
                pose.local[node].rotation = Quaternion::from_axis_angle(Vector3f{1, 0, 0}, angle).normalized();
            }
            Animator(document, model).compose(pose);
        };
        return hooks;
    }
};

} // namespace

TEST_CASE("G-S4 the soft bodies are stepped on the keys the bake just wrote, not on the clip's own",
          "[wem][soft][bake]") {
    // The tube's last bone is a body the host swings in a clip that keys
    // nothing of it. A squash read from the motion then has that swing to
    // read only after the node keys are written.
    Figure f = MakeTube(Skinned::Two, 19, 16);
    Model& model = f.model();
    {
        PhysicsRig rig;
        rig.id = model.physics.allocateId();
        rig.start = RigStart::Always;
        PhysicsBody body;
        body.id = model.physics.allocateId();
        body.node = f.bones[2];
        body.motion = BodyMotion::Dynamic;
        model.physics.bodies.push_back(body);
        rig.bodies.push_back(body.id);
        model.physics.rigs.push_back(rig);
    }
    const u32 id = SoftenAll(f, 0, true).id;
    {
        SoftBody& body = *f.model().physics.softBody(id);
        SoftEffect& squash = AddEffect(body, SoftEffectKind::SquashStretch);
        squash.squash->mode = SoftSquashMode::Auto;
        squash.squash->stretch = 0.6f;
        squash.amount = 1.0f;
        const SoftSpan span{0.0f, 1.0f, 0.0f, 0.0f};
        SetSoftSpans(f.document, 0, 1, id, squash.id, {&span, 1});
    }
    ClipPhysics& settings = f.document.clips[1].physics.emplace();
    settings.preheat = 0;
    settings.match = LoopMatch::Off;
    const Document source = f.document;
    Swing swing;
    const BakeRun run = SimulateClip(f.document, 1, swing.Hooks());
    INFO(run.error);
    REQUIRE(run.complete);
    const BakeReport report = BakeClip(f.document, run);
    INFO(report.error);
    REQUIRE(report.ok);
    // The bone was written, and the lumps moved with what it now plays.
    CHECK(report.nodes >= 1u);
    REQUIRE(report.soft.size() == 1u);
    CHECK(report.soft[0].keys > 0u);
    const Document& baked = f.document;
    const f32 accuracy = 0.001f * 6.0f;
    f32 stretched = 0.0f;
    for (const f32 seconds : {0.2f, 0.4f, 0.6f}) {
        // A step on the baked clip's own keys: every lump here is a copy, so
        // the pose holds none of the soft body's motion.
        const std::vector<Vector3f> drawn = Drawn(baked, 1, seconds);
        CHECK(Farthest(drawn, Stepped(baked, id, 1, seconds)) <= accuracy * 2.0f);
        const SoftDeformer deformer(baked.models[0]);
        stretched = std::max(stretched, Farthest(drawn, deformer.positions(0, deformer.palette(SkinAt(baked, 1, seconds), {}))));
    }
    CHECK(stretched > 0.02f);
    // On the clip's own keys the bone does not swing, and the squash has
    // nothing of it to read: what a pass before the node keys would have baked.
    f32 unswung = 0.0f;
    for (const f32 seconds : {0.2f, 0.4f, 0.6f}) {
        const SoftDeformer deformer(source.models[0]);
        unswung = std::max(unswung, Farthest(Stepped(source, id, 1, seconds),
                                             deformer.positions(0, deformer.palette(SkinAt(source, 1, seconds), {}))));
    }
    CHECK(unswung < stretched * 0.5f);
}

TEST_CASE("G-S4 a bone that is a rigid body's and a lump keyed in place: the body's keys are kept to step on",
          "[wem][soft][bake]") {
    // The tube's last bone is a body the host swings, and the soft body's own
    // lump: both bakes key its turn, the soft bodies' over the rigid body's.
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 tip = f.bones[2];
    {
        Model& model = f.model();
        PhysicsRig rig;
        rig.id = model.physics.allocateId();
        rig.start = RigStart::Always;
        PhysicsBody body;
        body.id = model.physics.allocateId();
        body.node = tip;
        body.motion = BodyMotion::Dynamic;
        model.physics.bodies.push_back(body);
        rig.bodies.push_back(body.id);
        model.physics.rigs.push_back(rig);
    }
    const u32 id = SoftenAll(f, 0, false).id;
    {
        SoftBody& body = *f.model().physics.softBody(id);
        REQUIRE(std::any_of(body.lumps.begin(), body.lumps.end(), [&](const SoftLump& lump) {
            return lump.kind == SoftLumpKind::Bone && lump.node == tip;
        }));
        AddEffect(body, SoftEffectKind::Jiggle).everywhere = true;
    }
    ClipPhysics& settings = f.document.clips[1].physics.emplace();
    settings.preheat = 0;
    settings.match = LoopMatch::Off;
    const Document source = f.document;
    const auto bake = [](Document& document) {
        Swing swing;
        const BakeRun run = SimulateClip(document, 1, swing.Hooks());
        INFO(run.error);
        REQUIRE(run.complete);
        const BakeReport report = BakeClip(document, run);
        INFO(report.error);
        REQUIRE(report.ok);
    };
    // The same clip with the Jiggle off: the keys the rigid body's bake leaves.
    Document rigid = source;
    rigid.models[0].physics.softBody(id)->effects.front().enabled = false;
    bake(rigid);
    Document baked = source;
    bake(baked);
    const PhysicsBake& record = *baked.clips[1].physics->baked;
    REQUIRE(record.softSource.size() == 1u);
    REQUIRE(!record.softSource.front().subTracks.empty());
    for (const SubTrack& track : record.softSource.front().subTracks) {
        CHECK(std::find(record.softChannels.begin(), record.softChannels.end(), track.channel) !=
              record.softChannels.end());
        const AnimChannel* channel = baked.models[0].animChannels.find(track.channel);
        REQUIRE(channel != nullptr);
        CHECK(channel->target.node == tip);
    }
    // The pose the bodies are stepped on is the rigid body's swing, with none
    // of the soft body's on it: a step on the baked clip does not swing twice.
    const Animator animator(baked, 0);
    const Vector3f point = baked.models[0].nodes.nodes[tip].pivot + Vector3f{0, 0, 1};
    f32 swung = 0.0f, off = 0.0f;
    for (const f32 seconds : {0.2f, 0.4f, 0.6f, 0.8f}) {
        Pose from;
        SoftSourcePose(baked, animator, 1, seconds, from);
        const Vector3f expected = transform_point(point, SkinAt(rigid, 1, seconds)[tip]);
        off = std::max(off, (transform_point(point, from.skinning[tip]) - expected).length());
        swung = std::max(swung, (transform_point(point, SkinAt(baked, 1, seconds)[tip]) - expected).length());
    }
    INFO("swung " << swung << ", off " << off);
    CHECK(swung > 0.02f);
    CHECK(off < 0.02f * swung);
    // A file holds them, and an Unbake gives the clip its own keys back.
    const std::vector<u8> bytes = Writer().write(baked);
    const std::optional<Document> read = Parser().parse(std::span<const u8>(bytes.data(), bytes.size()));
    REQUIRE(read.has_value());
    REQUIRE(read->clips[1].physics.has_value());
    REQUIRE(read->clips[1].physics->baked.has_value());
    CHECK((ReflectBytes(*read->clips[1].physics->baked) == ReflectBytes(record)));
    UnbakeClip(baked, 1);
    CHECK((ReflectBytes(baked) == ReflectBytes(source)));
}

TEST_CASE("G-S4 a chain used as it is bakes what the body was stepped to: a bone below a keyed one is held",
          "[wem][soft][bake]") {
    // No lump on the surface, where the three bones are used as they are; then
    // six, where a bone that carries one is copied below a bone that is not.
    for (const u32 detail : {0u, 6u}) {
        Figure f = MakeTube(Skinned::Two, 19, 16);
        const u32 id = SoftenAll(f, detail, false).id;
        {
            SoftBody& body = *f.model().physics.softBody(id);
            body.chainsAsTheyAre = true;
            MakeSoftLumps(f.model(), body, 4, false);
            // Fat, which the bound below was measured with.
            SoftEffect& jiggle = AddEffect(body, SoftEffectKind::Jiggle);
            jiggle.jiggle = SoftJigglePresetOf(SoftJigglePreset::Fat);
            jiggle.everywhere = true;
        }
        f.document.clips[0].physics.emplace();
        const Document source = f.document;
        REQUIRE(Bake(f.document, 0).ok);
        const Document baked = f.document;
        const SoftBody& stepped = *source.models[0].physics.softBody(id);
        // What the case is about: a lump whose bone hangs below one keyed in place.
        const NodeTree& tree = source.models[0].nodes;
        u32 below = 0;
        for (const SoftLump& lump : stepped.lumps) {
            for (u32 at = lump.family() < tree.size() ? tree.nodes[lump.family()].parent : kInvalidNode;
                 at < tree.size(); at = tree.nodes[at].parent) {
                if (std::any_of(stepped.lumps.begin(), stepped.lumps.end(), [&](const SoftLump& other) {
                        return other.kind == SoftLumpKind::Bone && other.node == at;
                    })) {
                    ++below;
                    break;
                }
            }
        }
        INFO("detail " << detail);
        REQUIRE(below > 0u);

        // What the tool shows: the body run over the clip, on the clip's keys.
        f32 dt = 0.0f;
        const u32 steps = BakeStepsOf(source, 0, dt);
        const auto poseAt = [&](f32 at, std::vector<Matrix44f>& skin) { skin = SkinAt(source, 0, at); };
        const SoftRun run = RunSoft(source, 0, id, 0, steps, dt, poseAt);
        REQUIRE(run.lumps == stepped.lumps.size());
        const SoftDeformer deformer(source.models[0]);
        f32 swung = 0.0f, apart = 0.0f;
        for (const u32 i : {steps / 5, steps * 2 / 5, steps / 2, steps * 7 / 10, steps * 9 / 10}) {
            const f32 seconds = static_cast<f32>(i) * dt;
            const std::vector<Matrix44f> from = SkinAt(source, 0, seconds);
            const std::vector<Vector3f> shown =
                deformer.positions(0, deformer.palette(from, std::span<const Transform>(run.at(i), run.lumps)));
            swung = std::max(swung, Farthest(shown, deformer.positions(0, deformer.palette(from, {}))));
            apart = std::max(apart, Farthest(Drawn(baked, 0, seconds), shown));
        }
        INFO("swung " << swung << ", apart " << apart);
        CHECK(swung > 0.05f);
        CHECK(apart <= 0.001f * 6.0f * 3.0f);
        // A bone held below a chain bone, that is no bone used as it is, is
        // said to be keyed in place, and is.
        const std::vector<u32> held = SoftBonesHeldBelow(source.models[0], id);
        CHECK(held.empty() == (detail == 0u));
        const PhysicsBake& record = *baked.clips[0].physics->baked;
        for (const u32 bone : held) {
            CHECK(std::any_of(record.softChannels.begin(), record.softChannels.end(), [&](u32 channel) {
                const AnimChannel* entry = baked.models[0].animChannels.find(channel);
                return entry != nullptr && entry->target.kind == TrackTarget::Kind::Node && entry->target.node == bone;
            }));
        }
    }
}
