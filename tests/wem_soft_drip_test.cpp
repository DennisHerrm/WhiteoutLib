// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

// Drip (EDIT_MODE_SOFTBODY_PLAN.md S7): where a liquid would run to, a strand's
// neck and its fall, and the bones and the tube the effect adds.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <whiteout/models/wem/anim/animator.h>
#include <whiteout/models/wem/physics/bake.h>
#include <whiteout/models/wem/physics/soft_bake.h>
#include <whiteout/models/wem/physics/soft_drip.h>
#include <whiteout/models/wem/physics/soft_run.h>
#include <whiteout/models/wem/physics/soft_step.h>
#include <whiteout/models/wem/skinning/deform.h>
#include <whiteout/models/wem/validate.h>

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

std::vector<Matrix44f> SkinAt(const Document& document, u32 clip, f32 seconds) {
    const Animator animator(document, 0);
    Mix mix;
    mix.plays.push_back(Play{clip, seconds, 1.0f, false});
    Pose pose;
    animator.evaluate(mix, pose);
    return pose.skinning;
}

/// The tube lifted off the floor and laid on its side, so it has an underside:
/// along +X, its axis two up.
Figure MakeLog() {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    Model& model = f.model();
    for (Node& node : model.nodes.nodes) {
        node.pivot = Vector3f{node.pivot.z, node.pivot.y, 2.0f};
        node.local.translation = node.parent == kInvalidNode ? node.pivot : Vector3f{2.0f, 0, 0};
    }
    const std::span<Vector3f> positions =
        model.meshes[0].attributes.getOrCreate<Vector3f>(geom::names::kPosition, geom::Domain::Vertex, geom::AttrType::F32x3);
    for (Vector3f& p : positions) {
        p = Vector3f{p.z, p.y, p.x + 2.0f};
    }
    // The clips' keys were for the standing tube: a log that lies still.
    for (Clip& clip : f.document.clips) {
        clip.containers.front().subTracks.clear();
    }
    SoftenAll(f, 0, true);
    return f;
}

/// A first UV set on the log: each corner's is its vertex's own. A corner's
/// is on the halfedge that leaves it.
void GiveUv(Figure& f) {
    Mesh& mesh = f.model().meshes[0];
    std::vector<Vector2f> made;
    {
        const Mesh& read = mesh;
        const geom::Topology& topology = read.topology();
        made.assign(topology.halfedgeCount(), Vector2f{0, 0});
        for (u32 face = 0; face < read.faceCount(); ++face) {
            const geom::HalfedgeId first = topology.halfedge(geom::FaceId(face));
            geom::HalfedgeId h = first;
            do {
                made[h.index()] = Vector2f{static_cast<f32>(topology.from(h).value()) / 1024.0f, 0.25f};
                h = topology.next(h);
            } while (h.index() != first.index());
        }
    }
    const std::span<Vector2f> uv =
        mesh.attributes.getOrCreate<Vector2f>(geom::names::uv(0), geom::Domain::Halfedge, geom::AttrType::F32x2);
    REQUIRE(uv.size() == made.size());
    std::copy(made.begin(), made.end(), uv.begin());
}

SoftEffect& AddDrip(Figure& f) {
    SoftBody& body = f.model().physics.soft[0];
    SoftEffect effect = DefaultSoftEffect(SoftEffectKind::Drip);
    effect.id = body.nextEffect++;
    effect.name = "Drip";
    // Plumb, and from a surface left as it is, for what these cases measure;
    // *Sway* and *Pull the surface* have cases of their own.
    effect.drip->sway = 0.0f;
    effect.drip->pull = 0.0f;
    body.effects.push_back(effect);
    return body.effects.back();
}

/// The strands built on @p f's body, and its lumps made again for them.
SoftStrandsBuilt Build(Figure& f) {
    SoftBody& body = f.model().physics.soft[0];
    SoftEffect& effect = body.effects.back();
    const std::vector<SoftDripSite> sites =
        SoftDripSites(f.model(), SoftRig(f.model(), body.id), *effect.drip);
    const SoftStrandsBuilt built = BuildSoftStrands(f.model(), body, effect, sites);
    MakeSoftLumps(f.model(), f.model().physics.soft[0], 4, false);
    return built;
}

/// The strand mesh's points as the body is stepped at @p seconds of @p clip.
std::vector<Vector3f> StrandsAt(const Document& document, u32 clip, f32 seconds) {
    const Model& model = document.models[0];
    const SoftBody& body = model.physics.soft[0];
    const SoftStepper stepper(document, 0, body.id);
    const SoftPoseAt poseAt = [&](f32 at, std::vector<Matrix44f>& skin) { skin = SkinAt(document, clip, at); };
    SoftStepResult out;
    const std::vector<Matrix44f> skin = SkinAt(document, clip, seconds);
    stepper.step(skin, SoftClock{clip, seconds}, poseAt, out);
    const SoftDeformer deformer(model);
    return deformer.positions(body.drips, deformer.palette(skin, out.departures));
}

} // namespace

TEST_CASE("G-S7 a strand's neck thins by its material's law, and never widens", "[wem][soft][drip]") {
    for (const SoftNeckLaw law : {SoftNeckLaw::Linear, SoftNeckLaw::Snap, SoftNeckLaw::Thread, SoftNeckLaw::Sets}) {
        CHECK(SoftNeckWidth(law, 0.4f, 0.0f) == Approx(1.0f));
        f32 before = 1.0f;
        for (f32 u = 0.05f; u <= 1.0f; u += 0.05f) {
            const f32 now = SoftNeckWidth(law, 0.4f, u);
            CHECK(now <= before + 1e-6f);
            CHECK(now >= 0.4f - 1e-5f);
            before = now;
        }
    }
    // In a straight line; slowly and then at once; most of it early; never half way.
    CHECK(SoftNeckWidth(SoftNeckLaw::Linear, 0.4f, 0.5f) == Approx(0.7f));
    CHECK(SoftNeckWidth(SoftNeckLaw::Linear, 0.4f, 1.0f) == Approx(0.4f));
    CHECK(SoftNeckWidth(SoftNeckLaw::Snap, 0.4f, 0.5f) > 0.95f);
    CHECK(SoftNeckWidth(SoftNeckLaw::Snap, 0.4f, 1.0f) == Approx(0.4f));
    CHECK(SoftNeckWidth(SoftNeckLaw::Thread, 0.4f, 0.5f) < 0.5f);
    CHECK(SoftNeckWidth(SoftNeckLaw::Thread, 0.4f, 1.0f) == Approx(0.4f));
    CHECK(SoftNeckWidth(SoftNeckLaw::Sets, 0.4f, 1.0f) == Approx(0.7f));

    // A preset is its law, speed, thickness and stagger; the rest is kept.
    SoftDrip mine;
    mine.sites = 9;
    mine.toGround = false;
    const SoftDrip honey = SoftDripPresetOf(SoftDripPreset::Honey, mine);
    CHECK(honey.preset == SoftDripPreset::Honey);
    CHECK(honey.neck == SoftNeckLaw::Linear);
    CHECK(honey.speed < SoftDripPresetOf(SoftDripPreset::Blood).speed);
    CHECK(honey.sites == 9u);
    CHECK(!honey.toGround);
    CHECK(SoftDripPresetOf(SoftDripPreset::Wax).neck == SoftNeckLaw::Sets);
    // Water: the thinnest of them, and it lets drops go.
    const SoftDrip water = SoftDripPresetOf(SoftDripPreset::Water);
    CHECK(water.pinches());
    CHECK(water.thickness < SoftDripPresetOf(SoftDripPreset::Blood).thickness);
    CHECK(std::string(ToString(SoftDripPreset::Water)) == "water");
}

TEST_CASE("G-S7 a strand's tip takes what its neck gives up, in step", "[wem][soft][drip]") {
    // What a neck's thinning has taken out of a plain tube, of the tube: the
    // strand's two middle rings ride the neck by two thirds, and the three
    // stretches between its four rings are frusta a third of it long.
    const auto givenUp = [](f32 neck) {
        const f32 r = (1.0f + 2.0f * neck) / 3.0f;
        return 1.0f - ((1.0f + r + r * r) + 3.0f * r * r + (r * r + r + 1.0f)) / 9.0f;
    };
    SoftDrip settings;
    settings.neckWidth = 0.4f;
    settings.tip = 1.6f;
    for (const SoftNeckLaw law : {SoftNeckLaw::Linear, SoftNeckLaw::Snap, SoftNeckLaw::Thread, SoftNeckLaw::Sets}) {
        settings.neck = law;
        const f32 narrowest = SoftNeckWidth(law, 0.4f, 1.0f);
        const f32 whole = givenUp(narrowest);
        f32 before = 0.0f;
        for (u32 k = 0; k <= 20; ++k) {
            const f32 u = static_cast<f32>(k) / 20.0f;
            const f32 neck = SoftNeckWidth(law, 0.4f, u);
            const f32 swell = SoftTipSwell(settings, u, neck);
            // One ratio from end to end: the tip has of its swell what the neck
            // has given up of all it will; and it never draws back.
            CHECK(swell == Approx(u * givenUp(neck) / whole).margin(1e-5));
            CHECK(swell >= before - 1e-6f);
            before = swell;
        }
        CHECK(SoftTipSwell(settings, 0.0f, 1.0f) == 0.0f);
        CHECK(SoftTipSwell(settings, 1.0f, narrowest) == Approx(1.0f).margin(1e-5));
    }
    // A neck that snaps has hardly thinned half way down, and its tip has
    // hardly swollen; a thread's has thinned early, and its tip with it.
    settings.neck = SoftNeckLaw::Snap;
    CHECK(SoftTipSwell(settings, 0.5f, SoftNeckWidth(SoftNeckLaw::Snap, 0.4f, 0.5f)) < 0.05f);
    settings.neck = SoftNeckLaw::Thread;
    CHECK(SoftTipSwell(settings, 0.5f, SoftNeckWidth(SoftNeckLaw::Thread, 0.4f, 0.5f)) > 0.4f);

    // On the strand: its tip's bone against its base's, half way down and at its reach.
    const Vector3f down{0, 0, -1}, sunk{0, 0, -0.01f};
    const f32 width = 0.05f, reach = 1.6f, height = 2.0f;
    settings.speed = 0.4f; // 0.8 a second on a body two high.
    const auto tipOf = [&](f32 since, const SoftPinch* pinch = nullptr) {
        const SoftStrandPose pose =
            SoftStrandAt(settings, width, reach, height, since, false, down, sunk, Vector2f{0, 0}, pinch);
        return pose.tip.scale.x / pose.base.scale.x;
    };
    settings.neck = SoftNeckLaw::Snap;
    CHECK(tipOf(1.0f) == Approx(1.0f + 0.6f * SoftTipSwell(settings, 0.5f, SoftNeckWidth(SoftNeckLaw::Snap, 0.4f, 0.5f)))
                             .margin(1e-4));
    CHECK(tipOf(1.0f) < 1.03f);
    CHECK(tipOf(2.0f) == Approx(1.6f).margin(1e-4));
    // Where it pinches off, the tip is whole as it lets go and gone once it has.
    SoftPinch hangs;
    hangs.length = 1.0f;
    hangs.thinned = 1.0f;
    hangs.swollen = 1.0f;
    CHECK(tipOf(0.0f, &hangs) == Approx(1.6f).margin(1e-4));
    hangs.away = true;
    CHECK(tipOf(0.0f, &hangs) == Approx(1.0f).margin(1e-4));

    // A neck that does not thin gives nothing up: its tip swells with its length.
    settings.neckWidth = 1.0f;
    CHECK(SoftTipSwell(settings, 0.5f, 1.0f) == Approx(0.5f));
    CHECK(SoftTipSwell(settings, 1.0f, 1.0f) == Approx(1.0f));
}

TEST_CASE("G-S7 sites land where a liquid would run to: the underside, kept apart, the largest first",
          "[wem][soft][drip]") {
    Figure f = MakeLog();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    SoftDrip settings;
    settings.sites = 4;
    settings.spacing = 0.15f;
    const std::vector<SoftDripSite> sites = SoftDripSites(f.model(), rig, settings);
    REQUIRE(sites.size() == 4u);
    const f32 size = rig.points().size;
    f32 share = 0.0f;
    for (std::size_t i = 0; i < sites.size(); ++i) {
        const SoftDripSite& site = sites[i];
        INFO("site " << i << " at " << site.at.x << ", " << site.at.y << ", " << site.at.z);
        // On the log's underside, which faces down: under its axis.
        CHECK(site.at.z < 2.0f - 0.4f);
        CHECK(site.normal.z < -0.4f);
        CHECK(site.bone != kInvalidNode);
        CHECK(site.point < rig.pointCount());
        CHECK(site.width > 0.0f);
        // The largest catchment first, and thicker for it.
        if (i > 0) {
            CHECK(site.catchment <= sites[i - 1].catchment + 1e-6f);
            CHECK(site.width <= sites[i - 1].width + 1e-6f);
        }
        for (std::size_t j = 0; j < i; ++j) {
            CHECK((site.at - sites[j].at).length() >= settings.spacing * size - 1e-4f);
        }
        share += site.catchment;
    }
    CHECK(share > 0.0f);
    CHECK(share <= 1.0f + 1e-4f);
    // Fewer asked for, fewer given; the same body twice gives the same.
    settings.sites = 2;
    const std::vector<SoftDripSite> two = SoftDripSites(f.model(), rig, settings);
    REQUIRE(two.size() == 2u);
    CHECK(two[0].point == sites[0].point);
    CHECK(SoftDripSites(f.model(), rig, settings)[1].point == two[1].point);

    // By hand: where they are put, whatever runs there.
    SoftDrip hand;
    hand.sitesByHand = true;
    hand.siteMesh = {0};
    hand.siteFace = {0};
    hand.siteWeights = {Vector3f{0.34f, 0.33f, 0.33f}};
    hand.siteThickness = {0.05f};
    const std::vector<SoftDripSite> placed = SoftDripSites(f.model(), rig, hand);
    REQUIRE(placed.size() == 1u);
    CHECK(placed[0].width == Approx(0.05f * size));
    CHECK(placed[0].catchment == 0.0f);

    // A strand needs room under it. The log leant up along its length: its
    // low end takes the most and has no room for a strand of 0.6 of the size,
    // so the sites are the higher of its underside.
    Figure leant = MakeLog();
    for (Vector3f& p : leant.model().meshes[0].attributes.getOrCreate<Vector3f>(
             geom::names::kPosition, geom::Domain::Vertex, geom::AttrType::F32x3)) {
        p.z += 0.25f * p.x;
    }
    const SoftRig slope(leant.model(), leant.model().physics.soft[0].id);
    SoftDrip hung;
    hung.sites = 4;
    hung.spacing = 0.15f;
    const std::vector<SoftDripSite> everywhere = SoftDripSites(leant.model(), slope, hung);
    hung.toGround = false;
    hung.length = 0.6f;
    const f32 room = 0.5f * hung.length * slope.points().size;
    REQUIRE(std::any_of(everywhere.begin(), everywhere.end(),
                        [&](const SoftDripSite& site) { return site.at.z < room; }));
    const std::vector<SoftDripSite> roomy = SoftDripSites(leant.model(), slope, hung);
    REQUIRE(!roomy.empty());
    for (const SoftDripSite& site : roomy) {
        CHECK(site.at.z >= room);
    }
    // With no place that has room, they are where a liquid runs to all the same.
    hung.length = 4.0f;
    const std::vector<SoftDripSite> none = SoftDripSites(leant.model(), slope, hung);
    REQUIRE(none.size() == everywhere.size());
    for (std::size_t i = 0; i < none.size(); ++i) {
        CHECK(none[i].point == everywhere[i].point);
    }
}

TEST_CASE("G-S7 a strand is three bones and a tube the effect adds, and the body's own points",
          "[wem][soft][drip]") {
    Figure f = MakeLog();
    const u32 nodes = f.model().nodes.size();
    SoftEffect& drip = AddDrip(f);
    drip.drip->sites = 3;
    f.model().physics.soft[0].detail = 12;
    GiveUv(f);
    const SoftStrandsBuilt built = Build(f);
    const Model& model = f.model();
    const SoftBody& body = model.physics.soft[0];
    const SoftDrip& made = *body.effects.back().drip;
    CHECK(built.strands == 3u);
    CHECK(built.bones == 9u);
    REQUIRE(built.mesh == 1u);
    CHECK(body.drips == 1u);
    CHECK(std::find(body.meshes.begin(), body.meshes.end(), 1u) != body.meshes.end());
    CHECK(model.nodes.size() == nodes + 9u);
    REQUIRE(made.strandNodes.size() == 9u);
    REQUIRE(made.strandAt.size() == 3u);
    REQUIRE(made.strandWidth.size() == 3u);
    const Mesh& mesh = model.meshes[1];
    // Six sides, four rings and a tip, each strand; one section for them all.
    CHECK(mesh.vertexCount() == 3u * (6u * 4u + 1u));
    CHECK(mesh.sections.size() == 1u);
    CHECK(mesh.sections[0].name == "Jelly Drips");
    for (std::size_t s = 0; s < 3; ++s) {
        // Side by side under the bone that carries the site.
        const Node& base = model.nodes.nodes[made.strandNodes[3 * s]];
        const Node& neck = model.nodes.nodes[made.strandNodes[3 * s + 1]];
        const Node& tip = model.nodes.nodes[made.strandNodes[3 * s + 2]];
        CHECK(base.parent < nodes);
        CHECK(neck.parent == base.parent);
        CHECK(tip.parent == base.parent);
        // Built a hundredth of itself, under the surface: within its site's own width.
        CHECK((base.pivot - made.strandAt[s]).length() < made.strandWidth[s]);
        CHECK((tip.pivot - base.pivot).length() == Approx(kSoftStrandRest * made.strandWidth[s] * kSoftStrandShrink));
    }
    // Every bone of a strand has a lump: the body's lanes carry its tube.
    const SoftStepper stepper(f.document, 0, body.id);
    const SoftRig& rig = stepper.rig();
    u32 carried = 0;
    for (const u32 node : made.strandNodes) {
        for (const SoftLump& lump : body.lumps) {
            carried += lump.kind != SoftLumpKind::Surface && lump.node == node ? 1 : 0;
        }
    }
    CHECK(carried == 9u);
    // Each is an end of the skeleton and the body's own making: used as it is
    // even on a body that copies every bone, so a file has three bones a strand.
    REQUIRE(body.copyAll);
    for (const SoftLump& lump : body.lumps) {
        if (std::find(made.strandNodes.begin(), made.strandNodes.end(), lump.node) != made.strandNodes.end()) {
            CHECK(lump.kind == SoftLumpKind::Bone);
        }
    }
    CHECK(rig.points().pointOfVertex[1].size() == mesh.vertexCount());
    // The body's surface lumps are on its own bones: one on a strand's would
    // hold part of the tube back from where the Drip puts it.
    u32 surface = 0, onStrand = 0;
    for (const SoftLump& lump : body.lumps) {
        if (lump.kind != SoftLumpKind::Surface) {
            continue;
        }
        ++surface;
        onStrand += std::find(made.strandNodes.begin(), made.strandNodes.end(), lump.carrier) != made.strandNodes.end();
    }
    CHECK(surface == 12u);
    CHECK(onStrand == 0u);
    // Wholly soft and marked painted, so a remake's blend spares it; and every
    // corner of a strand holds one UV, its site's.
    const std::span<const f32> softness = mesh.attributes.get<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex);
    const std::span<const u8> painted = mesh.attributes.get<u8>(geom::names::kSoftPainted, geom::Domain::Vertex);
    REQUIRE(softness.size() == mesh.vertexCount());
    REQUIRE(painted.size() == mesh.vertexCount());
    CHECK(std::all_of(softness.begin(), softness.end(), [](f32 s) { return s == 1.0f; }));
    CHECK(std::all_of(painted.begin(), painted.end(), [](u8 p) { return p != 0; }));
    const std::span<const Vector2f> uv = mesh.attributes.get<Vector2f>(geom::names::uv(0), geom::Domain::Halfedge);
    const std::vector<SoftDripSite> sited = [&] {
        Figure bare = MakeLog();
        bare.model().physics.soft[0].detail = 12;
        GiveUv(bare);
        const SoftRig over(bare.model(), bare.model().physics.soft[0].id);
        std::vector<SoftDripSite> found = SoftDripSites(bare.model(), over, made);
        // A site's UV is its own vertex's, which the log's UV set names.
        for (const SoftDripSite& site : found) {
            REQUIRE(site.point < over.pointCount());
            CHECK(site.uv.x == static_cast<f32>(over.points().points[site.point].vertex) / 1024.0f);
        }
        return found;
    }();
    REQUIRE(sited.size() == 3u);
    const geom::Topology& topology = mesh.topology();
    const u32 faces = mesh.faceCount() / 3;
    REQUIRE(faces * 3 == mesh.faceCount());
    REQUIRE(faces > 0u);
    CHECK(sited[0].uv.x != sited[1].uv.x);
    u32 wrong = 0;
    for (u32 face = 0; face < mesh.faceCount(); ++face) {
        const SoftDripSite& site = sited[face / faces];
        const geom::HalfedgeId first = topology.halfedge(geom::FaceId(face));
        geom::HalfedgeId h = first;
        do {
            wrong += uv[h.index()].x != site.uv.x || uv[h.index()].y != 0.25f ? 1 : 0;
            h = topology.next(h);
        } while (h.index() != first.index());
    }
    CHECK(wrong == 0u);
    // Nothing structural is wrong that was not before: the figure names no
    // material, and the strands take its tube's.
    Figure plain = MakeLog();
    CHECK(Validate(f.document, ValidateLevel::Structural).size() <=
          Validate(plain.document, ValidateLevel::Structural).size() + 1u);
    CHECK(mesh.sections[0].materialSlot == model.meshes[0].sections[0].materialSlot);
    // A second build makes nothing: the effect has its strands.
    const std::vector<SoftDripSite> again = SoftDripSites(f.model(), SoftRig(f.model(), body.id), made);
    CHECK(BuildSoftStrands(f.model(), f.model().physics.soft[0], f.model().physics.soft[0].effects.back(), again).strands == 0u);
}

TEST_CASE("G-S7 a strand draws nothing until it sets out, falls at its speed to the ground, and holds",
          "[wem][soft][drip]") {
    Figure f = MakeLog();
    SoftEffect& drip = AddDrip(f);
    drip.drip->sites = 2;
    drip.drip->stagger = 0.0f;
    drip.drip->neck = SoftNeckLaw::Linear;
    drip.drip->speed = 0.05f; // A twentieth of the body's size a second.
    const u32 effect = drip.id;
    Build(f);
    const u32 id = f.model().physics.soft[0].id;
    const u32 clip = 0;
    f.document.clips[clip].looping = false;
    f.document.clips[clip].duration = 5.0f;
    const SoftSpan span{0.25f, 5.0f, 0.0f, 0.0f};
    SetSoftSpans(f.document, 0, clip, id, effect, std::span<const SoftSpan>(&span, 1));
    const SoftDrip& made = *f.model().physics.soft[0].effects.back().drip;
    const f32 size = SoftRig(f.model(), id).points().size;
    const f32 width = made.strandWidth[0];
    const Vector3f site = made.strandAt[0];
    const u32 perStrand = 6u * 4u + 1u;

    // How far below its site a strand's lowest point is, and how wide its neck.
    const auto lowest = [&](f32 seconds) {
        const std::vector<Vector3f> at = StrandsAt(f.document, clip, seconds);
        f32 low = 0.0f;
        for (u32 v = 0; v < perStrand; ++v) {
            low = std::max(low, site.z - at[v].z);
        }
        return low;
    };
    // Before its span: as built, a hundredth of itself at its site.
    {
        const std::vector<Vector3f> at = StrandsAt(f.document, clip, 0.1f);
        for (u32 v = 0; v < perStrand; ++v) {
            CHECK((at[v] - site).length() < 3.0f * kSoftStrandRest * width * kSoftStrandShrink);
        }
    }
    // After it sets out, its tip is at speed times time under its site, and
    // its rounded end its swollen width past that.
    const f32 speed = 0.05f * size;
    const f32 reach = site.z - width * made.tip;
    REQUIRE(reach > 2.0f * speed);
    const auto swollen = [&](f32 fallen) {
        return 1.0f + (made.tip - 1.0f) * SoftTipSwell(made, fallen, SoftNeckWidth(made.neck, made.neckWidth, fallen));
    };
    for (const f32 since : {0.5f, 1.0f, 2.0f}) {
        const f32 length = speed * since;
        CHECK(lowest(0.25f + since) == Approx(length + width * swollen(length / reach)).epsilon(0.02));
    }
    // It reaches the ground and holds there.
    REQUIRE(speed * 4.5f > reach);
    CHECK(lowest(4.9f) == Approx(site.z).epsilon(0.01));
    CHECK(lowest(5.0f) == Approx(lowest(4.9f)).epsilon(1e-3));
    // It hangs straight down from its site, and its neck is thinner than its base.
    {
        const std::vector<Vector3f> at = StrandsAt(f.document, clip, 4.9f);
        f32 off = 0.0f, baseWide = 0.0f, neckWide = 0.0f;
        for (u32 v = 0; v < perStrand; ++v) {
            const f32 out = std::sqrt((at[v].x - site.x) * (at[v].x - site.x) + (at[v].y - site.y) * (at[v].y - site.y));
            off = std::max(off, out);
            if (v < 6) {
                baseWide = std::max(baseWide, out);
            } else if (v >= 6 && v < 18) {
                neckWide = std::max(neckWide, out);
            }
        }
        CHECK(off < 1.5f * width * made.tip);
        CHECK(baseWide == Approx(width).epsilon(0.02));
        CHECK(neckWide < 0.95f * width);
    }
    // Staggered, the second sets out after the first.
    f.model().physics.soft[0].effects.back().drip->stagger = 1.0f;
    {
        const std::vector<Vector3f> at = StrandsAt(f.document, clip, 1.25f);
        f32 first = 0.0f, second = 0.0f;
        for (u32 v = 0; v < perStrand; ++v) {
            first = std::max(first, made.strandAt[0].z - at[v].z);
            second = std::max(second, made.strandAt[1].z - at[perStrand + v].z);
        }
        CHECK(first > 10.0f * second);
    }
    // In a clip that does not place it, and switched off, it draws nothing.
    {
        const std::vector<Vector3f> at = StrandsAt(f.document, 1, 0.5f);
        for (u32 v = 0; v < perStrand; ++v) {
            CHECK((at[v] - site).length() < 3.0f * kSoftStrandRest * width * kSoftStrandShrink);
        }
    }
    f.model().physics.soft[0].effects.back().enabled = false;
    CHECK(lowest(4.9f) < 3.0f * kSoftStrandRest * width * kSoftStrandShrink);
}

TEST_CASE("G-S7 in a loop a strand hangs at its length from the first frame to the last, and bakes into a file",
          "[wem][soft][drip]") {
    Figure f = MakeLog();
    SoftEffect& drip = AddDrip(f);
    drip.drip->sites = 2;
    drip.drip->toGround = false;
    drip.drip->length = 0.2f;
    drip.everywhere = true;
    Build(f);
    const u32 id = f.model().physics.soft[0].id;
    const u32 clip = 0;
    REQUIRE(f.document.clips[clip].looping);
    const f32 size = SoftRig(f.model(), id).points().size;
    const SoftDrip& made = *f.model().physics.soft[0].effects.back().drip;
    const f32 width = made.strandWidth[0];
    const std::vector<Vector3f> first = StrandsAt(f.document, clip, 0.0f);
    const std::vector<Vector3f> last = StrandsAt(f.document, clip, f.document.clips[clip].duration);
    f32 apart = 0.0f, low = 0.0f;
    for (std::size_t v = 0; v < first.size(); ++v) {
        apart = std::max(apart, (first[v] - last[v]).length());
    }
    for (u32 v = 0; v < 25; ++v) {
        low = std::max(low, made.strandAt[0].z - first[v].z);
    }
    CHECK(apart < 1e-5f);
    // Its length, and its swollen end past that.
    CHECK(low == Approx(0.2f * size + width * made.tip).epsilon(0.02));

    // Baked, its bones' keys are the clip's; expanded, a file's mesh hangs where the editor's did.
    f.document.clips[clip].physics.emplace();
    BakeHooks hooks;
    hooks.reset = [](const Document&, u32, std::span<const u8>) {};
    hooks.step = [](const Document&, u32, const BakeStep&, Pose&) {};
    const BakeRun recorded = SimulateClip(f.document, clip, hooks);
    INFO(recorded.error);
    REQUIRE(recorded.complete);
    const BakeReport report = BakeClip(f.document, recorded);
    REQUIRE(report.ok);
    REQUIRE(report.soft.size() == 1u);
    CHECK(report.soft[0].keys > 0u);
    Document file = f.document;
    Diagnostics diagnostics;
    ExpandSoftToBones(file, ProfileId::Generic, diagnostics);
    const std::vector<Vector3f> played =
        skinning::DeformMesh(file.models[0].meshes[f.model().physics.soft[0].drips], SkinAt(file, clip, 0.5f));
    REQUIRE(played.size() == first.size());
    f32 off = 0.0f;
    for (std::size_t v = 0; v < played.size(); ++v) {
        off = std::max(off, (played[v] - first[v]).length());
    }
    CHECK(off < 0.02f * width + 2e-3f);
}

TEST_CASE("G-S7 a strand that hangs to the ground reaches it under a bone that scales the body", "[wem][soft][drip]") {
    Figure f = MakeLog();
    SoftEffect& drip = AddDrip(f);
    drip.drip->sites = 1;
    drip.everywhere = true;
    Build(f);
    const Model& model = f.model();
    const SoftBody& body = model.physics.soft[0];
    const SoftDrip& made = *body.effects.back().drip;
    REQUIRE(made.toGround);
    REQUIRE(f.document.clips[0].looping);
    const SoftStepper stepper(f.document, 0, body.id);
    const SoftDeformer deformer(model);
    for (const f32 scale : {1.0f, 0.5f, 2.0f}) {
        // The whole pose scaled about the ground's origin: the body half as
        // large, or twice, standing where it stood.
        const auto scaled = [&](f32 at, std::vector<Matrix44f>& skin) {
            skin = SkinAt(f.document, 0, at);
            for (Matrix44f& m : skin) {
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        m.data[r][c] *= scale;
                    }
                }
            }
        };
        std::vector<Matrix44f> skin;
        scaled(0.5f, skin);
        SoftStepResult out;
        stepper.step(skin, SoftClock{0, 0.5f}, scaled, out);
        const std::vector<Vector3f> at = deformer.positions(body.drips, deformer.palette(skin, out.departures));
        f32 low = 1e30f;
        for (const Vector3f& p : at) {
            low = std::min(low, p.z);
        }
        // Its site stands at the scaled height, and its end on the ground: the
        // strand is as long as that height, not that height scaled again.
        INFO("scale " << scale);
        const f32 height = made.strandAt[0].z * scale;
        CHECK(std::abs(low) < 0.03f * height);
    }
}

TEST_CASE("G-S7 Sway swings a strand as a chain of two links, each from its own phase, in whole swings a loop",
          "[wem][soft][drip]") {
    // The pose: a strand swung is no longer or shorter, its upper half half as far off the way down.
    const SoftDrip settings;
    const f32 width = 0.05f, reach = 1.0f;
    const Vector3f down{0, 0, -1}, sunk{0, 0, 0}, built{0, 0, -1};
    const f32 rest = kSoftStrandRest * width;
    const auto neckAt = [&](const SoftStrandPose& pose) {
        return pose.neck.translation + built * (0.5f * rest * kSoftStrandShrink);
    };
    const auto tipAt = [&](const SoftStrandPose& pose) { return pose.tip.translation + built * (rest * kSoftStrandShrink); };
    const SoftStrandPose still = SoftStrandAt(settings, width, reach, 2.0f, 0.0f, true, down, sunk);
    CHECK(tipAt(still).z == Approx(-1.0f));
    CHECK(std::hypot(tipAt(still).x, tipAt(still).y) < 1e-6f);
    const SoftStrandPose swung = SoftStrandAt(settings, width, reach, 2.0f, 0.0f, true, down, sunk, Vector2f{0.3f, 0.0f});
    CHECK(neckAt(swung).length() == Approx(0.5f).epsilon(1e-4));
    const Vector3f lower = tipAt(swung) - neckAt(swung);
    CHECK(lower.length() == Approx(0.5f).epsilon(1e-4));
    CHECK(std::acos(-neckAt(swung).z / 0.5f) == Approx(0.15f).margin(1e-3));
    CHECK(std::acos(-lower.z / 0.5f) == Approx(0.3f).margin(1e-3));
    CHECK((swung.base.translation - still.base.translation).length() < 1e-7f);
    // Its width is what it was: the swing turns it and does not thin it.
    CHECK(swung.neck.scale.x == still.neck.scale.x);
    CHECK(swung.tip.scale.z == still.tip.scale.z);

    // The swing: none at 0, twenty degrees at 1, each strand from a phase of its own.
    CHECK(SoftStrandSwing(0.0f, 1, 0, 1.0f, 0.3f).x == 0.0f);
    CHECK(SoftStrandSwing(0.0f, 1, 0, 1.0f, 0.3f).y == 0.0f);
    f32 most = 0.0f, across = 0.0f;
    for (u32 i = 0; i <= 200; ++i) {
        const Vector2f at = SoftStrandSwing(1.0f, 1, 0, 1.0f, static_cast<f32>(i) / 200.0f);
        most = std::max(most, std::abs(at.x));
        across = std::max(across, std::abs(at.y));
    }
    CHECK(most == Approx(0.35f).margin(1e-3));
    CHECK(across == Approx(0.21f).margin(1e-3));
    const Vector2f one = SoftStrandSwing(1.0f, 1, 0, 1.0f, 0.0f), two = SoftStrandSwing(1.0f, 1, 1, 1.0f, 0.0f);
    CHECK(std::hypot(one.x - two.x, one.y - two.y) > 0.05f);
    // A whole swing later it is where it was.
    const Vector2f later = SoftStrandSwing(1.0f, 1, 0, 2.0f, 0.5f);
    CHECK(later.x == Approx(one.x).margin(1e-5));
    CHECK(later.y == Approx(one.y).margin(1e-5));
    // A longer strand swings more slowly, as the root of its length.
    CHECK(SoftSwayRate(2.0f, 2.0f) == Approx(0.75f));
    CHECK(SoftSwayRate(0.5f, 2.0f) == Approx(1.5f));

    // Through a step, in a loop: the strands move, and end where they began.
    Figure f = MakeLog();
    AddDrip(f);
    {
        SoftEffect& drip = f.model().physics.soft[0].effects.back();
        drip.drip->sites = 2;
        drip.drip->toGround = false;
        drip.drip->length = 0.2f;
        drip.drip->sway = 1.0f;
        drip.everywhere = true;
    }
    Build(f);
    const u32 clip = 0;
    REQUIRE(f.document.clips[clip].looping);
    const f32 size = SoftRig(f.model(), f.model().physics.soft[0].id).points().size;
    const auto apart = [](const std::vector<Vector3f>& a, const std::vector<Vector3f>& b) {
        f32 far = 0.0f;
        for (std::size_t v = 0; v < a.size() && v < b.size(); ++v) {
            far = std::max(far, (a[v] - b[v]).length());
        }
        return far;
    };
    const std::vector<Vector3f> first = StrandsAt(f.document, clip, 0.0f);
    const std::vector<Vector3f> quarter = StrandsAt(f.document, clip, 0.25f * f.document.clips[clip].duration);
    const std::vector<Vector3f> last = StrandsAt(f.document, clip, f.document.clips[clip].duration);
    CHECK(apart(first, quarter) > 0.02f * size);
    CHECK(apart(first, last) < 1e-4f * size);
    // With no Sway they hang still.
    f.model().physics.soft[0].effects.back().drip->sway = 0.0f;
    CHECK(apart(StrandsAt(f.document, clip, 0.0f), StrandsAt(f.document, clip, 0.25f)) < 1e-6f * size);
}

TEST_CASE("G-S7 Pull the surface brings the skin under a strand down with it, and the strand rides it",
          "[wem][soft][drip]") {
    Figure f = MakeLog();
    // Surface lumps, so that the skin has something to come down by.
    f.model().physics.soft[0].detail = 48;
    AddDrip(f);
    {
        SoftEffect& drip = f.model().physics.soft[0].effects.back();
        drip.drip->sites = 2;
        // Far enough apart that one strand's pull does not reach the other's site.
        drip.drip->spacing = 0.3f;
        drip.drip->toGround = false;
        drip.drip->length = 0.1f;
        drip.drip->stagger = 0.0f;
        drip.everywhere = true;
    }
    Build(f);
    const u32 id = f.model().physics.soft[0].id;
    const u32 effect = f.model().physics.soft[0].effects.back().id;
    const auto stepAt = [&](u32 clip, f32 seconds) {
        const SoftStepper stepper(f.document, 0, id);
        const SoftPoseAt poseAt = [&](f32 at, std::vector<Matrix44f>& skin) { skin = SkinAt(f.document, clip, at); };
        SoftStepResult out;
        stepper.step(SkinAt(f.document, clip, seconds), SoftClock{clip, seconds}, poseAt, out);
        return out;
    };
    const u32 loop = 0;
    REQUIRE(f.document.clips[loop].looping);
    const SoftStepResult plain = stepAt(loop, 0.0f);
    const std::vector<Vector3f> hung = StrandsAt(f.document, loop, 0.0f);
    f.model().physics.soft[0].effects.back().drip->pull = 1.0f;
    const SoftDrip& drip = *f.model().physics.soft[0].effects.back().drip;
    const SoftStepResult pulled = stepAt(loop, 0.0f);
    const std::vector<Vector3f> rides = StrandsAt(f.document, loop, 0.0f);
    const SoftRig rig(f.model(), id);
    const u32 strandMesh = f.model().physics.soft[0].drips;
    const u32 perStrand = 6u * 4u + 1u;
    REQUIRE(drip.strandAt.size() == 2u);
    REQUIRE(pulled.moved);
    for (std::size_t s = 0; s < 2; ++s) {
        const f32 depth = kSoftPullDepth * drip.strandWidth[s];
        u32 nearest = kInvalidIndex;
        f32 least = 0.0f;
        for (u32 p = 0; p < rig.pointCount(); ++p) {
            const SoftPoint& point = rig.points().points[p];
            const f32 lowered = plain.goal[p].z - pulled.goal[p].z;
            // Down and nowhere else; and a strand's own points are not the skin.
            CHECK(std::abs(pulled.goal[p].x - plain.goal[p].x) < 1e-6f);
            CHECK(std::abs(pulled.goal[p].y - plain.goal[p].y) < 1e-6f);
            if (point.mesh == strandMesh) {
                CHECK(lowered == 0.0f);
                continue;
            }
            CHECK(lowered >= 0.0f);
            const f32 far = (point.at - drip.strandAt[s]).length();
            const f32 other = (point.at - drip.strandAt[1 - s]).length();
            // Out of both strands' reach the skin is left alone.
            if (far >= kSoftPullReach * drip.strandWidth[s] && other >= kSoftPullReach * drip.strandWidth[1 - s]) {
                CHECK(lowered == 0.0f);
            }
            if (nearest == kInvalidIndex || far < least) {
                nearest = p;
                least = far;
            }
        }
        REQUIRE(nearest != kInvalidIndex);
        // A site is a point of the body: there the skin is asked down by the pull's whole depth.
        CHECK(least < 1e-4f);
        CHECK(plain.goal[nearest].z - pulled.goal[nearest].z == Approx(depth).epsilon(1e-3));
        // The strand rides where the lumps put that point, every part of it by the same.
        const Vector3f rode = pulled.posed[nearest] - pulled.carried[nearest];
        INFO("strand " << s << " rode " << rode.z << " of " << depth);
        CHECK(rode.z < -0.05f * depth);
        for (u32 v = 0; v < perStrand; ++v) {
            const Vector3f moved = rides[s * perStrand + v] - hung[s * perStrand + v];
            CHECK((moved - rode).length() < 2e-4f);
        }
    }
    // Half the setting, half the depth.
    f.model().physics.soft[0].effects.back().drip->pull = 0.5f;
    {
        const SoftStepResult half = stepAt(loop, 0.0f);
        f32 most = 0.0f, whole = 0.0f;
        for (u32 p = 0; p < rig.pointCount(); ++p) {
            most = std::max(most, plain.goal[p].z - half.goal[p].z);
            whole = std::max(whole, plain.goal[p].z - pulled.goal[p].z);
        }
        CHECK(most == Approx(0.5f * whole).epsilon(1e-3));
    }
    f.model().physics.soft[0].effects.back().drip->pull = 1.0f;

    // Played once: nothing before a strand sets out, and then by how much of
    // its length it has.
    const u32 once = 1;
    f.document.clips[once].looping = false;
    f.document.clips[once].duration = 2.0f;
    f.model().physics.soft[0].effects.back().everywhere = false;
    const SoftSpan span{0.5f, 2.0f, 0.0f, 0.0f};
    SetSoftSpans(f.document, 0, once, id, effect, std::span<const SoftSpan>(&span, 1));
    CHECK(!stepAt(once, 0.25f).moved);
    {
        const f32 size = rig.points().size;
        const f32 reach = 0.1f * size, speed = drip.speed * size;
        const f32 since = 0.25f * reach / speed;
        const SoftStepResult early = stepAt(once, 0.5f + since);
        f32 most = 0.0f;
        for (u32 p = 0; p < rig.pointCount(); ++p) {
            most = std::max(most, early.carried[p].z - early.goal[p].z);
        }
        const f32 widest = std::max(drip.strandWidth[0], drip.strandWidth[1]);
        CHECK(most == Approx(0.25f * kSoftPullDepth * widest).epsilon(0.02));
    }
}

TEST_CASE("G-S7 Pinch off: a strand grows, lets its tip go as a droplet and draws back, in whole cycles a loop",
          "[wem][soft][drip]") {
    // The cycle. Played once, the first grows from nothing.
    const SoftPinch start = SoftPinchAt(0.0f, 2.0f, 0.3f, true);
    CHECK(start.length == 0.0f);
    CHECK(!start.away);
    const SoftPinch grown = SoftPinchAt(0.3f, 2.0f, 0.3f, true);
    CHECK(grown.length == Approx(1.0f));
    CHECK(grown.thinned == Approx(0.5f));
    const SoftPinch ripe = SoftPinchAt(kSoftPinchAt - 1e-4f, 2.0f, 0.3f, true);
    CHECK(ripe.length == Approx(1.0f));
    CHECK(ripe.thinned == Approx(1.0f).margin(1e-3));
    CHECK(ripe.swollen == Approx(1.0f).margin(1e-3));
    CHECK(!ripe.away);
    // It lets go: the swollen tip is the droplet's now.
    const SoftPinch let = SoftPinchAt(kSoftPinchAt, 2.0f, 0.3f, true);
    CHECK(let.away);
    CHECK(let.since == Approx(0.0f).margin(1e-6));
    CHECK(let.length == Approx(1.0f));
    CHECK(let.swollen == 0.0f);
    CHECK(let.fade == 1.0f);
    // A quarter of the way through what is left it is back at its stub, its
    // neck as wide as it was; the droplet is whole still.
    const SoftPinch back = SoftPinchAt(kSoftPinchAt + 0.25f * (1.0f - kSoftPinchAt), 2.0f, 0.3f, true);
    CHECK(back.length == Approx(kSoftPinchStub).margin(1e-4));
    CHECK(back.thinned == Approx(0.0f).margin(1e-4));
    CHECK(back.since == Approx(0.2f).margin(1e-4));
    CHECK(back.fade == 1.0f);
    // By the cycle's end the droplet is gone, and the next starts from the stub: no jump.
    const SoftPinch gone = SoftPinchAt(1.0f - 1e-4f, 2.0f, 0.3f, true);
    CHECK(gone.fade < 1e-3f);
    const SoftPinch next = SoftPinchAt(1.0f, 2.0f, 0.3f, true);
    CHECK(next.length == Approx(kSoftPinchStub));
    CHECK(gone.length == Approx(next.length).margin(1e-4));
    CHECK(!next.away);
    // In a loop every cycle is the same, the first too.
    CHECK(SoftPinchAt(0.0f, 2.0f, 0.3f, false).length == Approx(kSoftPinchStub));
    // No two strands let go together, and a seed moves them all.
    for (u32 s = 0; s < 6; ++s) {
        CHECK(SoftPinchPhase(1, s) >= 0.0f);
        CHECK(SoftPinchPhase(1, s) < 1.0f);
        for (u32 other = 0; other < s; ++other) {
            CHECK(std::abs(SoftPinchPhase(1, s) - SoftPinchPhase(1, other)) > 0.05f);
        }
    }
    CHECK(SoftPinchPhase(1, 0) != SoftPinchPhase(2, 0));

    // The droplet. Until it is away it is as built.
    {
        const SoftDrip settings;
        const f32 width = 0.05f, height = 2.0f, room = 0.5f;
        const Vector3f down{0, 0, -1}, from{0.1f, 0.0f, -0.3f};
        const Transform rest;
        const Transform held = SoftDropletAt(settings, width, room, height, ripe, down, from);
        CHECK(std::memcmp(&held, &rest, sizeof rest) == 0);
        // Let go, it is where the tip was, the tip's size.
        const f32 size = settings.tip / kSoftStrandShrink;
        const Transform born = SoftDropletAt(settings, width, room, height, let, down, from);
        CHECK((born.translation - from).length() < 1e-6f);
        CHECK(born.scale.x == Approx(size));
        CHECK(born.scale.z == Approx(size));
        // It falls freely,
        SoftPinch falling = let;
        falling.since = 0.1f;
        const f32 gravity = kSoftDropGravity * height;
        const Transform fell = SoftDropletAt(settings, width, room, height, falling, down, from);
        CHECK(fell.translation.z == Approx(from.z - 0.5f * gravity * 0.01f));
        CHECK(fell.translation.x == from.x);
        CHECK(fell.scale.z == Approx(size));
        // to the floor, where it lies flat with its volume kept,
        SoftPinch lying = let;
        lying.since = 5.0f;
        const Transform flat = SoftDropletAt(settings, width, room, height, lying, down, from);
        CHECK(flat.translation.z == Approx(from.z - (room - width * settings.tip * kSoftDropFlat)));
        CHECK(flat.scale.z == Approx(size * kSoftDropFlat));
        CHECK(flat.scale.x * flat.scale.y * flat.scale.z == Approx(size * size * size).epsilon(1e-4));
        // and shrinks away to what it was built at.
        lying.fade = 0.0f;
        const Transform away = SoftDropletAt(settings, width, room, height, lying, down, from);
        CHECK(away.scale.x == Approx(1.0f));
        CHECK(away.scale.z == Approx(1.0f));
    }

    // Built: a fourth bone a strand and a ball in the strands' mesh.
    Figure f = MakeLog();
    AddDrip(f);
    {
        SoftEffect& drip = f.model().physics.soft[0].effects.back();
        drip.drip->sites = 2;
        drip.drip->stagger = 0.0f;
        drip.drip->thickness = 0.01f;
        // Not a whole share of the loop, which is a second long: rounded, it is one cycle a loop.
        drip.drip->pinchEvery = 0.8f;
        drip.everywhere = true;
    }
    const u32 nodesBefore = f.model().nodes.size();
    const SoftStrandsBuilt built = Build(f);
    const SoftBody& body = f.model().physics.soft[0];
    const SoftDrip& drip = *body.effects.back().drip;
    REQUIRE(built.strands == 2u);
    CHECK(built.bones == 8u);
    CHECK(f.model().nodes.size() == nodesBefore + 8u);
    REQUIRE(drip.dropNodes.size() == 2u);
    const u32 perStrand = 6u * 4u + 1u, perDrop = 14u;
    CHECK(f.model().meshes[body.drips].vertexCount() == 2u * (perStrand + perDrop));
    for (std::size_t s = 0; s < 2; ++s) {
        const Node& drop = f.model().nodes.nodes[drip.dropNodes[s]];
        CHECK(drop.parent == f.model().nodes.nodes[drip.strandNodes[3 * s]].parent);
        CHECK((drop.pivot - f.model().nodes.nodes[drip.strandNodes[3 * s]].pivot).length() == 0.0f);
        // Used as it is, like its strand's bones.
        const auto lump = std::find_if(body.lumps.begin(), body.lumps.end(), [&](const SoftLump& each) {
            return each.kind == SoftLumpKind::Bone && each.node == drip.dropNodes[s];
        });
        CHECK(lump != body.lumps.end());
    }
    {
        Figure plain = MakeLog();
        CHECK(Validate(f.document, ValidateLevel::Structural).size() <=
              Validate(plain.document, ValidateLevel::Structural).size() + 1u);
    }
    // A material that sets lets nothing go, whatever the seconds say.
    {
        Figure wax = MakeLog();
        SoftEffect& set = AddDrip(wax);
        set.drip->sites = 2;
        set.drip->pinchEvery = 1.0f;
        set.drip->neck = SoftNeckLaw::Sets;
        CHECK(Build(wax).bones == 6u);
        CHECK(wax.model().physics.soft[0].effects.back().drip->dropNodes.empty());
    }

    // In a loop: a whole cycle a loop, so its last frame is its first.
    const u32 loop = 0;
    REQUIRE(f.document.clips[loop].looping);
    const f32 duration = f.document.clips[loop].duration;
    const f32 size = SoftRig(f.model(), body.id).points().size;
    const auto apart = [](const std::vector<Vector3f>& a, const std::vector<Vector3f>& b) {
        f32 far = 0.0f;
        for (std::size_t v = 0; v < a.size() && v < b.size(); ++v) {
            far = std::max(far, (a[v] - b[v]).length());
        }
        return far;
    };
    CHECK(apart(StrandsAt(f.document, loop, 0.0f), StrandsAt(f.document, loop, duration)) < 1e-4f * size);
    // Over the loop each strand is seen at its length and at its stub, and its
    // droplet away under it: let go where the tip was, down to the floor and
    // never under it, and gone again.
    for (u32 s = 0; s < 2; ++s) {
        const f32 width = drip.strandWidth[s];
        const Vector3f site = drip.strandAt[s];
        const f32 reach = (site.z - width * drip.tip) * kSoftPinchReach;
        f32 longest = 0.0f, shortest = 1e9f, lowest = 1e9f, largest = 0.0f, smallest = 1e9f;
        bool letGoAtTip = false;
        f32 before = 0.0f;
        const u32 samples = 240;
        for (u32 i = 0; i <= samples; ++i) {
            const std::vector<Vector3f> at = StrandsAt(f.document, loop, duration * static_cast<f32>(i) / samples);
            const u32 first = s * (perStrand + perDrop);
            // The strand's last ring stands at its tip's pivot; the droplet's poles are its first two points.
            f32 hangs = 0.0f;
            for (u32 v = 18; v < 24; ++v) {
                hangs += (site.z - at[first + v].z) / 6.0f;
            }
            longest = std::max(longest, hangs);
            shortest = std::min(shortest, hangs);
            const Vector3f over = at[first + perStrand], under = at[first + perStrand + 1];
            const f32 tall = (over - under).length();
            largest = std::max(largest, tall);
            smallest = std::min(smallest, tall);
            if (tall > width) {
                lowest = std::min(lowest, under.z);
                // The step it appears on, it is where the strand's tip just was.
                if (before <= width && std::abs(0.5f * (over.z + under.z) - (site.z - reach)) < 0.15f * reach) {
                    letGoAtTip = true;
                }
            }
            before = tall;
        }
        INFO("strand " << s << ": longest " << longest << " shortest " << shortest << " lowest " << lowest);
        CHECK(longest == Approx(reach).epsilon(0.03));
        CHECK(shortest == Approx(kSoftPinchStub * reach).epsilon(0.03));
        CHECK(largest == Approx(2.0f * width * drip.tip).epsilon(0.02));
        CHECK(smallest < 0.1f * width);
        CHECK(letGoAtTip);
        CHECK(lowest > -1e-3f);
        CHECK(lowest < 0.05f * size);
    }

    // Played once: nothing until it sets out, then the cycle from nothing.
    const u32 once = 1;
    f.document.clips[once].looping = false;
    f.document.clips[once].duration = 3.0f;
    f.model().physics.soft[0].effects.back().everywhere = false;
    const SoftSpan span{0.5f, 3.0f, 0.0f, 0.0f};
    SetSoftSpans(f.document, 0, once, body.id, body.effects.back().id, std::span<const SoftSpan>(&span, 1));
    {
        const std::vector<Vector3f> at = StrandsAt(f.document, once, 0.25f);
        for (std::size_t v = 0; v < at.size(); ++v) {
            const Vector3f site = drip.strandAt[v / (perStrand + perDrop)];
            CHECK((at[v] - site).length() < 3.0f * kSoftStrandRest * drip.strandWidth[0] * kSoftStrandShrink);
        }
        // A cycle later it has let one go: the droplet is as tall as the tip was wide.
        const std::vector<Vector3f> later = StrandsAt(f.document, once, 0.5f + kSoftPinchAt * 0.8f + 0.02f);
        CHECK((later[perStrand] - later[perStrand + 1]).length() == Approx(2.0f * drip.strandWidth[0] * drip.tip).epsilon(0.02));
    }
}
