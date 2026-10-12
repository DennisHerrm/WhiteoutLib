// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_run.h>

#include <whiteout/models/wem/anim/stages.h>
#include <whiteout/models/wem/anim/track_read.h>
#include <whiteout/models/wem/geometry/attributes.h>
#include <whiteout/models/wem/skinning/deform.h>

#include "soft_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace whiteout {
namespace models {
namespace wem {

SoftDeformer::SoftDeformer(const Model& model, SoftWeights weights) : model_(&model) {
    const u32 nodes = model.nodes.size();
    written_.resize(model.meshes.size());
    paletteSize_ = nodes;
    for (const SoftBody& body : model.physics.soft) {
        base_.push_back(paletteSize_);
        paletteSize_ += static_cast<u32>(body.lumps.size());
    }
    for (u32 b = 0; b < model.physics.soft.size(); ++b) {
        const SoftBody& body = model.physics.soft[b];
        SoftSkinner skinner(model, body, weights);
        for (const u32 m : body.meshes) {
            if (m >= model.meshes.size()) {
                continue;
            }
            const Mesh& mesh = model.meshes[m];
            const u32 count = mesh.vertexCount();
            std::vector<std::vector<geom::Influence>>& out = written_[m];
            bool any = false;
            for (u32 v = 0; v < count; ++v) {
                any = any || skinner.holds(m, v);
            }
            if (!any) {
                continue;
            }
            if (out.empty()) {
                // The mesh's other vertices by their own skin, under this
                // body's cap: a later body overwrites its own points alone.
                out.resize(count);
                for (u32 v = 0; v < count; ++v) {
                    const std::span<const geom::Influence> own = skinner.of(m, v);
                    out[v].assign(own.begin(), own.end());
                }
                meshes_.push_back(m);
            }
            for (u32 v = 0; v < count; ++v) {
                if (!skinner.holds(m, v)) {
                    continue;
                }
                const std::span<const geom::Influence> given = skinner.of(m, v);
                out[v].assign(given.begin(), given.end());
                for (geom::Influence& influence : out[v]) {
                    if (influence.bone >= nodes) {
                        influence.bone = base_[b] + (influence.bone - nodes);
                    }
                }
            }
        }
    }
    std::sort(meshes_.begin(), meshes_.end());
}

std::vector<Matrix44f> SoftDeformer::palette(std::span<const Matrix44f> skin,
                                             std::span<const Transform> departures) const {
    const u32 nodes = model_->nodes.size();
    std::vector<Matrix44f> out(paletteSize_, Matrix44f::identity());
    for (u32 n = 0; n < nodes && n < skin.size(); ++n) {
        out[n] = skin[n];
    }
    std::size_t at = 0;
    for (u32 b = 0; b < model_->physics.soft.size(); ++b) {
        const SoftBody& body = model_->physics.soft[b];
        for (u32 k = 0; k < body.lumps.size(); ++k, ++at) {
            const SoftLump& lump = body.lumps[k];
            const u32 family = lump.family();
            const Matrix44f carried = family < skin.size() ? skin[family] : Matrix44f::identity();
            const Matrix44f moved = at < departures.size()
                                        ? SoftDepartureMatrix(lump.rest, departures[at]) * carried
                                        : carried;
            out[base_[b] + k] = moved;
            if (lump.kind == SoftLumpKind::Bone && lump.node < nodes) {
                out[lump.node] = moved;
            }
        }
    }
    return out;
}

std::vector<Vector3f> SoftDeformer::positions(u32 mesh, std::span<const Matrix44f> palette) const {
    if (mesh >= model_->meshes.size()) {
        return {};
    }
    if (!soft(mesh)) {
        return skinning::DeformMesh(model_->meshes[mesh], palette);
    }
    return skinning::DeformMesh(model_->meshes[mesh], palette, written_[mesh]);
}

std::vector<Vector3f> SoftDeformer::normals(u32 mesh, std::span<const Matrix44f> palette) const {
    if (mesh >= model_->meshes.size()) {
        return {};
    }
    if (!soft(mesh)) {
        return skinning::DeformNormals(model_->meshes[mesh], palette);
    }
    return skinning::DeformNormals(model_->meshes[mesh], palette, written_[mesh]);
}

std::span<const geom::Influence> SoftDeformer::influences(u32 mesh, u32 vertex) const {
    if (!soft(mesh) || vertex >= written_[mesh].size()) {
        return {};
    }
    return written_[mesh][vertex];
}

bool SoftRun::moves() const {
    return std::find(acts.begin(), acts.end(), u8{1}) != acts.end();
}

const SoftModes& SoftModesCache::of(u32 model, const SoftRig& rig, const SoftVolume& volume,
                                    const SoftJiggle& settings) {
    const u32 body = rig.empty() ? 0 : rig.body().id;
    const u64 print = SoftRigPrint(rig);
    for (const Entry& entry : entries) {
        if (entry.model == model && entry.body == body && entry.print == print &&
            entry.holdShape == settings.holdShape && entry.keepVolume == settings.keepVolume &&
            entry.keepInPlace == settings.keepInPlace) {
            return entry.modes;
        }
    }
    // One mix a body: a slider dragged leaves no trail of them.
    std::erase_if(entries, [&](const Entry& entry) { return entry.model == model && entry.body == body; });
    entries.push_back(Entry{model, body, print, settings.holdShape, settings.keepVolume, settings.keepInPlace,
                            MakeSoftModes(rig, volume, settings)});
    return entries.back().modes;
}

namespace {

/// The Jiggle that acts on @p body: its first that is on.
const SoftEffect* JiggleOf(const SoftBody& body) {
    for (const SoftEffect& effect : body.effects) {
        if (effect.enabled && effect.kind == SoftEffectKind::Jiggle && effect.jiggle) {
            return &effect;
        }
    }
    return nullptr;
}

/// Whether @p effect is a Wobble on *Random* that is on.
bool IsRandom(const SoftEffect& effect) {
    return effect.enabled && effect.kind == SoftEffectKind::Wobble && effect.wobble &&
           effect.wobble->pattern == SoftWobblePattern::Random;
}

} // namespace

bool SoftRunsWhole(const SoftBody& body) {
    return JiggleOf(body) != nullptr || std::any_of(body.effects.begin(), body.effects.end(), IsRandom);
}

SoftGoalRun RunSoftGoal(const Document& document, u32 model, u32 body, u32 clip, u32 steps, f32 dt,
                        const SoftPoseAt& poseAt, const std::atomic<bool>* cancel) {
    SoftGoalRun goal;
    SoftRun& run = goal.run;
    run.body = body;
    run.steps = steps;
    run.dt = dt;
    const SoftStepper stepper(document, model, body);
    if (stepper.empty() || !poseAt || steps == 0 || !(dt > 0.0f)) {
        return goal;
    }
    const SoftRig& rig = stepper.rig();
    const SoftBody& soft = rig.body();
    run.lumps = rig.lumpCount();
    run.departures.assign(static_cast<std::size_t>(steps + 1) * run.lumps, Transform{});
    run.acts.assign(steps + 1, 0);
    const SoftEffect* jiggle = JiggleOf(soft);
    goal.duration = clip < document.clips.size() ? document.clips[clip].duration : 0.0f;
    goal.loops = clip < document.clips.size() && document.clips[clip].looping;
    // A Wobble on *Random*: how far it moves the body at each step.
    for (const SoftEffect& effect : soft.effects) {
        if (!IsRandom(effect)) {
            continue;
        }
        SoftGoalRun::Random random;
        random.effect = effect.id;
        bool acts = false;
        for (u32 i = 0; i <= steps; ++i) {
            const SoftTiming timing = SoftTimingAt(document, model, clip, soft, effect, static_cast<f32>(i) * dt);
            random.amount.push_back(effect.amount * timing.level * rig.points().size);
            acts = acts || random.amount.back() != 0.0f;
        }
        if (acts) {
            goal.random.push_back(std::move(random));
        }
    }
    // An effect that does not drive the Jiggle is left out of what throws it:
    // a second step, where there is one.
    bool undriving = false;
    for (const SoftEffect& effect : soft.effects) {
        undriving = undriving || (effect.enabled && !effect.drivesJiggle && effect.kind != SoftEffectKind::Jiggle);
    }
    std::vector<f32> level;
    bool any = false;
    for (u32 i = 0; jiggle != nullptr && i <= steps; ++i) {
        const SoftTiming timing = SoftTimingAt(document, model, clip, soft, *jiggle, static_cast<f32>(i) * dt);
        level.push_back(std::clamp(jiggle->amount * timing.level, 0.0f, 4.0f));
        any = any || level.back() > 0.0f;
    }
    SoftJiggleDrive& drive = goal.drive;
    if (any) {
        goal.jiggle = jiggle->id;
        drive.steps = steps;
        drive.dt = dt;
        drive.loops = clip < document.clips.size() && document.clips[clip].looping;
        drive.level = std::move(level);
        drive.driven.reserve(static_cast<std::size_t>(steps + 1) * rig.pointCount());
        drive.carriers.reserve(static_cast<std::size_t>(steps + 1) * run.lumps);
        // The ground the clip's run stands on (bake design §4.8): the grid,
        // or none. One that follows the root is no height a body can be held
        // over, and holds nothing.
        const bool grid = clip >= document.clips.size() || !document.clips[clip].physics.has_value() ||
                          document.clips[clip].physics->floor == BakeFloor::Grid;
        drive.floor = grid ? 0.0f : -std::numeric_limits<f32>::infinity();
        // The bones the points ride: what the swing's volume trim poses by.
        for (u32 p = 0; p < rig.pointCount(); ++p) {
            for (const SoftSlot& slot : rig.slotsOf(p)) {
                if (slot.node != kInvalidNode) {
                    goal.ridden.push_back(slot.node);
                }
            }
        }
        std::sort(goal.ridden.begin(), goal.ridden.end());
        goal.ridden.erase(std::unique(goal.ridden.begin(), goal.ridden.end()), goal.ridden.end());
        goal.skins.reserve(static_cast<std::size_t>(steps + 1) * goal.ridden.size());
        goal.volume.reserve(steps + 1);
    }
    // The clip's winds and blasts push a Jiggle that asks for them (§8.3).
    SoftWorld world;
    const bool blown = any && jiggle->jiggle->worldForces && clip < document.clips.size() &&
                       document.clips[clip].physics.has_value() && !document.clips[clip].physics->world.empty();
    if (blown) {
        world.forces = document.clips[clip].physics->world;
        world.size = rig.points().size;
        world.duration = document.clips[clip].duration;
        world.looping = drive.loops;
        world.lower = rig.points().low;
        world.upper = rig.points().high;
        world.centres.assign(world.forces.size(), Vector3f{0, 0, 0});
        drive.pushed.reserve(static_cast<std::size_t>(steps + 1) * rig.pointCount());
    }
    // Its kicks (§8.3): each an instant change of speed round its place, at
    // the start of each of its spans in this clip.
    struct Kick {
        std::vector<f32> share; ///< Per point.
        Vector3f direction{0, 0, 0};
        f32 strength = 0.0f;
        std::vector<u32> steps; ///< The step each of its marks lands in.
    };
    std::vector<Kick> kicks;
    if (any) {
        const SoftJiggle& settings = *jiggle->jiggle;
        const std::size_t count = settings.kickIds.size();
        const bool whole = settings.kickMesh.size() == count && settings.kickFace.size() == count &&
                           settings.kickWeights.size() == count && settings.kickDirection.size() == count &&
                           settings.kickStrength.size() == count;
        for (std::size_t k = 0; whole && k < count; ++k) {
            Kick kick;
            // A mark lands in the step that holds it; one on a step's own time, in that step.
            // A loop's last frame is its first: a mark on it lands there.
            for (const SoftSpan& span : SoftSpansOf(document, model, clip, soft.id, settings.kickIds[k])) {
                const u32 step = static_cast<u32>(std::floor(std::max(span.start, 0.0f) / dt + 1e-3f));
                kick.steps.push_back(drive.loops ? step % steps : step);
            }
            SoftPlace place;
            place.mesh = settings.kickMesh[k];
            place.face = settings.kickFace[k];
            place.weights = settings.kickWeights[k];
            const SoftPlaceAt at = ResolveSoftPlace(document.models[model], rig, place);
            const f32 length = settings.kickDirection[k].length();
            if (kick.steps.empty() || !at.valid || settings.kickStrength[k] == 0.0f) {
                continue;
            }
            // With no direction said, into the body where it lands.
            kick.direction = length > 1e-9f ? settings.kickDirection[k] * (1.0f / length) : at.normal * -1.0f;
            kick.strength = settings.kickStrength[k];
            for (const SoftPoint& point : rig.points().points) {
                kick.share.push_back(SoftKickShare((point.at - at.at).length(), rig.points().size));
            }
            kicks.push_back(std::move(kick));
        }
    }
    const bool pushed = blown || !kicks.empty();
    if (pushed && !blown) {
        drive.pushed.reserve(static_cast<std::size_t>(steps + 1) * rig.pointCount());
    }
    // And what it is kept out of: the body's colliders, where each step has
    // them. Gathered below with the rest of what drives a Jiggle that acts.
    const bool meets = !soft.colliders.empty();
    std::vector<Matrix44f> skin;
    SoftStepResult out, driving;
    // One pass: nothing here remembers the step before.
    for (u32 i = 0; i <= steps; ++i) {
        if (cancel != nullptr && cancel->load()) {
            return SoftGoalRun{};
        }
        const f32 seconds = static_cast<f32>(i) * dt;
        poseAt(seconds, skin);
        stepper.step(skin, SoftClock{clip, seconds}, poseAt, out);
        run.acts[i] = out.moved ? 1 : 0;
        for (u32 k = 0; k < run.lumps && k < out.departures.size(); ++k) {
            run.departures[static_cast<std::size_t>(i) * run.lumps + k] = out.departures[k];
        }
        run.worstMiss = std::max(run.worstMiss, out.fit.worstMiss);
        run.largestMove = std::max(run.largestMove, out.fit.largestMove);
        if (!any) {
            continue;
        }
        for (const u32 node : goal.ridden) {
            goal.skins.push_back(node < skin.size() ? skin[node] : Matrix44f::identity());
        }
        goal.volume.push_back(stepper.volume().encloses ? SoftVolumeAt(stepper.volume(), out.posed) : 0.0);
        const std::vector<Vector3f>* driven = &out.posed;
        if (undriving && out.moved) {
            stepper.step(skin, SoftClock{clip, seconds}, poseAt, driving, SoftFitMethod::Fast, true);
            driven = &driving.posed;
        }
        drive.driven.insert(drive.driven.end(), driven->begin(), driven->end());
        if (pushed) {
            const std::size_t first = drive.pushed.size();
            // A push is read as a line from each step to the next, so one that
            // lasts a step is felt over the two round it. At either end of a
            // clip played once there is one, and it gives all of itself there.
            const f32 whole = !drive.loops && (i == 0 || i == steps) ? 2.0f : 1.0f;
            Vector3f wind{0, 0, 0};
            if (blown) {
                const NodeTree& tree = document.models[model].nodes;
                for (std::size_t w = 0; w < world.forces.size(); ++w) {
                    const u32 node = world.forces[w].centreNode;
                    if (node < skin.size() && node < tree.size()) {
                        world.centres[w] = transform_point(SoftBonePivot(tree, node), skin[node]);
                    }
                }
                wind = SoftWindAt(world, seconds);
            }
            for (const Vector3f& at : *driven) {
                drive.pushed.push_back(blown ? wind + SoftBlastAt(world, seconds, dt, at) * whole : Vector3f{0, 0, 0});
            }
            for (const Kick& kick : kicks) {
                for (const u32 step : kick.steps) {
                    if (step != i) {
                        continue;
                    }
                    const f32 push = SoftKickPush(kick.strength, rig.points().size, dt) * whole;
                    for (std::size_t p = 0; p < kick.share.size(); ++p) {
                        drive.pushed[first + p] = drive.pushed[first + p] + kick.direction * (push * kick.share[p]);
                    }
                }
            }
        }
        if (meets) {
            const std::vector<SoftCollider> shapes = SoftCollidersAt(document.models[model], soft, skin);
            if (i == 0) {
                drive.colliderCount = static_cast<u32>(shapes.size());
            }
            drive.colliders.insert(drive.colliders.end(), shapes.begin(), shapes.end());
        }
        for (const SoftLump& lump : soft.lumps) {
            const u32 family = lump.family();
            drive.carriers.push_back(family < skin.size() ? skin[family] : Matrix44f::identity());
        }
    }
    return goal;
}

SoftRun SwingSoft(const Document& document, u32 model, u32 body, const SoftGoalRun& goal, SoftModesCache* cache) {
    SoftRun run = goal.run;
    if (!goal.swings() || model >= document.models.size()) {
        return run;
    }
    const SoftStepper stepper(document, model, body);
    if (stepper.empty() || stepper.rig().lumpCount() != run.lumps) {
        return run;
    }
    const SoftRig& rig = stepper.rig();
    const SoftEffect* jiggle = goal.jiggle != 0 ? rig.body().effect(goal.jiggle) : nullptr;
    if (jiggle != nullptr && !jiggle->jiggle) {
        jiggle = nullptr;
    }
    if (jiggle == nullptr && goal.random.empty()) {
        return run;
    }
    // The body's ways of swinging: its Jiggle's mix, whether or not the
    // Jiggle acts in this clip, or the default one's where it has none. A
    // Wobble on *Random* then swings one way in every clip, and the modes
    // found for one clip serve the next.
    const SoftEffect* mix = jiggle != nullptr ? jiggle : JiggleOf(rig.body());
    const SoftJiggle settings = mix != nullptr ? *mix->jiggle : SoftJiggle{};
    SoftModes found;
    const SoftModes* modes = nullptr;
    if (cache != nullptr) {
        modes = &cache->of(model, rig, stepper.volume(), settings);
    } else {
        found = MakeSoftModes(rig, stepper.volume(), settings);
        modes = &found;
    }
    if (modes->empty()) {
        return run;
    }
    const std::size_t lumps = run.lumps;
    // The volume trim (§7, §8.3): what a Jiggle's swing changed of the volume
    // the goal had at a step is taken back, by its *Keep volume*.
    const f32 keeps = jiggle != nullptr ? std::clamp(settings.keepVolume, 0.0f, 1.0f) : 0.0f;
    const std::size_t ridden = goal.ridden.size();
    const bool trims = keeps > 0.0f && stepper.volume().encloses && ridden != 0 &&
                       goal.skins.size() == static_cast<std::size_t>(run.steps + 1) * ridden &&
                       goal.volume.size() == static_cast<std::size_t>(run.steps + 1);
    std::vector<Matrix44f> skin(trims ? goal.ridden.back() + 1 : 0, Matrix44f::identity());
    // A swing a lump a step, on top of what the run holds. Not for a lump whose
    // carrier is scaled to nothing there: the step left it at rest, and it
    // stays so.
    const std::vector<Matrix44f>& carriers = goal.drive.carriers;
    const bool carried = carriers.size() == run.departures.size();
    const auto add = [&](const std::vector<Transform>& swings, bool trim) {
        if (swings.size() != run.departures.size()) {
            return false;
        }
        for (u32 i = 0; i <= run.steps; ++i) {
            Transform* at = run.departures.data() + i * lumps;
            bool moved = false;
            for (std::size_t k = 0; k < lumps; ++k) {
                const Transform& swing = swings[i * lumps + k];
                if (SoftAtRest(swing, rig.points().size) || (carried && softdetail::Collapsed(carriers[i * lumps + k]))) {
                    continue;
                }
                // What is there first, then the swing about the same pivot.
                at[k] = Compose(swing, at[k]);
                moved = true;
            }
            if (moved) {
                if (trim) {
                    for (std::size_t r = 0; r < ridden; ++r) {
                        skin[goal.ridden[r]] = goal.skins[i * ridden + r];
                    }
                    stepper.trimVolume(skin, goal.volume[i], keeps, std::span<Transform>(at, lumps));
                }
                stepper.keepClear(std::span<Transform>(at, lumps));
                // The trim is every lump's: one hidden there gives its share back.
                for (std::size_t k = 0; carried && k < lumps; ++k) {
                    if (softdetail::Collapsed(carriers[i * lumps + k])) {
                        at[k] = Transform{};
                    }
                }
                run.acts[i] = 1;
            }
        }
        return true;
    };
    if (jiggle != nullptr) {
        SoftJiggleResult swung = RunSoftJiggle(rig, *modes, settings, goal.drive);
        if (!add(swung.swing, trims)) {
            return run;
        }
        run.swings = true;
        swung.swing.clear();
        run.swing = std::move(swung);
    }
    // *Random* (§8.4): the same ways of swinging, at rates and phases of its own.
    for (const SoftGoalRun::Random& random : goal.random) {
        const SoftEffect* effect = rig.body().effect(random.effect);
        if (effect == nullptr || !effect->wobble) {
            continue;
        }
        run.swings = add(SoftRandomWobble(rig, *modes, effect->wobble->speed, effect->seed, random.amount, run.steps,
                                          run.dt, goal.duration, goal.loops),
                         false) ||
                     run.swings;
    }
    return run;
}

SoftRun RunSoft(const Document& document, u32 model, u32 body, u32 clip, u32 steps, f32 dt, const SoftPoseAt& poseAt,
                SoftModesCache* cache, const std::atomic<bool>* cancel) {
    const SoftGoalRun goal = RunSoftGoal(document, model, body, clip, steps, dt, poseAt, cancel);
    // The swing is not stopped once begun: asked before it, it is not begun.
    if (cancel != nullptr && cancel->load()) {
        return SoftRun{};
    }
    return goal.swings() ? SwingSoft(document, model, body, goal, cache) : goal.run;
}

bool SoftAtRest(const Transform& departure, f32 size) {
    const f32 small = 1e-5f;
    return departure.translation.length() <= small * std::max(size, 1e-6f) &&
           std::abs(departure.scale.x - 1.0f) <= small && std::abs(departure.scale.y - 1.0f) <= small &&
           std::abs(departure.scale.z - 1.0f) <= small &&
           std::abs(departure.rotation.x) + std::abs(departure.rotation.y) + std::abs(departure.rotation.z) <= small;
}

std::vector<u8> SoftLeadsToLump(const Model& model) {
    const NodeTree& tree = model.nodes;
    std::vector<u8> leads(tree.size(), 0);
    for (const SoftBody& body : model.physics.soft) {
        for (const SoftLump& lump : body.lumps) {
            u32 guard = 0;
            for (u32 at = lump.family(); at < tree.size() && leads[at] == 0 && guard <= tree.size();
                 at = tree.nodes[at].parent, ++guard) {
                leads[at] = 1;
            }
        }
    }
    return leads;
}

std::vector<u32> SoftBonesHeldBelow(const Model& model, u32 body) {
    std::vector<u32> held;
    const SoftBody* record = model.physics.softBody(body);
    if (record == nullptr) {
        return held;
    }
    const NodeTree& tree = model.nodes;
    // 1: a bone this body uses as it is. 2: one any body does.
    std::vector<u8> used(tree.size(), 0);
    for (const SoftBody& each : model.physics.soft) {
        for (const SoftLump& lump : each.lumps) {
            if (lump.kind == SoftLumpKind::Bone && lump.node < tree.size()) {
                used[lump.node] = each.id == body ? 1 : std::max<u8>(used[lump.node], 2);
            }
        }
    }
    const std::vector<u8> leads = SoftLeadsToLump(model);
    for (u32 n = 0; n < tree.size(); ++n) {
        const u32 parent = tree.nodes[n].parent;
        if (!tree.nodes[n].removed && leads[n] != 0 && used[n] == 0 && parent < tree.size() && parent != n &&
            used[parent] == 1) {
            held.push_back(n);
        }
    }
    return held;
}

u32 SoftLumpChannel(const Model& model, u32 body, u32 lump, Channel which) {
    const u32 sub = SoftSub(body, lump + 1);
    for (const AnimChannel& channel : model.animChannels.channels) {
        if (channel.target.kind == TrackTarget::Kind::Physics && channel.target.channel == which &&
            channel.target.sub == sub) {
            return channel.id;
        }
    }
    return kInvalidIndex;
}

namespace {

bool IsLumpChannel(Channel channel) {
    return channel == Channel::SoftLumpTranslation || channel == Channel::SoftLumpRotation ||
           channel == Channel::SoftLumpScale;
}

/// @p track's keys, each @p T, read at @p seconds: straight between two keys
/// (the shorter way round for a turn), held past either end.
template <class T>
T ReadLump(const SubTrack& track, f32 seconds, const T& rest) {
    const std::size_t count = track.times.size();
    if (count == 0 || track.values.size() < count * sizeof(T)) {
        return rest;
    }
    const auto key = [&](std::size_t k) {
        T value;
        std::memcpy(&value, track.values.data() + k * sizeof(T), sizeof(T));
        return value;
    };
    if (seconds <= track.times.front()) {
        return key(0);
    }
    if (seconds >= track.times.back()) {
        return key(count - 1);
    }
    const std::size_t after = static_cast<std::size_t>(
        std::upper_bound(track.times.begin(), track.times.end(), seconds) - track.times.begin());
    const f32 span = track.times[after] - track.times[after - 1];
    const f32 u = span > 0.0f ? (seconds - track.times[after - 1]) / span : 0.0f;
    if constexpr (std::is_same_v<T, Quaternion>) {
        return Quaternion::slerp(key(after - 1), key(after), u);
    } else {
        const T a = key(after - 1), b = key(after);
        return a + (b - a) * u;
    }
}

} // namespace

void SoftSourcePose(const Document& document, const Animator& animator, u32 clip, f32 seconds, Pose& out) {
    Mix mix;
    if (clip < document.clips.size()) {
        mix.plays.push_back(Play{clip, seconds, 1.0f, false});
    }
    animator.sample(mix, out, true);
    SoftSourceLocals(document, animator, clip, seconds, out);
    animator.compose(out);
    // The constraints and IK, as the bake's pass runs them (§11.1): a bone a
    // Link carries moves the body as the editor shows it.
    if (clip < document.clips.size()) {
        const StageRunner stages(document, document.clips[clip].model);
        if (!stages.empty()) {
            stages.constrain(animator, mix, out);
        }
    }
}

bool SoftSourceLocals(const Document& document, const Animator& animator, u32 clip, f32 seconds, Pose& out) {
    bool any = false;
    const Clip* played = clip < document.clips.size() ? &document.clips[clip] : nullptr;
    const PhysicsBake* record = played != nullptr && played->physics.has_value() && played->physics->baked.has_value()
                                    ? &*played->physics->baked
                                    : nullptr;
    if (record != nullptr && !record->softChannels.empty() && played->model < document.models.size()) {
        const AnimChannelTable& table = document.models[played->model].animChannels;
        // What a channel plays with no key of the clip's.
        Pose rest;
        animator.sample(Mix{}, rest, true);
        const SampleWindow window = ClipWindow(*played, Milliseconds(seconds), -1);
        for (const u32 id : record->softChannels) {
            const AnimChannel* channel = table.find(id);
            if (channel == nullptr || channel->target.kind != TrackTarget::Kind::Node ||
                channel->target.node >= out.local.size() || channel->target.node >= rest.local.size()) {
                continue;
            }
            // A rigid body's keys under the soft pass's first, where there are any.
            const SubTrack* source = nullptr;
            for (const std::vector<SubTrackContainer>* held : {&record->softSource, &record->source}) {
                for (const SubTrackContainer& container : *held) {
                    if (const SubTrack* found = container.find(id); found != nullptr && source == nullptr) {
                        source = found;
                    }
                }
            }
            Transform& local = out.local[channel->target.node];
            const Transform& still = rest.local[channel->target.node];
            if (channel->target.channel == Channel::Translation) {
                local.translation = source ? SampleVec3(*played, *source, window, still.translation) : still.translation;
            } else if (channel->target.channel == Channel::Rotation) {
                local.rotation = source ? SampleQuat(*played, *source, window, still.rotation) : still.rotation;
            } else if (channel->target.channel == Channel::Scale) {
                local.scale = source ? SampleVec3(*played, *source, window, still.scale) : still.scale;
            }
            any = true;
        }
    }
    return any;
}

bool ClipKeysSoftLumps(const Document& document, u32 clip) {
    if (clip >= document.clips.size() || document.clips[clip].model >= document.models.size()) {
        return false;
    }
    const AnimChannelTable& table = document.models[document.clips[clip].model].animChannels;
    for (const SubTrackContainer& container : document.clips[clip].containers) {
        for (const SubTrack& track : container.subTracks) {
            const AnimChannel* channel = table.find(track.channel);
            if (channel != nullptr && channel->target.kind == TrackTarget::Kind::Physics &&
                IsLumpChannel(channel->target.channel) && !track.times.empty()) {
                return true;
            }
        }
    }
    return false;
}

std::vector<Transform> SoftDeparturesAt(const Document& document, u32 model, u32 clip, f32 seconds) {
    std::vector<Transform> out;
    if (model >= document.models.size()) {
        return out;
    }
    const Model& owner = document.models[model];
    std::vector<std::size_t> base;
    for (const SoftBody& body : owner.physics.soft) {
        base.push_back(out.size());
        out.resize(out.size() + body.lumps.size());
    }
    if (clip >= document.clips.size()) {
        return out;
    }
    for (const SubTrackContainer& container : document.clips[clip].containers) {
        for (const SubTrack& track : container.subTracks) {
            const AnimChannel* channel = owner.animChannels.find(track.channel);
            if (channel == nullptr || channel->target.kind != TrackTarget::Kind::Physics ||
                !IsLumpChannel(channel->target.channel) || track.times.empty()) {
                continue;
            }
            const u32 id = SoftBodyOfSub(channel->target.sub);
            const u32 lump = SoftIndexOfSub(channel->target.sub) - 1;
            for (std::size_t b = 0; b < owner.physics.soft.size(); ++b) {
                const SoftBody& body = owner.physics.soft[b];
                // A bone used as it is has no track of its own; one left from
                // before a remake is not read.
                if (body.id != id || lump >= body.lumps.size() || body.lumps[lump].kind == SoftLumpKind::Bone) {
                    continue;
                }
                Transform& departure = out[base[b] + lump];
                if (channel->target.channel == Channel::SoftLumpTranslation) {
                    departure.translation = ReadLump<Vector3f>(track, seconds, Vector3f{0, 0, 0});
                } else if (channel->target.channel == Channel::SoftLumpRotation) {
                    departure.rotation = ReadLump<Quaternion>(track, seconds, Quaternion{0, 0, 0, 1});
                } else {
                    departure.scale = ReadLump<Vector3f>(track, seconds, Vector3f{1, 1, 1});
                }
            }
        }
    }
    return out;
}

} // namespace wem
} // namespace models
} // namespace whiteout
