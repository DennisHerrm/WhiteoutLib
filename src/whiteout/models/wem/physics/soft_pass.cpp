// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include "soft_pass.h"

#include <whiteout/models/wem/anim/animator.h>
#include <whiteout/models/wem/anim/stages.h>
#include <whiteout/models/wem/physics/soft_run.h>
#include <whiteout/models/wem/rigging/ik.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace whiteout {
namespace models {
namespace wem {
namespace softdetail {

namespace {

f32 Turn(const Quaternion& a, const Quaternion& b) {
    return 2.0f * std::acos(std::min(1.0f, std::abs(a.dot(b))));
}

/// The steps of @p first .. @p last to key so that every step is within
/// @p tolerance of the line between the keys around it, by @p off. With none,
/// every @p stride-th step and the last: the clip's key rate, as its bones
/// get. The farthest from each line is kept first.
template <class Off>
std::vector<u32> Thin(u32 first, u32 last, f32 tolerance, u32 stride, Off off) {
    std::vector<u32> kept;
    if (!(tolerance > 0.0f) || last <= first + 1) {
        for (u32 i = first; i <= last; ++i) {
            if (i == last || tolerance > 0.0f || (i - first) % std::max<u32>(stride, 1) == 0) {
                kept.push_back(i);
            }
        }
        return kept;
    }
    std::vector<u8> keep(last - first + 1, 0);
    keep.front() = keep.back() = 1;
    std::vector<std::pair<u32, u32>> lines{{first, last}};
    while (!lines.empty()) {
        const auto [a, b] = lines.back();
        lines.pop_back();
        f32 worst = 0.0f;
        u32 at = a;
        for (u32 m = a + 1; m < b; ++m) {
            const f32 e = off(a, b, static_cast<f32>(m - a) / static_cast<f32>(b - a), m);
            if (e > worst) {
                worst = e;
                at = m;
            }
        }
        if (worst > tolerance) {
            keep[at - first] = 1;
            lines.emplace_back(a, at);
            lines.emplace_back(at, b);
        }
    }
    for (u32 i = first; i <= last; ++i) {
        if (keep[i - first] != 0) {
            kept.push_back(i);
        }
    }
    return kept;
}

/// @p values at @p kept, read at step @p i: straight between the keys around it.
template <class T>
T Between(const std::vector<T>& values, const std::vector<u32>& kept, u32 i, const T& rest) {
    if (kept.empty()) {
        return rest;
    }
    if (i <= kept.front()) {
        return values[kept.front()];
    }
    if (i >= kept.back()) {
        return values[kept.back()];
    }
    const std::size_t after = static_cast<std::size_t>(std::upper_bound(kept.begin(), kept.end(), i) - kept.begin());
    const u32 a = kept[after - 1], b = kept[after];
    const f32 u = static_cast<f32>(i - a) / static_cast<f32>(b - a);
    if constexpr (std::is_same_v<T, Quaternion>) {
        return Quaternion::slerp(values[a], values[b], u);
    } else {
        return values[a] + (values[b] - values[a]) * u;
    }
}

template <class T>
SubTrack TrackOf(const std::vector<T>& values, const std::vector<u32>& kept, f32 dt) {
    SubTrack track;
    track.interp = std::is_same_v<T, Quaternion> ? Interpolation::Slerp : Interpolation::Linear;
    for (const u32 i : kept) {
        track.times.push_back(static_cast<f32>(i) * dt);
        const u8* bytes = reinterpret_cast<const u8*>(&values[i]);
        track.values.insert(track.values.end(), bytes, bytes + sizeof(T));
    }
    return track;
}

/// A bone keyed in place carries what hangs below it by its departure, and the
/// bodies were stepped with every lump departing from its carrier as the
/// animation has it (§6.1). So each child of such a bone that leads to a lump
/// is keyed too, to stand where it was stepped: its own departure from the
/// animation where it is a bone used as it is, the animation otherwise. A
/// chain used as it is (§6.2) then plays what the tool showed.
///
/// @p locals holds, per step, what the clip's keys play. A child that was not
/// keyed is added to @p bones under its parent's body.
void HoldBelow(const Model& owner, const Animator& animator, const std::vector<std::vector<Transform>>& locals,
               std::vector<SoftBoneKeys>& bones) {
    const NodeTree& tree = owner.nodes;
    const std::size_t lumps = bones.size();
    std::vector<u32> keysOf(tree.size(), kInvalidIndex);
    for (std::size_t k = 0; k < lumps; ++k) {
        if (bones[k].node < tree.size()) {
            keysOf[bones[k].node] = static_cast<u32>(k);
        }
    }
    struct Held {
        u32 child = kInvalidNode;
        u32 above = 0; ///< Its parent's place in `bones`.
        u32 keys = 0;  ///< Its own.
    };
    std::vector<Held> held;
    const std::vector<u8> leads = SoftLeadsToLump(owner);
    for (u32 n = 0; n < tree.size(); ++n) {
        const u32 parent = tree.nodes[n].parent;
        if (tree.nodes[n].removed || leads[n] == 0 || parent >= tree.size() || parent == n ||
            keysOf[parent] == kInvalidIndex) {
            continue;
        }
        held.push_back(Held{n, keysOf[parent], keysOf[n]});
    }
    if (held.empty()) {
        return;
    }
    const std::size_t steps = locals.size();
    // Where each lump's own departure is: a child's steps are added to below.
    std::vector<std::vector<u8>> own(lumps);
    for (std::size_t k = 0; k < lumps; ++k) {
        own[k] = bones[k].departs;
    }
    for (Held& hold : held) {
        if (hold.keys != kInvalidIndex) {
            continue;
        }
        SoftBoneKeys made;
        made.body = bones[hold.above].body;
        made.node = hold.child;
        made.departs.assign(steps, 0);
        for (std::size_t i = 0; i < steps; ++i) {
            made.plays.push_back(hold.child < locals[i].size() ? locals[i][hold.child] : Transform{});
        }
        made.samples = made.plays;
        hold.keys = static_cast<u32>(bones.size());
        bones.push_back(std::move(made));
    }
    const auto parentOf = [&](u32 node) {
        const u32 parent = tree.nodes[node].parent;
        return parent < tree.size() && parent != node ? parent : kInvalidNode;
    };
    Pose source, departed, file;
    for (std::size_t i = 0; i < steps; ++i) {
        if (std::none_of(held.begin(), held.end(), [&](const Held& hold) { return own[hold.above][i] != 0; })) {
            continue;
        }
        // The clip's keys as they play, and with each lump's departure on its
        // own bone: how that bone stands over its parent.
        source.local = locals[i];
        animator.compose(source);
        departed.local = locals[i];
        for (std::size_t k = 0; k < lumps; ++k) {
            if (bones[k].node < departed.local.size()) {
                departed.local[bones[k].node] = bones[k].samples[i];
            }
        }
        animator.compose(departed);
        // Where a lump's bone was stepped to: its departure over its parent as
        // the animation has it.
        const auto stepped = [&](u32 node) {
            const u32 parent = parentOf(node);
            return parent == kInvalidNode
                       ? departed.frame[node]
                       : departed.frame[node] * Matrix44f::inverse(departed.frame[parent]) * source.frame[parent];
        };
        file.frame = source.frame;
        for (const Held& hold : held) {
            if (own[hold.above][i] == 0) {
                continue;
            }
            const u32 above = bones[hold.above].node;
            file.frame[above] = stepped(above);
            SoftBoneKeys& keys = bones[hold.keys];
            keys.samples[i] =
                StageLocalFor(tree, file, hold.child, hold.keys < lumps ? stepped(hold.child) : source.frame[hold.child]);
            keys.departs[i] = 1;
            keys.moves = true;
            keys.turns = true;
        }
    }
}

} // namespace

Transform WithDeparture(const NodeTree& tree, u32 node, const Transform& local, const Transform& departure) {
    Transform inNode = departure;
    if (tree.rig == RigConvention::ExplicitBind && node < tree.size()) {
        // A departure is said along the model's axes at the bind; an explicit
        // rig's local is in the bone's own. A pivot rig's is an offset about
        // the pivot along the model's axes already, as a departure is. The
        // bind is the skin's, read from its inverse: a rest chain need not
        // compose to it (an `.m3`'s does not).
        const Quaternion bind = FromMatrix(Matrix44f::inverse(tree.inverseBindMatrix(node))).rotation;
        const Quaternion back = bind.conjugate();
        inNode.translation = Rotate(back, departure.translation);
        inNode.rotation = (back * departure.rotation * bind).normalized();
        // A scale is three numbers along the bone's own axes. Each takes what
        // the model's axes' scales come to along it: all of it where the bone
        // stands square to the model, and the nearest there is where it does
        // not, which no three numbers can say.
        const Vector3f axes[3] = {Rotate(back, Vector3f{1, 0, 0}), Rotate(back, Vector3f{0, 1, 0}),
                                  Rotate(back, Vector3f{0, 0, 1})};
        const f32 along[3] = {departure.scale.x, departure.scale.y, departure.scale.z};
        inNode.scale = Vector3f{0, 0, 0};
        for (int j = 0; j < 3; ++j) {
            inNode.scale.x += axes[j].x * axes[j].x * along[j];
            inNode.scale.y += axes[j].y * axes[j].y * along[j];
            inNode.scale.z += axes[j].z * axes[j].z * along[j];
        }
    }
    return Compose(local, inNode);
}

SoftPass RunSoftPass(Document& document, u32 clipIndex, u32 steps, f32 dt, f32 tolerance, u32 stride,
                     SoftModesCache* modes) {
    SoftPass pass;
    if (clipIndex >= document.clips.size() || document.clips[clipIndex].model >= document.models.size() || steps == 0) {
        return pass;
    }
    const u32 model = document.clips[clipIndex].model;
    Model& owner = document.models[model];
    if (owner.physics.soft.empty()) {
        return pass;
    }
    // What a lump's track is before its channel is found or declared.
    struct Pending {
        u32 body = 0;
        u32 lump = 0;
        Channel which = Channel::SoftLumpTranslation;
        SubTrack track;
    };
    std::vector<Pending> pending;
    {
        // The clip as its keys now play it. Nothing below writes until every
        // body has run: the animator and the steppers are cursors.
        const Animator animator(document, model);
        // After the constraints and IK (§11.1): a bone a Link carries moves the
        // body as the editor shows it. They rewrite the locals they place, so
        // what a bone's keys play is read before them.
        const StageRunner stages(document, model);
        std::vector<std::vector<Matrix44f>> skins(steps + 1);
        std::vector<std::vector<Transform>> locals(steps + 1);
        const auto evaluate = [&](f32 seconds, Pose& pose, std::vector<Transform>* keyed) {
            Mix mix;
            mix.plays.push_back(Play{clipIndex, seconds, 1.0f, false});
            animator.evaluate(mix, pose);
            if (keyed != nullptr) {
                *keyed = pose.local;
            }
            if (!stages.empty()) {
                stages.constrain(animator, mix, pose);
            }
        };
        for (u32 i = 0; i <= steps; ++i) {
            Pose pose;
            evaluate(static_cast<f32>(i) * dt, pose, &locals[i]);
            skins[i] = std::move(pose.skinning);
        }
        const SoftPoseAt poseAt = [&](f32 seconds, std::vector<Matrix44f>& skin) {
            // A step's own pose is at hand; what reads the motion asks between them.
            const f32 at = seconds / dt;
            const long step = std::lround(at);
            if (step >= 0 && step <= static_cast<long>(steps) && std::abs(at - static_cast<f32>(step)) < 1e-4f) {
                skin = skins[static_cast<std::size_t>(step)];
                return;
            }
            Pose pose;
            evaluate(seconds, pose, nullptr);
            skin = std::move(pose.skinning);
        };
        for (const SoftBody& body : owner.physics.soft) {
            if (body.effects.empty() || body.lumps.empty()) {
                continue;
            }
            const SoftRun run = RunSoft(document, model, body.id, clipIndex, steps, dt, poseAt, modes);
            if (run.lumps != body.lumps.size() || !run.moves()) {
                continue;
            }
            const SoftRig rig(owner, body.id);
            const f32 size = rig.points().size;
            const u32 lumps = run.lumps;
            // How far each lump's points lie from its pivot: what a turn or a
            // scale of it moves.
            std::vector<f32> reach(lumps, 0.0f);
            for (u32 p = 0; p < rig.pointCount(); ++p) {
                for (const SoftSlot& slot : rig.slotsOf(p)) {
                    if (slot.lump < lumps) {
                        reach[slot.lump] =
                            std::max(reach[slot.lump], (rig.points().points[p].at - body.lumps[slot.lump].rest).length());
                    }
                }
            }
            BakeReport::SoftLine line;
            line.body = body.id;
            line.follows = run.follows();
            line.push = run.swings ? run.swing.over : 0.0f;
            line.atEnd = run.swings ? run.swing.atEnd : 0.0f;
            // Each lump's three tracks, as the keys will play them: what the
            // measure below poses.
            std::vector<std::vector<Transform>> keyed(steps + 1, std::vector<Transform>(lumps));
            for (u32 k = 0; k < lumps; ++k) {
                const SoftLump& lump = body.lumps[k];
                std::vector<Vector3f> moves(steps + 1), scales(steps + 1);
                std::vector<Quaternion> turns(steps + 1);
                std::vector<u8> departs(steps + 1, 0);
                u32 first = steps + 1, last = 0;
                for (u32 i = 0; i <= steps; ++i) {
                    // At rest is said exactly, so a stretch of it is a line.
                    const Transform d = SoftAtRest(run.at(i)[k], size) ? Transform{} : run.at(i)[k];
                    moves[i] = d.translation;
                    turns[i] = i > 0 && d.rotation.dot(turns[i - 1]) < 0.0f
                                   ? Quaternion{-d.rotation.x, -d.rotation.y, -d.rotation.z, -d.rotation.w}
                                   : d.rotation;
                    scales[i] = d.scale;
                    keyed[i][k] = d;
                    if (!SoftAtRest(d, size)) {
                        departs[i] = 1;
                        first = std::min(first, i);
                        last = std::max(last, i);
                    }
                }
                if (first > steps) {
                    continue;
                }
                ++line.lumps;
                const auto any = [&](auto&& differs) {
                    for (u32 i = first; i <= last; ++i) {
                        if (differs(i)) {
                            return true;
                        }
                    }
                    return false;
                };
                const bool moved = any([&](u32 i) { return moves[i].length() > 1e-5f * std::max(size, 1e-6f); });
                const bool turned = any([&](u32 i) { return Turn(turns[i], Quaternion{0, 0, 0, 1}) > 1e-5f; });
                const bool scaled = any([&](u32 i) { return (scales[i] - Vector3f{1, 1, 1}).length() > 1e-5f; });
                if (lump.kind == SoftLumpKind::Bone) {
                    if (lump.node >= owner.nodes.size()) {
                        continue;
                    }
                    SoftBoneKeys bone;
                    bone.body = body.id;
                    bone.node = lump.node;
                    bone.samples.resize(steps + 1);
                    bone.plays.resize(steps + 1);
                    bone.departs = departs;
                    bone.moves = moved;
                    bone.turns = turned;
                    bone.scales = scaled;
                    bone.reach = reach[k];
                    for (u32 i = 0; i <= steps; ++i) {
                        bone.plays[i] = lump.node < locals[i].size() ? locals[i][lump.node] : Transform{};
                        bone.samples[i] = WithDeparture(owner.nodes, lump.node, bone.plays[i], keyed[i][k]);
                    }
                    pass.bones.push_back(std::move(bone));
                    continue;
                }
                // A key at rest one step outside each end, where the clip has room.
                const u32 from = first > 0 ? first - 1 : 0;
                const u32 to = std::min(last + 1, steps);
                // A third of the accuracy each: what a point a reach away moves.
                const f32 share = tolerance / 3.0f;
                const f32 arm = std::max(reach[k], 1e-6f);
                const auto write = [&](Channel which, SubTrack track, const std::vector<u32>& kept) {
                    line.keys += static_cast<u32>(kept.size());
                    pending.push_back(Pending{body.id, k, which, std::move(track)});
                };
                if (moved) {
                    const std::vector<u32> kept = Thin(from, to, share, stride, [&](u32 a, u32 b, f32 u, u32 m) {
                        return (moves[a] + (moves[b] - moves[a]) * u - moves[m]).length();
                    });
                    write(Channel::SoftLumpTranslation, TrackOf(moves, kept, dt), kept);
                    for (u32 i = 0; i <= steps; ++i) {
                        keyed[i][k].translation = Between(moves, kept, i, Vector3f{0, 0, 0});
                    }
                }
                if (turned) {
                    const std::vector<u32> kept = Thin(from, to, share, stride, [&](u32 a, u32 b, f32 u, u32 m) {
                        return Turn(Quaternion::slerp(turns[a], turns[b], u), turns[m]) * arm;
                    });
                    write(Channel::SoftLumpRotation, TrackOf(turns, kept, dt), kept);
                    for (u32 i = 0; i <= steps; ++i) {
                        keyed[i][k].rotation = Between(turns, kept, i, Quaternion{0, 0, 0, 1});
                    }
                }
                if (scaled) {
                    const std::vector<u32> kept = Thin(from, to, share, stride, [&](u32 a, u32 b, f32 u, u32 m) {
                        return (scales[a] + (scales[b] - scales[a]) * u - scales[m]).length() * arm;
                    });
                    write(Channel::SoftLumpScale, TrackOf(scales, kept, dt), kept);
                    for (u32 i = 0; i <= steps; ++i) {
                        keyed[i][k].scale = Between(scales, kept, i, Vector3f{1, 1, 1});
                    }
                }
            }
            if (line.lumps == 0) {
                continue;
            }
            // How far the keys leave any point from where a key every step had it.
            std::vector<Vector3f> exact, thinned;
            for (u32 i = 0; i <= steps; ++i) {
                rig.pose(skins[i], std::span<const Transform>(run.at(i), lumps), exact);
                rig.pose(skins[i], keyed[i], thinned);
                for (std::size_t p = 0; p < exact.size() && p < thinned.size(); ++p) {
                    line.within = std::max(line.within, (exact[p] - thinned[p]).length());
                }
            }
            pass.lines.push_back(line);
        }
        HoldBelow(owner, animator, locals, pass.bones);
    }
    // Now the table may grow: each lump's channel, found or declared.
    for (Pending& made : pending) {
        u32 channel = SoftLumpChannel(owner, made.body, made.lump, made.which);
        if (channel == kInvalidIndex) {
            AnimChannel declared;
            declared.id = owner.animChannels.nextFreeId();
            declared.target.kind = TrackTarget::Kind::Physics;
            declared.target.channel = made.which;
            declared.target.sub = SoftSub(made.body, made.lump + 1);
            declared.valueType = made.which == Channel::SoftLumpRotation ? geom::AttrType::Quat : geom::AttrType::F32x3;
            channel = owner.animChannels.add(declared);
            pass.declared.push_back(channel);
        }
        made.track.channel = channel;
        pass.tracks.push_back(std::move(made.track));
    }
    return pass;
}

} // namespace softdetail
} // namespace wem
} // namespace models
} // namespace whiteout
