// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_bake.h>

#include <whiteout/models/wem/geometry/attributes.h>
#include <whiteout/models/wem/physics/cloth_bake.h>

#include "soft_pass.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

namespace whiteout {
namespace models {
namespace wem {

namespace {

/// The groups a classic geoset holds (`GNDX` is a byte).
constexpr std::size_t kClassicGroups = 256;

/// A point's group as a classic file names it: the nodes it averages, sorted.
std::vector<u32> GroupOf(std::span<const geom::Influence> influences) {
    std::vector<u32> group;
    for (const geom::Influence& influence : influences) {
        group.push_back(influence.bone);
    }
    std::sort(group.begin(), group.end());
    return group;
}

/// @p wanted, or @p wanted numbered: a made bone's name is no other node's.
/// Two bodies that copy one bone would both make `<bone>_soft`.
std::string FreeNodeName(const NodeTree& tree, std::string wanted) {
    const auto taken = [&](const std::string& name) {
        return std::any_of(tree.nodes.begin(), tree.nodes.end(), [&](const Node& node) { return node.name == name; });
    };
    if (!taken(wanted)) {
        return wanted;
    }
    for (u32 n = 2;; ++n) {
        if (std::string tried = wanted + std::to_string(n); !taken(tried)) {
            return tried;
        }
    }
}

} // namespace

SoftWeights SoftWeightsFor(ProfileId target) {
    if (target == ProfileId::Wc3Classic) {
        return SoftWeights::EqualShares;
    }
    return target == ProfileId::Generic ? SoftWeights::Float : SoftWeights::Bytes;
}

SoftExpansion ExpandSoftToBones(Document& document, ProfileId target, Diagnostics& out) {
    SoftExpansion result;
    const SoftWeights weights = SoftWeightsFor(target);
    const u32 cap = std::max<u32>(1, Profile(target).maxBoneInfluences);
    result.boneOf.resize(document.models.size());
    for (u32 m = 0; m < document.models.size(); ++m) {
        Model& model = document.models[m];
        if (model.physics.soft.empty()) {
            continue;
        }
        const u32 nodesBefore = model.nodes.size();
        const u32 reducedBefore = result.reduced;
        result.boneOf[m].resize(model.physics.soft.size());

        // The lanes are for a number of weights a point; a target that holds
        // fewer, or shares them another way, gets them solved again by the
        // same rule (§6.4). One that holds more plays them as they are: what
        // the editor drew, and what the baked tracks were fitted to.
        for (SoftBody& body : model.physics.soft) {
            const bool equal = weights == SoftWeights::EqualShares;
            if (body.influences > cap || body.equalShares != equal) {
                BindLumps(model, body, cap, equal);
                ++result.rebound;
                out.info(DiagCode::PhysicsUnsupported,
                         "soft body '" + body.name + "': its lanes solved again for " + std::to_string(cap) +
                             " weights a point",
                         ElementRef(ElementKind::PhysicsRecord, body.id), target);
            }
        }

        // The skin first, while a slot past the nodes still means a lump.
        struct Bound {
            u32 mesh = 0;
            u32 vertex = 0;
            u32 body = 0;
            std::vector<geom::Influence> skin;
        };
        std::vector<Bound> bound;
        for (u32 b = 0; b < model.physics.soft.size(); ++b) {
            const SoftBody& body = model.physics.soft[b];
            SoftSkinner skinner(model, body, weights);
            for (const u32 meshIndex : body.meshes) {
                if (meshIndex >= model.meshes.size()) {
                    continue;
                }
                const u32 count = model.meshes[meshIndex].vertexCount();
                for (u32 v = 0; v < count; ++v) {
                    if (!skinner.holds(meshIndex, v)) {
                        continue;
                    }
                    const std::span<const geom::Influence> skin = skinner.of(meshIndex, v);
                    bound.push_back(Bound{meshIndex, v, b, {skin.begin(), skin.end()}});
                }
            }
        }

        // A classic geoset holds 256 groups, and every distinct set of lumps
        // is one. Past that, the rarest soft groups go back to their bones'
        // own lumps, a point at a time and inside its families, which is the
        // one reduction that cannot move the rest (§6.4).
        if (weights == SoftWeights::EqualShares) {
            for (u32 meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex) {
                const Mesh& mesh = model.meshes[meshIndex];
                std::map<std::vector<u32>, std::vector<std::size_t>> soft; // group -> entries of `bound`
                std::vector<u8> isSoft(mesh.vertexCount(), 0);
                for (std::size_t i = 0; i < bound.size(); ++i) {
                    if (bound[i].mesh == meshIndex) {
                        soft[GroupOf(bound[i].skin)].push_back(i);
                        isSoft[bound[i].vertex] = 1;
                    }
                }
                if (soft.empty()) {
                    continue;
                }
                std::map<std::vector<u32>, u32> all;
                for (const auto& [group, users] : soft) {
                    all[group] += static_cast<u32>(users.size());
                }
                for (u32 v = 0; v < mesh.vertexCount(); ++v) {
                    if (isSoft[v] == 0 && v < mesh.skin.vertexCount()) {
                        ++all[GroupOf(mesh.skin.forVertex(v))];
                    }
                }
                if (all.size() <= kClassicGroups) {
                    continue;
                }
                // Fewest users first; a group on own lumps alone is as small as it gets.
                std::vector<std::pair<u32, std::vector<u32>>> order;
                for (const auto& [group, users] : soft) {
                    order.push_back({static_cast<u32>(users.size()), group});
                }
                std::sort(order.begin(), order.end());
                for (const auto& [users, group] : order) {
                    if (all.size() <= kClassicGroups) {
                        break;
                    }
                    for (const std::size_t i : soft[group]) {
                        Bound& entry = bound[i];
                        const SoftBody& body = model.physics.soft[entry.body];
                        // Each family's lumps become its own lump (or its bone).
                        std::vector<u32> own;
                        for (const geom::Influence& influence : entry.skin) {
                            u32 node = influence.bone;
                            if (node >= nodesBefore) {
                                const SoftLump& lump = body.lumps[node - nodesBefore];
                                const u32 family = lump.family();
                                node = family;
                                for (u32 k = 0; k < body.lumps.size(); ++k) {
                                    if (body.lumps[k].kind == SoftLumpKind::Copy && body.lumps[k].node == family) {
                                        node = nodesBefore + k;
                                    }
                                }
                            }
                            if (std::find(own.begin(), own.end(), node) == own.end()) {
                                own.push_back(node);
                            }
                        }
                        std::vector<geom::Influence> reduced;
                        for (const u32 node : own) {
                            reduced.push_back({node, 1.0f / static_cast<f32>(own.size())});
                        }
                        const std::vector<u32> before = GroupOf(entry.skin);
                        entry.skin = std::move(reduced);
                        if (--all[before] == 0) {
                            all.erase(before);
                        }
                        ++all[GroupOf(entry.skin)];
                        ++result.reduced;
                    }
                }
            }
        }

        if (result.reduced != reducedBefore) {
            out.warn(DiagCode::PhysicsUnsupported,
                     std::to_string(result.reduced - reducedBefore) +
                         " soft points ride their bones' own lumps alone, with none of the surface's: a classic "
                         "geoset holds 256 groups of bones, and theirs were the least used",
                     ElementRef(ElementKind::Document, 0), target);
        }

        // A bone per copy and per surface lump, under its family's bone, at its pivot.
        std::vector<std::vector<u32>>& boneOf = result.boneOf[m];
        for (u32 b = 0; b < model.physics.soft.size(); ++b) {
            const SoftBody& body = model.physics.soft[b];
            u32 surface = 0;
            for (const SoftLump& lump : body.lumps) {
                if (lump.kind == SoftLumpKind::Bone) {
                    boneOf[b].push_back(lump.node);
                    continue;
                }
                std::string name;
                if (lump.kind == SoftLumpKind::Copy && lump.node < nodesBefore) {
                    name = model.nodes.nodes[lump.node].name + "_soft";
                } else {
                    char numbered[96];
                    std::snprintf(numbered, sizeof numbered, "%s_%02u", body.name.c_str(), ++surface);
                    name = numbered;
                }
                Matrix44f rest = Matrix44f::identity();
                rest.data[3][0] = lump.rest.x, rest.data[3][1] = lump.rest.y, rest.data[3][2] = lump.rest.z;
                const u32 family = lump.family();
                const u32 made = AddBoneAt(model.nodes, FreeNodeName(model.nodes, std::move(name)),
                                           family < nodesBefore ? family : kInvalidNode, rest);
                // At rest a made bone moves a point as its family's bone does.
                // The skin reads that bone through its inverse bind, which on an
                // explicit rig its rest chain need not compose to (an `.m3`'s).
                if (model.nodes.rig == RigConvention::ExplicitBind && family < nodesBefore) {
                    model.nodes.nodes[made].local = FromMatrix(rest * model.nodes.inverseBindMatrix(family));
                }
                boneOf[b].push_back(made);
                ++result.bones;
            }
        }

        // A baked clip's lump tracks become the made bones' own keys (§12).
        // On a pivot rig as they are: a departure is an offset about the
        // pivot along the model's axes, which is what a node's keys are there.
        // On an explicit rig, composed onto the bone's rest.
        const bool pivoted = model.nodes.rig == RigConvention::PivotRelative;
        for (Clip& clip : document.clips) {
            if (clip.model != m) {
                continue;
            }
            std::vector<SubTrack> keyed;
            for (const SubTrackContainer& container : clip.containers) {
                for (const SubTrack& track : container.subTracks) {
                    const AnimChannel* from = model.animChannels.find(track.channel);
                    if (from == nullptr || from->target.kind != TrackTarget::Kind::Physics || track.times.empty()) {
                        continue;
                    }
                    const Channel lumped = from->target.channel;
                    if (lumped != Channel::SoftLumpTranslation && lumped != Channel::SoftLumpRotation &&
                        lumped != Channel::SoftLumpScale) {
                        continue;
                    }
                    const u32 id = SoftBodyOfSub(from->target.sub);
                    const u32 lump = SoftIndexOfSub(from->target.sub) - 1;
                    u32 bone = kInvalidNode;
                    for (u32 b = 0; b < model.physics.soft.size(); ++b) {
                        const SoftBody& body = model.physics.soft[b];
                        // A bone used as it is holds its own keys already.
                        if (body.id == id && lump < body.lumps.size() && lump < boneOf[b].size() &&
                            body.lumps[lump].kind != SoftLumpKind::Bone) {
                            bone = boneOf[b][lump];
                        }
                    }
                    if (bone >= model.nodes.size()) {
                        continue;
                    }
                    const Channel which = lumped == Channel::SoftLumpTranslation ? Channel::Translation
                                          : lumped == Channel::SoftLumpRotation  ? Channel::Rotation
                                                                                 : Channel::Scale;
                    const bool quat = which == Channel::Rotation;
                    // One channel a bone and a kind of key, whatever the clips: a
                    // file's node has one track of each.
                    u32 channel = kInvalidIndex;
                    for (const AnimChannel& entry : model.animChannels.channels) {
                        if (entry.target.kind == TrackTarget::Kind::Node && entry.target.node == bone &&
                            entry.target.channel == which) {
                            channel = entry.id;
                        }
                    }
                    if (channel == kInvalidIndex) {
                        AnimChannel made;
                        made.id = model.animChannels.nextFreeId();
                        made.target.kind = TrackTarget::Kind::Node;
                        made.target.node = bone;
                        made.target.channel = which;
                        made.valueType = quat ? geom::AttrType::Quat : geom::AttrType::F32x3;
                        channel = model.animChannels.add(made);
                    }
                    SubTrack keys = track;
                    keys.channel = channel;
                    if (!pivoted) {
                        const Transform rest = model.nodes.nodes[bone].local;
                        const std::size_t size = quat ? sizeof(Quaternion) : sizeof(Vector3f);
                        for (std::size_t k = 0; k < keys.times.size() && (k + 1) * size <= keys.values.size(); ++k) {
                            Transform departure;
                            u8* at = keys.values.data() + k * size;
                            if (which == Channel::Translation) {
                                std::memcpy(&departure.translation, at, size);
                            } else if (quat) {
                                std::memcpy(&departure.rotation, at, size);
                            } else {
                                std::memcpy(&departure.scale, at, size);
                            }
                            const Transform local = softdetail::WithDeparture(model.nodes, bone, rest, departure);
                            if (which == Channel::Translation) {
                                std::memcpy(at, &local.translation, size);
                            } else if (quat) {
                                std::memcpy(at, &local.rotation, size);
                            } else {
                                std::memcpy(at, &local.scale, size);
                            }
                        }
                    }
                    result.keys += static_cast<u32>(keys.times.size());
                    keyed.push_back(std::move(keys));
                }
            }
            if (keyed.empty()) {
                continue;
            }
            if (clip.containers.empty()) {
                clip.containers.emplace_back();
            }
            for (SubTrack& keys : keyed) {
                clip.containers.front().subTracks.push_back(std::move(keys));
            }
        }

        // A rigid section binds its vertices to its node without asking the
        // skin (WEM_DESIGN §5.6), so one that holds a soft point states its
        // skin outright first.
        {
            std::vector<u8> touched(model.meshes.size(), 0);
            for (const Bound& entry : bound) {
                touched[entry.mesh] = 1;
            }
            for (u32 meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex) {
                Mesh& mesh = model.meshes[meshIndex];
                const bool rigid = std::any_of(mesh.sections.begin(), mesh.sections.end(),
                                               [](const MeshSection& s) { return s.rigidNode.has_value(); });
                if (touched[meshIndex] == 0 || (!rigid && !mesh.skin.empty())) {
                    continue;
                }
                SoftSkinSource own(model, cap);
                std::vector<std::vector<geom::Influence>> stated(mesh.vertexCount());
                for (u32 v = 0; v < stated.size(); ++v) {
                    const std::span<const geom::Influence> skin = own.of(meshIndex, v);
                    stated[v].assign(skin.begin(), skin.end());
                }
                if (mesh.skin.empty()) {
                    mesh.skin.reset(mesh.vertexCount());
                }
                for (u32 v = 0; v < stated.size(); ++v) {
                    mesh.skin.assignVertex(v, stated[v]);
                }
                for (MeshSection& section : mesh.sections) {
                    section.rigidNode.reset();
                }
            }
        }

        // The skin, each lump's slot now its bone.
        for (Bound& entry : bound) {
            bool lumped = false;
            for (geom::Influence& influence : entry.skin) {
                if (influence.bone >= nodesBefore) {
                    const u32 lump = influence.bone - nodesBefore;
                    influence.bone = lump < boneOf[entry.body].size() ? boneOf[entry.body][lump] : kInvalidNode;
                    lumped = true;
                }
            }
            std::erase_if(entry.skin, [](const geom::Influence& i) { return i.bone == kInvalidNode; });
            model.meshes[entry.mesh].skin.assignVertex(entry.vertex, entry.skin);
            result.points += lumped ? 1 : 0;
        }
    }
    return result;
}

} // namespace wem
} // namespace models
} // namespace whiteout
