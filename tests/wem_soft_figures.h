// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/// The figures the Soft Body tests share (wem_soft_test, wem_soft_fit_test):
/// boxes on a pivot rig, and a closed tube on a chain of bones with two clips.

#include <whiteout/models/wem/document.h>
#include <whiteout/models/wem/geometry/builder.h>
#include <whiteout/models/wem/nodes/node.h>
#include <whiteout/models/wem/physics/soft_lumps.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace softfig {

using namespace whiteout;
using namespace whiteout::models::wem;

/// A node at @p position under @p parent, on a pivot rig.
inline u32 AddNode(NodeTree& tree, const char* name, u32 parent, const Vector3f& position,
            NodeKind kind = NodeKind::Bone) {
    Node node;
    node.name = name;
    node.kind = kind;
    node.parent = parent;
    node.pivot = position;
    node.local.translation = position;
    if (parent != kInvalidNode) {
        const Vector3f& up = tree.nodes[parent].pivot;
        node.local.translation = Vector3f{position.x - up.x, position.y - up.y, position.z - up.z};
    }
    return tree.add(std::move(node));
}

using Skin = std::initializer_list<std::pair<u32, f32>>;

/// A closed box of side @p side with its low corner at @p low, every vertex on
/// @p skin. Wound outward, or inward with @p insideOut.
inline void AddBox(geom::MeshBuilder& builder, const Vector3f& low, f32 side, Skin skin, bool insideOut = false,
            u32 section = 0) {
    // Corner c is (x from bit 0, y from bit 1, z from bit 2).
    std::vector<geom::VertexId> corners;
    for (u32 corner = 0; corner < 8; ++corner) {
        corners.push_back(builder.addVertex(Vector3f{low.x + ((corner & 1) ? side : 0.0f),
                                                     low.y + ((corner & 2) ? side : 0.0f),
                                                     low.z + ((corner & 4) ? side : 0.0f)}));
        for (const auto& [bone, weight] : skin) {
            builder.addInfluence(corners.back(), bone, weight);
        }
    }
    static const u32 kFaces[12][3] = {{0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6}, {0, 1, 5}, {0, 5, 4},
                                      {2, 6, 7}, {2, 7, 3}, {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}};
    for (const auto& face : kFaces) {
        if (insideOut) {
            builder.addTriangle(corners[face[0]], corners[face[2]], corners[face[1]], section);
        } else {
            builder.addTriangle(corners[face[0]], corners[face[1]], corners[face[2]], section);
        }
    }
}

inline geom::MeshBuilder StartMesh(const char* name) {
    geom::MeshBuilder builder;
    MeshSection section;
    section.name = name;
    builder.addSection(std::move(section));
    return builder;
}

enum class Skinned { Rigid, Two, Four };

/// A closed tube along +Z, six long and one in radius, on a chain of bones two
/// apart, with two clips that turn, move and scale them.
struct Figure {
    Document document;
    std::vector<u32> bones;
    f32 size = 6.0f;

    Model& model() {
        return document.models[0];
    }
    const Model& model() const {
        return document.models[0];
    }
};

inline u32 NodeChannel(Model& model, u32 node, Channel channel) {
    AnimChannel made;
    made.id = model.animChannels.nextFreeId();
    made.target.kind = TrackTarget::Kind::Node;
    made.target.node = node;
    made.target.channel = channel;
    made.valueType = DefaultValueType(channel);
    return model.animChannels.add(made);
}

template <class T>
inline void Key(Clip& clip, u32 channel, std::initializer_list<f32> times, std::initializer_list<T> values,
         Interpolation interp = Interpolation::Linear) {
    SubTrack track;
    track.channel = channel;
    track.interp = interp;
    track.times = times;
    for (const T& value : values) {
        const u8* at = reinterpret_cast<const u8*>(&value);
        track.values.insert(track.values.end(), at, at + sizeof(T));
    }
    clip.containers.front().subTracks.push_back(std::move(track));
}

inline Figure MakeTube(Skinned skinned, u32 rings = 13, u32 segments = 12, ProfileId profile = ProfileId::Wc3Reforged) {
    Figure f;
    f.document.profiles = {profile};
    f.document.defaultProfile = profile;
    Model& model = f.document.models.emplace_back();
    const u32 count = skinned == Skinned::Four ? 4u : 3u;
    for (u32 b = 0; b < count; ++b) {
        f.bones.push_back(AddNode(model.nodes, ("b" + std::to_string(b)).c_str(), b == 0 ? kInvalidNode : f.bones[b - 1],
                                  {0, 0, 2.0f * static_cast<f32>(b)}));
    }
    geom::MeshBuilder builder = StartMesh("tube");
    std::vector<std::vector<geom::VertexId>> ring(rings);
    const auto bind = [&](geom::VertexId vertex, f32 z) {
        if (skinned == Skinned::Rigid) {
            builder.addInfluence(vertex, f.bones[z < 1.0f ? 0 : (z < 3.0f ? 1 : 2)], 1.0f);
        } else if (skinned == Skinned::Two) {
            const u32 low = std::min<u32>(static_cast<u32>(z / 2.0f), 1);
            const f32 t = std::clamp(z / 2.0f - static_cast<f32>(low), 0.0f, 1.0f);
            if (t < 1.0f) {
                builder.addInfluence(vertex, f.bones[low], 1.0f - t);
            }
            if (t > 0.0f) {
                builder.addInfluence(vertex, f.bones[low + 1], t);
            }
        } else {
            f32 total = 0.0f;
            f32 weights[4];
            for (u32 b = 0; b < 4; ++b) {
                const f32 d = z - 2.0f * static_cast<f32>(b);
                weights[b] = std::exp(-d * d / 8.0f);
                total += weights[b];
            }
            for (u32 b = 0; b < 4; ++b) {
                builder.addInfluence(vertex, f.bones[b], weights[b] / total);
            }
        }
    };
    for (u32 r = 0; r < rings; ++r) {
        const f32 z = 6.0f * static_cast<f32>(r) / static_cast<f32>(rings - 1);
        for (u32 s = 0; s < segments; ++s) {
            const f32 angle = 6.2831853f * static_cast<f32>(s) / static_cast<f32>(segments);
            ring[r].push_back(builder.addVertex(Vector3f{std::cos(angle), std::sin(angle), z}));
            bind(ring[r].back(), z);
        }
    }
    const geom::VertexId bottom = builder.addVertex(Vector3f{0, 0, 0});
    bind(bottom, 0.0f);
    const geom::VertexId top = builder.addVertex(Vector3f{0, 0, 6});
    bind(top, 6.0f);
    for (u32 s = 0; s < segments; ++s) {
        const u32 n = (s + 1) % segments;
        for (u32 r = 0; r + 1 < rings; ++r) {
            builder.addTriangle(ring[r][s], ring[r][n], ring[r + 1][n]);
            builder.addTriangle(ring[r][s], ring[r + 1][n], ring[r + 1][s]);
        }
        builder.addTriangle(bottom, ring[0][n], ring[0][s]);
        builder.addTriangle(top, ring[rings - 1][s], ring[rings - 1][n]);
    }
    model.meshes.push_back(builder.build().mesh);

    for (const char* name : {"Bend", "Twist"}) {
        Clip clip;
        clip.name = name;
        clip.model = 0;
        clip.duration = 1.0f;
        clip.looping = true;
        clip.containers.emplace_back();
        f.document.clips.push_back(clip);
    }
    Clip& bend = f.document.clips[0];
    const Quaternion none = Quaternion::identity();
    Key<Quaternion>(bend, NodeChannel(model, f.bones[1], Channel::Rotation), {0.0f, 0.5f, 1.0f},
                    {none, Quaternion::from_axis_angle(Vector3f{1, 0, 0}, 0.5f), none});
    Key<Quaternion>(bend, NodeChannel(model, f.bones[2], Channel::Rotation), {0.0f, 0.5f, 1.0f},
                    {none, Quaternion::from_axis_angle(Vector3f{0, 1, 0}, -0.4f), none});
    Key<Vector3f>(bend, NodeChannel(model, f.bones[0], Channel::Translation), {0.0f, 0.5f, 1.0f},
                  {Vector3f{0, 0, 0}, Vector3f{0.3f, 0, 0.2f}, Vector3f{0, 0, 0}});
    Key<Vector3f>(bend, NodeChannel(model, f.bones[2], Channel::Scale), {0.0f, 0.5f, 1.0f},
                  {Vector3f{1, 1, 1}, Vector3f{1.2f, 0.9f, 1.1f}, Vector3f{1, 1, 1}});
    Clip& twist = f.document.clips[1];
    Key<Quaternion>(twist, model.animChannels.channels[0].id, {0.0f, 0.5f, 1.0f},
                    {none, Quaternion::from_axis_angle(Vector3f{0, 0, 1}, 1.0f), none});
    return f;
}

/// Makes the part of the tube above z = 1 a soft body: softness @p soft
/// throughout, or rising from 0 at z = 1 to 1 at z = 3 when it is negative.
inline SoftBody& Soften(Figure& f, f32 soft, u32 detail, bool copyAll, u32 cap = 4, bool equalShares = false) {
    Model& model = f.model();
    Mesh& mesh = model.meshes[0];
    SoftBody made;
    made.id = model.physics.allocateId();
    made.name = "Jelly";
    made.meshes = {0};
    made.detail = detail;
    made.copyAll = copyAll;
    const std::span<const Vector3f> positions =
        mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    const std::span<u32> bodyOf =
        mesh.attributes.getOrCreate<u32>(geom::names::kSoftBody, geom::Domain::Vertex, geom::AttrType::U32);
    const std::span<f32> softness =
        mesh.attributes.getOrCreate<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex, geom::AttrType::F32);
    for (u32 v = 0; v < positions.size(); ++v) {
        if (positions[v].z >= 1.0f - 1e-4f) {
            bodyOf[v] = made.id;
            softness[v] = soft >= 0.0f ? soft : std::clamp((positions[v].z - 1.0f) / 2.0f, 0.0f, 1.0f);
        }
    }
    model.physics.soft.push_back(made);
    SoftBody& body = model.physics.soft.back();
    MakeSoftLumps(model, body, cap, equalShares);
    return body;
}

/// Makes the whole tube a soft body, fully soft.
inline SoftBody& SoftenAll(Figure& f, u32 detail, bool copyAll, u32 cap = 4) {
    Model& model = f.model();
    Mesh& mesh = model.meshes[0];
    SoftBody made;
    made.id = model.physics.allocateId();
    made.name = "Jelly";
    made.meshes = {0};
    made.detail = detail;
    made.copyAll = copyAll;
    const std::span<u32> bodyOf =
        mesh.attributes.getOrCreate<u32>(geom::names::kSoftBody, geom::Domain::Vertex, geom::AttrType::U32);
    const std::span<f32> softness =
        mesh.attributes.getOrCreate<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex, geom::AttrType::F32);
    for (u32 v = 0; v < bodyOf.size(); ++v) {
        bodyOf[v] = made.id;
        softness[v] = 1.0f;
    }
    model.physics.soft.push_back(made);
    SoftBody& body = model.physics.soft.back();
    MakeSoftLumps(model, body, cap, false);
    return body;
}

} // namespace softfig
