// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/mdx/loadable.h>
#include <whiteout/models/mdx/prune.h>
#include <whiteout/models/mdx/track_walk.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <unordered_map>

namespace whiteout {
namespace mdx {

namespace {

constexpr u32 kNoGlobalSequence = Track<f32>::kNoGlobalSequence;
constexpr u32 kNone = 0xFFFFFFFFu;

std::string TagText(u32 tag) {
    return std::string(reinterpret_cast<const char*>(&tag), 4);
}

/// Whether every float of @p value is a number. A key's value is a float, a
/// vector or a quaternion, each a run of floats.
template <class T>
bool Finite(const T& value) {
    f32 floats[sizeof(T) / sizeof(f32)];
    std::memcpy(floats, &value, sizeof(floats));
    return std::all_of(std::begin(floats), std::end(floats),
                       [](f32 f) { return std::isfinite(f); });
}

/// `makeLoadable`'s one pass over a model.
class Guard {
public:
    explicit Guard(Model& model) : model_(model) {}

    LoadReport run() {
        fixTracks();
        fixEvents();
        fixEmitterPools();
        fixObjectOrder();
        fixParents();
        fixGeosetAnimations();
        fixMaterials();
        fixTextures();
        fixAttachments();
        fixCollisionBoxes();
        reportLimits();
        return std::move(report_);
    }

private:
    void note(LoadRule rule, const std::string& where, u32 count = 1, bool repaired = true) {
        for (LoadFinding& finding : report_.findings) {
            if (finding.rule == rule) {
                finding.count += count;
                return;
            }
        }
        report_.findings.push_back({rule, count, repaired, where});
    }

    // ---- tracks -------------------------------------------------------------

    void fixTracks() {
        forEachTrack(model_, [this](u32 tag, auto& track) { fixTrack(tag, track); });
    }

    template <class T>
    void fixTrack(u32 tag, Track<T>& track) {
        if (!track.isUsed) {
            return;
        }
        const std::string where = TagText(tag);
        if (static_cast<u32>(track.interpolationType) >
            static_cast<u32>(InterpolationType::Bezier)) {
            track.interpolationType = InterpolationType::Linear;
            note(LoadRule::TrackInterpolation, where);
        }
        const std::size_t perKey = isSmoothInterpolation(track.interpolationType) ? 3 : 1;
        std::size_t keys = std::min(track.timestamps.size(), track.keys_data.size() / perKey);
        if (track.keyCount != keys || track.timestamps.size() != keys ||
            track.keys_data.size() != keys * perKey) {
            track.timestamps.resize(keys);
            track.keys_data.resize(keys * perKey);
            note(LoadRule::TrackKeyCount, where);
        }

        if constexpr (std::is_same_v<T, u32>) {
            if (perKey == 3) {
                for (std::size_t k = 0; k < keys; ++k) {
                    track.keys_data[k] = track.keys_data[k * 3];
                }
                track.keys_data.resize(keys);
                track.interpolationType = InterpolationType::None;
                note(LoadRule::IntegerTrackTangents, where);
            }
        } else {
            std::size_t kept = 0;
            for (std::size_t k = 0; k < keys; ++k) {
                bool finite = true;
                for (std::size_t v = 0; v < perKey; ++v) {
                    finite = finite && Finite(track.keys_data[k * perKey + v]);
                }
                if (!finite) {
                    continue;
                }
                if (kept != k) {
                    track.timestamps[kept] = track.timestamps[k];
                    for (std::size_t v = 0; v < perKey; ++v) {
                        track.keys_data[kept * perKey + v] = track.keys_data[k * perKey + v];
                    }
                }
                ++kept;
            }
            if (kept != keys) {
                note(LoadRule::NonFiniteKey, where, static_cast<u32>(keys - kept));
                keys = kept;
                track.timestamps.resize(keys);
                track.keys_data.resize(keys * perKey);
            }
        }
        track.keyCount = keys;

        if (track.globalSequenceId != kNoGlobalSequence &&
            track.globalSequenceId >= model_.globalSequences.size()) {
            track.globalSequenceId = kNoGlobalSequence;
            note(LoadRule::GlobalSequenceRange, where);
        }
        if (keys == 0) {
            track = Track<T>{};
            note(LoadRule::EmptyTrack, where);
        }
    }

    // ---- event objects ------------------------------------------------------

    void fixEvents() {
        bool keyless = false;
        for (EventObject& event : model_.eventObjects) {
            if (event.globalSequenceId != kNoGlobalSequence &&
                event.globalSequenceId >= model_.globalSequences.size()) {
                event.globalSequenceId = kNoGlobalSequence;
                note(LoadRule::GlobalSequenceRange, event.node.name);
            }
            keyless = keyless || event.eventTrackTimes.empty();
        }
        if (!keyless) {
            return;
        }
        std::vector<EventObject> kept;
        std::vector<Helper> held;
        for (EventObject& event : model_.eventObjects) {
            if (!event.eventTrackTimes.empty()) {
                kept.push_back(std::move(event));
                continue;
            }
            note(LoadRule::KeylessEvent, event.node.name);
            // What hangs from it still needs a node to hang from.
            if (isParent(event.node.objectId)) {
                Helper helper;
                helper.node = std::move(event.node);
                helper.node.type = Node::NodeType::Helper;
                helper.node.flags &= ~Node::NodeFlag::EventObject;
                held.push_back(std::move(helper));
            }
        }
        model_.eventObjects = std::move(kept);
        model_.helpers.insert(model_.helpers.end(), std::make_move_iterator(held.begin()),
                              std::make_move_iterator(held.end()));
    }

    bool isParent(u32 objectId) {
        bool found = false;
        forEachNode([&](Node& node) { found = found || node.parentId == objectId; });
        return found;
    }

    // ---- emitters -----------------------------------------------------------

    void fixEmitterPools() {
        bool clamped = false;
        const auto atLeastZero = [&clamped](f32& value) {
            if (!std::isfinite(value) || value < 0.0f) {
                value = 0.0f;
                clamped = true;
            }
        };
        const auto keysAtLeastZero = [&atLeastZero](Track<f32>& track) {
            const std::size_t perKey = isSmoothInterpolation(track.interpolationType) ? 3 : 1;
            for (std::size_t k = 0; k * perKey < track.keys_data.size(); ++k) {
                atLeastZero(track.keys_data[k * perKey]);
            }
        };
        const auto said = [&](const Node& node) {
            if (clamped) {
                note(LoadRule::EmitterPool, node.name);
            }
            clamped = false;
        };
        for (ParticleEmitter2& emitter : model_.particleEmitters2) {
            atLeastZero(emitter.emissionRate);
            atLeastZero(emitter.lifespan);
            keysAtLeastZero(emitter.emissionRateTracks);
            said(emitter.node);
        }
        for (ParticleEmitter& emitter : model_.particleEmitters) {
            atLeastZero(emitter.emissionRate);
            atLeastZero(emitter.lifespan);
            keysAtLeastZero(emitter.emissionRateTracks);
            keysAtLeastZero(emitter.lifespanTracks);
            said(emitter.node);
        }
    }

    // ---- nodes --------------------------------------------------------------

    /// Every node, in the order the game reads the node chunks: which is the
    /// order it numbers them in, whatever id each one stores.
    template <class Visit>
    void forEachNode(Visit&& visit) {
        const auto each = [&visit](auto& records) {
            for (auto& record : records) {
                visit(record.node);
            }
        };
        each(model_.bones);
        each(model_.lights);
        each(model_.helpers);
        each(model_.attachments);
        each(model_.particleEmitters);
        each(model_.particleEmitters2);
        each(model_.cornEmitters);
        each(model_.ribbonEmitters);
        each(model_.eventObjects);
        each(model_.collisionShapes);
    }

    void fixObjectOrder() {
        nodes_.clear();
        forEachNode([this](Node& node) { nodes_.push_back(&node); });
        const u32 count = static_cast<u32>(nodes_.size());

        u32 moved = 0;
        std::string first;
        for (u32 i = 0; i < count; ++i) {
            if (nodes_[i]->objectId != i) {
                if (moved++ == 0) {
                    first = nodes_[i]->name;
                }
            }
        }
        if (moved != 0) {
            // Where two nodes store one id, what names that id means the first.
            std::unordered_map<u32, u32> readAs;
            for (u32 i = 0; i < count; ++i) {
                readAs.try_emplace(nodes_[i]->objectId, i);
            }

            std::vector<Vector3f> pivots(count, Vector3f(0.0f, 0.0f, 0.0f));
            for (u32 i = 0; i < count; ++i) {
                if (nodes_[i]->objectId < model_.pivotPoints.size()) {
                    pivots[i] = model_.pivotPoints[nodes_[i]->objectId];
                }
            }
            // `BPOS` is a frame an object id and then one a camera.
            if (!model_.bindPoses.empty()) {
                const std::size_t cameras =
                    std::min(model_.cameras.size(), model_.bindPoses.size());
                const std::size_t nodeFrames = model_.bindPoses.size() - cameras;
                const std::array<f32, 12> identity{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
                std::vector<std::array<f32, 12>> frames(count, identity);
                for (u32 i = 0; i < count; ++i) {
                    if (nodes_[i]->objectId < nodeFrames) {
                        frames[i] = model_.bindPoses[nodes_[i]->objectId];
                    }
                }
                frames.insert(frames.end(),
                              model_.bindPoses.begin() + static_cast<std::ptrdiff_t>(nodeFrames),
                              model_.bindPoses.end());
                model_.bindPoses = std::move(frames);
            }
            for (Node* node : nodes_) {
                if (node->parentId == Node::NO_PARENT) {
                    continue;
                }
                const auto parent = readAs.find(node->parentId);
                if (parent == readAs.end()) {
                    node->parentId = Node::NO_PARENT;
                    note(LoadRule::ParentRange, node->name);
                } else {
                    node->parentId = parent->second;
                }
            }
            for (Geoset& geoset : model_.geosets) {
                for (u32& matrix : geoset.matrixIndices) {
                    const auto bone = readAs.find(matrix);
                    if (bone != readAs.end()) {
                        matrix = bone->second;
                    }
                }
            }
            for (u32 i = 0; i < count; ++i) {
                nodes_[i]->objectId = i;
            }
            model_.pivotPoints = std::move(pivots);
            note(LoadRule::ObjectOrder, first, moved);
        }
        if (model_.pivotPoints.size() < count) {
            note(LoadRule::PivotCount,
                 std::to_string(model_.pivotPoints.size()) + " of " + std::to_string(count));
            model_.pivotPoints.resize(count, Vector3f(0.0f, 0.0f, 0.0f));
        }
    }

    /// After `fixObjectOrder`: a node's id is its place in `nodes_`.
    void fixParents() {
        const u32 count = static_cast<u32>(nodes_.size());
        for (Node* node : nodes_) {
            if (node->parentId != Node::NO_PARENT && node->parentId >= count) {
                node->parentId = Node::NO_PARENT;
                note(LoadRule::ParentRange, node->name);
            }
        }
        for (u32 i = 0; i < count; ++i) {
            u32 at = i;
            for (u32 steps = 0; steps < count; ++steps) {
                const u32 parent = nodes_[at]->parentId;
                if (parent == Node::NO_PARENT) {
                    break;
                }
                if (parent == i) {
                    nodes_[at]->parentId = Node::NO_PARENT;
                    note(LoadRule::ParentCycle, nodes_[at]->name);
                    break;
                }
                at = parent;
            }
        }
    }

    // ---- indices ------------------------------------------------------------

    void fixGeosetAnimations() {
        std::vector<u8> remove(model_.geosetAnimations.size(), 0);
        for (std::size_t r = 0; r < remove.size(); ++r) {
            if (model_.geosetAnimations[r].geosetId >= model_.geosets.size()) {
                remove[r] = 1;
                note(LoadRule::GeosetAnimationTarget,
                     "geoset " + std::to_string(model_.geosetAnimations[r].geosetId));
            }
        }
        removeGeosetAnimations(model_, remove);
        for (Bone& bone : model_.bones) {
            if (bone.geosetAnimationId != kNone &&
                bone.geosetAnimationId >= model_.geosetAnimations.size()) {
                bone.geosetAnimationId = kNone;
                note(LoadRule::GeosetAnimationRange, bone.node.name);
            }
        }
    }

    /// A texture with no file, which is how a file says a stage has none.
    u32 blankTexture() {
        if (blankTexture_ == kNone) {
            blankTexture_ = static_cast<u32>(model_.textures.size());
            model_.textures.push_back(Texture{});
        }
        return blankTexture_;
    }

    /// One layer drawing the blank texture: what a geoset with nothing said
    /// about it is drawn with.
    u32 blankMaterial() {
        if (blankMaterial_ == kNone) {
            Layer layer;
            if (model_.version >= 1100) {
                Layer::SubTexture sub;
                sub.textureId = blankTexture();
                layer.subTextures.push_back(sub);
            } else {
                layer.textureId = blankTexture();
            }
            layer.textureAnimationId = kNone;
            Material material;
            material.layers.push_back(std::move(layer));
            blankMaterial_ = static_cast<u32>(model_.materials.size());
            model_.materials.push_back(std::move(material));
        }
        return blankMaterial_;
    }

    void fixMaterials() {
        for (std::size_t g = 0; g < model_.geosets.size(); ++g) {
            if (model_.geosets[g].materialId >= model_.materials.size()) {
                note(LoadRule::MaterialRange, "geoset " + std::to_string(g));
                model_.geosets[g].materialId = blankMaterial();
            }
        }
        for (RibbonEmitter& ribbon : model_.ribbonEmitters) {
            if (ribbon.materialId >= model_.materials.size()) {
                note(LoadRule::MaterialRange, ribbon.node.name);
                ribbon.materialId = blankMaterial();
            }
        }
        for (Material& material : model_.materials) {
            for (Layer& layer : material.layers) {
                if (layer.textureAnimationId != kNone &&
                    layer.textureAnimationId >= model_.textureAnimations.size()) {
                    layer.textureAnimationId = kNone;
                    note(LoadRule::TextureAnimationRange, "a layer");
                }
            }
        }
        for (std::size_t g = 0; g < model_.geosets.size(); ++g) {
            const Geoset& geoset = model_.geosets[g];
            const std::size_t sets = std::max<std::size_t>(1, geoset.textureCoordinateSets.size());
            for (Layer& layer : model_.materials[geoset.materialId].layers) {
                if (layer.coordId >= sets) {
                    layer.coordId = 0;
                    note(LoadRule::UvSetRange, "geoset " + std::to_string(g));
                }
            }
        }
    }

    /// After `fixMaterials`, which may have made the blank material.
    void fixTextures() {
        const auto named = [this](u32& id, const std::string& where) {
            if (id >= model_.textures.size()) {
                note(LoadRule::TextureRange, where);
                id = blankTexture();
            }
        };
        const auto keyed = [&named](Track<u32>& track, const std::string& where) {
            if (!track.isUsed) {
                return;
            }
            const std::size_t perKey = isSmoothInterpolation(track.interpolationType) ? 3 : 1;
            for (std::size_t k = 0; k * perKey < track.keys_data.size(); ++k) {
                named(track.keys_data[k * perKey], where);
            }
        };
        // Which of a layer's texture fields the file holds is its version's.
        for (std::size_t m = 0; m < model_.materials.size(); ++m) {
            const std::string where = "material " + std::to_string(m);
            for (Layer& layer : model_.materials[m].layers) {
                if (model_.version >= 1100) {
                    for (Layer::SubTexture& sub : layer.subTextures) {
                        named(sub.textureId, where);
                        keyed(sub.tracks, where);
                    }
                } else {
                    named(layer.textureId, where);
                    keyed(layer.textureIdTracks, where);
                }
            }
        }
        for (ParticleEmitter2& emitter : model_.particleEmitters2) {
            named(emitter.textureId, emitter.node.name);
        }
    }

    // ---- the rest -----------------------------------------------------------

    void fixAttachments() {
        std::vector<Attachment>& attachments = model_.attachments;
        const auto unordered = std::adjacent_find(attachments.begin(), attachments.end(),
                                                  [](const Attachment& a, const Attachment& b) {
                                                      return a.attachmentId >= b.attachmentId;
                                                  });
        if (unordered == attachments.end()) {
            return;
        }
        note(LoadRule::AttachmentIds, std::next(unordered)->node.name);
        for (std::size_t a = 0; a < attachments.size(); ++a) {
            attachments[a].attachmentId = static_cast<u32>(a);
        }
    }

    void fixCollisionBoxes() {
        for (CollisionShape& shape : model_.collisionShapes) {
            if (shape.type != CollisionShape::ShapeType::Box || shape.vertices.size() < 2) {
                continue;
            }
            Vector3f& lo = shape.vertices[0];
            Vector3f& hi = shape.vertices[1];
            if (lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z) {
                continue;
            }
            const Vector3f a = lo;
            const Vector3f b = hi;
            lo = Vector3f(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
            hi = Vector3f(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
            note(LoadRule::BoxCorners, shape.node.name);
        }
    }

    void reportLimits() {
        if (model_.sequences.size() > 255) {
            note(LoadRule::SequenceCount, std::to_string(model_.sequences.size()), 1, false);
        }
        for (std::size_t g = 0; g < model_.geosets.size(); ++g) {
            if (model_.geosets[g].vertexPositions.size() > 0xFFFFu) {
                note(LoadRule::GeosetVertexCount,
                     "geoset " + std::to_string(g) + ", " +
                         std::to_string(model_.geosets[g].vertexPositions.size()) + " vertices",
                     1, false);
            }
        }
    }

    Model& model_;
    LoadReport report_;
    std::vector<Node*> nodes_; ///< In read order, from `fixObjectOrder` on.
    u32 blankTexture_ = kNone;
    u32 blankMaterial_ = kNone;
};

} // namespace

bool LoadReport::repaired() const {
    return std::any_of(findings.begin(), findings.end(),
                       [](const LoadFinding& finding) { return finding.repaired; });
}

const LoadFinding* LoadReport::find(LoadRule rule) const {
    for (const LoadFinding& finding : findings) {
        if (finding.rule == rule) {
            return &finding;
        }
    }
    return nullptr;
}

std::string describe(const LoadFinding& finding) {
    const std::string n = std::to_string(finding.count);
    const std::string first = " (first: " + finding.first + ")";
    switch (finding.rule) {
    case LoadRule::TrackKeyCount:
        return n +
               " track(s) held a key count their key arrays did not; cut to the keys they hold" +
               first;
    case LoadRule::TrackInterpolation:
        return n + " track(s) had an interpolation type the file has no number for; made linear" +
               first;
    case LoadRule::IntegerTrackTangents:
        return n +
               " integer track(s) were keyed with tangents, which Warcraft III reads no bytes "
               "for; made stepped" +
               first;
    case LoadRule::NonFiniteKey:
        return n + " key(s) held a value that is not a number; left out" + first;
    case LoadRule::EmptyTrack:
        return n + " track(s) had no keys, which Warcraft III fails a model on; left out" + first;
    case LoadRule::GlobalSequenceRange:
        return n +
               " track(s) ran on a global sequence the model does not have; put on the "
               "sequences' clock" +
               first;
    case LoadRule::KeylessEvent:
        return n +
               " event object(s) fired at no time, which Warcraft III fails a model on; left "
               "out, or kept as a helper where a node hangs from one" +
               first;
    case LoadRule::ObjectOrder:
        return n +
               " object id(s) were not the order Warcraft III reads the nodes in, which is "
               "how it numbers them; renumbered" +
               first;
    case LoadRule::ParentRange:
        return n + " node(s) named a parent that is no node; made roots" + first;
    case LoadRule::ParentCycle:
        return n + " node(s) closed a loop of parents; made roots" + first;
    case LoadRule::PivotCount:
        return "the model held fewer pivot points than nodes; the missing ones are the origin" +
               first;
    case LoadRule::MaterialRange:
        return n +
               " geoset(s) or ribbon emitter(s) named a material the model does not have; "
               "given a blank one" +
               first;
    case LoadRule::TextureRange:
        return n +
               " layer(s) or particle emitter(s) named a texture the model does not have; "
               "given a blank one" +
               first;
    case LoadRule::TextureAnimationRange:
        return n + " layer(s) named a texture animation the model does not have; unlinked";
    case LoadRule::UvSetRange:
        return n + " layer(s) read a UV set their geoset does not hold; they read the first" +
               first;
    case LoadRule::GeosetAnimationTarget:
        return n + " geoset animation(s) named a geoset the model does not have; left out" + first;
    case LoadRule::GeosetAnimationRange:
        return n + " bone(s) named a geoset animation the model does not have; unlinked" + first;
    case LoadRule::AttachmentIds:
        return "the attachment ids did not climb through the file, which Warcraft III needs to "
               "find them all; numbered by place" +
               first;
    case LoadRule::EmitterPool:
        return n +
               " particle emitter(s) had a negative emission rate or lifespan; written as 0, "
               "which is all Warcraft III can take" +
               first;
    case LoadRule::BoxCorners:
        return n + " collision box(es) had their corners out of order; ordered" + first;
    case LoadRule::SequenceCount:
        return finding.first + " sequences: Warcraft III counts them in a byte, and plays none "
                               "past the 255th. Not repaired";
    case LoadRule::GeosetVertexCount:
        return n +
               " geoset(s) hold more than 65535 vertices, which Warcraft III builds no "
               "geoset for. Not repaired" +
               first;
    }
    return {};
}

LoadReport makeLoadable(Model& model) {
    return Guard(model).run();
}

} // namespace mdx
} // namespace whiteout
