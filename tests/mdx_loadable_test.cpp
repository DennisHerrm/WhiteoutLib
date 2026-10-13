// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

/// What Warcraft III 3.0's loader requires of an `.mdx`, on hand-built models.
///
/// A corpus sweep cannot say any of this: no shipped file holds a collision
/// cylinder, a track of no keys or a geoset past the index width, and each one
/// fails the whole model in the game. So every case here states a rule the
/// client's reader has and asserts the BYTES the writer produces for it, read
/// with a walk of its own rather than through `mdx::Parser`: a layout mistake
/// the writer and the parser share reads back as correct.

#include <catch2/catch_test_macros.hpp>

#include <whiteout/models/mdx/loadable.h>
#include <whiteout/models/mdx/parser.h>
#include <whiteout/models/mdx/prune.h>
#include <whiteout/models/mdx/structures.h>
#include <whiteout/models/mdx/track_walk.h>
#include <whiteout/models/mdx/types.h>
#include <whiteout/models/mdx/writer.h>

#include "whiteout/models/mdx/mdl_converter.h"
#include "whiteout/models/mdx/mdl_writer.h"

#include "wem_corpus_files.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <vector>

using namespace whiteout;
using namespace whiteout::mdx;

namespace {

u32 U32At(std::span<const u8> bytes, std::size_t at) {
    u32 value = 0;
    REQUIRE(at + 4 <= bytes.size());
    std::memcpy(&value, bytes.data() + at, 4);
    return value;
}

f32 F32At(std::span<const u8> bytes, std::size_t at) {
    f32 value = 0;
    REQUIRE(at + 4 <= bytes.size());
    std::memcpy(&value, bytes.data() + at, 4);
    return value;
}

/// The body of the top-level chunk @p tag, empty when the file has none.
std::span<const u8> ChunkOf(std::span<const u8> file, u32 tag) {
    std::size_t at = 4; // past "MDLX"
    while (at + 8 <= file.size()) {
        const u32 size = U32At(file, at + 4);
        if (U32At(file, at) == tag) {
            return file.subspan(at + 8, size);
        }
        at += 8 + size;
    }
    return {};
}

CollisionShape Shape(const char* name, u32 objectId, CollisionShape::ShapeType type,
                     std::vector<Vector3f> vertices, f32 radius = 0.0f) {
    CollisionShape shape;
    shape.node.name = name;
    shape.node.objectId = objectId;
    shape.node.flags = Node::NodeFlag::CollisionShape;
    shape.type = type;
    shape.vertices = std::move(vertices);
    shape.radius = radius;
    return shape;
}

/// A node with no tracks is 96 bytes: its size, a name of 80, three words.
constexpr std::size_t kBareNode = 96;

} // namespace

// ============================================================================
// CLID (MDL::ReadBinCollisions, 0x1404047D0)
// ============================================================================

TEST_CASE("mdx a collision shape is written in the bytes the game reads", "[mdx][loadable][clid]") {
    Model model;
    using Type = CollisionShape::ShapeType;
    model.collisionShapes = {
        Shape("Box", 0, Type::Box, {Vector3f(-1, -2, -3), Vector3f(1, 2, 3)}),
        Shape("Cylinder", 1, Type::Cylinder, {Vector3f(4, 5, 10), Vector3f(4, 5, 50)}, 8.0f),
        Shape("Sphere", 2, Type::Sphere, {Vector3f(1, 2, 3)}, 4.0f),
        Shape("Plane", 3, Type::Plane, {Vector3f(0, 0, 0), Vector3f(64, 32, 0)}),
    };
    model.pivotPoints.assign(4, Vector3f(0, 0, 0));

    Writer writer;
    const std::vector<u8> file = writer.write(model);
    const std::span<const u8> clid = ChunkOf(file, CLID_TAG);
    // Box 24, cylinder 20, sphere 16, plane 8, each after a node and a type.
    REQUIRE(clid.size() == 4 * (kBareNode + 4) + 24 + 20 + 16 + 8);

    std::size_t at = 0;
    const auto floats = [&](std::size_t count) {
        std::vector<f32> out;
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back(F32At(clid, at + 4 * i));
        }
        at += 4 * count;
        return out;
    };
    const auto type = [&] {
        REQUIRE(U32At(clid, at) == kBareNode);
        at += kBareNode;
        at += 4;
        return U32At(clid, at - 4);
    };

    CHECK(type() == 0);
    CHECK(floats(6) == std::vector<f32>{-1, -2, -3, 1, 2, 3});
    // A cylinder is 1, not the 3 other tools give it: base, height, radius.
    CHECK(type() == 1);
    CHECK(floats(5) == std::vector<f32>{4, 5, 10, 40, 8});
    CHECK(type() == 2);
    CHECK(floats(4) == std::vector<f32>{1, 2, 3, 4});
    // And a plane is 3: its two spans and nothing else.
    CHECK(type() == 3);
    CHECK(floats(2) == std::vector<f32>{64, 32});
    CHECK(at == clid.size());

    // Read back, each shape is what was written.
    Parser parser;
    const Model read = parser.parse(file);
    REQUIRE(read.collisionShapes.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        INFO("shape " << i);
        CHECK(read.collisionShapes[i].type == model.collisionShapes[i].type);
        CHECK(read.collisionShapes[i].vertices == model.collisionShapes[i].vertices);
        CHECK(read.collisionShapes[i].radius == model.collisionShapes[i].radius);
    }
    CHECK_FALSE(parser.hasIssues());
}

TEST_CASE("mdx a shape with the wrong number of vertices still writes its type's bytes",
          "[mdx][loadable][clid]") {
    // The chunk has no sizes: one vertex too many puts the game's reader in the
    // middle of the next shape, and the model does not load.
    Model model;
    using Type = CollisionShape::ShapeType;
    model.collisionShapes = {
        Shape("Short", 0, Type::Box, {Vector3f(1, 2, 3)}),
        Shape("Long", 1, Type::Box, {Vector3f(1, 1, 1), Vector3f(2, 2, 2), Vector3f(3, 3, 3)}),
        Shape("Flat", 2, Type::Cylinder, {Vector3f(0, 0, 7)}, 2.0f),
        Shape("Bare", 3, Type::Sphere, {}, 5.0f),
    };

    Writer writer;
    const std::vector<u8> file = writer.write(model);
    const std::span<const u8> clid = ChunkOf(file, CLID_TAG);
    REQUIRE(clid.size() == 4 * (kBareNode + 4) + 24 + 24 + 20 + 16);

    Parser parser;
    const Model read = parser.parse(file);
    REQUIRE(read.collisionShapes.size() == 4);
    CHECK(read.collisionShapes[0].vertices ==
          std::vector<Vector3f>{Vector3f(1, 2, 3), Vector3f(0, 0, 0)});
    CHECK(read.collisionShapes[1].vertices ==
          std::vector<Vector3f>{Vector3f(1, 1, 1), Vector3f(2, 2, 2)});
    // One end is a cylinder of no height.
    CHECK(read.collisionShapes[2].vertices ==
          std::vector<Vector3f>{Vector3f(0, 0, 7), Vector3f(0, 0, 7)});
    CHECK(read.collisionShapes[3].radius == 5.0f);
    CHECK(read.collisionShapes[3].node.name == "Bare");
}

TEST_CASE("mdx a collision type the game does not know reads no bytes", "[mdx][loadable][clid]") {
    // `ReadBinCollisions` has no case for a type past 3: it reads nothing and
    // goes on to the next shape.
    Model model;
    using Type = CollisionShape::ShapeType;
    model.collisionShapes = {
        Shape("Odd", 0, Type::Box, {Vector3f(9, 9, 9), Vector3f(9, 9, 9)}),
        Shape("Next", 1, Type::Sphere, {Vector3f(1, 2, 3)}, 4.0f),
    };
    Writer writer;
    std::vector<u8> file = writer.write(model);

    // Make the first shape a type 7 with nothing after it.
    const std::span<const u8> clid = ChunkOf(file, CLID_TAG);
    const std::size_t body = static_cast<std::size_t>(clid.data() - file.data());
    const std::size_t typeAt = body + kBareNode;
    const u32 seven = 7;
    std::memcpy(file.data() + typeAt, &seven, 4);
    file.erase(file.begin() + static_cast<std::ptrdiff_t>(typeAt + 4),
               file.begin() + static_cast<std::ptrdiff_t>(typeAt + 4 + 24));
    const u32 size = static_cast<u32>(clid.size() - 24);
    std::memcpy(file.data() + body - 4, &size, 4);

    Parser parser;
    const Model read = parser.parse(file);
    REQUIRE(read.collisionShapes.size() == 2);
    CHECK(read.collisionShapes[0].node.name == "Odd");
    CHECK(read.collisionShapes[0].type == Type::Box);
    CHECK(read.collisionShapes[0].vertices ==
          std::vector<Vector3f>{Vector3f(0, 0, 0), Vector3f(0, 0, 0)});
    CHECK(read.collisionShapes[1].node.name == "Next");
    CHECK(read.collisionShapes[1].type == Type::Sphere);
    CHECK(read.collisionShapes[1].vertices == std::vector<Vector3f>{Vector3f(1, 2, 3)});
    CHECK(read.collisionShapes[1].radius == 4.0f);
    CHECK(parser.hasIssues());
}

TEST_CASE("mdl a plane is its length and its width", "[mdx][loadable][clid][mdl]") {
    Model model;
    using Type = CollisionShape::ShapeType;
    model.collisionShapes = {
        Shape("Floor", 0, Type::Plane, {Vector3f(0, 0, 0), Vector3f(64, 32, 0)}),
        Shape("Post", 1, Type::Cylinder, {Vector3f(0, 0, 10), Vector3f(0, 0, 50)}, 8.0f),
    };

    // As `MDL::WriteCollisions` spells them.
    const std::string text = writeModelToMdl(model);
    CHECK(text.find("Plane,") != std::string::npos);
    CHECK(text.find("Length 64") != std::string::npos);
    CHECK(text.find("Width 32") != std::string::npos);
    CHECK(text.find("Vertices 2") != std::string::npos); // the cylinder's
    CHECK(text.find("BoundsRadius 8") != std::string::npos);

    std::vector<std::string> issues;
    const Model read = convertMdlToModel(text, issues);
    REQUIRE(read.collisionShapes.size() == 2);
    CHECK(read.collisionShapes[0].type == Type::Plane);
    CHECK(read.collisionShapes[0].vertices == model.collisionShapes[0].vertices);
    CHECK(read.collisionShapes[1].type == Type::Cylinder);
    CHECK(read.collisionShapes[1].vertices == model.collisionShapes[1].vertices);
    CHECK(read.collisionShapes[1].radius == 8.0f);

    // The two corners other writers give a plane are taken as they stand.
    const Model other = convertMdlToModel(R"(
CollisionShape "Floor" {
	ObjectId 0,
	Plane,
	Vertices 2 {
		{ -4, -2, 0 },
		{ 4, 2, 0 },
	}
}
)",
                                          issues);
    REQUIRE(other.collisionShapes.size() == 1);
    CHECK(other.collisionShapes[0].vertices ==
          std::vector<Vector3f>{Vector3f(-4, -2, 0), Vector3f(4, 2, 0)});
}

// ============================================================================
// The loader's guard (mdx::makeLoadable)
// ============================================================================

namespace {

template <class T>
Track<T> Keyed(std::vector<u32> times, std::vector<T> values,
               InterpolationType interp = InterpolationType::Linear) {
    Track<T> track;
    track.isUsed = true;
    track.interpolationType = interp;
    track.keyCount = times.size();
    track.timestamps = std::move(times);
    track.keys_data = std::move(values);
    return track;
}

Node NodeOf(const char* name, u32 objectId, u32 parentId = Node::NO_PARENT) {
    Node node;
    node.name = name;
    node.objectId = objectId;
    node.parentId = parentId;
    return node;
}

/// One triangle drawn with material @p material.
Geoset Triangle(u32 material) {
    Geoset geoset;
    geoset.vertexPositions = {Vector3f(0, 0, 0), Vector3f(1, 0, 0), Vector3f(0, 1, 0)};
    geoset.vertexNormals.assign(3, Vector3f(0, 0, 1));
    geoset.textureCoordinateSets.push_back({Vector2f(0, 0), Vector2f(1, 0), Vector2f(0, 1)});
    geoset.faceTypeGroups = {4};
    geoset.faceGroups = {3};
    geoset.faces = {0, 1, 2};
    geoset.vertexGroups = {0, 0, 0};
    geoset.matrixGroups = {1};
    geoset.matrixIndices = {0};
    geoset.materialId = material;
    return geoset;
}

/// A model the game loads as it is: one bone, one textured triangle.
Model SoundModel(u32 version = 800) {
    Model model;
    model.version = version;
    model.modelName = "sound";
    model.textures.push_back(Texture{0, "a.blp", Texture::Flag::None});
    Layer layer;
    layer.textureAnimationId = 0xFFFFFFFFu;
    if (version >= 1100) {
        layer.subTextures.push_back(Layer::SubTexture{});
    }
    Material material;
    material.layers.push_back(layer);
    model.materials.push_back(material);
    Bone bone;
    bone.node = NodeOf("root", 0);
    model.bones.push_back(bone);
    model.pivotPoints = {Vector3f(0, 0, 0)};
    model.geosets.push_back(Triangle(0));
    return model;
}

u32 Count(const LoadReport& report, LoadRule rule) {
    const LoadFinding* finding = report.find(rule);
    return finding != nullptr ? finding->count : 0u;
}

std::size_t UsedTracks(const Model& model) {
    std::size_t used = 0;
    forEachTrack(model, [&used](u32, const auto& track) { used += track.isUsed ? 1 : 0; });
    return used;
}

} // namespace

TEST_CASE("mdx the track walk is the list the writer writes and the parser reads",
          "[mdx][loadable][tracks]") {
    // One list for the writer, the guard and the pruning. A track the walk
    // left out would be one the guard never checked.
    for (const u32 version : {800u, 1000u, 1800u}) {
        CAPTURE(version);
        Model model = SoundModel(version);
        model.globalSequences = {1000};
        model.soundEmitters.emplace_back().node = NodeOf("snem", 0);
        model.textureAnimations.emplace_back();
        model.geosetAnimations.emplace_back();
        model.lights.emplace_back().node = NodeOf("light", 1);
        model.helpers.emplace_back().node = NodeOf("helper", 2);
        model.attachments.emplace_back().node = NodeOf("attachment", 3);
        model.particleEmitters.emplace_back().node = NodeOf("prem", 4);
        model.particleEmitters2.emplace_back().node = NodeOf("pre2", 5);
        model.cornEmitters.emplace_back().node = NodeOf("corn", 6);
        model.ribbonEmitters.emplace_back().node = NodeOf("ribbon", 7);
        model.cameras.emplace_back().name = "camera";
        model.eventObjects.emplace_back().node = NodeOf("SNDxEVNT", 8);
        model.eventObjects.back().eventTrackTimes = {0};
        model.collisionShapes.push_back(Shape("box", 9, CollisionShape::ShapeType::Box,
                                              {Vector3f(0, 0, 0), Vector3f(1, 1, 1)}));
        model.pivotPoints.assign(10, Vector3f(0, 0, 0));

        // Every track the file holds at this version, keyed once.
        std::size_t visited = 0;
        forEachTrack(model, [&visited](u32, auto& track) {
            track.isUsed = true;
            track.keyCount = 1;
            track.timestamps = {0};
            track.keys_data.assign(1, {});
            ++visited;
        });
        REQUIRE(visited > 60);

        Writer writer;
        const std::vector<u8> file = writer.write(model);
        CHECK(writer.lastReport().empty());
        Parser parser(Parser::UpgradeMode::PreserveOriginal);
        const Model read = parser.parse(file);
        CHECK_FALSE(parser.hasIssues());
        CHECK(UsedTracks(read) == visited);
    }
}

TEST_CASE("mdx a model with nothing to repair is written as it is", "[mdx][loadable]") {
    Model model = SoundModel();
    model.bones[0].node.translationTracks =
        Keyed<Vector3f>({0, 1000}, {Vector3f(0, 0, 0), Vector3f(0, 0, 5)});

    Model copy = model;
    CHECK(makeLoadable(copy).empty());

    Writer repairing;
    Writer raw;
    raw.setRepairs(false);
    CHECK(repairing.write(model) == raw.write(model));
    CHECK(repairing.lastReport().empty());
}

TEST_CASE("mdx a track with no keys is no track", "[mdx][loadable][tracks]") {
    // `ReadBinGenObject`: every key reader answers 0 for a count of 0, and a
    // chunk reader that answers 0 fails the whole model.
    Model model = SoundModel();
    model.bones[0].node.translationTracks.isUsed = true; // and nothing else
    // A count the arrays do not hold is cut to the keys they do.
    Track<Quaternion> rotation =
        Keyed<Quaternion>({0, 500}, {Quaternion{0, 0, 0, 1}, Quaternion{0, 0, 0, 1}});
    rotation.keyCount = 5;
    model.bones[0].node.rotationTracks = rotation;

    Writer writer;
    const std::vector<u8> file = writer.write(model);
    CHECK(Count(writer.lastReport(), LoadRule::EmptyTrack) == 1u);
    CHECK(Count(writer.lastReport(), LoadRule::TrackKeyCount) == 1u);
    // The bone: a node, a KGRT of two keys, and its two geoset words.
    const std::span<const u8> bone = ChunkOf(file, BONE_TAG);
    REQUIRE(bone.size() == kBareNode + 16 + 2 * (4 + 16) + 8);
    CHECK(U32At(bone, kBareNode) == KGRT_TAG);
    CHECK(U32At(bone, kBareNode + 4) == 2u);

    // The caller's model is its own.
    CHECK(model.bones[0].node.translationTracks.isUsed);
    CHECK(model.bones[0].node.rotationTracks.keyCount == 5u);

    // Told to write the model as given, the writer writes the empty track
    // and still no key it does not hold.
    writer.setRepairs(false);
    const std::vector<u8> raw = writer.write(model);
    const std::span<const u8> rawBone = ChunkOf(raw, BONE_TAG);
    REQUIRE(rawBone.size() == kBareNode + 16 + 16 + 2 * (4 + 16) + 8);
    CHECK(U32At(rawBone, kBareNode) == KGTR_TAG);
    CHECK(U32At(rawBone, kBareNode + 4) == 0u);
}

TEST_CASE("mdx an integer track is a time and a value a key", "[mdx][loadable][tracks]") {
    // `ReadBinUintKeyFrames` reads eight bytes a key whatever the header's
    // interpolation says, so tangents written for one put the layer out of
    // step: the reader does not even fail cleanly there.
    Model model = SoundModel();
    model.textures.push_back(Texture{0, "b.blp", Texture::Flag::None});
    model.materials[0].layers[0].textureIdTracks =
        Keyed<u32>({0, 500}, {0, 9, 9, 1, 9, 9}, InterpolationType::Hermite);

    const LoadReport report = makeLoadable(model);
    CHECK(Count(report, LoadRule::IntegerTrackTangents) == 1u);
    const Track<u32>& flipbook = model.materials[0].layers[0].textureIdTracks;
    CHECK(flipbook.interpolationType == InterpolationType::None);
    CHECK(flipbook.keys_data == std::vector<u32>{0, 1});
    CHECK(flipbook.keyCount == 2u);

    // And the parser reads one the way the game does: a header that claims
    // tangents over keys that hold none.
    Writer writer;
    std::vector<u8> file = writer.write(model);
    const u32 flip = KMTF_TAG;
    const u8* tag = reinterpret_cast<const u8*>(&flip);
    const auto at = std::search(file.begin(), file.end(), tag, tag + 4);
    REQUIRE(at != file.end());
    const u32 hermite = 2;
    std::memcpy(&*(at + 8), &hermite, 4);
    Parser parser;
    const Model read = parser.parse(file);
    CHECK(read.materials.at(0).layers.at(0).textureIdTracks.keys_data == std::vector<u32>{0, 1});
    CHECK(read.materials[0].layers[0].textureIdTracks.interpolationType == InterpolationType::None);
    CHECK(parser.hasIssues());
}

TEST_CASE("mdx a key that is not a number is left out", "[mdx][loadable][tracks]") {
    Model model = SoundModel();
    const f32 nan = std::numeric_limits<f32>::quiet_NaN();
    const f32 inf = std::numeric_limits<f32>::infinity();
    model.bones[0].node.translationTracks = Keyed<Vector3f>(
        {0, 100, 200}, {Vector3f(0, 0, 0), Vector3f(nan, 0, 0), Vector3f(0, 0, 2)});
    // A tangent counts: the game blends through it.
    model.bones[0].node.scalingTracks = Keyed<Vector3f>(
        {0}, {Vector3f(1, 1, 1), Vector3f(0, inf, 0), Vector3f(0, 0, 0)}, InterpolationType::Hermite);

    const LoadReport report = makeLoadable(model);
    CHECK(Count(report, LoadRule::NonFiniteKey) == 2u);
    CHECK(model.bones[0].node.translationTracks.timestamps == std::vector<u32>{0, 200});
    CHECK(model.bones[0].node.translationTracks.keys_data ==
          std::vector<Vector3f>{Vector3f(0, 0, 0), Vector3f(0, 0, 2)});
    // Left with no key, it is no track.
    CHECK_FALSE(model.bones[0].node.scalingTracks.isUsed);
    CHECK(Count(report, LoadRule::EmptyTrack) == 1u);
}

TEST_CASE("mdx an event object that fires at no time is left out", "[mdx][loadable][events]") {
    // `ReadBinEventObjects`: `KEVT` with a count of 0 fails the model, and
    // with no `KEVT` the reader takes the next object's size for a tag.
    Model model = SoundModel();
    EventObject fires;
    fires.node = NodeOf("SNDxFIRE", 1);
    fires.eventTrackTimes = {100};
    EventObject silent;
    silent.node = NodeOf("SNDxNONE", 2);
    EventObject holds;
    holds.node = NodeOf("SNDxHOLD", 3);
    holds.node.flags = Node::NodeFlag::EventObject;
    model.eventObjects = {fires, silent, holds};
    Helper under;
    under.node = NodeOf("under", 4, 3); // hangs from the third
    model.helpers.push_back(under);
    model.pivotPoints = {Vector3f(0, 0, 0), Vector3f(1, 0, 0), Vector3f(2, 0, 0),
                         Vector3f(3, 0, 0), Vector3f(4, 0, 0)};

    const LoadReport report = makeLoadable(model);
    CHECK(Count(report, LoadRule::KeylessEvent) == 2u);
    REQUIRE(model.eventObjects.size() == 1);
    CHECK(model.eventObjects[0].node.name == "SNDxFIRE");
    // What a node hangs from stays, as a helper, and the ids are once more
    // the order the game reads the nodes in: bone, the two helpers, the event.
    REQUIRE(model.helpers.size() == 2);
    CHECK(model.helpers[0].node.name == "under");
    CHECK(model.helpers[1].node.name == "SNDxHOLD");
    CHECK_FALSE(hasFlag(model.helpers[1].node.flags, Node::NodeFlag::EventObject));
    CHECK(model.helpers[0].node.objectId == 1u);
    CHECK(model.helpers[1].node.objectId == 2u);
    CHECK(model.helpers[0].node.parentId == 2u);
    CHECK(model.eventObjects[0].node.objectId == 3u);
    // Each pivot went with its node; the one left out took its own with it.
    CHECK(model.pivotPoints == std::vector<Vector3f>{Vector3f(0, 0, 0), Vector3f(4, 0, 0),
                                                     Vector3f(3, 0, 0), Vector3f(1, 0, 0)});

    // With none left there is no chunk.
    model.eventObjects.clear();
    Writer writer;
    CHECK(ChunkOf(writer.write(model), EVTS_TAG).empty());
}

TEST_CASE("mdx object ids are the order the nodes are read in", "[mdx][loadable][nodes]") {
    // The game ignores a node's stored id and numbers it as it reads it, so a
    // file whose ids say another order has every parent, pivot and skin entry
    // name another node.
    Model model = SoundModel(1000);
    model.bones.clear();
    Bone second;
    second.node = NodeOf("stored 1", 1);
    Bone first;
    first.node = NodeOf("stored 0", 0, 1);
    model.bones = {second, first};
    Helper helper;
    helper.node = NodeOf("stored 5", 5, 0);
    model.helpers.push_back(helper);
    model.pivotPoints = {Vector3f(10, 0, 0), Vector3f(11, 0, 0), Vector3f(0, 0, 0),
                         Vector3f(0, 0, 0),  Vector3f(0, 0, 0),  Vector3f(15, 0, 0)};
    model.geosets[0].matrixIndices = {1};
    model.cameras.emplace_back().name = "camera";
    const auto frame = [](f32 mark) {
        return std::array<f32, 12>{1, 0, 0, 0, 1, 0, 0, 0, 1, mark, 0, 0};
    };
    model.bindPoses = {frame(10), frame(11), frame(12), frame(13), frame(14), frame(15), frame(99)};

    const LoadReport report = makeLoadable(model);
    CHECK(Count(report, LoadRule::ObjectOrder) == 3u);
    CHECK(model.bones[0].node.objectId == 0u);
    CHECK(model.bones[1].node.objectId == 1u);
    CHECK(model.helpers[0].node.objectId == 2u);
    // "stored 0" was under "stored 1", and the helper under "stored 0".
    CHECK(model.bones[0].node.parentId == Node::NO_PARENT);
    CHECK(model.bones[1].node.parentId == 0u);
    CHECK(model.helpers[0].node.parentId == 1u);
    CHECK(model.pivotPoints ==
          std::vector<Vector3f>{Vector3f(11, 0, 0), Vector3f(10, 0, 0), Vector3f(15, 0, 0)});
    CHECK(model.geosets[0].matrixIndices == std::vector<u32>{0});
    // A frame a node, then the camera's.
    REQUIRE(model.bindPoses.size() == 4);
    CHECK(model.bindPoses[0][9] == 11.0f);
    CHECK(model.bindPoses[1][9] == 10.0f);
    CHECK(model.bindPoses[2][9] == 15.0f);
    CHECK(model.bindPoses[3][9] == 99.0f);
    CHECK(makeLoadable(model).empty());
}

TEST_CASE("mdx a parent is a node, and no node is its own ancestor", "[mdx][loadable][nodes]") {
    Model model = SoundModel();
    model.bones.clear();
    for (u32 b = 0; b < 4; ++b) {
        Bone bone;
        bone.node = NodeOf(("bone " + std::to_string(b)).c_str(), b);
        model.bones.push_back(bone);
    }
    model.pivotPoints.assign(4, Vector3f(0, 0, 0));
    model.bones[0].node.parentId = 40; // no such node
    model.bones[1].node.parentId = 2;  // 1 -> 2 -> 3 -> 1
    model.bones[2].node.parentId = 3;
    model.bones[3].node.parentId = 1;

    const LoadReport report = makeLoadable(model);
    CHECK(Count(report, LoadRule::ParentRange) == 1u);
    CHECK(Count(report, LoadRule::ParentCycle) == 1u);
    CHECK(model.bones[0].node.parentId == Node::NO_PARENT);
    // Cut where it closes: the link that leads back to the first of them.
    CHECK(model.bones[1].node.parentId == 2u);
    CHECK(model.bones[2].node.parentId == 3u);
    CHECK(model.bones[3].node.parentId == Node::NO_PARENT);
    CHECK(model.pivotPoints.size() == 4);

    // Fewer pivots than nodes is padded: the game reads one an id.
    model.pivotPoints.resize(2);
    CHECK(Count(makeLoadable(model), LoadRule::PivotCount) == 1u);
    CHECK(model.pivotPoints.size() == 4);
}

TEST_CASE("mdx an index names something the model has", "[mdx][loadable][indices]") {
    // The game follows each of these unchecked.
    for (const u32 version : {800u, 1800u}) {
        CAPTURE(version);
        Model model = SoundModel(version);
        model.globalSequences = {500};
        model.geosets.push_back(Triangle(7)); // no material 7
        if (version >= 1100) {
            model.materials[0].layers[0].subTextures[0].textureId = 9; // no texture 9
        } else {
            model.materials[0].layers[0].textureId = 9;
        }
        model.materials[0].layers[0].textureAnimationId = 3; // no texture animation
        model.materials[0].layers[0].coordId = 2;            // the geosets hold one UV set
        ParticleEmitter2 emitter;
        emitter.node = NodeOf("emitter", 1);
        emitter.textureId = 12;
        model.particleEmitters2.push_back(emitter);
        RibbonEmitter ribbon;
        ribbon.node = NodeOf("ribbon", 2);
        ribbon.materialId = 30;
        model.ribbonEmitters.push_back(ribbon);
        model.pivotPoints.assign(4, Vector3f(0, 0, 0));
        // Geoset animations: one for a geoset that is not there, one that is.
        GeosetAnimation stray;
        stray.geosetId = 50;
        GeosetAnimation real;
        real.geosetId = 1;
        real.alpha = 0.0f;
        model.geosetAnimations = {stray, real};
        model.bones[0].geosetId = 1;
        model.bones[0].geosetAnimationId = 1; // the real one, second in the file
        Bone lost;
        lost.node = NodeOf("lost", 3);
        lost.geosetId = 0;
        lost.geosetAnimationId = 20; // no such record
        model.bones.push_back(lost);
        model.bones[0].node.translationTracks = Keyed<Vector3f>({0}, {Vector3f(0, 0, 0)});
        model.bones[0].node.translationTracks.globalSequenceId = 4; // no such clock

        const LoadReport report = makeLoadable(model);
        CHECK(Count(report, LoadRule::MaterialRange) == 2u);
        CHECK(Count(report, LoadRule::TextureRange) == 2u);
        CHECK(Count(report, LoadRule::TextureAnimationRange) == 1u);
        CHECK(Count(report, LoadRule::UvSetRange) == 1u);
        CHECK(Count(report, LoadRule::GeosetAnimationTarget) == 1u);
        CHECK(Count(report, LoadRule::GeosetAnimationRange) == 1u);
        CHECK(Count(report, LoadRule::GlobalSequenceRange) == 1u);
        CHECK(Count(report, LoadRule::ObjectOrder) == 3u); // "lost" is a bone, read second

        // A geoset is drawn with something: one blank material for both.
        REQUIRE(model.materials.size() == 2);
        CHECK(model.geosets[1].materialId == 1u);
        CHECK(model.ribbonEmitters[0].materialId == 1u);
        // And one blank texture for the layer, the emitter and that material.
        REQUIRE(model.textures.size() == 2);
        CHECK(model.textures[1].fileName.empty());
        const Layer& layer = model.materials[0].layers[0];
        CHECK((version >= 1100 ? layer.subTextures[0].textureId : layer.textureId) == 1u);
        CHECK(model.particleEmitters2[0].textureId == 1u);
        CHECK(layer.textureAnimationId == 0xFFFFFFFFu);
        CHECK(layer.coordId == 0u);
        // The stray record is gone, and the bone still follows its own.
        REQUIRE(model.geosetAnimations.size() == 1);
        CHECK(model.geosetAnimations[0].geosetId == 1u);
        CHECK(model.bones[0].geosetAnimationId == 0u);
        CHECK(model.bones[1].geosetAnimationId == 0xFFFFFFFFu);
        CHECK(model.bones[0].node.translationTracks.globalSequenceId == 0xFFFFFFFFu);

        // Once is enough, and what it wrote reads back clean.
        CHECK(makeLoadable(model).empty());
        Writer writer;
        Parser parser(Parser::UpgradeMode::PreserveOriginal);
        const Model read = parser.parse(writer.write(model));
        CHECK_FALSE(parser.hasIssues());
        CHECK(read.materials.size() == 2);
    }
}

TEST_CASE("mdx the rules that came from the export", "[mdx][loadable]") {
    Model model = SoundModel();
    // Attachment ids that do not climb: the game sizes its table by the last.
    for (const u32 id : {0u, 2u, 1u}) {
        Attachment attachment;
        attachment.node = NodeOf("point", static_cast<u32>(model.attachments.size()) + 1);
        attachment.attachmentId = id;
        model.attachments.push_back(attachment);
    }
    // A negative rate or lifespan, which the game sizes a pool by, unsigned.
    ParticleEmitter2 emitter;
    emitter.node = NodeOf("emitter", 4);
    emitter.emissionRate = -11.5f;
    emitter.lifespan = 2.0f;
    emitter.emissionRateTracks = Keyed<f32>({0, 100}, {5.0f, -1.0f});
    model.particleEmitters2.push_back(emitter);
    // A box whose corners are not its minimum and its maximum.
    model.collisionShapes.push_back(Shape("box", 5, CollisionShape::ShapeType::Box,
                                          {Vector3f(4, -2, 6), Vector3f(-4, 2, 0)}));
    model.pivotPoints.assign(6, Vector3f(0, 0, 0));

    const LoadReport report = makeLoadable(model);
    CHECK(Count(report, LoadRule::AttachmentIds) == 1u);
    CHECK(Count(report, LoadRule::EmitterPool) == 1u);
    CHECK(Count(report, LoadRule::BoxCorners) == 1u);
    CHECK(model.attachments[0].attachmentId == 0u);
    CHECK(model.attachments[1].attachmentId == 1u);
    CHECK(model.attachments[2].attachmentId == 2u);
    CHECK(model.particleEmitters2[0].emissionRate == 0.0f);
    CHECK(model.particleEmitters2[0].lifespan == 2.0f);
    CHECK(model.particleEmitters2[0].emissionRateTracks.keys_data == std::vector<f32>{5.0f, 0.0f});
    CHECK(model.collisionShapes[0].vertices ==
          std::vector<Vector3f>{Vector3f(-4, -2, 0), Vector3f(4, 2, 6)});
    for (const LoadFinding& finding : report.findings) {
        CHECK(finding.repaired);
        CHECK_FALSE(describe(finding).empty());
    }
}

TEST_CASE("mdx what the guard reports and leaves", "[mdx][loadable]") {
    Model model = SoundModel();
    model.sequences.resize(256);
    for (std::size_t s = 0; s < model.sequences.size(); ++s) {
        model.sequences[s].name = "Stand - " + std::to_string(s);
        model.sequences[s].intervalStart = static_cast<u32>(s * 100);
        model.sequences[s].intervalEnd = static_cast<u32>(s * 100 + 50);
    }
    Model copy = model;
    const LoadReport report = makeLoadable(copy);
    REQUIRE(report.find(LoadRule::SequenceCount) != nullptr);
    CHECK_FALSE(report.find(LoadRule::SequenceCount)->repaired);
    // Said, and the model is the caller's to cut: the writer writes it as given.
    CHECK_FALSE(report.repaired());
    CHECK(copy.sequences.size() == 256);
    Writer writer;
    Writer raw;
    raw.setRepairs(false);
    CHECK(writer.write(model) == raw.write(model));
    CHECK(writer.lastReport().find(LoadRule::SequenceCount) != nullptr);
}

TEST_CASE("mdx every corpus file is loadable once the guard has run", "[mdx][loadable][corpus]") {
    // What the audit's sweep found by reading bytes, here by the rules: files
    // other tools wrote hold keyless events, ids out of read order, attachment
    // ids that do not climb. Re-saved, each used to be written as it was read.
    const auto files = test::gather("WEM_MDX_CORPUS_DIR", ".mdx", {"MDL", "Wc3Mdx"});
    if (files.empty()) {
        WARN("no .mdx corpus found; set WEM_MDX_CORPUS_DIR");
        return;
    }
    const std::size_t limit = test::sweepLimit(files.size(), 200);
    std::map<LoadRule, u32> filesWith;
    u32 read = 0;
    u32 repaired = 0;
    for (std::size_t i = 0; i < limit; ++i) {
        if (test::isKnownBad(files[i])) {
            continue;
        }
        test::trace(files[i]);
        const std::vector<u8> bytes = test::readCorpusFile(files[i]);
        if (bytes.size() < 4 || std::memcmp(bytes.data(), "MDLX", 4) != 0) {
            continue; // another container under the same extension
        }
        INFO(test::pathText(files[i].filename()));
        Parser parser(Parser::UpgradeMode::PreserveOriginal);
        const Model source = parser.parse(bytes);
        ++read;

        Model fixed = source;
        const LoadReport report = makeLoadable(fixed);
        for (const LoadFinding& finding : report.findings) {
            ++filesWith[finding.rule];
        }
        repaired += report.repaired() ? 1u : 0u;
        // Once is enough: nothing is left to repair.
        Model again = fixed;
        CHECK_FALSE(makeLoadable(again).repaired());

        // The writer writes the repaired model for the one it was given, and
        // a model with nothing to repair exactly as it is.
        Writer writer;
        const std::vector<u8> written = writer.write(source);
        Writer raw;
        raw.setRepairs(false);
        CHECK(written == raw.write(report.repaired() ? fixed : source));

        // And it reads back whole: every record, every track.
        Parser back(Parser::UpgradeMode::PreserveOriginal);
        const Model reread = back.parse(written);
        CHECK(back.getIssues().size() <= parser.getIssues().size());
        CHECK(reread.bones.size() == fixed.bones.size());
        CHECK(reread.helpers.size() == fixed.helpers.size());
        CHECK(reread.eventObjects.size() == fixed.eventObjects.size());
        CHECK(reread.collisionShapes.size() == fixed.collisionShapes.size());
        CHECK(reread.geosets.size() == fixed.geosets.size());
        CHECK(reread.geosetAnimations.size() == fixed.geosetAnimations.size());
        CHECK(reread.materials.size() == fixed.materials.size());
        CHECK(UsedTracks(reread) == UsedTracks(fixed));
    }
    std::cout << "mdx loadable: " << read << " files read, " << repaired << " repaired";
    for (const auto& [rule, count] : filesWith) {
        std::cout << "; rule " << static_cast<u32>(rule) << " in " << count;
    }
    std::cout << std::endl;
    REQUIRE(read > 0u);
}

// ============================================================================
// What a model need not hold (mdx::pruneModel)
// ============================================================================

namespace {

GeosetAnimation Record(u32 geoset, f32 alpha = 1.0f, u32 flags = 0x2) {
    GeosetAnimation record;
    record.geosetId = geoset;
    record.alpha = alpha;
    record.flags = static_cast<GeosetAnimation::Flag>(flags);
    return record;
}

Bone BoneOf(const char* name, u32 objectId, u32 geoset, u32 record) {
    Bone bone;
    bone.node = NodeOf(name, objectId);
    bone.geosetId = geoset;
    bone.geosetAnimationId = record;
    return bone;
}

/// The model `SoundModel` makes, with @p geosets triangles and no bone links.
Model ManyGeosets(std::size_t geosets) {
    Model model = SoundModel();
    model.geosets.assign(geosets, Triangle(0));
    return model;
}

} // namespace

TEST_CASE("mdx a geoset animation that states nothing is left out", "[mdx][prune]") {
    // The game multiplies a record's colour and alpha into its geoset, plays
    // its tracks, and reads `DropShadow` through a bone. One that is white,
    // opaque, unkeyed and unlinked says nothing -- whatever else its flags
    // word holds, which nothing reads.
    Model model = ManyGeosets(4);
    GeosetAnimation tinted = Record(0);
    tinted.color = Vector3f(1.0f, 0.5f, 0.5f);
    GeosetAnimation keyed = Record(1);
    keyed.alphaTracks = Keyed<f32>({0, 100}, {1.0f, 1.0f});
    model.geosetAnimations = {
        Record(0),             // nothing
        Record(1, 0.0f),       // hidden
        Record(2, 1.0f, 0x1A), // nothing but unread bits
        Record(3, 1.0f, 0x3),  // a drop shadow
        tinted,
        keyed,
        Record(2, 0.5f), // half seen
    };

    PruneReport report;
    pruneGeosetAnimations(model, report);
    CHECK(report.geosetAnimations == 2u);
    CHECK(report.boneLinks == 0u);
    REQUIRE(model.geosetAnimations.size() == 5);
    CHECK(model.geosetAnimations[0].alpha == 0.0f);
    CHECK(hasFlag(model.geosetAnimations[1].flags, GeosetAnimation::Flag::DropShadow));
    CHECK(model.geosetAnimations[2].color == Vector3f(1.0f, 0.5f, 0.5f));
    CHECK(model.geosetAnimations[3].alphaTracks.isUsed);
    CHECK(model.geosetAnimations[4].alpha == 0.5f);

    // And a model with none to spare is left as it is.
    PruneReport none;
    pruneGeosetAnimations(model, none);
    CHECK(none.empty());
}

TEST_CASE("mdx a bone does not follow a record that never hides", "[mdx][prune]") {
    // A bone follows its record's alpha. One that never leaves 1 hides
    // nothing, so the link is no link, and the record may then be nothing too.
    Model model = ManyGeosets(4);
    GeosetAnimation keyed = Record(2);
    keyed.alphaTracks = Keyed<f32>({0, 100}, {1.0f, 0.0f});
    model.geosetAnimations = {Record(0), Record(1, 0.0f), keyed, Record(3, 1.0f, 0x3)};
    model.bones = {BoneOf("always", 0, 0, 0), BoneOf("hidden", 1, 1, 1), BoneOf("keyed", 2, 2, 2),
                   BoneOf("shadow", 3, 3, 3)};
    model.pivotPoints.assign(4, Vector3f(0, 0, 0));

    PruneReport report;
    pruneGeosetAnimations(model, report);
    CHECK(report.boneLinks == 1u);
    CHECK(report.geosetAnimations == 1u);
    REQUIRE(model.geosetAnimations.size() == 3);
    CHECK(model.bones[0].geosetAnimationId == 0xFFFFFFFFu);
    // The rest follow their records to where they moved.
    CHECK(model.bones[1].geosetAnimationId == 0u);
    CHECK(model.bones[2].geosetAnimationId == 1u);
    // `DropShadow` is read THROUGH the bone, so that link is one.
    CHECK(model.bones[3].geosetAnimationId == 2u);
    CHECK(model.geosetAnimations[0].geosetId == 1u);
}

TEST_CASE("mdx the records bones follow fit the byte the game keeps", "[mdx][prune]") {
    // The link is kept in a byte, 255 for none, so record 254 is the last a
    // bone can follow. Nothing else reads a record's place.
    Model model = ManyGeosets(300);
    for (u32 g = 0; g < 300; ++g) {
        model.geosetAnimations.push_back(Record(g, 0.0f));
    }
    model.bones = {BoneOf("far", 0, 280, 280), BoneOf("near", 1, 3, 3)};
    model.pivotPoints.assign(2, Vector3f(0, 0, 0));

    PruneReport report;
    pruneGeosetAnimations(model, report);
    CHECK(report.linksPastByte == 0u);
    REQUIRE(model.geosetAnimations.size() == 300);
    // The two linked records first, in their order; every bone still follows
    // the record of the geoset it did.
    CHECK(model.bones[1].geosetAnimationId == 0u);
    CHECK(model.bones[0].geosetAnimationId == 1u);
    CHECK(model.geosetAnimations[0].geosetId == 3u);
    CHECK(model.geosetAnimations[1].geosetId == 280u);
    CHECK(model.geosetAnimations[2].geosetId == 0u);

    // More linked records than the byte reaches is said, not hidden.
    Model crowded = ManyGeosets(300);
    crowded.bones.clear();
    for (u32 g = 0; g < 300; ++g) {
        crowded.geosetAnimations.push_back(Record(g, 0.0f));
        crowded.bones.push_back(BoneOf("bone", g, g, g));
    }
    PruneReport over;
    pruneGeosetAnimations(crowded, over);
    CHECK(over.linksPastByte == 300u - 255u);
}

TEST_CASE("mdx a geoset with no faces is left out", "[mdx][prune]") {
    Model model = ManyGeosets(4);
    model.geosets[1] = Geoset{};
    model.geosets[3] = Geoset{};
    // A record a bone follows keeps the geoset it names.
    model.geosetAnimations = {Record(1, 0.0f), Record(2, 0.0f), Record(3, 0.0f)};
    model.bones = {BoneOf("on two", 0, 2, 1), BoneOf("on three", 1, 3, 2), BoneOf("on one", 2, 1, 0xFFFFFFFFu)};
    model.pivotPoints.assign(3, Vector3f(0, 0, 0));

    CHECK(pruneEmptyGeosets(model) == 1u);
    REQUIRE(model.geosets.size() == 3);
    CHECK_FALSE(model.geosets[0].faces.empty());
    CHECK_FALSE(model.geosets[1].faces.empty());
    CHECK(model.geosets[2].faces.empty()); // kept for its record
    // Its record went with it, and every number after it moved down.
    REQUIRE(model.geosetAnimations.size() == 2);
    CHECK(model.geosetAnimations[0].geosetId == 1u);
    CHECK(model.geosetAnimations[1].geosetId == 2u);
    CHECK(model.bones[0].geosetAnimationId == 0u);
    CHECK(model.bones[0].geosetId == 1u);
    CHECK(model.bones[1].geosetAnimationId == 1u);
    CHECK(model.bones[1].geosetId == 2u);
    CHECK(model.bones[2].geosetId == Bone::MULTIPLE_GEOSETS);
    CHECK(pruneEmptyGeosets(model) == 0u);

    // A ladder keeps every geoset: its levels are matched by count, and a
    // level that simplified to nothing is empty on purpose.
    Model ladder = ManyGeosets(4);
    ladder.version = 1800;
    ladder.geosets[2].lod = 1;
    ladder.geosets[3] = Geoset{};
    ladder.geosets[3].lod = 1;
    CHECK(pruneEmptyGeosets(ladder) == 0u);
    CHECK(ladder.geosets.size() == 4);
}

TEST_CASE("mdx a material identical to another is the other", "[mdx][prune]") {
    for (const u32 version : {800u, 1800u}) {
        CAPTURE(version);
        Model model = SoundModel(version);
        Material same = model.materials[0];
        Material other = model.materials[0];
        other.layers[0].alpha = 0.5f;
        Material unused = model.materials[0];
        unused.layers[0].alpha = 0.25f;
        model.materials = {model.materials[0], same, other, unused};
        model.geosets = {Triangle(0), Triangle(1), Triangle(2)};
        RibbonEmitter ribbon;
        ribbon.node = NodeOf("ribbon", 1);
        ribbon.materialId = 1;
        model.ribbonEmitters.push_back(ribbon);
        model.pivotPoints.assign(2, Vector3f(0, 0, 0));

        const PruneReport report = pruneModel(model);
        CHECK(report.materialsUnused == 1u);
        CHECK(report.materialsMerged == 1u);
        REQUIRE(model.materials.size() == 2);
        CHECK(model.materials[1].layers[0].alpha == 0.5f);
        CHECK(model.geosets[0].materialId == 0u);
        CHECK(model.geosets[1].materialId == 0u);
        CHECK(model.geosets[2].materialId == 1u);
        CHECK(model.ribbonEmitters[0].materialId == 0u);
        CHECK(pruneModel(model).empty());
    }
}

TEST_CASE("mdx a global sequence nothing runs on is left out", "[mdx][prune]") {
    Model model = SoundModel();
    model.globalSequences = {100, 200, 300};
    model.bones[0].node.translationTracks = Keyed<Vector3f>({0}, {Vector3f(0, 0, 0)});
    model.bones[0].node.translationTracks.globalSequenceId = 2;
    EventObject event;
    event.node = NodeOf("SNDxLOOP", 1);
    event.globalSequenceId = 0;
    event.eventTrackTimes = {50};
    model.eventObjects.push_back(event);
    model.pivotPoints.assign(2, Vector3f(0, 0, 0));

    CHECK(pruneUnusedGlobalSequences(model) == 1u);
    CHECK(model.globalSequences == std::vector<u32>{100, 300});
    CHECK(model.bones[0].node.translationTracks.globalSequenceId == 1u);
    CHECK(model.eventObjects[0].globalSequenceId == 0u);
    CHECK(pruneUnusedGlobalSequences(model) == 0u);
}
