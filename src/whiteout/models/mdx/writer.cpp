// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <whiteout/models/mdx/track_walk.h>
#include <whiteout/models/mdx/writer.h>
#include "../../common/binary_writer.h"
#include "../../common/streams.h"
#include "../../common/unicode_path.h"
#include "mdl_writer.h"

namespace whiteout {
namespace mdx {

using common::BinaryWriter;

namespace {

struct SizeEnclosure {
    u32 startPos;
    u32 sizePos;
    BinaryWriter& writer;
    u32 sizeFlags = 0; ///< ORed into the size word (a camera's variant byte)

    SizeEnclosure(BinaryWriter& writer, u32 tag = 0, u32 sizeFlags = 0) : writer(writer), sizeFlags(sizeFlags) {

        if (tag != 0) {
            writer.write(tag);
        }
        startPos = writer.getPosition();
        writer.write<u32>(0); // Placeholder for size
        if (tag == 0) {
            sizePos = startPos; // this is inclusive
            return;
        }
        sizePos = writer.getPosition();
    }

    ~SizeEnclosure() {
        u32 const endPos = writer.getPosition();
        u32 const actualSize = endPos - sizePos;
        writer.setPosition(startPos); // Move back to size field
        writer.write(actualSize | sizeFlags);
        writer.setPosition(endPos); // Move back to end
    }
};

} // namespace

template <typename T>
static void writeTrackChunk(BinaryWriter& writer, u32 tag, const Track<T>& track) {
    if (!track.isUsed)
        return; // Skip unused tracks

    // The in-memory layout splits timestamps from key-value data, but the on-disk
    // layout interleaves them per keyframe -- mirror Parser::Impl::readTrackChunk:
    //   non-smooth: { u32 timestamp, T value }                * keyCount
    //   smooth:     { u32 timestamp, T value, T inTan, T outTan } * keyCount
    const size_t keyComponents = isSmoothInterpolation(track.interpolationType) ? 3 : 1;
    const size_t valueStride = keyComponents * sizeof(T);
    const size_t stride = sizeof(u32) + valueStride;
    // The keys the arrays hold, whatever `keyCount` claims: it is the count
    // written, and the reader takes exactly that many.
    const size_t keyCount =
        std::min({track.keyCount, track.timestamps.size(), track.keys_data.size() / keyComponents});

    // Write track header
    writer.write(tag);
    writer.write(static_cast<u32>(keyCount));
    writer.write(static_cast<u32>(track.interpolationType));
    writer.write(track.globalSequenceId);

    std::vector<u8> buffer(keyCount * stride);
    for (size_t i = 0; i < keyCount; ++i) {
        std::memcpy(buffer.data() + i * stride, &track.timestamps[i], sizeof(u32));
        std::memcpy(buffer.data() + i * stride + sizeof(u32), &track.keys_data[i * keyComponents],
                    valueStride);
    }
    writer.writeBytes(reinterpret_cast<const char*>(buffer.data()),
                      static_cast<u32>(buffer.size()));
}

/// A record's own tracks. The list, and its order, is the walk's
/// (`track_walk.h`), which the loader's guard reads through too.
template <class Record>
static void writeOwnTracks(BinaryWriter& writer, const Record& record, u32 version) {
    forEachOwnTrack(record, version,
                    [&writer](u32 tag, const auto& track) { writeTrackChunk(writer, tag, track); });
}

// ============================================================================
// WriterImpl - Implementation class using PImpl idiom
// ============================================================================

class Writer::Impl {
public:
    bool repairs = true;
    LoadReport report;

    /// The model `write` writes for @p mdx: itself, or @p scratch holding its
    /// repaired copy when the loader's rules asked for one (`makeLoadable`).
    const Model& loadable(const Model& mdx, Model& scratch);

    struct ChunkHeader {
        u32 tag;  ///< Chunk identifier
        u32 size; ///< Chunk data size in bytes
    };

    void writeChunkHeader(BinaryWriter& writer, u32 tag, size_t size);

    void write(BinaryWriter& writer, const Model& mdx);

    // Chunk writers - each writes one MDX chunk type
    void writeVERS(BinaryWriter& writer, const Model& mdx);
    void writeMODL(BinaryWriter& writer, const Model& mdx);
    void writeSEQS(BinaryWriter& writer, const Model& mdx);
    void writeGLBS(BinaryWriter& writer, const Model& mdx);
    void writeTEXS(BinaryWriter& writer, const Model& mdx);
    void writeSNDS(BinaryWriter& writer, const Model& mdx);
    void writeSNEM(BinaryWriter& writer, const Model& mdx);
    void writeMTLS(BinaryWriter& writer, const Model& mdx);
    void writeTXAN(BinaryWriter& writer, const Model& mdx);
    void writeGEOS(BinaryWriter& writer, const Model& mdx);
    void writeGEOA(BinaryWriter& writer, const Model& mdx);
    void writeBONE(BinaryWriter& writer, const Model& mdx);
    void writeLITE(BinaryWriter& writer, const Model& mdx);
    void writeHELP(BinaryWriter& writer, const Model& mdx);
    void writeATCH(BinaryWriter& writer, const Model& mdx);
    void writePIVT(BinaryWriter& writer, const Model& mdx);
    void writePREM(BinaryWriter& writer, const Model& mdx);
    void writePRE2(BinaryWriter& writer, const Model& mdx);
    void writeRIBB(BinaryWriter& writer, const Model& mdx);
    void writeEVTS(BinaryWriter& writer, const Model& mdx);
    void writeCAMS(BinaryWriter& writer, const Model& mdx);
    void writeCLID(BinaryWriter& writer, const Model& mdx);
    void writeBPOS(BinaryWriter& writer, const Model& mdx);
    void writeFFX(BinaryWriter& writer, const Model& mdx);
    void writeCORN(BinaryWriter& writer, const Model& mdx);

    // Structure writers - write individual structure types
    void writeMaterial(BinaryWriter& writer, const Material& mat, u32 version);
    void writeLayer(BinaryWriter& writer, const Layer& layer, u32 version);
    void writeTextureAnimation(BinaryWriter& writer, const TextureAnimation& anim);
    void writeGeoset(BinaryWriter& writer, const Geoset& geoset, const Model& mdx);
    void writeGeosetAnimation(BinaryWriter& writer, const GeosetAnimation& anim);
    void writeBone(BinaryWriter& writer, const Bone& bone);
    void writeNode(BinaryWriter& writer, const Node& node);
    void writeLight(BinaryWriter& writer, const Light& light, const Model& mdx);
    void writeHelper(BinaryWriter& writer, const Helper& helper);
    void writeAttachment(BinaryWriter& writer, const Attachment& att);
    void writeParticleEmitter(BinaryWriter& writer, const ParticleEmitter& pem);
    void writeParticleEmitter2(BinaryWriter& writer, const ParticleEmitter2& pem2);
    void writeRibbonEmitter(BinaryWriter& writer, const RibbonEmitter& ribb);
    void writeEventObject(BinaryWriter& writer, const EventObject& evt);
    void writeCamera(BinaryWriter& writer, const Camera& cam, const Model& mdx);
    void writeCollisionShape(BinaryWriter& writer, const CollisionShape& shape);
    void writeSoundEmitter(BinaryWriter& writer, const SoundEmitter& snem);
    void writeCornEmitter(BinaryWriter& writer, const CornEmitter& corn);

    /**
     * @brief Write an animation track
     * @tparam T Track value type (f32, Vector3f, etc.)
     * @param writer Binary writer
     * @param tag Track chunk tag
     * @param track Track data to write
     */
    template <typename T>
    void writeTrack(BinaryWriter& writer, u32 tag, const Track<T>& track);

    void writeNodeTracks(BinaryWriter& writer, const Node& node);
};

void Writer::Impl::writeChunkHeader(BinaryWriter& writer, u32 tag, size_t size) {
    writer.write(tag);
    writer.write(static_cast<u32>(size));
}

void Writer::Impl::write(BinaryWriter& writer, const Model& mdx) {
    // Write magic number
    writer.write(MDLX_TAG);

    // Chunks in 3.0.0's own writer order. The node chunks' order is load-bearing:
    // the game's binary reader ignores each node's stored objectId and numbers
    // nodes in the order it reads them, so CORN must sit between PRE2 and RIBB,
    // where toMdx numbers it, or every later node reads with a shifted id.
    writeVERS(writer, mdx);
    writeMODL(writer, mdx);

    if (!mdx.sequences.empty())
        writeSEQS(writer, mdx);
    if (!mdx.globalSequences.empty())
        writeGLBS(writer, mdx);
    if (!mdx.materials.empty())
        writeMTLS(writer, mdx);
    if (!mdx.textures.empty())
        writeTEXS(writer, mdx);
    if (!mdx.sounds.empty())
        writeSNDS(writer, mdx);
    if (!mdx.soundEmitters.empty())
        writeSNEM(writer, mdx);
    if (!mdx.textureAnimations.empty())
        writeTXAN(writer, mdx);
    if (!mdx.geosets.empty())
        writeGEOS(writer, mdx);
    if (!mdx.geosetAnimations.empty())
        writeGEOA(writer, mdx);
    if (!mdx.bones.empty())
        writeBONE(writer, mdx);
    if (!mdx.lights.empty())
        writeLITE(writer, mdx);
    if (!mdx.helpers.empty())
        writeHELP(writer, mdx);
    if (!mdx.attachments.empty())
        writeATCH(writer, mdx);
    if (!mdx.pivotPoints.empty())
        writePIVT(writer, mdx);
    if (!mdx.particleEmitters.empty())
        writePREM(writer, mdx);
    if (!mdx.particleEmitters2.empty())
        writePRE2(writer, mdx);
    if (!mdx.cornEmitters.empty())
        writeCORN(writer, mdx);
    if (!mdx.ribbonEmitters.empty())
        writeRIBB(writer, mdx);
    if (!mdx.cameras.empty())
        writeCAMS(writer, mdx);
    if (!mdx.eventObjects.empty())
        writeEVTS(writer, mdx);
    if (!mdx.collisionShapes.empty())
        writeCLID(writer, mdx);
    if (!mdx.faceEffects.empty())
        writeFFX(writer, mdx);
    if (!mdx.bindPoses.empty())
        writeBPOS(writer, mdx);
}

// ============================================================================
// Writer Public Interface (using PImpl)
// ============================================================================

Writer::Writer() : pImpl(std::make_unique<Impl>()) {}

Writer::~Writer() = default;

void Writer::setRepairs(bool repairs) {
    pImpl->repairs = repairs;
}

const LoadReport& Writer::lastReport() const {
    return pImpl->report;
}

const Model& Writer::Impl::loadable(const Model& mdx, Model& scratch) {
    report = {};
    if (!repairs) {
        return mdx;
    }
    scratch = mdx;
    report = makeLoadable(scratch);
    return report.repaired() ? scratch : mdx;
}

std::vector<u8> Writer::materialRecord(const Material& material, u32 version) {
    std::vector<u8> buffer;
    common::vector_streambuf streambuf(buffer);
    std::ostream out(&streambuf);
    BinaryWriter writer(out);
    Impl().writeMaterial(writer, material, version);
    return buffer;
}

void Writer::write(const std::string& filePath, const Model& given, MdlFormat mdlFormat) {
    Model scratch;
    const Model& mdx = pImpl->loadable(given, scratch);

    // Detect MDL text format from extension
    bool isMdl = false;
    if (filePath.size() >= 4) {
        auto ext = filePath.substr(filePath.size() - 4);
        isMdl = (ext == ".mdl" || ext == ".MDL" || ext == ".Mdl");
    }

    if (isMdl) {
        std::string text = writeModelToMdl(mdx, mdlFormat);
        auto file = common::open_ofstream(filePath, std::ios::binary);
        if (!file.is_open()) {
            return; // silently fail; caller can check filesystem
        }
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        return;
    }

    auto file = common::open_ofstream(filePath, std::ios::binary);
    if (!file.is_open()) {
        return;
    }
    BinaryWriter writer(file);

    pImpl->write(writer, mdx);
}

std::vector<u8> Writer::write(const Model& given, MDLXFormat format, MdlFormat mdlFormat) {
    Model scratch;
    const Model& mdx = pImpl->loadable(given, scratch);
    if (format == MDLXFormat::MDL) {
        std::string text = writeModelToMdl(mdx, mdlFormat);
        return std::vector<u8>(text.begin(), text.end());
    }

    std::vector<u8> buffer;
    buffer.reserve(1 * 1024 * 1024); // Reserve 1MB to avoid frequent reallocations
    common::vector_streambuf streambuf(buffer);
    std::ostream out(&streambuf);
    BinaryWriter writer(out);

    pImpl->write(writer, mdx);

    buffer.shrink_to_fit(); // Reduce capacity to actual size
    return buffer;
}

// ============================================================================
// WriterImpl Implementation - Moved all private methods here
// ============================================================================

void Writer::Impl::writeVERS(BinaryWriter& writer, const Model& mdx) {
    writeChunkHeader(writer, VERS_TAG, 4);
    writer.write(mdx.version);
}

void Writer::Impl::writeMODL(BinaryWriter& writer, const Model& mdx) {
    writeChunkHeader(writer, MODL_TAG, 372);
    writer.writeString(mdx.modelName, 80);
    writer.writeString(mdx.animationFileName, 260);
    writer.write(mdx.modelExtent);
    writer.write(mdx.blendTime);
}

void Writer::Impl::writeSEQS(BinaryWriter& writer, const Model& mdx) {
    size_t const size = mdx.sequences.size() * 132;
    writeChunkHeader(writer, SEQS_TAG, size);

    for (const auto& seq : mdx.sequences) {
        writer.writeString(seq.name, 80);
        writer.write(seq.intervalStart);
        writer.write(seq.intervalEnd);
        writer.write(seq.moveSpeed);
        writer.write(seq.flags);
        writer.write(seq.rarity);
        writer.write(seq.syncPoint);
        writer.write(seq.extent);
    }
}

void Writer::Impl::writeGLBS(BinaryWriter& writer, const Model& mdx) {
    size_t const size = mdx.globalSequences.size() * 4;
    writeChunkHeader(writer, GLBS_TAG, size);
    writer.write(mdx.globalSequences);
}

void Writer::Impl::writeTEXS(BinaryWriter& writer, const Model& mdx) {
    size_t const size = mdx.textures.size() * 268;
    writeChunkHeader(writer, TEXS_TAG, size);

    for (const auto& tex : mdx.textures) {
        writer.write(tex.replaceableId);
        writer.writeString(tex.fileName, 260);
        writer.write(tex.flags);
    }
}

void Writer::Impl::writeSNDS(BinaryWriter& writer, const Model& mdx) {
    size_t const size = mdx.sounds.size() * 56;
    writeChunkHeader(writer, SNDS_TAG, size);

    for (const auto& snd : mdx.sounds) {
        writer.writeString(snd.soundFile, 44);
        writer.write(snd.maximumDistance);
        writer.write(snd.minimumDistance);
        writer.write(snd.soundChannel);
    }
}

void Writer::Impl::writeSNEM(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, SNEM_TAG);

    for (const auto& snem : mdx.soundEmitters) {
        writeSoundEmitter(writer, snem);
    }
}

void Writer::Impl::writeSoundEmitter(BinaryWriter& writer, const SoundEmitter& snem) {
    SizeEnclosure const sizeEnclosure(writer);

    writeNode(writer, snem.node);

    writeOwnTracks(writer, snem, 0);
}

void Writer::Impl::writeMTLS(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, MTLS_TAG);

    for (const auto& mat : mdx.materials) {
        writeMaterial(writer, mat, mdx.version);
    }
}

void Writer::Impl::writeMaterial(BinaryWriter& writer, const Material& mat, u32 version) {
    SizeEnclosure const sizeEnclosure(writer);

    writer.write(mat.priorityPlane);
    writer.write(mat.flags);

    if (version > 800 && version < 1100) {
        writer.writeString(mat.shader, 80);
    }

    // Write LAYS chunk
    writer.write(LAYS_TAG);
    writer.write(static_cast<u32>(mat.layers.size()));

    for (const auto& layer : mat.layers) {
        writeLayer(writer, layer, version);
    }
}

void Writer::Impl::writeLayer(BinaryWriter& writer, const Layer& layer, u32 version) {
    SizeEnclosure const sizeEnclosure(writer);

    writer.write(static_cast<u32>(layer.filterMode));
    writer.write(static_cast<u32>(layer.shadingFlags));
    writer.write(layer.textureId);
    writer.write(layer.textureAnimationId);
    writer.write(layer.coordId);
    writer.write(layer.alpha);

    // emissiveGain exists from v900, the fresnel fields only from v1000 —
    // must match parseLayer's gating (and the KFC3/KFCA/KFTC track gate below).
    if (version > 800) {
        writer.write(layer.emissiveGain);
    }
    if (version > 900) {
        writer.write(layer.fresnelColor);
        writer.write(layer.fresnelOpacity);
        writer.write(layer.fresnelTeamColor);
    }

    if (version >= 1100) {
        writer.write(layer.shader);
        writer.write(static_cast<u32>(layer.subTextures.size()));

        for (const auto& subTex : layer.subTextures) {
            writer.write(subTex.textureId);
            writer.write(static_cast<u32>(subTex.slot));
            writeTrackChunk(writer, KMTF_TAG, subTex.tracks);
        }
    }

    // The layer's own tracks; a sub-texture's flipbook went out with it above.
    writeOwnTracks(writer, layer, version);
}

void Writer::Impl::writeTXAN(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, TXAN_TAG);

    for (const auto& anim : mdx.textureAnimations) {
        writeTextureAnimation(writer, anim);
    }
}

void Writer::Impl::writeTextureAnimation(BinaryWriter& writer, const TextureAnimation& anim) {
    SizeEnclosure const sizeEnclosure(writer);

    writeOwnTracks(writer, anim, 0);
}

void Writer::Impl::writeGEOS(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, GEOS_TAG);

    for (const auto& geo : mdx.geosets) {
        writeGeoset(writer, geo, mdx);
    }
}

void Writer::Impl::writeGeoset(BinaryWriter& writer, const Geoset& geo, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer);

    // Write VRTX
    writeChunkHeader(writer, VRTX_TAG, geo.vertexPositions.size());
    writer.write(geo.vertexPositions);

    // Write NRMS
    writeChunkHeader(writer, NRMS_TAG, geo.vertexNormals.size());
    writer.write(geo.vertexNormals);

    // Write PTYP
    writeChunkHeader(writer, PTYP_TAG, geo.faceTypeGroups.size());
    writer.write(geo.faceTypeGroups);

    // Write PCNT
    writeChunkHeader(writer, PCNT_TAG, geo.faceGroups.size());
    writer.write(geo.faceGroups);

    // Write PVTX
    writeChunkHeader(writer, PVTX_TAG, geo.faces.size());
    writer.write(geo.faces);

    // Write GNDX
    writeChunkHeader(writer, GNDX_TAG, geo.vertexGroups.size());
    writer.write(geo.vertexGroups);

    // Write MTGC
    writeChunkHeader(writer, MTGC_TAG, geo.matrixGroups.size());
    writer.write(geo.matrixGroups);

    // Write MATS
    writeChunkHeader(writer, MATS_TAG, geo.matrixIndices.size());
    writer.write(geo.matrixIndices);

    writer.write(geo.materialId);
    writer.write(geo.selectionGroup);
    writer.write(geo.selectionFlags);

    if (mdx.version > 800) {
        writer.write(geo.lod);
        writer.writeString(geo.lodName, 80);
    }

    writer.write(geo.extent);

    writer.write(static_cast<u32>(geo.sequenceExtents.size()));
    for (const auto& ext : geo.sequenceExtents) {
        writer.write(ext);
    }

    if (mdx.version > 800) {
        // Write optional TANG
        if (!geo.tangents.empty()) {
            writeChunkHeader(writer, TANG_TAG, geo.tangents.size());
            writer.write(geo.tangents);
        }

        // Write optional SKIN. The count is an element count in both
        // conventions, so only the stream width changes: v1400 and later spell
        // each influence as a u16 (see the parser's SKIN case). Below 1400 a
        // bone index past 255 has no spelling at all.
        if (!geo.skinData.empty()) {
            writeChunkHeader(writer, SKIN_TAG, geo.skinData.size());
            if (mdx.version >= 1400) {
                writer.write(geo.skinData);
            } else {
                std::vector<u8> narrow(geo.skinData.size());
                for (std::size_t i = 0; i < narrow.size(); ++i)
                    narrow[i] = static_cast<u8>(geo.skinData[i]);
                writer.write(narrow);
            }
        }
    }

    // Write optional UVAS
    if (!geo.textureCoordinateSets.empty()) {
        writeChunkHeader(writer, UVAS_TAG, geo.textureCoordinateSets.size());

        for (const auto& uvSet : geo.textureCoordinateSets) {
            writeChunkHeader(writer, UVBS_TAG, uvSet.size());
            writer.write(uvSet);
        }
    }
}

void Writer::Impl::writeGEOA(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, GEOA_TAG);

    for (const auto& anim : mdx.geosetAnimations) {
        writeGeosetAnimation(writer, anim);
    }
}

void Writer::Impl::writeGeosetAnimation(BinaryWriter& writer, const GeosetAnimation& anim) {
    SizeEnclosure const sizeEnclosure(writer);

    writer.write(anim.alpha);
    writer.write(anim.flags);
    writer.write(anim.color);
    writer.write(anim.geosetId);

    writeOwnTracks(writer, anim, 0);
}

void Writer::Impl::writeBONE(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, BONE_TAG);

    for (const auto& bone : mdx.bones) {
        writeBone(writer, bone);
    }
}

void Writer::Impl::writeBone(BinaryWriter& writer, const Bone& bone) {
    writeNode(writer, bone.node);
    writer.write(bone.geosetId);
    writer.write(bone.geosetAnimationId);
}

void Writer::Impl::writeNode(BinaryWriter& writer, const Node& node) {
    SizeEnclosure const sizeEnclosure(writer);

    writer.writeString(node.name, 80);
    writer.write(node.objectId);
    writer.write(node.parentId);
    writer.write(static_cast<u32>(node.flags));

    writeNodeTracks(writer, node);
}

void Writer::Impl::writeNodeTracks(BinaryWriter& writer, const Node& node) {
    forEachNodeTrack(
        node, [&writer](u32 tag, const auto& track) { writeTrackChunk(writer, tag, track); });
}

void Writer::Impl::writeLITE(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, LITE_TAG);

    for (const auto& light : mdx.lights) {
        writeLight(writer, light, mdx);
    }
}

void Writer::Impl::writeLight(BinaryWriter& writer, const Light& light, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer);

    writeNode(writer, light.node);
    writer.write(static_cast<u32>(light.type));
    if (mdx.version >= 1300)
        writer.write<u32>(light.shadowCasting ? 1u : 0u);
    writer.write(light.attenuationStart);
    writer.write(light.attenuationEnd);
    writer.write(light.color);
    writer.write(light.intensity);
    writer.write(light.ambientColor);
    writer.write(light.ambientIntensity);
    if (mdx.version >= 1200) {
        writer.write(light.shadowIntensity);
    }
    if (mdx.version >= 1300) {
        writer.write(light.shadowCastingStart);
        writer.write(light.shadowCastingEnd);
    }
    if (mdx.version >= 1600) {
        writer.write(light.quadraticFalloff);
        writer.write(light.linearFalloff);
        writer.write(light.damping);
    }

    writeOwnTracks(writer, light, mdx.version);
}

void Writer::Impl::writeHELP(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, HELP_TAG);

    for (const auto& helper : mdx.helpers) {
        writeHelper(writer, helper);
    }
}

void Writer::Impl::writeHelper(BinaryWriter& writer, const Helper& helper) {
    writeNode(writer, helper.node);
}

void Writer::Impl::writeATCH(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, ATCH_TAG);

    for (const auto& att : mdx.attachments) {
        writeAttachment(writer, att);
    }
}

void Writer::Impl::writeAttachment(BinaryWriter& writer, const Attachment& att) {
    SizeEnclosure const sizeEnclosure(writer);

    writeNode(writer, att.node);
    writer.writeString(att.path, 260);
    writer.write(att.attachmentId);

    writeOwnTracks(writer, att, 0);
}

void Writer::Impl::writePIVT(BinaryWriter& writer, const Model& mdx) {
    size_t const size = mdx.pivotPoints.size() * 12;
    writeChunkHeader(writer, PIVT_TAG, size);
    writer.write(mdx.pivotPoints);
}

void Writer::Impl::writePREM(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, PREM_TAG);

    for (const auto& pem : mdx.particleEmitters) {
        writeParticleEmitter(writer, pem);
    }
}

void Writer::Impl::writeParticleEmitter(BinaryWriter& writer, const ParticleEmitter& pem) {
    SizeEnclosure const sizeEnclosure(writer);

    writeNode(writer, pem.node);
    writer.write(pem.emissionRate);
    writer.write(pem.gravity);
    writer.write(pem.longitude);
    writer.write(pem.latitude);
    writer.writeString(pem.spawnModelFileName, 260);
    writer.write(pem.lifespan);
    writer.write(pem.initialVelocity);

    writeOwnTracks(writer, pem, 0);
}

void Writer::Impl::writePRE2(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, PRE2_TAG);

    for (const auto& pem2 : mdx.particleEmitters2) {
        writeParticleEmitter2(writer, pem2);
    }
}

void Writer::Impl::writeParticleEmitter2(BinaryWriter& writer, const ParticleEmitter2& pem2) {
    SizeEnclosure const sizeEnclosure(writer);

    writeNode(writer, pem2.node);
    writer.write(pem2.speed);
    writer.write(pem2.variation);
    writer.write(pem2.latitude);
    writer.write(pem2.gravity);
    writer.write(pem2.lifespan);
    writer.write(pem2.emissionRate);
    writer.write(pem2.length);
    writer.write(pem2.width);

    writer.write(pem2.filterMode);
    writer.write(pem2.rows);
    writer.write(pem2.columns);
    writer.write(pem2.headOrTail);

    writer.write(pem2.tailLength);
    writer.write(pem2.time);

    for (int i = 0; i < 3; i++) {
        writer.write(pem2.segmentColor[i]);
    }
    for (int i = 0; i < 3; i++) {
        writer.write(pem2.segmentAlpha[i]);
    }
    for (int i = 0; i < 3; i++) {
        writer.write(pem2.segmentScaling[i]);
    }

    for (int i = 0; i < 3; i++) {
        writer.write(pem2.headInterval[i]);
    }
    for (int i = 0; i < 3; i++) {
        writer.write(pem2.headDecayInterval[i]);
    }
    for (int i = 0; i < 3; i++) {
        writer.write(pem2.tailInterval[i]);
    }
    for (int i = 0; i < 3; i++) {
        writer.write(pem2.tailDecayInterval[i]);
    }

    writer.write(pem2.textureId);
    writer.write(pem2.squirt);
    writer.write(pem2.priorityPlane);
    writer.write(pem2.replaceableId);

    writeOwnTracks(writer, pem2, 0);
}

void Writer::Impl::writeRIBB(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, RIBB_TAG);

    for (const auto& ribb : mdx.ribbonEmitters) {
        writeRibbonEmitter(writer, ribb);
    }
}

void Writer::Impl::writeRibbonEmitter(BinaryWriter& writer, const RibbonEmitter& ribb) {
    SizeEnclosure const sizeEnclosure(writer);

    writeNode(writer, ribb.node);
    writer.write(ribb.heightAbove);
    writer.write(ribb.heightBelow);
    writer.write(ribb.alpha);
    writer.write(ribb.color);
    writer.write(ribb.lifespan);
    writer.write(ribb.textureSlot);
    writer.write(ribb.emissionRate);
    writer.write(ribb.rows);
    writer.write(ribb.columns);
    writer.write(ribb.materialId);
    writer.write(ribb.gravity);

    writeOwnTracks(writer, ribb, 0);
}

void Writer::Impl::writeEVTS(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, EVTS_TAG);

    for (const auto& evt : mdx.eventObjects) {
        writeEventObject(writer, evt);
    }
}

void Writer::Impl::writeEventObject(BinaryWriter& writer, const EventObject& evt) {
    writeNode(writer, evt.node);

    writer.write(KEVT_TAG);
    writer.write(static_cast<u32>(evt.eventTrackTimes.size()));
    writer.write(evt.globalSequenceId);
    writer.write(evt.eventTrackTimes);
}

void Writer::Impl::writeCAMS(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, CAMS_TAG);

    for (const auto& cam : mdx.cameras) {
        writeCamera(writer, cam, mdx);
    }
}

void Writer::Impl::writeCamera(BinaryWriter& writer, const Camera& cam, const Model& mdx) {
    // 3.0.0 stamps variant 3 on every camera it writes; a 2.x client reads the
    // whole word as the size, so earlier versions keep it clear.
    SizeEnclosure const sizeEnclosure(writer, 0, mdx.version >= 1800 ? kCameraVariant3 : 0u);

    writer.writeString(cam.name, 80);
    writer.write(cam.position);
    writer.write(cam.fieldOfView);
    writer.write(cam.farClippingPlane);
    writer.write(cam.nearClippingPlane);
    writer.write(cam.targetPosition);

    writeOwnTracks(writer, cam, mdx.version);
}

void Writer::Impl::writeCLID(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, CLID_TAG);

    for (const auto& shape : mdx.collisionShapes) {
        writeCollisionShape(writer, shape);
    }
}

void Writer::Impl::writeCollisionShape(BinaryWriter& writer, const CollisionShape& shape) {
    writeNode(writer, shape.node);
    writer.write(collisionShapeFileType(shape.type));

    // Each type's exact bytes, whatever the vector holds: the chunk has no
    // sizes, so one float too many puts the reader in the middle of the next
    // shape (structures.h has the layouts).
    const auto vertex = [&shape](std::size_t i) {
        return i < shape.vertices.size() ? shape.vertices[i] : Vector3f{};
    };
    switch (shape.type) {
    case CollisionShape::ShapeType::Cylinder:
        writer.write(vertex(0));
        writer.write(shape.vertices.size() > 1 ? vertex(1).z - vertex(0).z : 0.0f);
        writer.write(shape.radius);
        break;
    case CollisionShape::ShapeType::Sphere:
        writer.write(vertex(0));
        writer.write(shape.radius);
        break;
    case CollisionShape::ShapeType::Plane:
        writer.write(vertex(1).x - vertex(0).x);
        writer.write(vertex(1).y - vertex(0).y);
        break;
    case CollisionShape::ShapeType::Box:
    default:
        writer.write(vertex(0));
        writer.write(vertex(1));
        break;
    }
}

void Writer::Impl::writeBPOS(BinaryWriter& writer, const Model& mdx) {
    size_t const size = 4 + mdx.bindPoses.size() * 48; // count + matrices
    writeChunkHeader(writer, BPOS_TAG, size);

    writer.write(static_cast<u32>(mdx.bindPoses.size()));
    for (const auto& pose : mdx.bindPoses) {
        for (int i = 0; i < 12; i++) {
            writer.write(pose[i]);
        }
    }
}

void Writer::Impl::writeFFX(BinaryWriter& writer, const Model& mdx) {
    size_t const size = mdx.faceEffects.size() * 340;
    writeChunkHeader(writer, FAFX_TAG, size);

    for (const auto& fx : mdx.faceEffects) {
        writer.writeString(fx.name, 80);
        writer.writeString(fx.path, 260);
    }
}

void Writer::Impl::writeCORN(BinaryWriter& writer, const Model& mdx) {
    SizeEnclosure const sizeEnclosure(writer, CORN_TAG);

    for (const auto& corn : mdx.cornEmitters) {
        writeCornEmitter(writer, corn);
    }
}

void Writer::Impl::writeCornEmitter(BinaryWriter& writer, const CornEmitter& corn) {
    SizeEnclosure const sizeEnclosure(writer);

    writeNode(writer, corn.node);
    writer.write(corn.lifeSpan);
    writer.write(corn.emissionRate);
    writer.write(corn.speed);
    writer.write(corn.color.x);
    writer.write(corn.color.y);
    writer.write(corn.color.z);
    writer.write(corn.alpha);
    writer.write(corn.replaceableId);
    writer.writeString(corn.path, 260);
    writer.writeString(corn.animVisibilityGuide, 260);

    writeOwnTracks(writer, corn, 0);
}

} // namespace mdx
} // namespace whiteout
