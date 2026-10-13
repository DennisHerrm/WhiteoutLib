// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file track_walk.h
 * @brief Every animation track of a model, in the order its file holds them.
 *
 * One list for whatever has to see every track. The writer writes each track
 * as it is visited, so the list cannot drift from the file; the loader's guard
 * (`loadable.h`) and the export's pruning (`prune.h`) read and renumber
 * through the same walk.
 *
 * A walk calls `visit(tag, track)` with the track's chunk tag, and `track` is
 * a `Track<f32>`, `Track<u32>`, `Track<Vector3f>` or `Track<Quaternion>`: the
 * visitor is a generic callable. Which tracks a record's bytes hold depends on
 * the file's version, so the walks that vary take it, and visit only what a
 * file of that version holds. A track the model carries for another version is
 * not in the file, and is not visited.
 *
 * Templates on the record, so a const model yields const tracks.
 */

#include <concepts>
#include <type_traits>

#include "structures.h"
#include "types.h"

namespace whiteout {
namespace mdx {

/// @p R is `T` or `const T`.
template <class R, class T>
concept RecordOf = std::same_as<std::remove_const_t<R>, T>;

/// A node's three transform tracks. Every node record starts with them.
template <RecordOf<Node> R, class Visit>
void forEachNodeTrack(R& node, Visit&& visit) {
    visit(KGTR_TAG, node.translationTracks);
    visit(KGRT_TAG, node.rotationTracks);
    visit(KGSC_TAG, node.scalingTracks);
}

// ----------------------------------------------------------------------------
// A record's own tracks: the ones after its fixed fields, node tracks left out
// ----------------------------------------------------------------------------

/// The tracks at the end of a layer. Below v1100 the flipbook is one of them;
/// from v1100 it sits on each sub-texture instead (`forEachLayerTrack`).
template <RecordOf<Layer> R, class Visit>
void forEachOwnTrack(R& layer, u32 version, Visit&& visit) {
    if (version < 1100) {
        visit(KMTF_TAG, layer.textureIdTracks);
    }
    visit(KMTA_TAG, layer.alphaTracks);
    // The emissive gain is v900's and the fresnel triple v1000's, as the
    // fields they animate are (`Parser::parseLayer`).
    if (version > 800) {
        visit(KMTE_TAG, layer.emissiveGainTracks);
    }
    if (version > 900) {
        visit(KFC3_TAG, layer.fresnelColorTracks);
        visit(KFCA_TAG, layer.fresnelAlphaTracks);
        visit(KFTC_TAG, layer.fresnelTeamColorTracks);
    }
}

template <RecordOf<TextureAnimation> R, class Visit>
void forEachOwnTrack(R& animation, u32 /*version*/, Visit&& visit) {
    visit(KTAT_TAG, animation.translationTracks);
    visit(KTAR_TAG, animation.rotationTracks);
    visit(KTAS_TAG, animation.scalingTracks);
}

template <RecordOf<GeosetAnimation> R, class Visit>
void forEachOwnTrack(R& animation, u32 /*version*/, Visit&& visit) {
    visit(KGAO_TAG, animation.alphaTracks);
    visit(KGAC_TAG, animation.colorTracks);
}

template <RecordOf<Light> R, class Visit>
void forEachOwnTrack(R& light, u32 version, Visit&& visit) {
    visit(KLAS_TAG, light.attenuationStartTracks);
    visit(KLAE_TAG, light.attenuationEndTracks);
    visit(KLAC_TAG, light.colorTracks);
    visit(KLAI_TAG, light.intensityTracks);
    visit(KLBI_TAG, light.ambientIntensityTracks);
    visit(KLBC_TAG, light.ambientColorTracks);
    visit(KLAV_TAG, light.visibilityTracks);
    if (version >= 1300) {
        visit(KLSS_TAG, light.shadowCastingStartTracks);
        visit(KLSE_TAG, light.shadowCastingEndTracks);
    }
    if (version >= 1600) {
        visit(KLQF_TAG, light.quadraticFalloffTracks);
        visit(KLLF_TAG, light.linearFalloffTracks);
        visit(KLDA_TAG, light.dampingTracks);
    }
}

template <RecordOf<Attachment> R, class Visit>
void forEachOwnTrack(R& attachment, u32 /*version*/, Visit&& visit) {
    visit(KATV_TAG, attachment.visibilityTracks);
}

template <RecordOf<ParticleEmitter> R, class Visit>
void forEachOwnTrack(R& emitter, u32 /*version*/, Visit&& visit) {
    visit(KPEE_TAG, emitter.emissionRateTracks);
    visit(KPEG_TAG, emitter.gravityTracks);
    visit(KPLN_TAG, emitter.longitudeTracks);
    visit(KPLT_TAG, emitter.latitudeTracks);
    visit(KPEL_TAG, emitter.lifespanTracks);
    visit(KPES_TAG, emitter.speedTracks);
    visit(KPEV_TAG, emitter.visibilityTracks);
}

template <RecordOf<ParticleEmitter2> R, class Visit>
void forEachOwnTrack(R& emitter, u32 /*version*/, Visit&& visit) {
    visit(KP2S_TAG, emitter.speedTracks);
    visit(KP2R_TAG, emitter.variationTracks);
    visit(KP2L_TAG, emitter.latitudeTracks);
    visit(KP2G_TAG, emitter.gravityTracks);
    visit(KP2E_TAG, emitter.emissionRateTracks);
    visit(KP2N_TAG, emitter.lengthTracks);
    visit(KP2W_TAG, emitter.widthTracks);
    visit(KP2V_TAG, emitter.visibilityTracks);
}

template <RecordOf<RibbonEmitter> R, class Visit>
void forEachOwnTrack(R& emitter, u32 /*version*/, Visit&& visit) {
    visit(KRHA_TAG, emitter.heightAboveTracks);
    visit(KRHB_TAG, emitter.heightBelowTracks);
    visit(KRAL_TAG, emitter.alphaTracks);
    visit(KRCO_TAG, emitter.colorTracks);
    visit(KRTX_TAG, emitter.textureSlotTracks);
    visit(KRVS_TAG, emitter.visibilityTracks);
}

template <RecordOf<CornEmitter> R, class Visit>
void forEachOwnTrack(R& emitter, u32 /*version*/, Visit&& visit) {
    visit(KPPL_TAG, emitter.lifeSpanTracks);
    visit(KPPE_TAG, emitter.emissionRateTracks);
    visit(KPPS_TAG, emitter.speedTracks);
    visit(KPPC_TAG, emitter.colorTracks);
    visit(KPPA_TAG, emitter.alphaTracks);
    visit(KPPV_TAG, emitter.visibilityTracks);
}

/// A camera is no node: these are all the tracks it has.
template <RecordOf<Camera> R, class Visit>
void forEachOwnTrack(R& camera, u32 version, Visit&& visit) {
    visit(KCTR_TAG, camera.positionTracks);
    visit(KCRL_TAG, camera.targetRotationTracks);
    visit(KTTR_TAG, camera.targetPositionTracks);
    // The 3.0 client reads these at any version; Classic knows none of them.
    if (version > 800) {
        visit(KCVS_TAG, camera.visibilityTracks);
        visit(IDUF_TAG, camera.focusDistanceTracks);
        visit(ELAF_TAG, camera.focalLengthTracks);
        visit(PTSF_TAG, camera.fStopTracks);
    }
}

template <RecordOf<SoundEmitter> R, class Visit>
void forEachOwnTrack(R& emitter, u32 /*version*/, Visit&& visit) {
    visit(KSEK_TAG, emitter.soundTrack);
}

// ----------------------------------------------------------------------------
// Whole records, and the whole model
// ----------------------------------------------------------------------------

/// Every track of a layer: from v1100 each sub-texture's flipbook, which the
/// file interleaves with the sub-texture's fields, then the layer's own.
template <RecordOf<Layer> R, class Visit>
void forEachLayerTrack(R& layer, u32 version, Visit&& visit) {
    if (version >= 1100) {
        for (auto& sub : layer.subTextures) {
            visit(KMTF_TAG, sub.tracks);
        }
    }
    forEachOwnTrack(layer, version, visit);
}

/// Every track @p model's file holds, in the file's order.
template <RecordOf<Model> R, class Visit>
void forEachTrack(R& model, Visit&& visit) {
    const u32 version = model.version;
    // A node record: the node's tracks, then its own.
    const auto nodes = [&](auto& records) {
        for (auto& record : records) {
            forEachNodeTrack(record.node, visit);
        }
    };
    const auto withOwn = [&](auto& records) {
        for (auto& record : records) {
            forEachNodeTrack(record.node, visit);
            forEachOwnTrack(record, version, visit);
        }
    };

    for (auto& material : model.materials) {
        for (auto& layer : material.layers) {
            forEachLayerTrack(layer, version, visit);
        }
    }
    withOwn(model.soundEmitters);
    for (auto& animation : model.textureAnimations) {
        forEachOwnTrack(animation, version, visit);
    }
    for (auto& animation : model.geosetAnimations) {
        forEachOwnTrack(animation, version, visit);
    }
    nodes(model.bones);
    withOwn(model.lights);
    nodes(model.helpers);
    withOwn(model.attachments);
    withOwn(model.particleEmitters);
    withOwn(model.particleEmitters2);
    withOwn(model.cornEmitters);
    withOwn(model.ribbonEmitters);
    for (auto& camera : model.cameras) {
        forEachOwnTrack(camera, version, visit);
    }
    nodes(model.eventObjects);
    nodes(model.collisionShapes);
}

} // namespace mdx
} // namespace whiteout
