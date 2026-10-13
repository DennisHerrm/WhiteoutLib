// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/mdx/prune.h>
#include <whiteout/models/mdx/track_walk.h>
#include <whiteout/models/mdx/writer.h>

#include <algorithm>
#include <map>

namespace whiteout {
namespace mdx {

namespace {

constexpr u32 kNone = 0xFFFFFFFFu;
constexpr u32 kNoGlobalSequence = Track<f32>::kNoGlobalSequence;

/// The last record a bone can link: the game keeps the link in a byte, and
/// 255 there means none (`IAnimCreateObjects`).
constexpr u32 kLastLinkableRecord = 254;

/// Whether @p record never hides what follows it: opaque, unkeyed, and no
/// drop shadow, which is the one flag the game reads and reads through a bone.
bool AlwaysVisible(const GeosetAnimation& record) {
    return record.alpha == 1.0f && !record.alphaTracks.isUsed &&
           !hasFlag(record.flags, GeosetAnimation::Flag::DropShadow);
}

/// Whether @p record states nothing at all once no bone links it.
bool Inert(const GeosetAnimation& record) {
    return AlwaysVisible(record) && !record.colorTracks.isUsed &&
           record.color == Vector3f(1.0f, 1.0f, 1.0f);
}

/// Per geoset animation, whether a bone is linked to it. A link counts only
/// on a bone that names a geoset: the game keeps none for one that does not.
std::vector<u8> LinkedRecords(const Model& model) {
    std::vector<u8> linked(model.geosetAnimations.size(), 0);
    for (const Bone& bone : model.bones) {
        if (bone.geosetId != Bone::MULTIPLE_GEOSETS && bone.geosetAnimationId < linked.size()) {
            linked[bone.geosetAnimationId] = 1;
        }
    }
    return linked;
}

} // namespace

void removeGeosetAnimations(Model& model, const std::vector<u8>& remove) {
    if (std::none_of(remove.begin(), remove.end(), [](u8 flag) { return flag != 0; })) {
        return;
    }
    std::vector<u32> movedTo(model.geosetAnimations.size(), kNone);
    std::vector<GeosetAnimation> kept;
    for (std::size_t r = 0; r < model.geosetAnimations.size(); ++r) {
        if (r < remove.size() && remove[r] != 0) {
            continue;
        }
        movedTo[r] = static_cast<u32>(kept.size());
        kept.push_back(std::move(model.geosetAnimations[r]));
    }
    model.geosetAnimations = std::move(kept);
    for (Bone& bone : model.bones) {
        if (bone.geosetAnimationId < movedTo.size()) {
            bone.geosetAnimationId = movedTo[bone.geosetAnimationId];
        }
    }
}

void pruneGeosetAnimations(Model& model, PruneReport& report) {
    // A link to a record that never hides: the bone follows an alpha that
    // never leaves 1.
    for (Bone& bone : model.bones) {
        if (bone.geosetAnimationId < model.geosetAnimations.size() &&
            AlwaysVisible(model.geosetAnimations[bone.geosetAnimationId])) {
            if (bone.geosetId != Bone::MULTIPLE_GEOSETS) {
                ++report.boneLinks;
            }
            bone.geosetAnimationId = kNone;
        }
    }

    std::vector<u8> linked = LinkedRecords(model);
    std::vector<u8> remove(model.geosetAnimations.size(), 0);
    for (std::size_t r = 0; r < remove.size(); ++r) {
        if (linked[r] == 0 && Inert(model.geosetAnimations[r])) {
            remove[r] = 1;
            ++report.geosetAnimations;
        }
    }
    removeGeosetAnimations(model, remove);

    // The records bones link go first when one of them is out of the byte's
    // reach: nothing else reads a record's place.
    linked = LinkedRecords(model);
    bool pastByte = false;
    for (std::size_t r = kLastLinkableRecord + 1; r < linked.size(); ++r) {
        pastByte = pastByte || linked[r] != 0;
    }
    if (!pastByte) {
        return;
    }
    std::vector<u32> order(linked.size());
    for (u32 r = 0; r < order.size(); ++r) {
        order[r] = r;
    }
    std::stable_partition(order.begin(), order.end(), [&linked](u32 r) { return linked[r] != 0; });
    std::vector<u32> movedTo(order.size(), kNone);
    std::vector<GeosetAnimation> sorted;
    sorted.reserve(order.size());
    for (const u32 r : order) {
        movedTo[r] = static_cast<u32>(sorted.size());
        sorted.push_back(std::move(model.geosetAnimations[r]));
    }
    model.geosetAnimations = std::move(sorted);
    for (Bone& bone : model.bones) {
        if (bone.geosetAnimationId < movedTo.size()) {
            bone.geosetAnimationId = movedTo[bone.geosetAnimationId];
        }
    }
    const std::size_t links =
        static_cast<std::size_t>(std::count(linked.begin(), linked.end(), u8{1}));
    if (links > kLastLinkableRecord + 1) {
        report.linksPastByte = static_cast<u32>(links - (kLastLinkableRecord + 1));
    }
}

u32 pruneEmptyGeosets(Model& model) {
    // A ladder of levels of detail is matched by count: the game switches
    // levels only while each holds as many geosets as the first. A level that
    // simplified to nothing is an empty geoset there on purpose.
    for (const Geoset& geoset : model.geosets) {
        if (geoset.lod != model.geosets.front().lod) {
            return 0;
        }
    }
    const std::vector<u8> linked = LinkedRecords(model);
    std::vector<u8> gone(model.geosets.size(), 0);
    for (std::size_t g = 0; g < gone.size(); ++g) {
        gone[g] = model.geosets[g].faces.empty() ? 1 : 0;
    }
    // A record a bone is linked to keeps the geoset it names.
    for (std::size_t r = 0; r < model.geosetAnimations.size(); ++r) {
        const u32 geoset = model.geosetAnimations[r].geosetId;
        if (linked[r] != 0 && geoset < gone.size()) {
            gone[geoset] = 0;
        }
    }
    const u32 removed = static_cast<u32>(std::count(gone.begin(), gone.end(), u8{1}));
    if (removed == 0) {
        return 0;
    }

    std::vector<u32> movedTo(gone.size(), kNone);
    std::vector<Geoset> kept;
    kept.reserve(gone.size() - removed);
    for (std::size_t g = 0; g < gone.size(); ++g) {
        if (gone[g] != 0) {
            continue;
        }
        movedTo[g] = static_cast<u32>(kept.size());
        kept.push_back(std::move(model.geosets[g]));
    }
    model.geosets = std::move(kept);

    std::vector<u8> remove(model.geosetAnimations.size(), 0);
    for (std::size_t r = 0; r < remove.size(); ++r) {
        u32& geoset = model.geosetAnimations[r].geosetId;
        if (geoset >= movedTo.size()) {
            continue; // names no geoset, before and after
        }
        if (movedTo[geoset] == kNone) {
            remove[r] = 1;
        } else {
            geoset = movedTo[geoset];
        }
    }
    removeGeosetAnimations(model, remove);
    // A bone's own geoset is read against "none" alone, so it only has to stay
    // a geoset: the one its record names where its own went.
    for (Bone& bone : model.bones) {
        if (bone.geosetId >= movedTo.size()) {
            continue;
        }
        if (movedTo[bone.geosetId] != kNone) {
            bone.geosetId = movedTo[bone.geosetId];
        } else if (bone.geosetAnimationId < model.geosetAnimations.size()) {
            bone.geosetId = model.geosetAnimations[bone.geosetAnimationId].geosetId;
        } else {
            bone.geosetId = Bone::MULTIPLE_GEOSETS;
        }
    }
    return removed;
}

namespace {

std::vector<u8> UsedMaterials(const Model& model) {
    std::vector<u8> used(model.materials.size(), 0);
    for (const Geoset& geoset : model.geosets) {
        if (geoset.materialId < used.size()) {
            used[geoset.materialId] = 1;
        }
    }
    for (const RibbonEmitter& ribbon : model.ribbonEmitters) {
        if (ribbon.materialId < used.size()) {
            used[ribbon.materialId] = 1;
        }
    }
    return used;
}

} // namespace

u32 mergeIdenticalMaterials(Model& model) {
    const std::vector<u8> usedBefore = UsedMaterials(model);
    // The first material with each record, by the bytes the file would hold.
    std::map<std::vector<u8>, u32> firstOf;
    std::vector<u32> sameAs(model.materials.size());
    bool any = false;
    for (u32 m = 0; m < sameAs.size(); ++m) {
        sameAs[m] =
            firstOf.try_emplace(Writer::materialRecord(model.materials[m], model.version), m)
                .first->second;
        any = any || sameAs[m] != m;
    }
    if (!any) {
        return 0;
    }
    for (Geoset& geoset : model.geosets) {
        if (geoset.materialId < sameAs.size()) {
            geoset.materialId = sameAs[geoset.materialId];
        }
    }
    for (RibbonEmitter& ribbon : model.ribbonEmitters) {
        if (ribbon.materialId < sameAs.size()) {
            ribbon.materialId = sameAs[ribbon.materialId];
        }
    }
    const std::vector<u8> usedAfter = UsedMaterials(model);
    u32 freed = 0;
    for (std::size_t m = 0; m < usedAfter.size(); ++m) {
        freed += usedBefore[m] != 0 && usedAfter[m] == 0 ? 1u : 0u;
    }
    return freed;
}

u32 pruneUnusedMaterials(Model& model) {
    const std::vector<u8> used = UsedMaterials(model);
    const u32 pruned = static_cast<u32>(std::count(used.begin(), used.end(), u8{0}));
    if (pruned == 0) {
        return 0;
    }
    std::vector<u32> movedTo(used.size(), 0);
    std::vector<Material> kept;
    kept.reserve(used.size() - pruned);
    for (std::size_t m = 0; m < used.size(); ++m) {
        movedTo[m] = static_cast<u32>(kept.size());
        if (used[m] != 0) {
            kept.push_back(std::move(model.materials[m]));
        }
    }
    model.materials = std::move(kept);
    // An id past the table named nothing before and keeps naming nothing.
    for (Geoset& geoset : model.geosets) {
        if (geoset.materialId < movedTo.size()) {
            geoset.materialId = movedTo[geoset.materialId];
        }
    }
    for (RibbonEmitter& ribbon : model.ribbonEmitters) {
        if (ribbon.materialId < movedTo.size()) {
            ribbon.materialId = movedTo[ribbon.materialId];
        }
    }
    return pruned;
}

u32 pruneUnusedGlobalSequences(Model& model) {
    std::vector<u8> used(model.globalSequences.size(), 0);
    forEachTrack(model, [&used](u32, const auto& track) {
        if (track.isUsed && track.globalSequenceId < used.size()) {
            used[track.globalSequenceId] = 1;
        }
    });
    for (const EventObject& event : model.eventObjects) {
        if (event.globalSequenceId < used.size()) {
            used[event.globalSequenceId] = 1;
        }
    }
    const u32 pruned = static_cast<u32>(std::count(used.begin(), used.end(), u8{0}));
    if (pruned == 0) {
        return 0;
    }
    std::vector<u32> movedTo(used.size(), kNoGlobalSequence);
    std::vector<u32> kept;
    for (std::size_t g = 0; g < used.size(); ++g) {
        if (used[g] != 0) {
            movedTo[g] = static_cast<u32>(kept.size());
            kept.push_back(model.globalSequences[g]);
        }
    }
    model.globalSequences = std::move(kept);
    // An unused track is renumbered with the rest: marked used later, it must
    // not come back on another's clock.
    forEachTrack(model, [&movedTo](u32, auto& track) {
        if (track.globalSequenceId < movedTo.size()) {
            track.globalSequenceId = movedTo[track.globalSequenceId];
        }
    });
    for (EventObject& event : model.eventObjects) {
        if (event.globalSequenceId < movedTo.size()) {
            event.globalSequenceId = movedTo[event.globalSequenceId];
        }
    }
    return pruned;
}

PruneReport pruneModel(Model& model) {
    PruneReport report;
    // The geoset animations first: an empty geoset goes only once no bone is
    // linked to its record, and a cleared link is what frees it.
    pruneGeosetAnimations(model, report);
    report.geosets = pruneEmptyGeosets(model);
    // Then the materials, which the geosets that went no longer hold on to.
    report.materialsUnused = pruneUnusedMaterials(model);
    report.materialsMerged = mergeIdenticalMaterials(model);
    pruneUnusedMaterials(model);
    report.globalSequences = pruneUnusedGlobalSequences(model);
    return report;
}

} // namespace mdx
} // namespace whiteout
