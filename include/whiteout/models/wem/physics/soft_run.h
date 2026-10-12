// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_run.h
 * @brief Where a model's soft points are, given its pose and its lumps'
 *        departures (EDIT_MODE_SOFTBODY_DESIGN.md §6.3, §10).
 *
 * `SoftDeformer` is the one place a soft mesh is posed: the editor's view, the
 * bake's measure and the extents all read it, so what is measured is what is
 * drawn. It skins by `SoftSkinner`'s influences, which are a file's.
 *
 * `RunSoft` is a body over a whole clip (§11.1): what the bake writes and a
 * preview plays. `SoftDeparturesAt` reads a baked clip's lump tracks back, so a
 * baked clip is drawn by the same deformer.
 */

#include <atomic>
#include <span>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "../anim/animator.h"
#include "../document.h"
#include "../model.h"
#include "../nodes/node.h"
#include "soft.h"
#include "soft_lumps.h"
#include "soft_modes.h"
#include "soft_step.h"

namespace whiteout {
namespace models {
namespace wem {

/// A model's soft meshes, ready to pose. A cursor over the model, as
/// `Animator` is: it must not outlive it, an edit of its skin, or a remake.
class SoftDeformer {
public:
    explicit SoftDeformer(const Model& model, SoftWeights weights = SoftWeights::Float);

    /// Whether any vertex of @p mesh is a soft body's point.
    bool soft(u32 mesh) const {
        return mesh < written_.size() && !written_[mesh].empty();
    }
    /// The meshes that are, ascending.
    std::span<const u32> meshes() const {
        return meshes_;
    }

    /// The slot of lump @p lump of the model's @p body-th soft body in a
    /// palette: after the model's nodes, the bodies' lumps in order.
    u32 slot(u32 body, u32 lump) const {
        return body < base_.size() ? base_[body] + lump : kInvalidIndex;
    }
    /// How many slots a palette has: the nodes, then every lump.
    u32 paletteSize() const {
        return paletteSize_;
    }

    /**
     * @brief The matrices the soft meshes are skinned by: @p skin, a pose's
     *        skinning matrices by node, then each lump's.
     *
     * @p departures holds, body after body in the set's order, one transform a
     * lump: its departure from its family's bone, the identity at rest. A span
     * shorter than that reads as rest from where it ends. A bone used as it is
     * has no slot of its own: its departure is applied to the bone's matrix
     * here, so pass none for it when the pose already carries it.
     */
    std::vector<Matrix44f> palette(std::span<const Matrix44f> skin, std::span<const Transform> departures) const;

    /// Every vertex of @p mesh posed by @p palette: a soft body's points by
    /// their lumps, the rest by their own skin under the body's cap.
    std::vector<Vector3f> positions(u32 mesh, std::span<const Matrix44f> palette) const;

    /// @p mesh's corner normals posed the same way, indexed like its Halfedge
    /// `normal` layer. They turn with the lumps; they are not computed again
    /// from the moved surface, because a game does not (§6.6).
    std::vector<Vector3f> normals(u32 mesh, std::span<const Matrix44f> palette) const;

    /// What @p vertex of @p mesh binds, in palette slots. Empty for a mesh
    /// that is not soft.
    std::span<const geom::Influence> influences(u32 mesh, u32 vertex) const;

private:
    const Model* model_ = nullptr;
    std::vector<u32> meshes_;
    std::vector<u32> base_; ///< Per body, its first lump's slot.
    u32 paletteSize_ = 0;
    /// Per mesh, per vertex, its influences in palette slots; empty for a mesh
    /// with no soft point.
    std::vector<std::vector<std::vector<geom::Influence>>> written_;
};

/// One soft body's lumps over a clip, a step at a time (§11.1).
struct SoftRun {
    u32 body = 0;
    u32 lumps = 0;
    u32 steps = 0; ///< `steps + 1` are recorded: the clip's frame 0 to its end.
    f32 dt = 0.0f;
    /// Per recorded step, each lump's departure from its family's bone.
    std::vector<Transform> departures;
    /// Per recorded step, whether any effect asked for anything.
    std::vector<u8> acts;
    /// The fit over the clip: the worst miss of any step, and the largest move.
    f32 worstMiss = 0.0f;
    f32 largestMove = 0.0f;
    /// What its Jiggle did, where one acts in the clip (§8.3).
    bool swings = false;
    SoftJiggleResult swing; ///< Its measures; `swing.swing` is left empty.

    const Transform* at(u32 step) const {
        return step <= steps && lumps != 0 ? departures.data() + static_cast<std::size_t>(step) * lumps : nullptr;
    }
    bool moves() const;
    f32 follows() const {
        return largestMove > 1e-12f ? worstMiss / largestMove : 0.0f;
    }
};

/// The modes a caller found for a body, kept across the clips it runs and the
/// edits between them: finding them is the one slow part of a Jiggle, and they
/// depend on neither the clip nor the Jiggle's other sliders. An entry is read
/// only while the body's `SoftRigPrint` is the one it was made for. One thread's.
struct SoftModesCache {
    struct Entry {
        u32 model = 0;
        u32 body = 0;
        u64 print = 0;
        f32 holdShape = 0.0f;
        f32 keepVolume = 0.0f;
        bool keepInPlace = false;
        SoftModes modes;
    };
    std::vector<Entry> entries;

    /// @p rig's modes under @p settings, found once.
    const SoftModes& of(u32 model, const SoftRig& rig, const SoftVolume& volume, const SoftJiggle& settings);
};

/// What a body's swing is thrown by over a clip, and what it is added to: the
/// part of `RunSoft` that does not depend on the Jiggle's settings.
struct SoftGoalRun {
    SoftRun run;           ///< The goal: every effect but the Jiggle and a Wobble on *Random*.
    SoftJiggleDrive drive; ///< Empty when no Jiggle acts in the clip.
    u32 jiggle = 0;        ///< The Jiggle's effect id, or 0.
    /// A Wobble on *Random* that acts in the clip (§8.4): per step, how far it
    /// moves the body there, model units (its Amount by its level).
    struct Random {
        u32 effect = 0;
        std::vector<f32> amount;
    };
    std::vector<Random> random;
    f32 duration = 0.0f; ///< The clip's, and whether it loops.
    bool loops = false;
    /// What the swing's volume trim poses the body by again (§8.3): the bones
    /// its points ride, per step each one's skinning matrix, and per step the
    /// volume the goal has there. Empty when no Jiggle acts in the clip.
    std::vector<u32> ridden;
    std::vector<Matrix44f> skins;
    std::vector<f64> volume;

    /// Whether `SwingSoft` has anything to add: what a body's own ways of
    /// swinging are needed for.
    bool swings() const {
        return jiggle != 0 || !random.empty();
    }
};

/// Whether @p body's motion over a clip is found over the whole of it
/// (`SwingSoft`) and not a step at a time: it has a Jiggle that is on, or a
/// Wobble on *Random* that is, which is its own ways of swinging (§8.4).
bool SoftRunsWhole(const SoftBody& body);

/// The first half of `RunSoft`: the stack with no memory, a step at a time.
/// @p cancel, set, stops it between two steps: what it returns then is no
/// run of the body (`run.lumps` is 0).
SoftGoalRun RunSoftGoal(const Document& document, u32 model, u32 body, u32 clip, u32 steps, f32 dt,
                        const SoftPoseAt& poseAt, const std::atomic<bool>* cancel = nullptr);

/// The second half: @p goal with the body's Jiggle on top (§7's step 6), and
/// its Wobbles on *Random* on top of that. It reads the Jiggle's settings as
/// they now are, so a slider of it runs this alone. @p cache may be null. The
/// body's ways of swinging are its Jiggle's mix, or the default one's where it
/// has none.
///
/// What the Jiggle's swing changes of the volume the goal had at a step is
/// taken back by its *Keep volume*, all of it at 1 (§7's trim, on what comes
/// out): an even change of the lumps' scales, with no memory, so a loop stays
/// closed.
SoftRun SwingSoft(const Document& document, u32 model, u32 body, const SoftGoalRun& goal, SoftModesCache* cache);

/**
 * @brief Body @p body of @p model over @p clip, @p steps steps of @p dt.
 *
 * @p poseAt gives the pose the clip plays at a time: in a bake, as the keys it
 * just wrote play it. The stack with no memory is one pass; what swings is
 * solved over the whole clip on top of it, a loop from the state it maps onto
 * itself (§11.3).
 */
SoftRun RunSoft(const Document& document, u32 model, u32 body, u32 clip, u32 steps, f32 dt, const SoftPoseAt& poseAt,
                SoftModesCache* cache = nullptr, const std::atomic<bool>* cancel = nullptr);

/// Whether @p departure leaves a lump where its family's bone has it, for a
/// body of @p size: under a hundred-thousandth of it.
bool SoftAtRest(const Transform& departure, f32 size);

/// Per node of @p model, whether a lump's family bone is it or hangs below it.
std::vector<u8> SoftLeadsToLump(const Model& model);

/// The bones a bake keys below @p body's bones used as they are (§6.5): each
/// child of one that leads to a lump and is no bone used as it is itself. A
/// bake stands it where the bodies were stepped on it, wherever the bone above
/// departs. Ascending.
std::vector<u32> SoftBonesHeldBelow(const Model& model, u32 body);

/// The channel that holds lump @p lump of body @p body's @p which track
/// (`SoftLumpTranslation`, `SoftLumpRotation` or `SoftLumpScale`), or
/// `kInvalidIndex` when no bake declared it.
u32 SoftLumpChannel(const Model& model, u32 body, u32 lump, Channel which);

/// Whether @p clip holds a lump's track: a bake left its soft bodies in it.
bool ClipKeysSoftLumps(const Document& document, u32 clip);

/**
 * @brief The pose @p clip plays at @p seconds, as its soft bodies are stepped
 *        on it: `sample`, `compose` and the constraint and IK stages, which
 *        keep no state. No physics stage.
 *
 * In a baked clip, a bone a soft body keyed in place is read from what that
 * bake replaced, so a step on this pose gives the bake's own departures and
 * not those on top of themselves. @p animator is @p document's, for the
 * clip's model.
 */
void SoftSourcePose(const Document& document, const Animator& animator, u32 clip, f32 seconds, Pose& out);

/// `SoftSourcePose`'s own part, on a pose already sampled: in @p out's locals,
/// each bone a soft body's bake keyed in place in @p clip put back to what
/// that bake replaced, at @p seconds. Returns whether it put any back: the
/// frames are then the caller's to compose again. False where the clip is not
/// so baked.
bool SoftSourceLocals(const Document& document, const Animator& animator, u32 clip, f32 seconds, Pose& out);

/**
 * @brief Every lump's departure at @p seconds of @p clip as its baked tracks
 *        hold it, body after body in the set's order: what
 *        `SoftDeformer::palette` takes.
 *
 * A lump with no track is at rest. So is a bone used as it is: its bake is the
 * bone's own keys, and the pose carries it.
 */
std::vector<Transform> SoftDeparturesAt(const Document& document, u32 model, u32 clip, f32 seconds);

} // namespace wem
} // namespace models
} // namespace whiteout
