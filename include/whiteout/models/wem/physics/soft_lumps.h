// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#pragma once

/**
 * @file soft_lumps.h
 * @brief A soft body's lumps, its lanes, and the skin a file holds
 *        (EDIT_MODE_SOFTBODY_DESIGN.md §5.3, §6).
 *
 * **Lumps.** One for every bone the region rides: the bone itself where it is
 * an end of the skeleton and the region's alone, else a copy that hangs from
 * it (`BoneLumps`). *Detail* more are spread over the surface of the largest
 * bones (`MakeSurfaceLumps`). A bone a lump of the surface hangs from is
 * copied too (`MakeSoftLumps`): a lump has no child bone.
 *
 * **Lanes.** A point's weight on a bone is shared out among that bone's lumps
 * and nowhere else (`BindLumps`): with every lump at rest the model is its own
 * skin, for any lanes and any softness. How many lumps a bone's share may go
 * to is what the point's weight slots leave (§5.3's table).
 *
 * **One place each.** A point's lanes are `BindLumps`' alone. A point's
 * influences, for the editor's deform and for a file, are `SoftSkinner`'s.
 */

#include <array>
#include <span>
#include <vector>

#include <whiteout/common_types.h>
#include <whiteout/vector_types.h>

#include "../model.h"
#include "../nodes/node.h"
#include "../skinning/solid.h"
#include "soft.h"
#include "soft_region.h"

namespace whiteout {
namespace models {
namespace wem {

/// Where @p node's bind frame puts its origin, in mesh space: a pivot rig's
/// pivot.
Vector3f SoftBonePivot(const NodeTree& tree, u32 node);

/// The matrix a lump moves its points by before its family's bone does (row
/// vectors): @p departure's scale and turn about @p pivot, then its
/// translation, all along the model's axes at the bind. The identity at rest.
Matrix44f SoftDepartureMatrix(const Vector3f& pivot, const Transform& departure);

/// The softness each of a point's bones gives up (§5.3): the point's softness
/// is handed over heaviest bone first, so the shares sum to it, and a point
/// with no slot to spare (@p influences holds @p cap bones or more) is all
/// stiff below a half and all soft from it. One value per influence.
std::array<f32, 8> SoftShares(std::span<const geom::Influence> influences, f32 softness, u32 cap);

/// Whether a point riding @p bones bones under @p cap weights a point can only
/// be all soft or all stiff.
constexpr bool SoftAllOrNone(u32 bones, u32 cap) {
    return bones >= cap;
}

/// One point of a soft body: a welded point of one mesh, its seam twins
/// together.
struct SoftPoint {
    u32 mesh = 0;
    u32 point = 0;   ///< In the mesh's `PointTable`.
    u32 vertex = 0;  ///< Its first member.
    Vector3f at{0, 0, 0};
    Vector3f outward{0, 0, 1}; ///< Unit, out of the body's solid (§5.4).
    f32 area = 0.0f;           ///< A third of each of its region triangles.
    f32 softness = 0.0f;       ///< As asked for; 1 on a part that rides rigidly, whatever its layer holds.
    u32 skinStart = 0;         ///< Into `SoftPoints::skin`.
    u32 skinCount = 0;
    u16 rider = 0;             ///< The part that carries it rigidly: its place in the body's `riders` + 1.
};

/// A soft body's points, gathered from its meshes' `soft.body` and
/// `soft.softness` layers, with their skin under a cap.
struct SoftPoints {
    std::vector<SoftPoint> points;
    std::vector<geom::Influence> skin; ///< Heaviest first per point.
    /// Per mesh of the model, per vertex: its entry in `points`, or `kInvalidIndex`.
    std::vector<std::vector<u32>> pointOfVertex;
    Vector3f low{0, 0, 0};
    Vector3f high{0, 0, 0};
    f32 size = 0.0f; ///< The largest side of the box.
    f32 area = 0.0f;
    u32 cap = 4;     ///< The weights a point the skin was read under.
    /// The solid of the body's meshes (`skinning/solid.h`): what "outward" and
    /// a distance through the body are read from.
    skinning::SolidGrid grid;
    std::vector<u8> solid;

    /// Whether @p position is in the solid.
    bool inside(const Vector3f& position) const;

    std::span<const geom::Influence> skinOf(const SoftPoint& point) const {
        return std::span<const geom::Influence>(skin).subspan(point.skinStart, point.skinCount);
    }
};

/// @p body's points on @p model, their skin under @p cap weights a point. A
/// point a face of one of the body's `riders` draws is that part's (§5.4).
SoftPoints GatherSoftPoints(const Model& model, u32 body, u32 cap);

/// §6.2's first two columns, in node order: a lump for every bone @p points
/// ride. A bone is used as it is when it is an end of the skeleton and the
/// region's alone; else it is copied. @p copyAll copies every one;
/// @p chainsAsTheyAre uses a bone with skinned bones below it as it is, when
/// nothing outside the region rides it or them. A bone in @p own is the
/// body's own making (a Drip's strand): no animation keys it, so it is used
/// as it is whatever @p copyAll says.
///
/// **A point of the region that is not fully soft counts as one outside it**
/// for the bones it keeps weight on: in the edge blend or painted down, it
/// keeps that skin on the bone as the animation has it, so the bone is copied
/// and the copy takes the soft share. Used as it is, the bone would carry the
/// stiff point wherever an effect sends the soft ones.
std::vector<SoftLump> BoneLumps(const Model& model, const SoftPoints& points, bool copyAll, bool chainsAsTheyAre,
                                std::span<const u32> own = {});

/// What `MakeSurfaceLumps` did.
struct SoftSurfaceReport {
    u32 made = 0;
    u32 bones = 0;      ///< How many bones took some.
    f32 reach = 0.0f;   ///< The share of the region's soft area those bones carry.
};

/**
 * @brief §6.2's third column: @p detail lumps spread over the surface, appended
 *        to @p lumps.
 *
 * They go to the region's bones by soft area, largest first, three each, and
 * what is left one more each to the largest again. Within a bone's share they
 * are spread farthest-first from the softest point, by distance through the
 * body's solid, then evened by two passes that move each to the middle of the
 * points nearest it. The order is fixed: the same points give the same lumps.
 * A bone in @p without takes none: a Drip's strand is its three bones' alone.
 * A rider's points take none either: they are their part's own lumps'.
 */
SoftSurfaceReport MakeSurfaceLumps(const Model& model, const SoftPoints& points, u32 detail,
                                   std::vector<SoftLump>& lumps, std::span<const u32> without = {});

/// How far from a place's middle its six lumps stand, of its radius (§6.2).
constexpr f32 kSoftPlaceSpoke = 0.6f;
/// The lumps a placed effect claims: one at its middle and six round it.
constexpr u32 kSoftPlaceLumps = 7;

/// The most lumps a travelling Bulge's rows take in all (§6.2).
constexpr u32 kSoftRowLumps = 24;

/// Whether @p effect asks for lumps of its own at its place (§6.2): a Bulge, or
/// an Inflate toward a place, said along the surface and placed. Not a Bulge
/// from contact, whose place is wherever a collider presses, nor one that
/// travels, which asks for rows (`SoftEffectClaimsRow`).
bool SoftEffectClaims(const SoftEffect& effect);

/// Whether @p effect asks for rows of lumps along its line (§6.2): a Bulge
/// that travels, said along the surface, down a chain of bones or a line drawn
/// over the surface.
bool SoftEffectClaimsRow(const SoftEffect& effect);

/// Whether two states of an effect differ in what it claims, at its place or
/// along its line: what a body's lumps are made again for.
bool SoftClaimsDiffer(const SoftEffect& before, const SoftEffect& after);

/// What lumps of its own would do for an effect's place (§6.2).
struct SoftPlaceRoom {
    bool valid = false;      ///< The place is on the body.
    u32 bone = kInvalidNode; ///< The bone that carries most of its disc: what its lumps hang from.
    /// The share of the disc's soft area that can give them two lanes or more
    /// on that bone. The seven serve the one bone they hang from, and a point
    /// of a crowded skin has no lane to give them.
    f32 room = 0.0f;

    /// They pay on a small place with room: where half the disc or more can.
    bool pays() const {
        return valid && room >= 0.5f;
    }
};

/// @p effect's place on @p points, and whether lumps of its own would pay.
SoftPlaceRoom SoftPlaceRoomOf(const Model& model, const SoftPoints& points, const SoftEffect& effect);

/**
 * @brief The lumps @p body's placed effects claim (§6.2), appended to @p lumps:
 *        seven for each effect that asks (`SoftEffectClaims`) and whose place
 *        has room (`SoftPlaceRoom::pays`), in the effects' order.
 *
 * One at the place's middle and six round it at `kSoftPlaceSpoke` of its
 * radius, each on the body's point nearest there that faces as the place does,
 * hung from the bone that carries most of the place. None under a classic
 * group's equal shares, which cannot say their pattern. Returns how many.
 *
 * **A Bulge that travels takes rows along its line** (`SoftEffectClaimsRow`):
 * for each bone whose soft points its swell reaches, lumps on the line over
 * the stretch of it those points lie by, one every half radius, hung from that
 * bone. A bone whose points lie by less than two such steps is at that grain
 * already and takes none, as does a place on the line no point lies by. The
 * step widens until the rows are `kSoftRowLumps` in all or fewer.
 */
u32 MakePlaceLumps(const Model& model, const SoftPoints& points, const SoftBody& body, bool equalShares,
                   std::vector<SoftLump>& lumps);

/// What `BindLumps` found.
struct SoftBindReport {
    u32 points = 0;
    u32 allOrNone = 0;        ///< Points with a softness strictly between 0 and 1 that can only be all soft or all stiff.
    std::array<u32, 4> byBones{}; ///< Points riding one, two, three, and four or more bones.
    f32 worstReturn = 0.0f;   ///< The farthest any point's lanes land from it, model units.
};

/**
 * @brief Solves @p body's lanes for @p cap weights a point (§5.3, §6.4), and
 *        writes them: the `soft.bind.lump` and `soft.bind.weight` layers of its
 *        meshes, `influences`, `equalShares`, and each lump's pivot.
 *
 * Within a family a point's lanes are non-negative and sum to the share of
 * the bone's weight its softness gives up. One lane is the bone's own lump.
 * Two or three are solved so that the lumps' anchors, weighed by the lanes,
 * land on the point; the families of one point are solved together.
 *
 * A rider's point is not solved (§5.4): each of its bones' weights goes whole
 * to its part's lump of that bone, so the part moves as one piece a bone.
 *
 * **A place's seven bind their disc by a pattern,** not by solved lanes (§6.2):
 * a hub and six spokes. A point of the disc takes the hub and the two spokes it
 * lies between, as many as it has slots for, and its bone's own lump the rest,
 * which is all of it at the rim. Only on the bone the seven hang from.
 *
 * **A travelling Bulge's row binds by where a point lies along the line:** all
 * of its bone's share on the two lumps of that bone's row it lies between, or
 * on the nearer where it has one slot. The bone's own lump keeps none of it,
 * so its slot is the row's.
 */
SoftBindReport BindLumps(Model& model, SoftBody& body, u32 cap, bool equalShares);

/// A part of the model outside a soft body that lies in or against it (§5.4):
/// a slime's skeleton, an elemental's bracelets.
struct SoftInside {
    SoftPart part;  ///< A mesh, or the faces of one material in it.
    u32 points = 0; ///< How many of its vertices are in the body's solid.
};

/// The parts of @p model that lie in or against the solid of @p points' own
/// faces and are not those points': what the tool offers the body to keep
/// clear of. By mesh, then by material.
std::vector<SoftInside> SoftPartsInside(const Model& model, const SoftPoints& points);

/// Sets each of @p body's lumps' `clear`: the least distance at the bind
/// between a point the lump carries and a point of the parts in
/// `body.inside`; 0 when it keeps clear of nothing.
void MeasureSoftClear(const Model& model, SoftBody& body);

/// What `MakeSoftLumps` made.
struct SoftMakeReport {
    u32 asTheyAre = 0; ///< Bones used as they are.
    u32 copies = 0;
    u32 riders = 0;    ///< Lumps made for the parts that ride rigidly: one a part a bone.
    u32 claimed = 0;   ///< Lumps the effects claimed: seven a place, and a travelling Bulge's rows (§6.2).
    SoftSurfaceReport surface;
    SoftBindReport bind;
};

/**
 * @brief Makes @p body's lumps and lanes from its points, which its meshes'
 *        `soft.body` and `soft.softness` layers hold: `BoneLumps` by
 *        `copyAll` and `chainsAsTheyAre`, `MakeSurfaceLumps` for `detail`,
 *        a rigid lump for each bone each of `riders` rides (§5.4), then
 *        `BindLumps`. What Make and Remake both run.
 *
 * The lumps the effects' places claim (`MakePlaceLumps`) come last, made
 * again from the places as they stand: a place moved moves its seven, and the
 * lumps before them keep their numbers.
 *
 * **A bone that carries a lump of the surface is copied,** whatever
 * `BoneLumps` found (§6.5, a lump has no child bone): a bake keys a bone used
 * as it is in place, and a surface lump, a rider's or a place's that hangs
 * from it would then take the bone's departure on top of its own.
 */
SoftMakeReport MakeSoftLumps(Model& model, SoftBody& body, u32 cap, bool equalShares);

/**
 * @brief What @p body's effects' places claim made again as they now stand,
 *        and its lanes bound again, by the weights a point and the shares it
 *        was last bound for.
 *
 * Every other lump keeps its number: a place set, moved, sized or taken away
 * is not a remake. A bone the claimed lumps come to hang from is copied, and
 * one they leave is used as it is again where `BoneLumps` says so. Returns
 * whether any lump changed: a clip baked with the body then holds tracks of
 * lumps that are no more, or keys of a bone that is no lump now.
 */
bool ClaimSoftPlaces(Model& model, SoftBody& body);

/**
 * @brief What a softness stroke ends with: each bone's lump given its kind
 *        again for the softness as it stands, and the lanes bound again.
 *
 * Every lump keeps its number. A bone a point was painted down on becomes a
 * copy, and one painted up again is used as it is where `BoneLumps` says so.
 * Returns whether any lump changed, as `ClaimSoftPlaces` does and for the same
 * reason.
 */
bool SettleSoftLumps(Model& model, SoftBody& body, u32 cap, bool equalShares);

/// A file's weight for a profile: floats, bytes that sum to 255, or a classic
/// group's equal shares.
enum class SoftWeights : u8 { Float, Bytes, EqualShares };

/**
 * @brief A soft body's points as a file, or the editor's deform, skins them
 *        (§6.3).
 *
 * The palette is the model's nodes, then one slot a lump at `lumpBase() +`
 * its index. A bone used as it is has no slot of its own: its points stay on
 * the bone. A cursor over the model, as `Animator` is: it must not outlive it
 * or an edit.
 */
class SoftSkinner {
public:
    SoftSkinner(const Model& model, const SoftBody& body, SoftWeights weights = SoftWeights::Float);

    /// The first lump's slot: the model's node count.
    u32 lumpBase() const {
        return lumpBase_;
    }

    /// Whether @p vertex of @p mesh is one of the body's points.
    bool holds(u32 mesh, u32 vertex) const;

    /// @p vertex's influences, heaviest first. A vertex that is no point of
    /// the body gets its own skin under the body's cap. Valid until the next
    /// call.
    std::span<const geom::Influence> of(u32 mesh, u32 vertex);

private:
    struct MeshLayers {
        std::span<const u32> body;
        std::span<const f32> softness;
        std::span<const std::array<u32, 4>> lumps;
        std::span<const std::array<f32, 4>> weights;
    };

    const SoftBody* body_ = nullptr;
    SoftWeights weights_ = SoftWeights::Float;
    u32 lumpBase_ = 0;
    SoftSkinSource skin_;
    std::vector<MeshLayers> layers_;
    std::vector<u32> ownLump_; ///< Per node: its own lump's index, or `kInvalidIndex`.
    std::vector<geom::Influence> scratch_;
};

} // namespace wem
} // namespace models
} // namespace whiteout
