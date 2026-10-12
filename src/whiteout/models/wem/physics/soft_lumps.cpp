// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_lumps.h>

#include <whiteout/models/wem/geometry/attributes.h>
#include <whiteout/models/wem/reflect_bytes.h>
#include <whiteout/models/wem/skinning/points.h>
#include <whiteout/models/wem/skinning/quantize.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "soft_common.h"

namespace whiteout {
namespace models {
namespace wem {

namespace {

using softdetail::ForEachTriangle;
using softdetail::kNoWeight;
using softdetail::kTravelReach;
using softdetail::OnLine;
using softdetail::PositionsOf;

/// The cells along the longest side of a body's solid.
constexpr u32 kSolidResolution = 48;

/// The candidates a family's lanes are chosen among: its own lump and the
/// surface lumps nearest the point.
constexpr std::size_t kLaneCandidates = 6;

/// How much a lane's distance from its point counts against it, so that of
/// two answers that land alike the nearer lumps win (§6.4).
constexpr f64 kNearness = 1e-6;

f64 Dot(const Vector3f& a, const Vector3f& b) {
    return static_cast<f64>(a.x) * b.x + static_cast<f64>(a.y) * b.y + static_cast<f64>(a.z) * b.z;
}

/// The closest point to @p q of the segment @p a @p b, as weights on its ends.
std::array<f64, 2> OnSegment(const Vector3f& a, const Vector3f& b, const Vector3f& q) {
    const Vector3f ab = b - a;
    const f64 length = Dot(ab, ab);
    if (length <= 1e-30) {
        return {1.0, 0.0};
    }
    const f64 t = std::clamp(Dot(q - a, ab) / length, 0.0, 1.0);
    return {1.0 - t, t};
}

/// The closest point to @p q of the triangle @p a @p b @p c, as weights on
/// its corners (Ericson's regions).
std::array<f64, 3> OnTriangle(const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& q) {
    const Vector3f ab = b - a, ac = c - a, aq = q - a;
    const f64 d1 = Dot(ab, aq), d2 = Dot(ac, aq);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return {1.0, 0.0, 0.0};
    }
    const Vector3f bq = q - b;
    const f64 d3 = Dot(ab, bq), d4 = Dot(ac, bq);
    if (d3 >= 0.0 && d4 <= d3) {
        return {0.0, 1.0, 0.0};
    }
    const f64 vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const f64 v = d1 / (d1 - d3);
        return {1.0 - v, v, 0.0};
    }
    const Vector3f cq = q - c;
    const f64 d5 = Dot(ab, cq), d6 = Dot(ac, cq);
    if (d6 >= 0.0 && d5 <= d6) {
        return {0.0, 0.0, 1.0};
    }
    const f64 vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const f64 w = d2 / (d2 - d6);
        return {1.0 - w, 0.0, w};
    }
    const f64 va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const f64 w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return {0.0, 1.0 - w, w};
    }
    const f64 sum = va + vb + vc;
    if (std::abs(sum) <= 1e-30) {
        // A sliver: its longest side answers.
        const std::array<f64, 2> on = OnSegment(a, b, q);
        return {on[0], on[1], 0.0};
    }
    const f64 v = vb / sum, w = vc / sum;
    return {1.0 - v - w, v, w};
}

/// One family of a point while its lanes are solved.
struct Family {
    u32 bone = kInvalidNode;
    f64 handed = 0.0;            ///< The share of the point's weight its softness gives up.
    u32 lanes = 0;               ///< How many lumps the share may go to.
    u32 own = kInvalidIndex;     ///< The bone's own lump.
    std::vector<u32> candidates; ///< Lump indices: the own lump first, then surface lumps, nearest first.
    std::vector<std::pair<u32, f64>> chosen; ///< Lump and its share of `handed`, summing to 1.
    Vector3f lands{0, 0, 0};     ///< Where the chosen lanes' anchors land.
};

/// The @p lanes lumps of @p family, and their shares, whose anchors land
/// nearest @p target; of answers that land alike, the lumps nearer @p point.
void SolveFamily(Family& family, std::span<const SoftLump> lumps, const Vector3f& target, const Vector3f& point) {
    family.chosen.clear();
    const std::vector<u32>& c = family.candidates;
    if (c.empty()) {
        family.lands = point;
        return;
    }
    if (family.lanes <= 1 || c.size() == 1) {
        // One lane is the bone's own lump.
        family.chosen.push_back({c[0], 1.0});
        family.lands = lumps[c[0]].anchor;
        return;
    }
    f64 best = std::numeric_limits<f64>::infinity();
    std::array<u32, 3> bestLumps{};
    std::array<f64, 3> bestShares{};
    u32 bestCount = 0;
    const auto consider = [&](u32 count, const std::array<u32, 3>& which, const std::array<f64, 3>& shares) {
        Vector3f lands{0, 0, 0};
        f64 near = 0.0;
        for (u32 k = 0; k < count; ++k) {
            const Vector3f& anchor = lumps[which[k]].anchor;
            lands = lands + anchor * static_cast<f32>(shares[k]);
            near += shares[k] * Dot(anchor - point, anchor - point);
        }
        const f64 cost = Dot(lands - target, lands - target) + kNearness * near;
        if (cost < best) {
            best = cost;
            bestLumps = which;
            bestShares = shares;
            bestCount = count;
        }
    };
    for (std::size_t i = 0; i < c.size(); ++i) {
        consider(1, {c[i], 0, 0}, {1.0, 0.0, 0.0});
        for (std::size_t j = i + 1; j < c.size(); ++j) {
            const std::array<f64, 2> two = OnSegment(lumps[c[i]].anchor, lumps[c[j]].anchor, target);
            consider(2, {c[i], c[j], 0}, {two[0], two[1], 0.0});
            for (std::size_t k = j + 1; family.lanes >= 3 && k < c.size(); ++k) {
                consider(3, {c[i], c[j], c[k]},
                         OnTriangle(lumps[c[i]].anchor, lumps[c[j]].anchor, lumps[c[k]].anchor, target));
            }
        }
    }
    family.lands = Vector3f{0, 0, 0};
    for (u32 k = 0; k < bestCount; ++k) {
        if (bestShares[k] > 1e-9) {
            family.chosen.push_back({bestLumps[k], bestShares[k]});
        }
        family.lands = family.lands + lumps[bestLumps[k]].anchor * static_cast<f32>(bestShares[k]);
    }
}

/// How many slots a point's families take before any lane beyond the own
/// lump: a stiff bone one, a soft one its own lump, and a bone that has only
/// partly given way both. A bone used as it is counts two then as well: what
/// it keeps stays on it, and its lanes may all go to the surface.
u32 SlotsTaken(std::span<const Family> families, std::span<const f32> shares) {
    u32 taken = 0;
    for (std::size_t i = 0; i < families.size(); ++i) {
        taken += families[i].own != kInvalidIndex && shares[i] > 0.0f && shares[i] < 1.0f ? 2 : 1;
    }
    return taken;
}

} // namespace

Vector3f SoftBonePivot(const NodeTree& tree, u32 node) {
    if (node >= tree.size()) {
        return Vector3f{0, 0, 0};
    }
    const Matrix44f bind = Matrix44f::inverse(tree.inverseBindMatrix(node));
    return Vector3f{bind.data[3][0], bind.data[3][1], bind.data[3][2]};
}

Matrix44f SoftDepartureMatrix(const Vector3f& pivot, const Transform& departure) {
    Transform turn;
    turn.rotation = departure.rotation;
    turn.scale = departure.scale;
    Matrix44f m = ToMatrix(turn);
    // p' = (p - pivot) * S * R + pivot + t.
    const Vector3f back{pivot.x * m.data[0][0] + pivot.y * m.data[1][0] + pivot.z * m.data[2][0],
                        pivot.x * m.data[0][1] + pivot.y * m.data[1][1] + pivot.z * m.data[2][1],
                        pivot.x * m.data[0][2] + pivot.y * m.data[1][2] + pivot.z * m.data[2][2]};
    m.data[3][0] = pivot.x - back.x + departure.translation.x;
    m.data[3][1] = pivot.y - back.y + departure.translation.y;
    m.data[3][2] = pivot.z - back.z + departure.translation.z;
    return m;
}

std::array<f32, 8> SoftShares(std::span<const geom::Influence> influences, f32 softness, u32 cap) {
    std::array<f32, 8> shares{};
    const std::size_t count = std::min<std::size_t>(influences.size(), shares.size());
    if (!(softness > 0.0f) || count == 0) {
        return shares;
    }
    if (SoftAllOrNone(static_cast<u32>(influences.size()), cap)) {
        for (std::size_t i = 0; i < count; ++i) {
            shares[i] = softness >= 0.5f ? 1.0f : 0.0f;
        }
        return shares;
    }
    // Heaviest first: the shares times the weights sum to the softness.
    f32 left = std::min(softness, 1.0f);
    for (std::size_t i = 0; i < count; ++i) {
        const f32 weight = influences[i].weight;
        if (!(weight > 0.0f)) {
            continue;
        }
        const f32 share = left >= weight - 1e-6f ? 1.0f : std::max(left, 0.0f) / weight;
        shares[i] = share;
        left -= share * weight;
    }
    return shares;
}

bool SoftPoints::inside(const Vector3f& position) const {
    const std::array<i32, 3> cell = grid.cellOf(position);
    return grid.contains(cell[0], cell[1], cell[2]) && solid[grid.index(cell[0], cell[1], cell[2])] != 0;
}

namespace {

/// The points of @p out a face of one of @p riders draws take that part's
/// place in the list, the first that draws them (§5.4). Such a point is wholly
/// soft, whatever its layer holds: a part that rides follows the effects whole.
void MarkRiders(const Model& model, std::span<const SoftPart> riders, SoftPoints& out) {
    for (std::size_t r = 0; r < riders.size() && r < 0xFFFE; ++r) {
        const SoftPart& part = riders[r];
        if (part.mesh >= model.meshes.size() || part.mesh >= out.pointOfVertex.size()) {
            continue;
        }
        const std::vector<u32>& entryOf = out.pointOfVertex[part.mesh];
        const Mesh& mesh = model.meshes[part.mesh];
        const std::span<const u32> sectionOf = mesh.faceSections();
        const geom::FaceSet& faces = mesh.faceSet();
        std::size_t corner = 0;
        for (std::size_t f = 0; f < faces.faceCount(); ++f) {
            const u32 valence = faces.faceValence[f];
            const u32 section = f < sectionOf.size() ? sectionOf[f] : 0;
            const u32 material = section < mesh.sections.size() ? mesh.sections[section].materialSlot : kInvalidIndex;
            if (part.material == kInvalidIndex || part.material == material) {
                for (u32 k = 0; k < valence; ++k) {
                    const u32 v = faces.cornerVertex[corner + k];
                    if (v < entryOf.size() && entryOf[v] != kInvalidIndex && out.points[entryOf[v]].rider == 0) {
                        out.points[entryOf[v]].rider = static_cast<u16>(r + 1);
                        out.points[entryOf[v]].softness = 1.0f;
                    }
                }
            }
            corner += valence;
        }
    }
}

SoftPoints GatherWith(const Model& model, u32 body, u32 cap, std::span<const SoftPart> riders);

} // namespace

SoftPoints GatherSoftPoints(const Model& model, u32 body, u32 cap) {
    const SoftBody* record = model.physics.softBody(body);
    return GatherWith(model, body, cap, record ? std::span<const SoftPart>(record->riders) : std::span<const SoftPart>());
}

namespace {

SoftPoints GatherWith(const Model& model, u32 body, u32 cap, std::span<const SoftPart> riders) {
    SoftPoints out;
    out.cap = std::max<u32>(1, cap);
    out.pointOfVertex.resize(model.meshes.size());
    SoftSkinSource skin(model, out.cap);
    std::vector<const Mesh*> meshes;
    std::vector<Vector3f> normals; // Per entry of `points`, by the faces' winding.
    bool any = false;
    Vector3f boxLow{0, 0, 0}, boxHigh{0, 0, 0}; // Of the whole meshes: the solid's grid.
    bool anyBox = false;

    for (u32 m = 0; m < model.meshes.size(); ++m) {
        const Mesh& mesh = model.meshes[m];
        const std::span<const u32> bodyOf = mesh.attributes.get<u32>(geom::names::kSoftBody, geom::Domain::Vertex);
        if (std::find(bodyOf.begin(), bodyOf.end(), body) == bodyOf.end()) {
            continue;
        }
        meshes.push_back(&mesh);
        const std::span<const Vector3f> positions = PositionsOf(mesh);
        const std::span<const f32> softness =
            mesh.attributes.get<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex);
        const skinning::PointTable table = skinning::BuildPointTable(mesh);
        std::vector<u32>& entryOf = out.pointOfVertex[m];
        entryOf.assign(positions.size(), kInvalidIndex);
        std::vector<u32> entryOfPoint(table.pointCount, kInvalidIndex);
        for (const Vector3f& p : positions) {
            boxLow = anyBox ? Vector3f{std::min(boxLow.x, p.x), std::min(boxLow.y, p.y), std::min(boxLow.z, p.z)} : p;
            boxHigh = anyBox ? Vector3f{std::max(boxHigh.x, p.x), std::max(boxHigh.y, p.y), std::max(boxHigh.z, p.z)} : p;
            anyBox = true;
        }
        for (u32 v = 0; v < positions.size() && v < bodyOf.size(); ++v) {
            if (bodyOf[v] != body || v >= table.pointOf.size()) {
                continue;
            }
            const u32 point = table.pointOf[v];
            u32& entry = entryOfPoint[point];
            if (entry == kInvalidIndex) {
                entry = static_cast<u32>(out.points.size());
                SoftPoint made;
                made.mesh = m;
                made.point = point;
                made.vertex = v;
                made.at = positions[v];
                const std::span<const geom::Influence> influences = skin.of(m, v);
                made.skinStart = static_cast<u32>(out.skin.size());
                made.skinCount = static_cast<u32>(influences.size());
                out.skin.insert(out.skin.end(), influences.begin(), influences.end());
                out.points.push_back(made);
                normals.push_back(Vector3f{0, 0, 0});
                const Vector3f& p = made.at;
                out.low = any ? Vector3f{std::min(out.low.x, p.x), std::min(out.low.y, p.y), std::min(out.low.z, p.z)} : p;
                out.high = any ? Vector3f{std::max(out.high.x, p.x), std::max(out.high.y, p.y), std::max(out.high.z, p.z)} : p;
                any = true;
            }
            entryOf[v] = entry;
            if (v < softness.size()) {
                out.points[entry].softness = std::max(out.points[entry].softness, std::clamp(softness[v], 0.0f, 1.0f));
            }
        }
        ForEachTriangle(mesh, [&](u32, u32 a, u32 b, u32 c) {
            if (a >= entryOf.size() || b >= entryOf.size() || c >= entryOf.size() || entryOf[a] == kInvalidIndex ||
                entryOf[b] == kInvalidIndex || entryOf[c] == kInvalidIndex) {
                return;
            }
            const Vector3f normal = cross(positions[b] - positions[a], positions[c] - positions[a]);
            const f32 third = 0.5f * normal.length() / 3.0f;
            for (const u32 v : {a, b, c}) {
                out.points[entryOf[v]].area += third;
                normals[entryOf[v]] = normals[entryOf[v]] + normal;
            }
        });
    }
    if (out.points.empty()) {
        return out;
    }
    const Vector3f extent = out.high - out.low;
    out.size = std::max({extent.x, extent.y, extent.z});
    for (const SoftPoint& point : out.points) {
        out.area += point.area;
    }

    // Outward: the stored winding's normal, turned to point out of the solid.
    out.grid = skinning::MakeSolidGrid(boxLow, boxHigh, kSolidResolution);
    out.solid = skinning::BuildSolid(out.grid, meshes);
    for (std::size_t i = 0; i < out.points.size(); ++i) {
        SoftPoint& point = out.points[i];
        const f32 length = normals[i].length();
        Vector3f n = length > 1e-20f ? normals[i] * (1.0f / length) : Vector3f{0, 0, 1};
        u32 ahead = 0, behind = 0;
        for (const f32 cells : {1.5f, 2.5f}) {
            const Vector3f step = n * (cells * out.grid.cell);
            ahead += out.inside(point.at + step) ? 1 : 0;
            behind += out.inside(point.at - step) ? 1 : 0;
        }
        if (ahead > behind) {
            n = n * -1.0f;
        }
        point.outward = n;
    }
    MarkRiders(model, riders, out);
    return out;
}

/// §5.4: a lump for each bone each rider's points ride, appended to @p lumps
/// by part, then by bone: at the middle of what it carries, and rigid. Returns
/// how many.
u32 MakeRiderLumps(const SoftPoints& points, std::vector<SoftLump>& lumps) {
    struct Held {
        u16 rider = 0;
        u32 bone = kInvalidNode;
        f64 mass = 0.0;
        f64 sum[3] = {0.0, 0.0, 0.0};
    };
    std::vector<Held> held;
    for (const SoftPoint& point : points.points) {
        if (point.rider == 0) {
            continue;
        }
        for (const geom::Influence& influence : points.skinOf(point)) {
            if (!(influence.weight > 0.0f)) {
                continue;
            }
            auto at = std::find_if(held.begin(), held.end(), [&](const Held& h) {
                return h.rider == point.rider && h.bone == influence.bone;
            });
            if (at == held.end()) {
                Held made;
                made.rider = point.rider;
                made.bone = influence.bone;
                held.push_back(made);
                at = held.end() - 1;
            }
            const f64 mass = std::max(static_cast<f64>(point.area), 1e-12) * influence.weight;
            at->mass += mass;
            at->sum[0] += mass * point.at.x;
            at->sum[1] += mass * point.at.y;
            at->sum[2] += mass * point.at.z;
        }
    }
    std::sort(held.begin(), held.end(),
              [](const Held& a, const Held& b) { return a.rider != b.rider ? a.rider < b.rider : a.bone < b.bone; });
    for (const Held& h : held) {
        SoftLump lump;
        lump.kind = SoftLumpKind::Surface;
        lump.carrier = h.bone;
        lump.anchor = Vector3f{static_cast<f32>(h.sum[0] / h.mass), static_cast<f32>(h.sum[1] / h.mass),
                               static_cast<f32>(h.sum[2] / h.mass)};
        lump.rest = lump.anchor;
        lump.rigid = true;
        lump.rider = h.rider;
        lumps.push_back(lump);
    }
    return static_cast<u32>(held.size());
}

} // namespace

std::vector<SoftLump> BoneLumps(const Model& model, const SoftPoints& points, bool copyAll, bool chainsAsTheyAre,
                                std::span<const u32> own) {
    // The region as vertices, for the count that says what else rides each bone.
    std::vector<SoftMeshPoints> region;
    for (u32 m = 0; m < points.pointOfVertex.size(); ++m) {
        SoftMeshPoints part;
        part.mesh = m;
        for (u32 v = 0; v < points.pointOfVertex[m].size(); ++v) {
            if (points.pointOfVertex[m][v] != kInvalidIndex) {
                part.vertices.push_back(v);
            }
        }
        if (!part.vertices.empty()) {
            region.push_back(std::move(part));
        }
    }
    const SoftCount count = CountSoftRegion(model, region, points.cap);
    std::vector<SoftBoneCount> bones = count.bones;
    std::sort(bones.begin(), bones.end(),
              [](const SoftBoneCount& a, const SoftBoneCount& b) { return a.node < b.node; });
    const NodeTree& tree = model.nodes;
    // A point that keeps some of its own skin, in the edge blend or painted
    // down, keeps it on the bone as the animation has it. That bone is not the
    // soft part's alone, and neither is one above it: used as it is, an effect
    // would move the stiff point with the soft ones.
    std::vector<u8> keeps(tree.size(), 0);
    for (const SoftPoint& point : points.points) {
        if (point.rider != 0 || !(point.softness < 1.0f)) {
            continue;
        }
        const std::span<const geom::Influence> skin = points.skinOf(point);
        const std::array<f32, 8> shares = SoftShares(skin, point.softness, points.cap);
        for (std::size_t i = 0; i < std::min<std::size_t>(skin.size(), shares.size()); ++i) {
            if (skin[i].weight > 0.0f && shares[i] < 1.0f - 1e-4f && skin[i].bone < keeps.size()) {
                keeps[skin[i].bone] |= 1;
            }
        }
    }
    for (u32 n = 0; n < tree.size(); ++n) {
        u32 guard = 0;
        for (u32 at = tree.nodes[n].parent; (keeps[n] & 1) != 0 && at < tree.size() && guard <= tree.size();
             at = tree.nodes[at].parent, ++guard) {
            keeps[at] |= 2;
        }
    }
    std::vector<SoftLump> lumps;
    for (const SoftBoneCount& bone : bones) {
        const bool alone = !bone.outside && !bone.outsideBelow && keeps[bone.node] == 0;
        const bool made = std::find(own.begin(), own.end(), bone.node) != own.end();
        // A bone a stage places or reads is copied even in a chain used as it
        // is: its keys are not what plays, or not all that moves with them.
        const bool asItIs = made || (!copyAll && alone && !bone.staged && (!bone.skinBelow || chainsAsTheyAre));
        SoftLump lump;
        lump.kind = asItIs ? SoftLumpKind::Bone : SoftLumpKind::Copy;
        lump.node = bone.node;
        const u32 parent = tree.nodes[bone.node].parent;
        lump.carrier = asItIs ? (parent < tree.size() ? parent : kInvalidNode) : bone.node;
        lump.anchor = SoftBonePivot(tree, bone.node);
        lump.rest = lump.anchor;
        lumps.push_back(lump);
    }
    return lumps;
}

namespace {

constexpr f32 kTurn = 6.2831853f;

/// A place at the bind, read from its mesh: where it is, which way it faces,
/// and two ways across it.
struct PlaceFrame {
    bool valid = false;
    Vector3f at{0, 0, 0};
    Vector3f normal{0, 0, 1};
    Vector3f across{1, 0, 0};
    Vector3f along{0, 1, 0};
    f32 radius = 0.0f;
};

PlaceFrame FrameOfPlace(const Model& model, const SoftPoints& points, const SoftPlace& place) {
    PlaceFrame out;
    if (place.mesh >= model.meshes.size() || place.mesh >= points.pointOfVertex.size()) {
        return out;
    }
    const Mesh& mesh = model.meshes[place.mesh];
    const geom::FaceSet& faces = mesh.faceSet();
    if (place.face >= faces.faceCount() || faces.faceValence[place.face] < 3) {
        return out;
    }
    std::size_t corner = 0;
    for (u32 f = 0; f < place.face; ++f) {
        corner += faces.faceValence[f];
    }
    const std::span<const Vector3f> positions = PositionsOf(mesh);
    const std::vector<u32>& entryOf = points.pointOfVertex[place.mesh];
    const f32 weights[3] = {place.weights.x, place.weights.y, place.weights.z};
    Vector3f normal{0, 0, 0};
    for (int k = 0; k < 3; ++k) {
        const u32 v = faces.cornerVertex[corner + static_cast<std::size_t>(k)];
        if (v >= positions.size()) {
            return out;
        }
        out.at = out.at + positions[v] * weights[k];
        if (v < entryOf.size() && entryOf[v] != kInvalidIndex) {
            normal = normal + points.points[entryOf[v]].outward;
        }
    }
    if (!(normal.length() > 1e-9f)) {
        return out;
    }
    out.normal = normal * (1.0f / normal.length());
    const Vector3f beside = std::abs(out.normal.x) < 0.9f ? Vector3f{1, 0, 0} : Vector3f{0, 1, 0};
    out.across = cross(out.normal, beside);
    out.across = out.across * (1.0f / out.across.length());
    out.along = cross(out.normal, out.across);
    // A place of no size takes §8.2's default.
    out.radius = place.radius > 0.0f ? place.radius : 0.3f * points.size;
    out.valid = out.radius > 0.0f;
    return out;
}

/// Whether @p point is in @p frame's disc: within its radius and facing as it
/// does, so the far face of a thin part is out of it (§8.2).
bool InDisc(const PlaceFrame& frame, const SoftPoint& point) {
    return point.rider == 0 && (point.at - frame.at).length() < frame.radius && Dot(point.outward, frame.normal) > 0.0;
}

/// The slots @p bones bones take before any lane beyond their own lumps, as
/// `SlotsTaken` counts them where every bone has a lump of its own.
u32 SlotsOf(const std::array<f32, 8>& shares, std::size_t bones) {
    u32 taken = 0;
    for (std::size_t i = 0; i < bones; ++i) {
        taken += shares[i] > 0.0f && shares[i] < 1.0f ? 2 : 1;
    }
    return taken;
}

/// A travelling Bulge's line at the bind (§8.2), as the step reads it: its
/// chain's pivots, or the line drawn over the surface; and the radius its
/// swell is said by.
struct TravelLine {
    std::vector<Vector3f> points;
    f32 length = 0.0f;
    f32 radius = 0.0f;

    bool valid() const {
        return points.size() >= 2 && length > 0.0f && radius > 0.0f;
    }

    /// Where the line is, @p along it.
    Vector3f at(f32 along) const {
        for (std::size_t k = 0; k + 1 < points.size(); ++k) {
            const Vector3f edge = points[k + 1] - points[k];
            const f32 span = edge.length();
            if (along <= span || k + 2 == points.size()) {
                return span > 0.0f ? points[k] + edge * std::clamp(along / span, 0.0f, 1.0f) : points[k];
            }
            along -= span;
        }
        return points.empty() ? Vector3f{0, 0, 0} : points.back();
    }
};

TravelLine LineOf(const Model& model, const SoftPoints& points, const SoftEffect& effect) {
    TravelLine out;
    if (!effect.bulge) {
        return out;
    }
    for (const u32 bone : effect.bulge->chain) {
        out.points.push_back(SoftBonePivot(model.nodes, bone));
    }
    if (out.points.size() < 2) {
        out.points = effect.bulge->line;
    }
    for (std::size_t k = 0; k + 1 < out.points.size(); ++k) {
        out.length += (out.points[k + 1] - out.points[k]).length();
    }
    out.radius = effect.place.radius > 0.0f ? effect.place.radius : 0.3f * points.size;
    return out;
}

/// §6.2's rows for a travelling Bulge, appended to @p lumps: for each bone
/// whose soft points its swell reaches, lumps on the line over the stretch of
/// it those points lie by. Returns how many.
u32 MakeRowLumps(const Model& model, const SoftPoints& points, const SoftEffect& effect, std::vector<SoftLump>& lumps) {
    const TravelLine line = LineOf(model, points, effect);
    if (!line.valid()) {
        return 0;
    }
    struct Stretch {
        u32 bone = kInvalidNode;
        std::vector<f32> along; ///< Its points, each how far along the line.
        f32 from = 0.0f, to = 0.0f;
    };
    std::vector<Stretch> stretches;
    for (const SoftPoint& point : points.points) {
        if (point.rider != 0) {
            continue;
        }
        f32 along = 0.0f;
        const Vector3f on = OnLine(line.points, point.at, along);
        if (!((point.at - on).length() < kTravelReach * line.radius)) {
            continue;
        }
        const std::span<const geom::Influence> skin = points.skinOf(point);
        const std::array<f32, 8> shares = SoftShares(skin, point.softness, points.cap);
        for (std::size_t i = 0; i < skin.size() && i < shares.size(); ++i) {
            if (!(shares[i] > 0.0f) || !(skin[i].weight > kNoWeight)) {
                continue;
            }
            auto at = std::find_if(stretches.begin(), stretches.end(),
                                   [&](const Stretch& s) { return s.bone == skin[i].bone; });
            if (at == stretches.end()) {
                stretches.push_back(Stretch{skin[i].bone, {}, along, along});
                at = stretches.end() - 1;
            }
            at->along.push_back(along);
            at->from = std::min(at->from, along);
            at->to = std::max(at->to, along);
        }
    }
    // Only a bone with a lump of its own gives its weight up (`BindLumps`).
    std::erase_if(stretches, [&](const Stretch& s) {
        return std::none_of(lumps.begin(), lumps.end(), [&](const SoftLump& lump) {
            return lump.kind != SoftLumpKind::Surface && lump.node == s.bone;
        });
    });
    std::sort(stretches.begin(), stretches.end(), [](const Stretch& a, const Stretch& b) { return a.bone < b.bone; });

    // One every half radius. A bone whose points lie by less than two steps
    // is at that grain already; and the step widens until the rows fit.
    f32 step = 0.5f * line.radius;
    const auto stepsOf = [&](const Stretch& s) {
        const u32 steps = static_cast<u32>(std::lround((s.to - s.from) / step));
        return steps >= 2 ? steps : 0u;
    };
    for (;;) {
        u32 knots = 0;
        for (const Stretch& s : stretches) {
            knots += stepsOf(s) != 0 ? stepsOf(s) + 1 : 0;
        }
        if (knots <= kSoftRowLumps) {
            break;
        }
        step *= 1.25f;
    }
    u32 made = 0;
    for (const Stretch& s : stretches) {
        const u32 steps = stepsOf(s);
        const auto knot = [&](u32 j) { return s.from + (s.to - s.from) * static_cast<f32>(j) / static_cast<f32>(steps); };
        for (u32 j = 0; steps != 0 && j <= steps; ++j) {
            // Only where a point would lean on it: between the knots either side.
            const f32 low = j == 0 ? -1.0f : knot(j - 1), high = j == steps ? line.length + 1.0f : knot(j + 1);
            if (std::none_of(s.along.begin(), s.along.end(), [&](f32 a) { return a > low && a < high; })) {
                continue;
            }
            SoftLump lump;
            lump.kind = SoftLumpKind::Surface;
            lump.carrier = s.bone;
            lump.effect = effect.id;
            lump.anchor = line.at(knot(j));
            lump.rest = lump.anchor;
            lumps.push_back(lump);
            ++made;
        }
    }
    return made;
}

} // namespace

bool SoftEffectClaims(const SoftEffect& effect) {
    if (effect.place.mesh == kInvalidIndex) {
        return false;
    }
    if (effect.kind == SoftEffectKind::Bulge && effect.bulge) {
        return effect.bulge->along == SoftAlong::Surface && effect.bulge->from != SoftComesFrom::Contact &&
               !effect.bulge->travels;
    }
    if (effect.kind == SoftEffectKind::Inflate && effect.inflate) {
        return effect.inflate->along == SoftAlong::Surface && effect.inflate->towardPlace;
    }
    return false;
}

bool SoftEffectClaimsRow(const SoftEffect& effect) {
    if (effect.kind != SoftEffectKind::Bulge || !effect.bulge) {
        return false;
    }
    const SoftBulge& bulge = *effect.bulge;
    return bulge.along == SoftAlong::Surface && bulge.from != SoftComesFrom::Contact && bulge.travels &&
           (bulge.chain.size() >= 2 || bulge.line.size() >= 2);
}

bool SoftClaimsDiffer(const SoftEffect& before, const SoftEffect& after) {
    const bool was = SoftEffectClaims(before), is = SoftEffectClaims(after);
    const bool wasRow = SoftEffectClaimsRow(before), isRow = SoftEffectClaimsRow(after);
    if (was != is || wasRow != isRow) {
        return true;
    }
    if (isRow) {
        // Its line, and the radius its lumps are spaced by.
        const SoftBulge& from = *before.bulge;
        const SoftBulge& to = *after.bulge;
        bool same = before.place.radius == after.place.radius && from.chain == to.chain &&
                    from.line.size() == to.line.size();
        for (std::size_t k = 0; same && k < from.line.size(); ++k) {
            same = (from.line[k] - to.line[k]).length() == 0.0f;
        }
        return !same;
    }
    const SoftPlace& a = before.place;
    const SoftPlace& b = after.place;
    return is && (a.mesh != b.mesh || a.face != b.face || a.radius != b.radius || a.weights.x != b.weights.x ||
                  a.weights.y != b.weights.y || a.weights.z != b.weights.z);
}

SoftPlaceRoom SoftPlaceRoomOf(const Model& model, const SoftPoints& points, const SoftEffect& effect) {
    SoftPlaceRoom out;
    const PlaceFrame frame = FrameOfPlace(model, points, effect.place);
    if (!frame.valid) {
        return out;
    }
    // The bone that carries most of the disc.
    std::vector<std::pair<u32, f64>> carried;
    f64 whole = 0.0;
    for (const SoftPoint& point : points.points) {
        if (!InDisc(frame, point)) {
            continue;
        }
        for (const geom::Influence& influence : points.skinOf(point)) {
            const f64 soft = static_cast<f64>(point.softness) * point.area * influence.weight;
            auto at = std::find_if(carried.begin(), carried.end(),
                                   [&](const auto& entry) { return entry.first == influence.bone; });
            if (at == carried.end()) {
                carried.push_back({influence.bone, 0.0});
                at = carried.end() - 1;
            }
            at->second += soft;
            whole += soft;
        }
    }
    if (!(whole > 0.0)) {
        return out;
    }
    std::sort(carried.begin(), carried.end(),
              [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
    out.valid = true;
    out.bone = carried.front().first;
    f64 roomy = 0.0;
    for (const SoftPoint& point : points.points) {
        if (!InDisc(frame, point)) {
            continue;
        }
        const std::span<const geom::Influence> skin = points.skinOf(point);
        const std::array<f32, 8> shares = SoftShares(skin, point.softness, points.cap);
        const std::size_t bones = std::min<std::size_t>(skin.size(), shares.size());
        const u32 taken = SlotsOf(shares, bones);
        for (std::size_t i = 0; i < bones; ++i) {
            if (skin[i].bone == out.bone && points.cap >= taken + 2) {
                roomy += static_cast<f64>(point.area) * shares[i] * skin[i].weight;
            }
        }
    }
    out.room = static_cast<f32>(roomy / whole);
    return out;
}

u32 MakePlaceLumps(const Model& model, const SoftPoints& points, const SoftBody& body, bool equalShares,
                   std::vector<SoftLump>& lumps) {
    u32 made = 0;
    // A classic group's equal shares cannot say the pattern (§6.4).
    if (equalShares) {
        return made;
    }
    for (const SoftEffect& effect : body.effects) {
        if (SoftEffectClaimsRow(effect)) {
            made += MakeRowLumps(model, points, effect, lumps);
            continue;
        }
        if (!SoftEffectClaims(effect)) {
            continue;
        }
        const SoftPlaceRoom room = SoftPlaceRoomOf(model, points, effect);
        if (!room.pays()) {
            continue;
        }
        const PlaceFrame frame = FrameOfPlace(model, points, effect.place);
        for (u32 k = 0; k < kSoftPlaceLumps; ++k) {
            // The hub, then the six spokes in turn.
            Vector3f target = frame.at;
            if (k > 0) {
                const f32 angle = kTurn * static_cast<f32>(k - 1) / static_cast<f32>(kSoftPlaceLumps - 1);
                target = target + (frame.across * std::cos(angle) + frame.along * std::sin(angle)) *
                                      (kSoftPlaceSpoke * frame.radius);
            }
            // Onto the surface: the body's point nearest there that faces as the place does.
            const SoftPoint* nearest = nullptr;
            f32 least = 0.0f;
            for (const SoftPoint& point : points.points) {
                if (point.rider != 0 || !(Dot(point.outward, frame.normal) > 0.0)) {
                    continue;
                }
                const f32 far = (point.at - target).length_squared();
                if (nearest == nullptr || far < least) {
                    nearest = &point;
                    least = far;
                }
            }
            SoftLump lump;
            lump.kind = SoftLumpKind::Surface;
            lump.carrier = room.bone;
            lump.effect = effect.id;
            lump.anchor = nearest != nullptr ? nearest->at : target;
            lump.rest = lump.anchor;
            if (nearest != nullptr) {
                // Half the body's thickness under it, as a surface lump's.
                const f32 step = points.grid.cell * 0.5f;
                f32 through = step;
                while (through < points.size && points.inside(nearest->at - nearest->outward * (through + step))) {
                    through += step;
                }
                lump.depth = 0.5f * through;
            }
            lumps.push_back(lump);
            ++made;
        }
    }
    return made;
}

SoftSurfaceReport MakeSurfaceLumps(const Model&, const SoftPoints& points, u32 detail, std::vector<SoftLump>& lumps,
                                   std::span<const u32> without) {
    SoftSurfaceReport report;
    if (detail == 0 || points.points.empty()) {
        return report;
    }
    // Each bone's soft area, and the points that ride it.
    struct Share {
        u32 bone = kInvalidNode;
        f64 area = 0.0;
        std::vector<u32> members; ///< Entries of `points.points`.
        u32 count = 0;            ///< Lumps it takes.
    };
    std::vector<Share> shares;
    f64 total = 0.0;
    for (u32 i = 0; i < points.points.size(); ++i) {
        const SoftPoint& point = points.points[i];
        if (point.rider != 0) {
            continue;
        }
        for (const geom::Influence& influence : points.skinOf(point)) {
            const f64 soft = static_cast<f64>(point.softness) * point.area * influence.weight;
            if (!(soft > 0.0) || std::find(without.begin(), without.end(), influence.bone) != without.end()) {
                continue;
            }
            auto at = std::find_if(shares.begin(), shares.end(),
                                   [&](const Share& s) { return s.bone == influence.bone; });
            if (at == shares.end()) {
                shares.push_back(Share{influence.bone, 0.0, {}, 0});
                at = shares.end() - 1;
            }
            at->area += soft;
            at->members.push_back(i);
            total += soft;
        }
    }
    std::sort(shares.begin(), shares.end(),
              [](const Share& a, const Share& b) { return a.area != b.area ? a.area > b.area : a.bone < b.bone; });

    // Three each to the largest, then one more each to the largest again.
    u32 left = detail;
    for (Share& share : shares) {
        const u32 take = std::min<u32>({3u, left, static_cast<u32>(share.members.size())});
        if (left < 3 && &share != &shares.front()) {
            break;
        }
        share.count = take;
        left -= take;
        if (left == 0) {
            break;
        }
    }
    for (bool gave = true; left != 0 && gave;) {
        gave = false;
        for (Share& share : shares) {
            if (left == 0 || share.count == 0) {
                break;
            }
            if (share.count < share.members.size()) {
                ++share.count;
                --left;
                gave = true;
            }
        }
    }

    const auto weightOf = [&](const Share& share, u32 entry) {
        const SoftPoint& point = points.points[entry];
        for (const geom::Influence& influence : points.skinOf(point)) {
            if (influence.bone == share.bone) {
                return static_cast<f64>(point.softness) * influence.weight;
            }
        }
        return 0.0;
    };
    // The distance through the solid from @p from to each of @p members.
    const auto walk = [&](const Vector3f& from, const std::vector<u32>& members, std::vector<f32>& distance) {
        const std::array<i32, 3> cell = points.grid.cellOf(from);
        distance.assign(members.size(), 0.0f);
        skinning::SolidWalk reached;
        if (points.grid.contains(cell[0], cell[1], cell[2])) {
            const skinning::SolidSeed seed{points.grid.index(cell[0], cell[1], cell[2]), 0.0f};
            reached = skinning::WalkSolid(points.grid, points.solid, std::span<const skinning::SolidSeed>(&seed, 1));
        }
        for (std::size_t k = 0; k < members.size(); ++k) {
            const Vector3f& at = points.points[members[k]].at;
            const f32 straight = (at - from).length();
            f32 through = 0.0f;
            // A walk cannot be shorter than the straight line; the grid can say so by half a cell.
            distance[k] = reached.reachedAny() && skinning::ReadSolidWalk(points.grid, reached, at, through)
                              ? std::max(through, straight)
                              : straight;
        }
    };

    for (Share& share : shares) {
        if (share.count == 0) {
            continue;
        }
        ++report.bones;
        report.reach += total > 0.0 ? static_cast<f32>(share.area / total) : 0.0f;
        const std::vector<u32>& members = share.members;
        // Farthest first, from the softest point.
        std::vector<u32> chosen; ///< Places in `members`.
        std::vector<std::vector<f32>> fields;
        u32 first = 0;
        for (u32 k = 1; k < members.size(); ++k) {
            const f64 a = weightOf(share, members[k]), b = weightOf(share, members[first]);
            if (a > b || (a == b && points.points[members[k]].area > points.points[members[first]].area)) {
                first = k;
            }
        }
        chosen.push_back(first);
        fields.emplace_back();
        walk(points.points[members[first]].at, members, fields.back());
        std::vector<f32> nearest = fields.back();
        while (chosen.size() < share.count) {
            // Of the points not yet taken: where several stand on one spot,
            // the farthest is no farther than one already chosen.
            u32 far = kInvalidIndex;
            for (u32 k = 0; k < members.size(); ++k) {
                if (std::find(chosen.begin(), chosen.end(), k) == chosen.end() &&
                    (far == kInvalidIndex || nearest[k] > nearest[far])) {
                    far = k;
                }
            }
            if (far == kInvalidIndex) {
                break;
            }
            chosen.push_back(far);
            fields.emplace_back();
            walk(points.points[members[far]].at, members, fields.back());
            for (u32 k = 0; k < members.size(); ++k) {
                nearest[k] = std::min(nearest[k], fields.back()[k]);
            }
        }
        // Two evening passes: each lump to the middle of the points nearest it.
        for (u32 pass = 0; pass < 2; ++pass) {
            std::vector<Vector3f> middle(chosen.size(), Vector3f{0, 0, 0});
            std::vector<f64> mass(chosen.size(), 0.0);
            std::vector<u32> owner(members.size(), 0);
            for (u32 k = 0; k < members.size(); ++k) {
                for (u32 c = 1; c < chosen.size(); ++c) {
                    if (fields[c][k] < fields[owner[k]][k]) {
                        owner[k] = c;
                    }
                }
                const SoftPoint& point = points.points[members[k]];
                const f64 m = weightOf(share, members[k]) * point.area;
                middle[owner[k]] = middle[owner[k]] + point.at * static_cast<f32>(m);
                mass[owner[k]] += m;
            }
            for (u32 c = 0; c < chosen.size(); ++c) {
                if (!(mass[c] > 0.0)) {
                    continue;
                }
                const Vector3f centre = middle[c] * static_cast<f32>(1.0 / mass[c]);
                // Onto the surface: the nearest of its own points.
                u32 to = chosen[c];
                f32 best = std::numeric_limits<f32>::max();
                for (u32 k = 0; k < members.size(); ++k) {
                    const f32 d = (points.points[members[k]].at - centre).length();
                    if (owner[k] == c && d < best) {
                        best = d;
                        to = k;
                    }
                }
                if (to != chosen[c] && std::find(chosen.begin(), chosen.end(), to) == chosen.end()) {
                    chosen[c] = to;
                    walk(points.points[members[to]].at, members, fields[c]);
                }
            }
        }
        for (const u32 place : chosen) {
            const SoftPoint& point = points.points[members[place]];
            SoftLump lump;
            lump.kind = SoftLumpKind::Surface;
            lump.carrier = share.bone;
            lump.anchor = point.at;
            lump.rest = point.at;
            // Half the body's thickness under it: inward until the solid ends.
            const f32 step = points.grid.cell * 0.5f;
            f32 through = step;
            while (through < points.size && points.inside(point.at - point.outward * (through + step))) {
                through += step;
            }
            lump.depth = 0.5f * through;
            lumps.push_back(lump);
            ++report.made;
        }
    }
    return report;
}

SoftBindReport BindLumps(Model& model, SoftBody& body, u32 cap, bool equalShares) {
    SoftBindReport report;
    cap = std::max<u32>(1, cap);
    body.influences = static_cast<u8>(std::min<u32>(cap, 255));
    body.equalShares = equalShares;
    const SoftPoints points = GatherWith(model, body.id, cap, body.riders);
    const std::span<const SoftLump> lumps = body.lumps;

    // Each bone's own lump and its surface lumps. A lump a place claimed binds
    // by its pattern, not here; a rider's carries its own part and no other.
    std::vector<u32> ownOf(model.nodes.size(), kInvalidIndex);
    std::vector<std::vector<u32>> surfaceOf(model.nodes.size());
    std::vector<u32> riderLumps;
    for (u32 k = 0; k < lumps.size(); ++k) {
        const SoftLump& lump = lumps[k];
        if (lump.kind != SoftLumpKind::Surface) {
            if (lump.node < ownOf.size() && ownOf[lump.node] == kInvalidIndex) {
                ownOf[lump.node] = k;
            }
        } else if (lump.rider != 0) {
            riderLumps.push_back(k);
        } else if (lump.effect == 0 && lump.carrier < surfaceOf.size()) {
            surfaceOf[lump.carrier].push_back(k);
        }
    }

    // The lumps the effects' places claimed (§6.2): a hub and six spokes each,
    // on the bone they hang from, the spokes in the order they stand round it.
    struct Claim {
        u32 bone = kInvalidNode;
        PlaceFrame frame;
        u32 hub = 0;
        std::array<u32, 6> spokes{};
        std::array<f32, 6> angle{};
    };
    std::vector<Claim> claims;
    for (const SoftEffect& effect : body.effects) {
        std::vector<u32> own;
        for (u32 k = 0; k < lumps.size(); ++k) {
            if (lumps[k].kind == SoftLumpKind::Surface && lumps[k].effect == effect.id && effect.id != 0) {
                own.push_back(k);
            }
        }
        Claim claim;
        claim.frame = FrameOfPlace(model, points, effect.place);
        if (!SoftEffectClaims(effect) || own.size() != kSoftPlaceLumps || !claim.frame.valid) {
            continue;
        }
        claim.bone = lumps[own[0]].carrier;
        claim.hub = own[0];
        // Measured from the hub's own anchor, which is where the pattern's middle is.
        claim.frame.at = lumps[own[0]].anchor;
        std::array<std::pair<f32, u32>, 6> round{};
        for (u32 s = 0; s < 6; ++s) {
            const Vector3f from = lumps[own[s + 1]].anchor - claim.frame.at;
            round[s] = {std::atan2(static_cast<f32>(Dot(from, claim.frame.along)),
                                   static_cast<f32>(Dot(from, claim.frame.across))),
                        own[s + 1]};
        }
        std::sort(round.begin(), round.end());
        for (u32 s = 0; s < 6; ++s) {
            claim.angle[s] = round[s].first;
            claim.spokes[s] = round[s].second;
        }
        claims.push_back(claim);
    }

    // A travelling Bulge's rows (§6.2): a bone's lumps along the line, in order.
    struct Row {
        u32 effect = 0;
        u32 bone = kInvalidNode;
        std::vector<std::pair<f32, u32>> knots; ///< How far along the line, and the lump.
    };
    std::vector<TravelLine> travels;
    std::vector<std::pair<u32, Row>> rows; ///< Its place in `travels`, and the row.
    for (const SoftEffect& effect : body.effects) {
        if (!SoftEffectClaimsRow(effect) || effect.id == 0) {
            continue;
        }
        TravelLine line = LineOf(model, points, effect);
        if (!line.valid()) {
            continue;
        }
        const u32 of = static_cast<u32>(travels.size());
        for (u32 k = 0; k < lumps.size(); ++k) {
            if (lumps[k].kind != SoftLumpKind::Surface || lumps[k].effect != effect.id) {
                continue;
            }
            auto at = std::find_if(rows.begin(), rows.end(), [&](const auto& row) {
                return row.second.effect == effect.id && row.second.bone == lumps[k].carrier;
            });
            if (at == rows.end()) {
                rows.push_back({of, Row{effect.id, lumps[k].carrier, {}}});
                at = rows.end() - 1;
            }
            f32 along = 0.0f;
            OnLine(line.points, lumps[k].anchor, along);
            at->second.knots.push_back({along, k});
        }
        travels.push_back(std::move(line));
    }
    for (auto& row : rows) {
        std::sort(row.second.knots.begin(), row.second.knots.end());
    }

    // The layers, cleared for this body's points.
    struct Written {
        std::span<std::array<u32, 4>> lumps;
        std::span<std::array<f32, 4>> weights;
        skinning::PointTable table;
    };
    std::vector<Written> written(model.meshes.size());
    for (u32 m = 0; m < model.meshes.size(); ++m) {
        if (points.pointOfVertex[m].empty()) {
            continue;
        }
        Mesh& mesh = model.meshes[m];
        written[m].table = skinning::BuildPointTable(mesh);
        written[m].lumps = mesh.attributes.getOrCreate<std::array<u32, 4>>(
            geom::names::kSoftBindLump, geom::Domain::Vertex, geom::AttrType::U32x4);
        written[m].weights = mesh.attributes.getOrCreate<std::array<f32, 4>>(
            geom::names::kSoftBindWeight, geom::Domain::Vertex, geom::AttrType::F32x4);
    }

    std::vector<Vector3f> pivotSum(lumps.size(), Vector3f{0, 0, 0});
    std::vector<f64> pivotMass(lumps.size(), 0.0);
    std::vector<Family> families;
    const auto store = [&](const SoftPoint& point, const std::array<u32, 4>& laneLump,
                           const std::array<f32, 4>& laneWeight) {
        const Written& out = written[point.mesh];
        for (const u32 vertex : out.table.membersOf(point.point)) {
            if (vertex < out.lumps.size() && vertex < out.weights.size() &&
                points.pointOfVertex[point.mesh][vertex] != kInvalidIndex) {
                out.lumps[vertex] = laneLump;
                out.weights[vertex] = laneWeight;
            }
        }
    };
    for (const SoftPoint& point : points.points) {
        const std::span<const geom::Influence> skin = points.skinOf(point);
        ++report.points;
        if (!skin.empty()) {
            ++report.byBones[std::min<std::size_t>(skin.size(), 4) - 1];
        }
        if (point.rider != 0) {
            // A rider's point: each bone's weight, whole, to its part's lump of
            // that bone. Heaviest first, as the skin is.
            std::array<u32, 4> laneLump{};
            std::array<f32, 4> laneWeight{};
            std::size_t used = 0;
            for (const geom::Influence& influence : skin) {
                const auto at = std::find_if(riderLumps.begin(), riderLumps.end(), [&](u32 k) {
                    return lumps[k].rider == point.rider && lumps[k].carrier == influence.bone;
                });
                if (used >= laneLump.size() || at == riderLumps.end() || !(influence.weight > kNoWeight)) {
                    continue;
                }
                laneLump[used] = *at + 1;
                laneWeight[used] = influence.weight;
                ++used;
                pivotSum[*at] = pivotSum[*at] + point.at * (influence.weight * point.area);
                pivotMass[*at] += static_cast<f64>(influence.weight) * point.area;
            }
            store(point, laneLump, laneWeight);
            continue;
        }
        if (point.softness > 0.0f && point.softness < 1.0f && SoftAllOrNone(static_cast<u32>(skin.size()), cap)) {
            ++report.allOrNone;
        }
        const std::array<f32, 8> shares = SoftShares(skin, point.softness, cap);
        const std::size_t bones = std::min<std::size_t>(skin.size(), shares.size());
        families.assign(bones, Family{});
        for (std::size_t i = 0; i < bones; ++i) {
            Family& family = families[i];
            family.bone = skin[i].bone;
            family.own = family.bone < ownOf.size() ? ownOf[family.bone] : kInvalidIndex;
            // A bone with no lump of its own gives nothing up.
            family.handed = family.own != kInvalidIndex ? static_cast<f64>(shares[i]) * skin[i].weight : 0.0;
            family.lanes = family.handed > 0.0 ? 1 : 0;
            if (family.handed > 0.0) {
                family.candidates.push_back(family.own);
                std::vector<u32> near = family.bone < surfaceOf.size() ? surfaceOf[family.bone] : std::vector<u32>{};
                std::sort(near.begin(), near.end(), [&](u32 a, u32 b) {
                    const f32 da = (lumps[a].anchor - point.at).length_squared();
                    const f32 db = (lumps[b].anchor - point.at).length_squared();
                    return da != db ? da < db : a < b;
                });
                if (near.size() > kLaneCandidates - 1) {
                    near.resize(kLaneCandidates - 1);
                }
                family.candidates.insert(family.candidates.end(), near.begin(), near.end());
            }
        }
        // The slots the bones and their own lumps leave go round, heaviest
        // first, up to three lanes a family and the four a layer holds.
        const u32 taken = SlotsTaken(families, std::span<const f32>(shares.data(), bones));
        u32 spare = cap > taken ? cap - taken : 0;
        u32 stored = 0;
        // A place's seven take their disc by a pattern: the hub and the two
        // spokes the point lies between, heaviest first, as many as it has
        // slots for; its bone's own lump has the rest, all of it at the rim.
        std::array<std::vector<std::pair<u32, f64>>, 8> pattern;
        for (const Claim& claim : claims) {
            if (!InDisc(claim.frame, point)) {
                continue;
            }
            for (std::size_t i = 0; i < bones; ++i) {
                Family& family = families[i];
                if (family.bone != claim.bone || !(family.handed > 0.0) || !pattern[i].empty()) {
                    continue;
                }
                const Vector3f from = point.at - claim.frame.at;
                const f32 rho = from.length() / claim.frame.radius;
                const f64 hub = rho < kSoftPlaceSpoke ? 1.0 - rho / kSoftPlaceSpoke : 0.0;
                const f64 ring = rho < kSoftPlaceSpoke ? rho / kSoftPlaceSpoke : (1.0 - rho) / (1.0 - kSoftPlaceSpoke);
                const f32 theta = std::atan2(static_cast<f32>(Dot(from, claim.frame.along)),
                                             static_cast<f32>(Dot(from, claim.frame.across)));
                u32 after = 0;
                while (after < 6 && claim.angle[after] <= theta) {
                    ++after;
                }
                const u32 a = (after + 5) % 6, b = after % 6;
                f32 span = claim.angle[b] - claim.angle[a], turned = theta - claim.angle[a];
                span += span <= 0.0f ? kTurn : 0.0f;
                turned += turned < 0.0f ? kTurn : 0.0f;
                const f64 t = span > 1e-6f ? std::clamp(turned / span, 0.0f, 1.0f) : 0.5;
                std::vector<std::pair<u32, f64>> lanes;
                for (const std::pair<u32, f64>& lane : {std::pair<u32, f64>{claim.hub, hub},
                                                         std::pair<u32, f64>{claim.spokes[a], ring * (1.0 - t)},
                                                         std::pair<u32, f64>{claim.spokes[b], ring * t}}) {
                    if (lane.second > 1e-6) {
                        lanes.push_back(lane);
                    }
                }
                std::sort(lanes.begin(), lanes.end(), [](const auto& x, const auto& y) {
                    return x.second != y.second ? x.second > y.second : x.first < y.first;
                });
                lanes.resize(std::min<std::size_t>({lanes.size(), spare, 4u - stored}));
                spare -= static_cast<u32>(lanes.size());
                stored += static_cast<u32>(lanes.size());
                pattern[i] = std::move(lanes);
                // No other lane of this family is solved for: the rest is its own lump's.
                family.candidates.resize(1);
            }
        }
        // A travelling Bulge's row takes a bone's share by where the point lies
        // along the line: the two lumps it lies between, or the nearer where it
        // has one slot. They have all of it, so the own lump's slot is theirs.
        for (const auto& [of, row] : rows) {
            const TravelLine& line = travels[of];
            for (std::size_t i = 0; i < bones; ++i) {
                Family& family = families[i];
                if (family.bone != row.bone || !(family.handed > 0.0) || !pattern[i].empty() || stored >= 4) {
                    continue;
                }
                f32 along = 0.0f;
                const Vector3f on = OnLine(line.points, point.at, along);
                if (!((point.at - on).length() < kTravelReach * line.radius)) {
                    continue;
                }
                std::size_t after = 0;
                while (after < row.knots.size() && row.knots[after].first <= along) {
                    ++after;
                }
                std::vector<std::pair<u32, f64>> lanes;
                if (after == 0 || after == row.knots.size()) {
                    lanes.push_back({row.knots[after == 0 ? 0 : after - 1].second, 1.0});
                } else {
                    const std::pair<f32, u32>& a = row.knots[after - 1];
                    const std::pair<f32, u32>& b = row.knots[after];
                    const f64 t = b.first > a.first ? (along - a.first) / (b.first - a.first) : 0.5;
                    for (const std::pair<u32, f64>& lane : {std::pair<u32, f64>{a.second, 1.0 - t},
                                                             std::pair<u32, f64>{b.second, t}}) {
                        if (lane.second > 1e-6) {
                            lanes.push_back(lane);
                        }
                    }
                }
                std::sort(lanes.begin(), lanes.end(), [](const auto& x, const auto& y) {
                    return x.second != y.second ? x.second > y.second : x.first < y.first;
                });
                lanes.resize(std::min<std::size_t>({lanes.size(), spare + 1u, 4u - stored}));
                f64 given = 0.0;
                for (const std::pair<u32, f64>& lane : lanes) {
                    given += lane.second;
                }
                for (std::pair<u32, f64>& lane : lanes) {
                    lane.second /= given;
                }
                spare -= static_cast<u32>(lanes.size()) - 1;
                stored += static_cast<u32>(lanes.size());
                pattern[i] = std::move(lanes);
                family.candidates.resize(1);
            }
        }
        for (bool gave = true; spare != 0 && gave;) {
            gave = false;
            for (Family& family : families) {
                if (spare == 0) {
                    break;
                }
                const u32 next = family.lanes + 1;
                if (family.lanes == 0 || family.lanes >= 3 || next > family.candidates.size()) {
                    continue;
                }
                // A family past its own lump may put every lane on a surface lump.
                const u32 more = family.lanes == 1 ? 2 : 1;
                if (stored + more > 4) {
                    continue;
                }
                family.lanes = next;
                stored += more;
                --spare;
                gave = true;
            }
        }

        // Solved together: each family's lanes land where the others leave the point.
        f64 handed = 0.0;
        for (Family& family : families) {
            handed += family.handed;
            SolveFamily(family, lumps, point.at, point.at);
        }
        for (std::size_t i = 0; i < bones; ++i) {
            if (pattern[i].empty()) {
                continue;
            }
            Family& family = families[i];
            f64 given = 0.0;
            family.chosen.clear();
            for (const std::pair<u32, f64>& lane : pattern[i]) {
                family.chosen.push_back(lane);
                given += lane.second;
            }
            if (1.0 - given > 1e-9) {
                family.chosen.push_back({family.own, 1.0 - given});
            }
            // Not solved: the other families solve for themselves, as if it
            // had landed on the point.
            family.lands = point.at;
        }
        for (u32 round = 0; round < 3 && bones > 1 && handed > 0.0; ++round) {
            for (Family& family : families) {
                if (family.lanes < 2 || family.chosen.empty()) {
                    continue;
                }
                Vector3f others{0, 0, 0};
                for (const Family& other : families) {
                    if (&other != &family && other.handed > 0.0) {
                        others = others + other.lands * static_cast<f32>(other.handed);
                    }
                }
                const Vector3f target = (point.at * static_cast<f32>(handed) - others) * static_cast<f32>(1.0 / family.handed);
                SolveFamily(family, lumps, target, point.at);
            }
        }
        if (handed > 0.0) {
            Vector3f lands{0, 0, 0};
            for (const Family& family : families) {
                lands = lands + family.lands * static_cast<f32>(family.handed);
            }
            report.worstReturn = std::max(report.worstReturn, (lands * static_cast<f32>(1.0 / handed) - point.at).length());
        }

        // The surface lanes, heaviest first; the own lumps' shares are what is left.
        std::array<u32, 4> laneLump{};
        std::array<f32, 4> laneWeight{};
        std::vector<std::pair<f32, u32>> lanes;
        for (const Family& family : families) {
            for (const auto& [lump, share] : family.chosen) {
                const f32 weight = static_cast<f32>(family.handed * share);
                pivotSum[lump] = pivotSum[lump] + point.at * (weight * point.area);
                pivotMass[lump] += static_cast<f64>(weight) * point.area;
                if (lumps[lump].kind == SoftLumpKind::Surface && weight > kNoWeight) {
                    lanes.push_back({weight, lump});
                }
            }
        }
        std::sort(lanes.begin(), lanes.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        for (std::size_t k = 0; k < lanes.size() && k < 4; ++k) {
            laneLump[k] = lanes[k].second + 1;
            laneWeight[k] = lanes[k].first;
        }
        store(point, laneLump, laneWeight);
    }

    // A surface lump's pivot is the middle of what it carries (§6.1).
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        SoftLump& lump = body.lumps[k];
        if (lump.kind == SoftLumpKind::Surface) {
            lump.rest = pivotMass[k] > 0.0 ? pivotSum[k] * static_cast<f32>(1.0 / pivotMass[k]) : lump.anchor;
        } else {
            lump.rest = SoftBonePivot(model.nodes, lump.node);
        }
    }
    return report;
}

std::vector<SoftInside> SoftPartsInside(const Model& model, const SoftPoints& points) {
    std::vector<SoftInside> out;
    if (points.points.empty()) {
        return out;
    }
    // The solid of the body's own faces alone: a part of the same mesh that
    // is not the body's must not count as the body.
    std::vector<Mesh> own;
    for (u32 m = 0; m < model.meshes.size(); ++m) {
        const std::vector<u32>& entryOf = points.pointOfVertex[m];
        if (entryOf.empty()) {
            continue;
        }
        const Mesh& mesh = model.meshes[m];
        const geom::FaceSet& faces = mesh.faceSet();
        geom::FaceSet kept;
        kept.vertexCount = faces.vertexCount;
        std::size_t corner = 0;
        for (std::size_t f = 0; f < faces.faceCount(); ++f) {
            const u32 valence = faces.faceValence[f];
            bool all = valence >= 3;
            for (u32 k = 0; k < valence; ++k) {
                const u32 v = faces.cornerVertex[corner + k];
                all = all && v < entryOf.size() && entryOf[v] != kInvalidIndex;
            }
            if (all) {
                kept.addFace(std::span<const u32>(faces.cornerVertex.data() + corner, valence));
            }
            corner += valence;
        }
        Mesh& made = own.emplace_back();
        made.setFaceSet(std::move(kept));
        const std::span<const Vector3f> from = PositionsOf(mesh);
        const std::span<Vector3f> to =
            made.attributes.getOrCreate<Vector3f>(geom::names::kPosition, geom::Domain::Vertex, geom::AttrType::F32x3);
        std::copy(from.begin(), from.begin() + static_cast<std::ptrdiff_t>(std::min(from.size(), to.size())), to.begin());
    }
    std::vector<const Mesh*> meshes;
    for (const Mesh& mesh : own) {
        meshes.push_back(&mesh);
    }
    const skinning::SolidGrid grid = skinning::MakeSolidGrid(points.low, points.high, kSolidResolution);
    const std::vector<u8> solid = skinning::BuildSolid(grid, meshes);
    const auto inside = [&](const Vector3f& at) {
        const std::array<i32, 3> cell = grid.cellOf(at);
        return grid.contains(cell[0], cell[1], cell[2]) && solid[grid.index(cell[0], cell[1], cell[2])] != 0;
    };

    for (u32 m = 0; m < model.meshes.size(); ++m) {
        const Mesh& mesh = model.meshes[m];
        const std::span<const Vector3f> positions = PositionsOf(mesh);
        const std::span<const u32> sectionOf = mesh.faceSections();
        const std::vector<u32>& entryOf = points.pointOfVertex[m];
        // Per material of the mesh, its vertices that are not the body's.
        std::vector<std::pair<u32, std::vector<u8>>> byMaterial;
        const geom::FaceSet& faces = mesh.faceSet();
        std::size_t corner = 0;
        for (std::size_t f = 0; f < faces.faceCount(); ++f) {
            const u32 valence = faces.faceValence[f];
            const u32 section = f < sectionOf.size() ? sectionOf[f] : 0;
            const u32 material = section < mesh.sections.size() ? mesh.sections[section].materialSlot : kInvalidIndex;
            auto at = std::find_if(byMaterial.begin(), byMaterial.end(),
                                   [&](const auto& entry) { return entry.first == material; });
            if (at == byMaterial.end()) {
                byMaterial.push_back({material, std::vector<u8>(positions.size(), 0)});
                at = byMaterial.end() - 1;
            }
            for (u32 k = 0; k < valence; ++k) {
                const u32 v = faces.cornerVertex[corner + k];
                if (v < positions.size() && (v >= entryOf.size() || entryOf[v] == kInvalidIndex)) {
                    at->second[v] = 1;
                }
            }
            corner += valence;
        }
        std::sort(byMaterial.begin(), byMaterial.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& [material, used] : byMaterial) {
            u32 count = 0;
            for (u32 v = 0; v < used.size(); ++v) {
                count += used[v] != 0 && inside(positions[v]) ? 1 : 0;
            }
            if (count != 0) {
                out.push_back(SoftInside{SoftPart{m, material}, count});
            }
        }
    }
    return out;
}

void MeasureSoftClear(const Model& model, SoftBody& body) {
    for (SoftLump& lump : body.lumps) {
        lump.clear = 0.0f;
    }
    if (body.inside.empty()) {
        return;
    }
    // The points to keep clear of: each part's vertices that are not the body's.
    std::vector<Vector3f> kept;
    for (const SoftPart& part : body.inside) {
        if (part.mesh >= model.meshes.size()) {
            continue;
        }
        const Mesh& mesh = model.meshes[part.mesh];
        const std::span<const Vector3f> positions = PositionsOf(mesh);
        const std::span<const u32> bodyOf = mesh.attributes.get<u32>(geom::names::kSoftBody, geom::Domain::Vertex);
        const std::span<const u32> sectionOf = mesh.faceSections();
        std::vector<u8> used(positions.size(), 0);
        const geom::FaceSet& faces = mesh.faceSet();
        std::size_t corner = 0;
        for (std::size_t f = 0; f < faces.faceCount(); ++f) {
            const u32 valence = faces.faceValence[f];
            const u32 section = f < sectionOf.size() ? sectionOf[f] : 0;
            const u32 material = section < mesh.sections.size() ? mesh.sections[section].materialSlot : kInvalidIndex;
            if (part.material == kInvalidIndex || part.material == material) {
                for (u32 k = 0; k < valence; ++k) {
                    const u32 v = faces.cornerVertex[corner + k];
                    if (v < used.size() && (v >= bodyOf.size() || bodyOf[v] != body.id)) {
                        used[v] = 1;
                    }
                }
            }
            corner += valence;
        }
        for (u32 v = 0; v < used.size(); ++v) {
            if (used[v] != 0) {
                kept.push_back(positions[v]);
            }
        }
    }
    if (kept.empty()) {
        return;
    }
    // Along their longest side, in order: a point of the body then asks only
    // the kept ones nearer along it than the nearest found, not every one.
    Vector3f low = kept.front(), high = kept.front();
    for (const Vector3f& at : kept) {
        low = Vector3f{std::min(low.x, at.x), std::min(low.y, at.y), std::min(low.z, at.z)};
        high = Vector3f{std::max(high.x, at.x), std::max(high.y, at.y), std::max(high.z, at.z)};
    }
    const Vector3f side = high - low;
    const int axis = side.x >= side.y && side.x >= side.z ? 0 : (side.y >= side.z ? 1 : 2);
    const auto along = [axis](const Vector3f& at) { return axis == 0 ? at.x : (axis == 1 ? at.y : at.z); };
    std::sort(kept.begin(), kept.end(), [&](const Vector3f& a, const Vector3f& b) { return along(a) < along(b); });
    const auto nearestTo = [&](const Vector3f& at) {
        const f32 here = along(at);
        const std::size_t from = static_cast<std::size_t>(
            std::lower_bound(kept.begin(), kept.end(), here, [&](const Vector3f& each, f32 value) {
                return along(each) < value;
            }) - kept.begin());
        f32 best = std::numeric_limits<f32>::max();
        for (std::size_t i = from; i < kept.size(); ++i) {
            const f32 apart = along(kept[i]) - here;
            if (apart * apart >= best) {
                break;
            }
            best = std::min(best, (kept[i] - at).length_squared());
        }
        for (std::size_t i = from; i-- > 0;) {
            const f32 apart = here - along(kept[i]);
            if (apart * apart >= best) {
                break;
            }
            best = std::min(best, (kept[i] - at).length_squared());
        }
        return std::sqrt(best);
    };
    std::vector<f32> clear(body.lumps.size(), std::numeric_limits<f32>::max());
    std::vector<u32> ownOf(model.nodes.size(), kInvalidIndex);
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        if (body.lumps[k].kind != SoftLumpKind::Surface && body.lumps[k].node < ownOf.size()) {
            ownOf[body.lumps[k].node] = k;
        }
    }
    SoftSkinner skinner(model, body);
    for (const u32 m : body.meshes) {
        if (m >= model.meshes.size()) {
            continue;
        }
        const std::span<const Vector3f> positions = PositionsOf(model.meshes[m]);
        for (u32 v = 0; v < positions.size(); ++v) {
            if (!skinner.holds(m, v)) {
                continue;
            }
            const f32 nearest = nearestTo(positions[v]);
            for (const geom::Influence& influence : skinner.of(m, v)) {
                const u32 lump = influence.bone >= skinner.lumpBase()
                                     ? influence.bone - skinner.lumpBase()
                                     : (influence.bone < ownOf.size() ? ownOf[influence.bone] : kInvalidIndex);
                if (lump < clear.size() && (influence.bone >= skinner.lumpBase() ||
                                            body.lumps[lump].kind == SoftLumpKind::Bone)) {
                    clear[lump] = std::min(clear[lump], nearest);
                }
            }
        }
    }
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        body.lumps[k].clear = clear[k] == std::numeric_limits<f32>::max() ? 0.0f : clear[k];
    }
}

namespace {

/// The bones of @p body's Drips' strands and droplets: its own making.
std::vector<u32> StrandBones(const SoftBody& body) {
    std::vector<u32> strands;
    for (const SoftEffect& effect : body.effects) {
        if (effect.drip) {
            strands.insert(strands.end(), effect.drip->strandNodes.begin(), effect.drip->strandNodes.end());
            strands.insert(strands.end(), effect.drip->dropNodes.begin(), effect.drip->dropNodes.end());
        }
    }
    return strands;
}

/// §6.5: a lump has no child bone. A bake keys a bone used as it is in place,
/// and a lump that hangs from it would take the bone's departure on top of its
/// own, in a file and in every view of the keys. So a bone that carries a lump
/// of the surface is copied, whatever `BoneLumps` found: @p base is its answer
/// for the body as it stands, and each bone's lump of @p lumps gets its kind
/// from it, or a copy's where the bone carries one.
void SettleBoneLumps(const NodeTree& tree, std::span<const SoftLump> base, std::vector<SoftLump>& lumps) {
    for (SoftLump& lump : lumps) {
        if (lump.kind == SoftLumpKind::Surface) {
            continue;
        }
        const auto found = std::find_if(base.begin(), base.end(),
                                        [&](const SoftLump& other) { return other.node == lump.node; });
        const bool carries = std::any_of(lumps.begin(), lumps.end(), [&](const SoftLump& other) {
            return other.kind == SoftLumpKind::Surface && other.carrier == lump.node;
        });
        const bool own = found != base.end() ? found->kind == SoftLumpKind::Bone : lump.kind == SoftLumpKind::Bone;
        const bool asItIs = own && !carries;
        const u32 parent = lump.node < tree.size() ? tree.nodes[lump.node].parent : kInvalidNode;
        // A copy always swings (§6.5); whether a bone does is its caller's to say.
        lump.swings = lump.swings || (!asItIs && lump.kind == SoftLumpKind::Bone);
        lump.kind = asItIs ? SoftLumpKind::Bone : SoftLumpKind::Copy;
        lump.carrier = asItIs ? (parent < tree.size() ? parent : kInvalidNode) : lump.node;
    }
}

} // namespace

SoftMakeReport MakeSoftLumps(Model& model, SoftBody& body, u32 cap, bool equalShares) {
    SoftMakeReport report;
    const SoftPoints points = GatherWith(model, body.id, cap, body.riders);
    // A Drip's strands are the body's own making: their bones are used as
    // they are, and take no lump of the surface.
    const std::vector<u32> strands = StrandBones(body);
    const std::vector<SoftLump> base = BoneLumps(model, points, body.copyAll, body.chainsAsTheyAre, strands);
    body.lumps = base;
    report.surface = MakeSurfaceLumps(model, points, body.detail, body.lumps, strands);
    report.riders = MakeRiderLumps(points, body.lumps);
    // The effects' places claim theirs last, so the lumps before keep their numbers.
    report.claimed = MakePlaceLumps(model, points, body, equalShares, body.lumps);
    SettleBoneLumps(model.nodes, base, body.lumps);
    for (const SoftLump& lump : body.lumps) {
        report.asTheyAre += lump.kind == SoftLumpKind::Bone ? 1 : 0;
        report.copies += lump.kind == SoftLumpKind::Copy ? 1 : 0;
    }
    report.bind = BindLumps(model, body, cap, equalShares);
    MeasureSoftClear(model, body);
    return report;
}

bool ClaimSoftPlaces(Model& model, SoftBody& body) {
    const std::vector<SoftLump> was = body.lumps;
    const u32 cap = std::max<u32>(1, body.influences);
    std::erase_if(body.lumps, [](const SoftLump& lump) { return lump.effect != 0; });
    const SoftPoints points = GatherWith(model, body.id, cap, body.riders);
    MakePlaceLumps(model, points, body, body.equalShares, body.lumps);
    // A bone a place's lumps came to hang from, or no longer do.
    SettleBoneLumps(model.nodes, BoneLumps(model, points, body.copyAll, body.chainsAsTheyAre, StrandBones(body)),
                    body.lumps);
    // The lanes read a place's radius and a line as well as where the lumps
    // stand, so they are solved again whether or not a lump moved.
    BindLumps(model, body, cap, body.equalShares);
    MeasureSoftClear(model, body);
    return !SameReflected(was, body.lumps);
}

bool SettleSoftLumps(Model& model, SoftBody& body, u32 cap, bool equalShares) {
    const std::vector<SoftLump> was = body.lumps;
    const SoftPoints points = GatherWith(model, body.id, cap, body.riders);
    SettleBoneLumps(model.nodes, BoneLumps(model, points, body.copyAll, body.chainsAsTheyAre, StrandBones(body)),
                    body.lumps);
    BindLumps(model, body, cap, equalShares);
    MeasureSoftClear(model, body);
    return !SameReflected(was, body.lumps);
}

// ============================================================================
// SoftSkinner
// ============================================================================

SoftSkinner::SoftSkinner(const Model& model, const SoftBody& body, SoftWeights weights)
    : body_(&body), weights_(weights), lumpBase_(model.nodes.size()),
      skin_(model, std::max<u32>(1, body.influences)) {
    layers_.resize(model.meshes.size());
    for (u32 m = 0; m < model.meshes.size(); ++m) {
        const geom::AttributeSet& attributes = model.meshes[m].attributes;
        layers_[m].body = attributes.get<u32>(geom::names::kSoftBody, geom::Domain::Vertex);
        layers_[m].softness = attributes.get<f32>(geom::names::kSoftSoftness, geom::Domain::Vertex);
        layers_[m].lumps = attributes.get<std::array<u32, 4>>(geom::names::kSoftBindLump, geom::Domain::Vertex);
        layers_[m].weights = attributes.get<std::array<f32, 4>>(geom::names::kSoftBindWeight, geom::Domain::Vertex);
    }
    ownLump_.assign(model.nodes.size(), kInvalidIndex);
    for (u32 k = 0; k < body.lumps.size(); ++k) {
        const SoftLump& lump = body.lumps[k];
        if (lump.kind != SoftLumpKind::Surface && lump.node < ownLump_.size() && ownLump_[lump.node] == kInvalidIndex) {
            ownLump_[lump.node] = k;
        }
    }
}

bool SoftSkinner::holds(u32 mesh, u32 vertex) const {
    return mesh < layers_.size() && vertex < layers_[mesh].body.size() && layers_[mesh].body[vertex] == body_->id;
}

std::span<const geom::Influence> SoftSkinner::of(u32 mesh, u32 vertex) {
    const std::span<const geom::Influence> source = skin_.of(mesh, vertex);
    scratch_.assign(source.begin(), source.end());
    if (!holds(mesh, vertex) || scratch_.empty()) {
        return scratch_;
    }
    const std::vector<geom::Influence> skin = scratch_;
    scratch_.clear();
    const MeshLayers& layers = layers_[mesh];
    const f32 softness = vertex < layers.softness.size() ? std::clamp(layers.softness[vertex], 0.0f, 1.0f) : 0.0f;
    const u32 cap = std::max<u32>(1, body_->influences);
    const std::array<f32, 8> shares = SoftShares(skin, softness, cap);
    const std::size_t bones = std::min<std::size_t>(skin.size(), shares.size());
    const std::span<const SoftLump> lumps = body_->lumps;

    // The stored lanes, by family.
    struct Lane {
        u32 lump = 0;
        f32 weight = 0.0f;
        u32 family = kInvalidNode;
    };
    std::array<Lane, 4> lanes{};
    u32 laneCount = 0;
    if (vertex < layers.lumps.size() && vertex < layers.weights.size()) {
        for (std::size_t k = 0; k < 4; ++k) {
            const u32 lump = layers.lumps[vertex][k];
            if (lump != 0 && lump <= lumps.size() && layers.weights[vertex][k] > 0.0f) {
                lanes[laneCount++] = Lane{lump - 1, layers.weights[vertex][k], lumps[lump - 1].family()};
            }
        }
    }

    // One family at a time: its members, and what each takes of the bone's
    // weight. They sum to the bone's own, whatever the lanes say.
    struct Member {
        u32 slot = 0;
        f32 weight = 0.0f;
    };
    // A rider's point (§5.4): each bone's weight is its part's lump of that
    // bone, whole, whatever the softness says. A bone the part has no lump for
    // keeps its own.
    bool rides = false;
    for (u32 k = 0; k < laneCount; ++k) {
        rides = rides || lumps[lanes[k].lump].rider != 0;
    }
    const auto ridden = [&](u32 bone) {
        for (u32 k = 0; k < laneCount; ++k) {
            if (lanes[k].family == bone && lumps[lanes[k].lump].rider != 0) {
                return lumpBase_ + lanes[k].lump;
            }
        }
        return bone;
    };
    std::vector<std::vector<Member>> families(bones);
    for (std::size_t i = 0; i < bones; ++i) {
        const u32 bone = skin[i].bone;
        const f32 weight = skin[i].weight;
        if (rides) {
            families[i].push_back({ridden(bone), weight});
            continue;
        }
        const u32 own = bone < ownLump_.size() ? ownLump_[bone] : kInvalidIndex;
        const f32 handed = own != kInvalidIndex ? shares[i] * weight : 0.0f;
        f32 onLanes = 0.0f;
        for (u32 k = 0; k < laneCount; ++k) {
            onLanes += lanes[k].family == bone ? lanes[k].weight : 0.0f;
        }
        // Lanes solved for more softness than the point has now are scaled back.
        const f32 scale = onLanes > handed ? (onLanes > 0.0f ? handed / onLanes : 0.0f) : 1.0f;
        onLanes *= scale;
        std::vector<Member>& members = families[i];
        if (own == kInvalidIndex || lumps[own].kind == SoftLumpKind::Bone) {
            // The bone is its own lump, or has none: what the lanes leave stays on it.
            members.push_back({bone, weight - onLanes});
        } else {
            members.push_back({bone, weight - handed});
            members.push_back({lumpBase_ + own, handed - onLanes});
        }
        for (u32 k = 0; k < laneCount; ++k) {
            if (lanes[k].family == bone) {
                members.push_back({lumpBase_ + lanes[k].lump, lanes[k].weight * scale});
            }
        }
    }

    if (weights_ == SoftWeights::EqualShares) {
        // A classic group averages its nodes equally and names none twice. The
        // point's own group is the source's; each of its bones is then traded
        // for the same number of members, so every family keeps its share.
        std::vector<std::vector<geom::Influence>> one(1, skin);
        const skinning::ClassicSkin snapped = skinning::QuantizeClassic(one, {});
        std::vector<u32> group = snapped.groups.empty() ? std::vector<u32>{} : snapped.groups[snapped.groupOf.empty() ? 0 : snapped.groupOf[0]];
        if (group.empty()) {
            return scratch_ = skin;
        }
        if (rides) {
            // Each bone of the group traded for the part's lump of it: the
            // group names none twice, and every family keeps its share.
            for (const u32 bone : group) {
                scratch_.push_back({ridden(bone), 1.0f / static_cast<f32>(group.size())});
            }
            std::sort(scratch_.begin(), scratch_.end(), [](const geom::Influence& a, const geom::Influence& b) {
                return a.weight != b.weight ? a.weight > b.weight : a.bone < b.bone;
            });
            return scratch_;
        }
        const u32 most = std::max<u32>(1, 8u / static_cast<u32>(group.size()));
        // What each bone of the group may be traded for: its lumps, heaviest first.
        std::vector<std::vector<Member>> trade(group.size());
        bool copies = true;
        u32 fewest = most;
        for (std::size_t g = 0; g < group.size(); ++g) {
            const auto at = std::find_if(skin.begin(), skin.end(),
                                         [&](const geom::Influence& i) { return i.bone == group[g]; });
            const std::size_t i = static_cast<std::size_t>(at - skin.begin());
            const u32 own = group[g] < ownLump_.size() ? ownLump_[group[g]] : kInvalidIndex;
            copies = copies && own != kInvalidIndex && lumps[own].kind == SoftLumpKind::Copy;
            if (i < bones) {
                for (const Member& member : families[i]) {
                    if (member.slot != group[g] && member.weight > kNoWeight) {
                        trade[g].push_back(member);
                    }
                }
            }
            if (own != kInvalidIndex && lumps[own].kind != SoftLumpKind::Surface) {
                // The bone's own lump stands first, whatever the lanes leave
                // it: the bone itself where it is the lump, else its copy. A
                // point traded for one member then moves as its bone's lump
                // does, and not wholly as one lump of the surface.
                const u32 slot = lumps[own].kind == SoftLumpKind::Bone ? group[g] : lumpBase_ + own;
                std::erase_if(trade[g], [&](const Member& member) { return member.slot == slot; });
                trade[g].insert(trade[g].begin(), Member{slot, 1.0f});
            }
            std::stable_sort(trade[g].begin(), trade[g].end(),
                             [](const Member& a, const Member& b) { return a.weight > b.weight; });
            fewest = std::min<u32>(fewest, static_cast<u32>(trade[g].size()));
        }
        const f32 soft = softness;
        if (soft < 0.25f || fewest == 0) {
            for (const u32 bone : group) {
                scratch_.push_back({bone, 1.0f / static_cast<f32>(group.size())});
            }
        } else if (soft < 0.75f && copies && group.size() * 2 <= 8) {
            // Half soft: each bone beside its copy.
            for (const u32 bone : group) {
                scratch_.push_back({bone, 0.5f / static_cast<f32>(group.size())});
                scratch_.push_back({lumpBase_ + ownLump_[bone], 0.5f / static_cast<f32>(group.size())});
            }
        } else {
            for (std::size_t g = 0; g < group.size(); ++g) {
                for (u32 k = 0; k < fewest; ++k) {
                    scratch_.push_back({trade[g][k].slot, 1.0f / static_cast<f32>(group.size() * fewest)});
                }
            }
        }
    } else if (weights_ == SoftWeights::Bytes && bones <= 4) {
        // The bytes the bone had are shared out inside its family by largest
        // remainder, so each family's bytes sum to the bone's own.
        std::array<f32, 4> floats{};
        for (std::size_t i = 0; i < bones; ++i) {
            floats[i] = skin[i].weight;
        }
        const std::array<u8, 4> bytes = skinning::QuantizeWeights(floats, bones);
        for (std::size_t i = 0; i < bones; ++i) {
            const std::vector<Member>& members = families[i];
            f32 total = 0.0f;
            for (const Member& member : members) {
                total += std::max(member.weight, 0.0f);
            }
            std::vector<u32> given(members.size(), 0);
            std::vector<f32> remainder(members.size(), 0.0f);
            u32 sum = 0;
            for (std::size_t k = 0; k < members.size(); ++k) {
                const f32 exact = total > 0.0f ? std::max(members[k].weight, 0.0f) / total * static_cast<f32>(bytes[i])
                                               : (k == 0 ? static_cast<f32>(bytes[i]) : 0.0f);
                given[k] = static_cast<u32>(std::floor(exact + 1e-3f));
                remainder[k] = exact - static_cast<f32>(given[k]);
                sum += given[k];
            }
            while (sum < bytes[i]) {
                std::size_t to = 0;
                for (std::size_t k = 1; k < members.size(); ++k) {
                    if (remainder[k] > remainder[to]) {
                        to = k;
                    }
                }
                ++given[to];
                remainder[to] = -1.0f;
                ++sum;
            }
            while (sum > bytes[i]) {
                std::size_t from = 0;
                for (std::size_t k = 1; k < members.size(); ++k) {
                    if (given[k] > given[from]) {
                        from = k;
                    }
                }
                --given[from];
                --sum;
            }
            for (std::size_t k = 0; k < members.size(); ++k) {
                if (given[k] != 0) {
                    scratch_.push_back({members[k].slot, static_cast<f32>(given[k]) / 255.0f});
                }
            }
        }
    } else {
        for (const std::vector<Member>& members : families) {
            for (const Member& member : members) {
                if (member.weight > kNoWeight) {
                    scratch_.push_back({member.slot, member.weight});
                }
            }
        }
    }
    std::sort(scratch_.begin(), scratch_.end(), [](const geom::Influence& a, const geom::Influence& b) {
        return a.weight != b.weight ? a.weight > b.weight : a.bone < b.bone;
    });
    return scratch_;
}

} // namespace wem
} // namespace models
} // namespace whiteout
