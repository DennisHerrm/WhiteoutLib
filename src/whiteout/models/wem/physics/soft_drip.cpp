// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_drip.h>

#include <whiteout/models/wem/geometry/attributes.h>
#include <whiteout/models/wem/geometry/builder.h>
#include <whiteout/models/wem/physics/cloth_bake.h>
#include <whiteout/models/wem/physics/soft_fields.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

namespace whiteout {
namespace models {
namespace wem {

namespace {

constexpr f32 kPi = 3.14159265f;
constexpr f32 kTau = 6.2831853f;
/// A strand's tube has six sides and four rings (§8.6).
constexpr u32 kSides = 6;
constexpr u32 kRings = 4;

/// The bone that carries most of point @p point.
u32 CarrierOf(const SoftRig& rig, u32 point) {
    u32 bone = kInvalidNode;
    f32 most = 0.0f;
    for (const SoftSlot& slot : rig.slotsOf(point)) {
        if (slot.node != kInvalidNode && slot.weight > most) {
            most = slot.weight;
            bone = slot.node;
        }
    }
    return bone;
}

/// The first UV set at point @p point's vertex: any corner of it. A corner's
/// is on the halfedge that leaves it.
Vector2f UvOf(const Model& model, const SoftPoint& point) {
    if (point.mesh >= model.meshes.size()) {
        return Vector2f{0, 0};
    }
    const Mesh& mesh = model.meshes[point.mesh];
    // A corner's layer is read through the half-edges: a mesh that has none
    // built (one just read from a file) says nothing here.
    if (!mesh.hasConnectivity()) {
        return Vector2f{0, 0};
    }
    const std::span<const Vector2f> uv = mesh.attributes.get<Vector2f>(geom::names::uv(0), geom::Domain::Halfedge);
    const geom::Topology& topology = mesh.topology();
    for (u32 face = 0; face < mesh.faceCount(); ++face) {
        const geom::HalfedgeId first = topology.halfedge(geom::FaceId(face));
        geom::HalfedgeId h = first;
        do {
            if (topology.from(h).value() == point.vertex && h.index() < uv.size()) {
                return uv[h.index()];
            }
            h = topology.next(h);
        } while (h.index() != first.index());
    }
    return Vector2f{0, 0};
}

} // namespace

std::vector<SoftDripSite> SoftDripSites(const Model& model, const SoftRig& rig, const SoftDrip& settings) {
    std::vector<SoftDripSite> sites;
    if (rig.empty()) {
        return sites;
    }
    const std::vector<SoftPoint>& points = rig.points().points;
    const u32 count = static_cast<u32>(points.size());
    const f32 size = std::max(rig.points().size, 1e-6f);
    const f32 width = std::clamp(settings.thickness, 0.002f, 0.5f) * size;

    if (settings.sitesByHand) {
        for (std::size_t i = 0; i < settings.siteMesh.size() && i < settings.siteFace.size() &&
                                i < settings.siteWeights.size();
             ++i) {
            SoftPlace place;
            place.mesh = settings.siteMesh[i];
            place.face = settings.siteFace[i];
            place.weights = settings.siteWeights[i];
            const SoftPlaceAt at = ResolveSoftPlace(model, rig, place);
            if (!at.valid) {
                continue;
            }
            SoftDripSite site;
            site.at = at.at;
            site.normal = at.normal;
            site.bone = at.bone;
            for (const u32 corner : at.points) {
                site.point = corner < count ? corner : site.point;
            }
            if (site.point < count) {
                site.uv = UvOf(model, points[site.point]);
            }
            const f32 own = i < settings.siteThickness.size() ? settings.siteThickness[i] : 0.0f;
            site.width = own > 0.0f ? own * size : width;
            sites.push_back(site);
        }
        return sites;
    }

    // Each point walked down the surface to where it cannot leave: a point
    // none of whose neighbours is lower, or one that faces down past the
    // *Overhang*, where a liquid lets go. A walk only ever goes down, so it ends.
    const f32 sheds = std::cos(std::clamp(settings.overhang, 0.0f, 0.5f * kPi));
    const f32 step = 1e-5f * size;
    std::vector<u32> ends(count, kInvalidIndex);
    std::vector<u32> path;
    for (u32 p = 0; p < count; ++p) {
        path.clear();
        u32 at = p;
        while (ends[at] == kInvalidIndex) {
            u32 lower = kInvalidIndex;
            if (-points[at].outward.z < sheds) {
                f32 lowest = points[at].at.z - step;
                for (const u32 other : rig.ringOf(at)) {
                    if (points[other].at.z < lowest) {
                        lowest = points[other].at.z;
                        lower = other;
                    }
                }
            }
            if (lower == kInvalidIndex) {
                ends[at] = at;
                break;
            }
            path.push_back(at);
            at = lower;
        }
        for (const u32 passed : path) {
            ends[passed] = ends[at];
        }
    }
    // What arrives at each place: its catchment, as skin.
    std::vector<f32> catchment(count, 0.0f);
    f32 skin = 0.0f;
    for (u32 p = 0; p < count; ++p) {
        catchment[ends[p]] += points[p].area;
        skin += points[p].area;
    }
    std::vector<u32> order;
    for (u32 p = 0; p < count; ++p) {
        if (catchment[p] > 0.0f) {
            order.push_back(p);
        }
    }
    // A strand needs room under it: half its length, or three of its widths
    // where it runs to the ground. The rim of a base that stands on the ground
    // has the largest catchment of all and none, so a place that low is no
    // site, unless the body has no other.
    const f32 room = settings.toGround ? 3.0f * width : 0.5f * std::max(settings.length, 0.0f) * size;
    if (std::any_of(order.begin(), order.end(), [&](u32 p) { return points[p].at.z >= room; })) {
        std::erase_if(order, [&](u32 p) { return points[p].at.z < room; });
    }
    std::sort(order.begin(), order.end(), [&](u32 a, u32 b) {
        return catchment[a] != catchment[b] ? catchment[a] > catchment[b] : a < b;
    });
    // Kept *Spacing* apart, the largest first: a place near a larger one runs
    // into it.
    const f32 apart = std::clamp(settings.spacing, 0.01f, 1.0f) * size;
    std::vector<std::pair<u32, f32>> kept;
    for (const u32 p : order) {
        std::pair<u32, f32>* into = nullptr;
        f32 nearest = apart;
        for (std::pair<u32, f32>& other : kept) {
            const f32 d = (points[other.first].at - points[p].at).length();
            if (d < nearest) {
                nearest = d;
                into = &other;
            }
        }
        if (into != nullptr) {
            into->second += catchment[p];
        } else {
            kept.emplace_back(p, catchment[p]);
        }
    }
    std::stable_sort(kept.begin(), kept.end(),
                     [](const std::pair<u32, f32>& a, const std::pair<u32, f32>& b) { return a.second > b.second; });
    kept.resize(std::min<std::size_t>(kept.size(), std::clamp<u32>(settings.sites, 1, 24)));
    f32 mean = 0.0f;
    for (const std::pair<u32, f32>& one : kept) {
        mean += one.second;
    }
    mean = kept.empty() ? 0.0f : mean / static_cast<f32>(kept.size());
    for (const std::pair<u32, f32>& one : kept) {
        SoftDripSite site;
        site.point = one.first;
        site.at = points[one.first].at;
        site.normal = points[one.first].outward;
        site.bone = CarrierOf(rig, one.first);
        site.uv = UvOf(model, points[one.first]);
        site.catchment = skin > 0.0f ? one.second / skin : 0.0f;
        // Thicker where more runs to it: by the root, as a tube's width goes by its flow.
        site.width = width * (mean > 0.0f ? std::clamp(std::sqrt(one.second / mean), 0.6f, 1.6f) : 1.0f);
        sites.push_back(site);
    }
    return sites;
}

f32 SoftNeckWidth(SoftNeckLaw law, f32 narrowest, f32 fallen) {
    const f32 u = std::clamp(fallen, 0.0f, 1.0f);
    const f32 thins = 1.0f - std::clamp(narrowest, 0.02f, 1.0f);
    switch (law) {
    case SoftNeckLaw::Linear:
        // Thick and even: in a straight line.
        return 1.0f - thins * u;
    case SoftNeckLaw::Snap:
        // Thin: slowly, and then at once.
        return 1.0f - thins * u * u * u * u;
    case SoftNeckLaw::Thread:
        // Stringy: most of it early, and a thread that lasts.
        return 1.0f - thins * (1.0f - std::exp(-4.0f * u)) / (1.0f - std::exp(-4.0f));
    case SoftNeckLaw::Sets:
    case SoftNeckLaw::Count:
        // Setting as it runs: it never gets half way.
        return 1.0f - 0.5f * thins * u;
    }
    return 1.0f;
}

namespace {

/// The share of a plain tube's volume a strand lacks with its neck @p neck
/// wide: its two middle rings ride the neck's bone by two thirds, so they are
/// `(1 + 2 neck) / 3` wide, and the three stretches between its four rings
/// are cones' frusta.
f32 GivenUp(f32 neck) {
    const f32 r = (1.0f + 2.0f * std::clamp(neck, 0.0f, 1.0f)) / 3.0f;
    return 1.0f - (2.0f + 2.0f * r + 5.0f * r * r) / 9.0f;
}

} // namespace

f32 SoftTipSwell(const SoftDrip& settings, f32 length, f32 neck) {
    const f32 along = std::clamp(length, 0.0f, 1.0f);
    const f32 whole = GivenUp(SoftNeckWidth(settings.neck, settings.neckWidth, 1.0f));
    if (!(whole > 1e-6f)) {
        // A neck that does not thin gives nothing up: by the strand's length.
        return along * along * (3.0f - 2.0f * along);
    }
    return std::clamp(along * GivenUp(neck) / whole, 0.0f, 1.0f);
}

SoftDrip SoftDripPresetOf(SoftDripPreset preset, const SoftDrip& from) {
    SoftDrip out = from;
    out.preset = preset;
    const auto set = [&](SoftNeckLaw neck, f32 speed, f32 thickness, f32 stagger, f32 pinchEvery) {
        out.neck = neck;
        out.speed = speed;
        out.thickness = thickness;
        out.stagger = stagger;
        out.pinchEvery = pinchEvery;
    };
    switch (preset) {
    case SoftDripPreset::Wax:
        set(SoftNeckLaw::Sets, 0.15f, 0.035f, 0.7f, 0.0f);
        break;
    case SoftDripPreset::Slime:
        set(SoftNeckLaw::Thread, 0.4f, 0.03f, 0.6f, 0.0f);
        break;
    case SoftDripPreset::Blood:
        set(SoftNeckLaw::Snap, 1.2f, 0.015f, 0.5f, 2.0f);
        break;
    case SoftDripPreset::Paint:
        set(SoftNeckLaw::Snap, 0.8f, 0.02f, 0.4f, 3.0f);
        break;
    case SoftDripPreset::Chocolate:
        set(SoftNeckLaw::Linear, 0.25f, 0.035f, 0.5f, 0.0f);
        break;
    case SoftDripPreset::Honey:
        set(SoftNeckLaw::Linear, 0.12f, 0.04f, 0.3f, 0.0f);
        break;
    case SoftDripPreset::Water:
        // A thread and a drop: nothing of it stays to hang.
        set(SoftNeckLaw::Snap, 1.2f, 0.006f, 0.5f, 0.6f);
        break;
    case SoftDripPreset::Custom:
    case SoftDripPreset::Count:
        break;
    }
    return out;
}

f32 SoftSwayRate(f32 reach, f32 height) {
    const f32 tall = std::max(height, 1e-6f);
    return std::clamp(0.75f * std::sqrt(tall / std::max(reach, 0.05f * tall)), 0.25f, 3.0f);
}

Vector2f SoftStrandSwing(f32 sway, u32 seed, u32 strand, f32 rate, f32 seconds) {
    const f32 most = 0.35f * std::clamp(sway, 0.0f, 1.0f);
    if (!(most > 0.0f)) {
        return Vector2f{0, 0};
    }
    // The golden angle apart, strand after strand: no two swing together.
    const f32 phase = 2.3999632f * (static_cast<f32>(seed % 1024u) + 7.0f * static_cast<f32>(strand));
    const f32 turns = 2.0f * kPi * rate * seconds;
    // Across at 0.6 of along and a quarter turn behind: the tip rounds an ellipse.
    return Vector2f{most * std::sin(turns + phase), 0.6f * most * std::cos(turns + phase)};
}

namespace {

f32 Eased(f32 u) {
    u = std::clamp(u, 0.0f, 1.0f);
    return u * u * (3.0f - 2.0f * u);
}

/// The turn that takes a strand's tube, which hangs along -Z as built, to @p way.
Quaternion TurnedTo(const Vector3f& way) {
    const Vector3f built{0, 0, -1};
    const Vector3f axis = cross(built, way);
    const f32 sine = axis.length(), cosine = built.dot(way);
    if (sine > 1e-6f) {
        return Quaternion::from_axis_angle(axis * (1.0f / sine), std::atan2(sine, cosine));
    }
    return cosine < 0.0f ? Quaternion::from_axis_angle(Vector3f{1, 0, 0}, kPi) : Quaternion::identity();
}

/// The way down @p fall, swung by @p share of @p swing (*Sway*).
Vector3f Swung(const Vector3f& fall, const Vector2f& swing, f32 share) {
    const Vector3f beside = std::abs(fall.x) < 0.9f ? Vector3f{1, 0, 0} : Vector3f{0, 1, 0};
    const Vector3f u = cross(fall, beside) * (1.0f / cross(fall, beside).length());
    const Vector3f v = cross(fall, u);
    const Vector3f off = u * (swing.x * share) + v * (swing.y * share);
    const f32 angle = off.length();
    return angle > 1e-7f ? fall * std::cos(angle) + off * (std::sin(angle) / angle) : fall;
}

Vector3f UnitOr(const Vector3f& v, const Vector3f& otherwise) {
    return v.length() > 1e-9f ? v * (1.0f / v.length()) : otherwise;
}

} // namespace

SoftPinch SoftPinchAt(f32 turn, f32 period, f32 grows, bool first) {
    SoftPinch out;
    const f32 whole = std::floor(turn);
    const f32 c = turn - whole;
    const f32 stub = first && whole < 1.0f ? 0.0f : kSoftPinchStub;
    if (c < kSoftPinchAt) {
        // It grows, then hangs while its neck thins and its tip swells.
        const f32 g = std::clamp(grows, 0.05f, kSoftPinchAt);
        out.length = stub + (1.0f - stub) * Eased(c / g);
        out.thinned = c / kSoftPinchAt;
        out.swollen = Eased(c / kSoftPinchAt);
        return out;
    }
    // Let go: the strand draws back over the first quarter of what is left of
    // the cycle, and the droplet shrinks away over the last.
    const f32 after = (c - kSoftPinchAt) / (1.0f - kSoftPinchAt);
    const f32 back = Eased(after / 0.25f);
    out.length = 1.0f + (kSoftPinchStub - 1.0f) * back;
    out.thinned = 1.0f - back;
    out.swollen = 0.0f;
    out.away = true;
    out.since = (c - kSoftPinchAt) * std::max(period, 0.0f);
    out.fade = 1.0f - Eased((after - 0.75f) / 0.25f);
    return out;
}

f32 SoftPinchPhase(u32 seed, u32 strand) {
    // The golden section apart, strand after strand.
    const f32 turn = 0.61803399f * static_cast<f32>(strand + 1) + 0.137f * static_cast<f32>(seed % 97u);
    return turn - std::floor(turn);
}

Vector3f SoftStrandTip(const Vector3f& down, f32 length, const Vector2f& swing) {
    const Vector3f fall = UnitOr(down, Vector3f{0, 0, -1});
    return (Swung(fall, swing, 0.5f) + Swung(fall, swing, 1.0f)) * (0.5f * length);
}

Transform SoftDropletAt(const SoftDrip& settings, f32 width, f32 room, f32 height, const SoftPinch& pinch,
                        const Vector3f& down, const Vector3f& from) {
    Transform out;
    if (!pinch.away) {
        return out;
    }
    const f32 gravity = kSoftDropGravity * std::max(height, 1e-6f);
    const f32 swell = std::clamp(settings.tip, 0.2f, 4.0f);
    // It falls until it touches the floor, then lies flat within a twelfth of
    // a second, the volume kept and its underside where it landed.
    const f32 radius = width * swell;
    const f32 lands = std::sqrt(2.0f * std::max(room - radius, 0.0f) / gravity);
    const bool landed = pinch.since >= lands;
    const f32 tall = 1.0f + (kSoftDropFlat - 1.0f) * (landed ? Eased((pinch.since - lands) / 0.08f) : 0.0f);
    const f32 wide = 1.0f / std::sqrt(tall);
    const f32 fallen = landed ? std::max(room - radius * tall, 0.0f) : 0.5f * gravity * pinch.since * pinch.since;
    // From the hundredth it is built at to its tip's size, and back as it fades.
    const f32 grown = swell / kSoftStrandShrink;
    const auto sized = [&](f32 by) { return 1.0f + (grown * by - 1.0f) * std::clamp(pinch.fade, 0.0f, 1.0f); };
    out.translation = from + down * fallen;
    out.rotation = TurnedTo(UnitOr(down, Vector3f{0, 0, -1}));
    out.scale = Vector3f{sized(wide), sized(wide), sized(tall)};
    return out;
}

SoftStrandPose SoftStrandAt(const SoftDrip& settings, f32 width, f32 reach, f32 height, f32 since, bool hangs,
                            const Vector3f& down, const Vector3f& sunk, const Vector2f& swing, const SoftPinch* pinch) {
    SoftStrandPose out;
    // Not yet: as it is built, which is a hundredth of itself under the surface.
    if (!hangs && since < 0.0f) {
        return out;
    }
    const f32 rest = kSoftStrandRest * width;
    const f32 shrunk = kSoftStrandShrink;
    const f32 speed = std::clamp(settings.speed, 0.01f, 10.0f) * std::max(height, 1e-6f);
    const f32 far = std::max(reach, 0.0f);
    f32 length = far;
    if (pinch != nullptr) {
        length = far * std::clamp(pinch->length, 0.0f, 1.0f);
    } else if (!hangs) {
        // Wax sets as it runs: its speed dies, and it never quite arrives.
        length = settings.neck == SoftNeckLaw::Sets && far > 0.0f ? far * (1.0f - std::exp(-speed * since / far))
                                                                 : std::min(far, speed * since);
    }
    out.length = length;
    const f32 fallen = far > 0.0f ? length / far : 1.0f;
    // A strand that pinches off thins by its cycle, not by its length.
    const f32 neck = SoftNeckWidth(settings.neck, settings.neckWidth, pinch != nullptr ? pinch->thinned : fallen);
    // Its tip takes what its neck gives up, in step; once let go it has none.
    const f32 swollen = pinch != nullptr && pinch->away ? 0.0f : SoftTipSwell(settings, fallen, neck);
    const f32 tip = 1.0f + (std::clamp(settings.tip, 0.2f, 4.0f) - 1.0f) * swollen;
    // Its tube hangs along the model's -Z as built: turned to the way it falls.
    const Vector3f fall = UnitOr(down, Vector3f{0, 0, -1});
    const Vector3f built{0, 0, -1};
    const auto turnedTo = [](const Vector3f& way) { return TurnedTo(way); };
    // *Sway*: a chain of two links, the upper swung half as far as the lower.
    const Vector3f upper = Swung(fall, swing, 0.5f), lower = Swung(fall, swing, 1.0f);
    // Each bone to where the strand has it: out from under the surface, then
    // down the strand; and from the hundredth it is built at to its size.
    const f32 grown = 1.0f / shrunk;
    out.base.translation = sunk;
    out.base.rotation = turnedTo(fall);
    out.base.scale = Vector3f{grown, grown, grown};
    out.neck.translation = sunk + upper * (0.5f * length) - built * (0.5f * rest * shrunk);
    out.neck.rotation = turnedTo(upper);
    out.neck.scale = Vector3f{grown * neck, grown * neck, grown};
    out.tip.translation = sunk + (upper + lower) * (0.5f * length) - built * (rest * shrunk);
    out.tip.rotation = turnedTo(lower);
    out.tip.scale = Vector3f{grown * tip, grown * tip, grown * tip};
    return out;
}

SoftStrandsBuilt BuildSoftStrands(Model& model, SoftBody& body, SoftEffect& effect,
                                  std::span<const SoftDripSite> sites) {
    SoftStrandsBuilt built;
    if (!effect.drip || sites.empty() || !effect.drip->strandNodes.empty()) {
        return built;
    }
    SoftDrip& drip = *effect.drip;
    drip.dropNodes.clear();
    geom::MeshBuilder builder;
    MeshSection section;
    section.name = body.name + " Drips";
    // The surface's material under the first site, unless the effect names one.
    section.materialSlot = drip.material;
    if (section.materialSlot == kInvalidIndex) {
        section.materialSlot = 0;
        const u32 point = sites.front().point;
        if (point != kInvalidIndex) {
            for (u32 m = 0; m < model.meshes.size(); ++m) {
                const std::span<const u32> bodyOf =
                    model.meshes[m].attributes.get<u32>(geom::names::kSoftBody, geom::Domain::Vertex);
                if (std::find(bodyOf.begin(), bodyOf.end(), body.id) != bodyOf.end() && !model.meshes[m].sections.empty()) {
                    section.materialSlot = model.meshes[m].sections.front().materialSlot;
                    break;
                }
            }
        }
    }
    builder.addSection(section);
    builder.declareAttr(geom::Domain::Vertex, geom::names::kSoftBody, geom::AttrType::U32);
    builder.declareAttr(geom::Domain::Vertex, geom::names::kSoftSoftness, geom::AttrType::F32);
    builder.declareAttr(geom::Domain::Vertex, geom::names::kSoftPainted, geom::AttrType::Bool);
    const std::string uv0 = geom::names::uv(0);
    builder.declareAttr(geom::Domain::Halfedge, uv0, geom::AttrType::F32x2);

    // Standing at @p pivot along the model's axes, on either kind of rig.
    const auto bone = [&](const std::string& name, u32 parent, const Vector3f& pivot) {
        Matrix44f rest = Matrix44f::identity();
        rest.data[3][0] = pivot.x, rest.data[3][1] = pivot.y, rest.data[3][2] = pivot.z;
        return AddBoneAt(model.nodes, name, parent, rest);
    };
    std::vector<Vector3f> normals; // By vertex, as the builder numbers them.
    for (std::size_t s = 0; s < sites.size(); ++s) {
        const SoftDripSite& site = sites[s];
        const f32 width = std::max(site.width, 1e-6f);
        const f32 rest = kSoftStrandRest * width;
        // Built a hundredth of itself, and under the surface by its own length.
        const f32 shrunk = kSoftStrandShrink;
        const Vector3f out = site.normal.length() > 1e-9f ? site.normal * (1.0f / site.normal.length()) : Vector3f{0, 0, -1};
        const Vector3f top = site.at - out * (1.5f * rest * shrunk);
        char number[16];
        std::snprintf(number, sizeof number, " %02u", static_cast<unsigned>(s + 1));
        const std::string stem = effect.name + number;
        // Side by side under the site's bone, not a chain: with no skinned bone
        // below it each is used as it is, and a file has three bones a strand.
        const u32 base = bone(stem + " base", site.bone, top);
        const u32 neck = bone(stem + " neck", site.bone, top + Vector3f{0, 0, -0.5f * rest * shrunk});
        const u32 tip = bone(stem + " tip", site.bone, top + Vector3f{0, 0, -rest * shrunk});
        drip.strandNodes.insert(drip.strandNodes.end(), {base, neck, tip});
        drip.strandAt.push_back(site.at);
        drip.strandWidth.push_back(width);
        built.bones += 3;

        std::array<std::array<geom::VertexId, kSides>, kRings> ring{};
        const auto vertex = [&](const Vector3f& at, const Vector3f& normal, std::initializer_list<std::pair<u32, f32>> skin) {
            const geom::VertexId made = builder.addVertex(at);
            for (const auto& [node, weight] : skin) {
                builder.addInfluence(made, node, weight);
            }
            builder.setVertexAttr<u32>(made, geom::names::kSoftBody, body.id);
            builder.setVertexAttr<f32>(made, geom::names::kSoftSoftness, 1.0f);
            // Wholly soft, and kept so: a remake's blend spares what is painted.
            builder.setVertexAttr<u8>(made, geom::names::kSoftPainted, u8{1});
            normals.push_back(normal);
            return made;
        };
        for (u32 r = 0; r < kRings; ++r) {
            const f32 along = static_cast<f32>(r) / static_cast<f32>(kRings - 1);
            for (u32 k = 0; k < kSides; ++k) {
                const f32 angle = kTau * static_cast<f32>(k) / static_cast<f32>(kSides);
                const Vector3f round{std::cos(angle), std::sin(angle), 0.0f};
                const Vector3f at = top + round * (width * shrunk) + Vector3f{0, 0, -along * rest * shrunk};
                // By where the ring stands between its two bones.
                ring[r][k] = r == 0   ? vertex(at, round, {{base, 1.0f}})
                             : r == 1 ? vertex(at, round, {{base, 1.0f / 3.0f}, {neck, 2.0f / 3.0f}})
                             : r == 2 ? vertex(at, round, {{neck, 2.0f / 3.0f}, {tip, 1.0f / 3.0f}})
                                      : vertex(at, round, {{tip, 1.0f}});
            }
        }
        const geom::VertexId end =
            vertex(top + Vector3f{0, 0, -(rest + width) * shrunk}, Vector3f{0, 0, -1}, {{tip, 1.0f}});
        const auto face = [&](geom::VertexId a, geom::VertexId b, geom::VertexId c) {
            const geom::FaceId made = builder.addTriangle(a, b, c);
            const geom::VertexId corners[3] = {a, b, c};
            for (u32 corner = 0; corner < 3; ++corner) {
                builder.setCornerAttr<Vector3f>(made, corner, geom::names::kNormal, normals[corners[corner].value()]);
                builder.setCornerAttr<Vector2f>(made, corner, uv0, site.uv);
            }
        };
        for (u32 k = 0; k < kSides; ++k) {
            const u32 next = (k + 1) % kSides;
            for (u32 r = 0; r + 1 < kRings; ++r) {
                face(ring[r][k], ring[r + 1][k], ring[r + 1][next]);
                face(ring[r][k], ring[r + 1][next], ring[r][next]);
            }
            face(ring[kRings - 1][k], end, ring[kRings - 1][next]);
        }
        if (drip.pinches()) {
            // Its droplet: one more bone where the base's is, and a ball of the
            // strand's width about it, two poles and two rings.
            const u32 drop = bone(stem + " drop", site.bone, top);
            drip.dropNodes.push_back(drop);
            built.bones += 1;
            const f32 radius = width * shrunk;
            const geom::VertexId over = vertex(top + Vector3f{0, 0, radius}, Vector3f{0, 0, 1}, {{drop, 1.0f}});
            const geom::VertexId under = vertex(top + Vector3f{0, 0, -radius}, Vector3f{0, 0, -1}, {{drop, 1.0f}});
            std::array<std::array<geom::VertexId, kSides>, 2> band{};
            for (u32 r = 0; r < 2; ++r) {
                const f32 z = r == 0 ? 0.5f : -0.5f;
                const f32 across = std::sqrt(1.0f - z * z);
                for (u32 k = 0; k < kSides; ++k) {
                    const f32 angle = kTau * static_cast<f32>(k) / static_cast<f32>(kSides);
                    const Vector3f way{across * std::cos(angle), across * std::sin(angle), z};
                    band[r][k] = vertex(top + way * radius, way, {{drop, 1.0f}});
                }
            }
            for (u32 k = 0; k < kSides; ++k) {
                const u32 next = (k + 1) % kSides;
                face(over, band[0][k], band[0][next]);
                face(band[0][k], band[1][k], band[1][next]);
                face(band[0][k], band[1][next], band[0][next]);
                face(band[1][k], under, band[1][next]);
            }
        }
        ++built.strands;
    }
    Mesh mesh = builder.build().mesh;
    mesh.name = section.name;
    built.mesh = static_cast<u32>(model.meshes.size());
    model.meshes.push_back(std::move(mesh));
    body.meshes.push_back(built.mesh);
    std::sort(body.meshes.begin(), body.meshes.end());
    body.meshes.erase(std::unique(body.meshes.begin(), body.meshes.end()), body.meshes.end());
    body.drips = built.mesh;
    return built;
}

} // namespace wem
} // namespace models
} // namespace whiteout
