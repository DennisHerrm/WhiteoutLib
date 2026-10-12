// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/physics/soft_fields.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <queue>

#include <whiteout/models/wem/geometry/attributes.h>

namespace whiteout {
namespace models {
namespace wem {

namespace {

/// A region that encloses under this share of its area times its size is a
/// flat patch (a hemisphere's is 0.17).
constexpr f64 kFlat = 0.005;

f64 Det(const Vector3f& a, const Vector3f& b, const Vector3f& c) {
    return static_cast<f64>(a.x) * (static_cast<f64>(b.y) * c.z - static_cast<f64>(b.z) * c.y) -
           static_cast<f64>(a.y) * (static_cast<f64>(b.x) * c.z - static_cast<f64>(b.z) * c.x) +
           static_cast<f64>(a.z) * (static_cast<f64>(b.x) * c.y - static_cast<f64>(b.y) * c.x);
}

Vector3f MiddleOf(std::span<const u32> loop, std::span<const Vector3f> values) {
    Vector3f sum{0, 0, 0};
    for (const u32 p : loop) {
        sum = sum + values[p];
    }
    return loop.empty() ? sum : sum * (1.0f / static_cast<f32>(loop.size()));
}

/// One connected piece of the surface, with its open borders fanned shut.
struct Shell {
    std::vector<std::array<u32, 3>> triangles;
    std::vector<std::vector<u32>> fans;
    f64 volume = 0.0; ///< Signed, as wound.
    u32 first = 0;    ///< A point of it.
};

f64 ShellVolume(const Shell& shell, std::span<const Vector3f> at) {
    f64 six = 0.0;
    for (const std::array<u32, 3>& t : shell.triangles) {
        six += Det(at[t[0]], at[t[1]], at[t[2]]);
    }
    for (const std::vector<u32>& loop : shell.fans) {
        const Vector3f middle = MiddleOf(loop, at);
        for (std::size_t i = 0; i < loop.size(); ++i) {
            six += Det(at[loop[i]], at[loop[(i + 1) % loop.size()]], middle);
        }
    }
    return six / 6.0;
}

/// Whether the ray from @p from along a fixed, skew direction crosses the
/// triangle: counted over a closed shell, an odd number is inside.
bool Crosses(const Vector3f& from, const Vector3f& a, const Vector3f& b, const Vector3f& c) {
    const Vector3f direction{0.9993f, 0.0131f, 0.0351f};
    const Vector3f e1 = b - a, e2 = c - a;
    const Vector3f h = cross(direction, e2);
    const f32 det = e1.dot(h);
    if (std::abs(det) < 1e-20f) {
        return false;
    }
    const f32 inv = 1.0f / det;
    const Vector3f s = from - a;
    const f32 u = inv * s.dot(h);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Vector3f q = cross(s, e1);
    const f32 v = inv * direction.dot(q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    return inv * e2.dot(q) > 1e-7f;
}

bool Inside(const Vector3f& point, const Shell& shell, std::span<const Vector3f> at) {
    u32 crossings = 0;
    for (const std::array<u32, 3>& t : shell.triangles) {
        crossings += Crosses(point, at[t[0]], at[t[1]], at[t[2]]) ? 1 : 0;
    }
    for (const std::vector<u32>& loop : shell.fans) {
        const Vector3f middle = MiddleOf(loop, at);
        for (std::size_t i = 0; i < loop.size(); ++i) {
            crossings += Crosses(point, at[loop[i]], at[loop[(i + 1) % loop.size()]], middle) ? 1 : 0;
        }
    }
    return (crossings & 1) != 0;
}

} // namespace

SoftVolume MakeSoftVolume(const SoftRig& rig) {
    SoftVolume out;
    if (rig.empty()) {
        return out;
    }
    const u32 count = rig.pointCount();
    std::vector<Vector3f> at(count);
    for (u32 p = 0; p < count; ++p) {
        at[p] = rig.points().points[p].at;
    }
    const std::span<const std::array<u32, 3>> triangles = rig.triangles();

    // The shells: pieces joined by a shared point.
    std::vector<u32> parent(count);
    std::iota(parent.begin(), parent.end(), 0u);
    const auto find = [&](u32 p) {
        while (parent[p] != p) {
            parent[p] = parent[parent[p]];
            p = parent[p];
        }
        return p;
    };
    for (const std::array<u32, 3>& t : triangles) {
        parent[find(t[1])] = find(t[0]);
        parent[find(t[2])] = find(t[0]);
    }
    std::map<u32, u32> shellOf;
    std::vector<Shell> shells;
    for (const std::array<u32, 3>& t : triangles) {
        const u32 root = find(t[0]);
        const auto [it, fresh] = shellOf.try_emplace(root, static_cast<u32>(shells.size()));
        if (fresh) {
            shells.emplace_back().first = t[0];
        }
        shells[it->second].triangles.push_back(t);
    }
    out.shells = static_cast<u32>(shells.size());

    // Each shell's open borders: the edges one face alone holds, chained into
    // loops, and wound to close the shell.
    for (Shell& shell : shells) {
        std::map<std::pair<u32, u32>, u32> uses;
        for (const std::array<u32, 3>& t : shell.triangles) {
            for (int k = 0; k < 3; ++k) {
                const u32 a = t[k], b = t[(k + 1) % 3];
                ++uses[{std::min(a, b), std::max(a, b)}];
            }
        }
        std::map<u32, u32> next; // Along the border, against the faces' own direction.
        for (const std::array<u32, 3>& t : shell.triangles) {
            for (int k = 0; k < 3; ++k) {
                const u32 a = t[k], b = t[(k + 1) % 3];
                if (uses[{std::min(a, b), std::max(a, b)}] == 1) {
                    next[b] = a;
                }
            }
        }
        while (!next.empty()) {
            std::vector<u32> loop;
            u32 here = next.begin()->first;
            const u32 start = here;
            while (true) {
                const auto step = next.find(here);
                if (step == next.end()) {
                    break;
                }
                loop.push_back(here);
                here = step->second;
                next.erase(step);
                if (here == start) {
                    break;
                }
            }
            if (loop.size() >= 3) {
                shell.fans.push_back(std::move(loop));
            }
        }
        shell.volume = ShellVolume(shell, at);
    }

    // The outermost: a shell is another's inside when a point of it is in the
    // other, and the other is the larger (or, of two that coincide, the first).
    f64 rest = 0.0;
    for (std::size_t a = 0; a < shells.size(); ++a) {
        bool inner = false;
        for (std::size_t b = 0; b < shells.size() && !inner; ++b) {
            if (a == b) {
                continue;
            }
            const f64 mine = std::abs(shells[a].volume), theirs = std::abs(shells[b].volume);
            const bool larger = theirs > mine * (1.0 + 1e-6) || (theirs >= mine * (1.0 - 1e-6) && b < a);
            inner = larger && Inside(at[shells[a].first], shells[b], at);
        }
        if (inner) {
            continue;
        }
        ++out.outer;
        // Wound outward, whatever its faces say (§5.4).
        const bool insideOut = shells[a].volume < 0.0;
        for (const std::array<u32, 3>& t : shells[a].triangles) {
            out.triangles.push_back(insideOut ? std::array<u32, 3>{t[0], t[2], t[1]} : t);
        }
        for (std::vector<u32>& loop : shells[a].fans) {
            if (insideOut) {
                std::reverse(loop.begin(), loop.end());
            }
            out.fans.push_back(std::move(loop));
        }
        rest += std::abs(shells[a].volume);
    }
    out.rest = static_cast<f32>(rest);
    out.encloses = rest > kFlat * static_cast<f64>(rig.points().area) * static_cast<f64>(rig.points().size);
    return out;
}

f64 SoftVolumeAt(const SoftVolume& volume, std::span<const Vector3f> positions) {
    f64 six = 0.0;
    for (const std::array<u32, 3>& t : volume.triangles) {
        six += Det(positions[t[0]], positions[t[1]], positions[t[2]]);
    }
    for (const std::vector<u32>& loop : volume.fans) {
        const Vector3f middle = MiddleOf(loop, positions);
        for (std::size_t i = 0; i < loop.size(); ++i) {
            six += Det(positions[loop[i]], positions[loop[(i + 1) % loop.size()]], middle);
        }
    }
    return six / 6.0;
}

std::vector<Vector3f> SoftVolumeGradient(const SoftVolume& volume, std::span<const Vector3f> positions) {
    std::vector<Vector3f> gradient(positions.size(), Vector3f{0, 0, 0});
    // A triangle's six volumes are a . (b x c): each corner's share is the
    // cross product of the other two, in turn.
    const auto add = [&](u32 a, const Vector3f& b, const Vector3f& c) { gradient[a] = gradient[a] + cross(b, c); };
    for (const std::array<u32, 3>& t : volume.triangles) {
        add(t[0], positions[t[1]], positions[t[2]]);
        add(t[1], positions[t[2]], positions[t[0]]);
        add(t[2], positions[t[0]], positions[t[1]]);
    }
    for (const std::vector<u32>& loop : volume.fans) {
        if (loop.empty()) {
            continue;
        }
        // The fan's middle is the mean of its loop: its share goes to them all.
        const Vector3f middle = MiddleOf(loop, positions);
        Vector3f through{0, 0, 0};
        for (std::size_t i = 0; i < loop.size(); ++i) {
            const u32 a = loop[i], b = loop[(i + 1) % loop.size()];
            add(a, positions[b], middle);
            add(b, middle, positions[a]);
            through = through + cross(positions[a], positions[b]);
        }
        through = through * (1.0f / static_cast<f32>(loop.size()));
        for (const u32 p : loop) {
            gradient[p] = gradient[p] + through;
        }
    }
    for (Vector3f& g : gradient) {
        g = g * (1.0f / 6.0f);
    }
    return gradient;
}

std::array<f64, 4> OffsetCubic(const SoftVolume& volume, std::span<const Vector3f> positions,
                               std::span<const Vector3f> moves) {
    std::array<f64, 4> six{0, 0, 0, 0};
    const auto add = [&](const Vector3f& xa, const Vector3f& xb, const Vector3f& xc, const Vector3f& ua,
                         const Vector3f& ub, const Vector3f& uc) {
        six[0] += Det(xa, xb, xc);
        six[1] += Det(ua, xb, xc) + Det(xa, ub, xc) + Det(xa, xb, uc);
        six[2] += Det(ua, ub, xc) + Det(ua, xb, uc) + Det(xa, ub, uc);
        six[3] += Det(ua, ub, uc);
    };
    for (const std::array<u32, 3>& t : volume.triangles) {
        add(positions[t[0]], positions[t[1]], positions[t[2]], moves[t[0]], moves[t[1]], moves[t[2]]);
    }
    for (const std::vector<u32>& loop : volume.fans) {
        // The fan's middle is the mean of its loop, and so moves by the mean.
        const Vector3f middle = MiddleOf(loop, positions);
        const Vector3f moved = MiddleOf(loop, moves);
        for (std::size_t i = 0; i < loop.size(); ++i) {
            const u32 a = loop[i], b = loop[(i + 1) % loop.size()];
            add(positions[a], positions[b], middle, moves[a], moves[b], moved);
        }
    }
    return {six[0] / 6.0, six[1] / 6.0, six[2] / 6.0, six[3] / 6.0};
}

f64 SolveOffset(const std::array<f64, 4>& cubic, f64 target, f64 limit) {
    const auto f = [&](f64 d) { return ((cubic[3] * d + cubic[2]) * d + cubic[1]) * d + cubic[0] - target; };
    const f64 at0 = f(0.0);
    if (at0 == 0.0 || !(limit > 0.0)) {
        return 0.0;
    }
    // Outward from 0 on both sides, for the first change of sign; then halved.
    constexpr int kSteps = 96;
    f64 best = 0.0;
    bool found = false;
    for (const f64 side : {1.0, -1.0}) {
        f64 a = 0.0, fa = at0;
        for (int i = 1; i <= kSteps; ++i) {
            // Fine near 0, where a small Amount's root is.
            const f64 t = static_cast<f64>(i) / kSteps;
            const f64 b = side * limit * t * t;
            const f64 fb = f(b);
            if ((fa < 0.0) != (fb < 0.0) || fb == 0.0) {
                f64 lo = a, hi = b, flo = fa;
                for (int pass = 0; pass < 80; ++pass) {
                    const f64 mid = 0.5 * (lo + hi);
                    const f64 fm = f(mid);
                    if ((flo < 0.0) != (fm < 0.0)) {
                        hi = mid;
                    } else {
                        lo = mid;
                        flo = fm;
                    }
                }
                const f64 root = 0.5 * (lo + hi);
                if (!found || std::abs(root) < std::abs(best)) {
                    best = root;
                    found = true;
                }
                break;
            }
            a = b;
            fa = fb;
        }
    }
    if (found) {
        return best;
    }
    return std::abs(f(limit)) < std::abs(f(-limit)) ? limit : -limit;
}

std::vector<f32> SurfaceDistances(const SoftRig& rig, std::span<const std::pair<u32, f32>> seeds, f32 cutoff) {
    const u32 count = rig.pointCount();
    std::vector<f32> distance(count, cutoff);
    using Entry = std::pair<f32, u32>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
    for (const auto& [point, from] : seeds) {
        if (point < count && from < distance[point]) {
            distance[point] = from;
            open.push({from, point});
        }
    }
    while (!open.empty()) {
        const auto [d, p] = open.top();
        open.pop();
        if (d > distance[p]) {
            continue;
        }
        const Vector3f& here = rig.points().points[p].at;
        for (const u32 other : rig.ringOf(p)) {
            const f32 through = d + (rig.points().points[other].at - here).length();
            if (through < distance[other]) {
                distance[other] = through;
                open.push({through, other});
            }
        }
    }
    return distance;
}

SoftPlaceAt ResolveSoftPlace(const Model& model, const SoftRig& rig, const SoftPlace& place) {
    SoftPlaceAt out;
    if (rig.empty() || place.mesh >= model.meshes.size() || place.mesh >= rig.points().pointOfVertex.size()) {
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
    const std::span<const Vector3f> positions =
        mesh.attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
    const std::vector<u32>& entryOf = rig.points().pointOfVertex[place.mesh];
    const f32 weights[3] = {place.weights.x, place.weights.y, place.weights.z};
    u32 held = 0;
    for (int k = 0; k < 3; ++k) {
        const u32 v = faces.cornerVertex[corner + static_cast<std::size_t>(k)];
        if (v >= positions.size()) {
            return out;
        }
        out.at = out.at + positions[v] * weights[k];
        out.points[k] = v < entryOf.size() ? entryOf[v] : kInvalidIndex;
        held += out.points[k] != kInvalidIndex ? 1 : 0;
    }
    if (held == 0) {
        return out;
    }
    out.weights = place.weights;
    if (held < 3) {
        // A face on the region's border: the place rides the body's point nearest it.
        u32 nearest = kInvalidIndex;
        f32 least = 0.0f;
        for (int k = 0; k < 3; ++k) {
            if (out.points[k] == kInvalidIndex) {
                continue;
            }
            const f32 d = (rig.points().points[out.points[k]].at - out.at).length();
            if (nearest == kInvalidIndex || d < least) {
                nearest = out.points[k];
                least = d;
            }
        }
        out.points = {nearest, nearest, nearest};
        out.weights = Vector3f{1, 0, 0};
    }
    Vector3f normal{0, 0, 0};
    std::map<u32, f32> bones;
    for (int k = 0; k < 3; ++k) {
        normal = normal + rig.points().points[out.points[k]].outward;
        for (const SoftSlot& slot : rig.slotsOf(out.points[k])) {
            bones[slot.node] += slot.weight;
        }
    }
    const f32 length = normal.length();
    out.normal = length > 1e-12f ? normal * (1.0f / length) : Vector3f{0, 0, 1};
    f32 most = -1.0f;
    for (const auto& [bone, weight] : bones) {
        if (weight > most) {
            most = weight;
            out.bone = bone;
        }
    }
    out.valid = true;
    return out;
}

Vector3f SoftPlacePosition(const SoftPlaceAt& place, std::span<const Vector3f> positions) {
    if (!place.valid) {
        return place.at;
    }
    return positions[place.points[0]] * place.weights.x + positions[place.points[1]] * place.weights.y +
           positions[place.points[2]] * place.weights.z;
}

} // namespace wem
} // namespace models
} // namespace whiteout
