// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

#include <whiteout/models/wem/skinning/solid.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <queue>

#include "generate_common.h"

namespace whiteout {
namespace models {
namespace wem {
namespace skinning {

namespace {

/// Where the parity lines run, off the cell centres, as a fraction of a cell.
/// A line through a centre would pass exactly along the edges of a mesh built
/// on the same spacing -- a box, say -- and count its crossings twice or not at
/// all.
constexpr f32 kLineJitterU = 0.0137f;
constexpr f32 kLineJitterV = 0.0291f;

constexpr f32 kUnreached = std::numeric_limits<f32>::infinity();

f32 Axis(const Vector3f& v, i32 axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

/// Akenine-Möller's triangle/box overlap: the separating-axis test over the
/// box's three faces, the triangle's plane and the nine edge crosses. The box
/// is centred on the origin, and the triangle given relative to its centre.
bool TriangleOverlapsBox(const Vector3f& half, const Vector3f& v0, const Vector3f& v1,
                         const Vector3f& v2) {
    // The box's faces.
    for (i32 axis = 0; axis < 3; ++axis) {
        const f32 a = Axis(v0, axis);
        const f32 b = Axis(v1, axis);
        const f32 c = Axis(v2, axis);
        const f32 h = Axis(half, axis);
        if (std::min({a, b, c}) > h || std::max({a, b, c}) < -h) {
            return false;
        }
    }
    // The nine crosses of a triangle edge with a box axis.
    const Vector3f edges[3]{v1 - v0, v2 - v1, v0 - v2};
    const Vector3f units[3]{Vector3f{1, 0, 0}, Vector3f{0, 1, 0}, Vector3f{0, 0, 1}};
    for (const Vector3f& edge : edges) {
        for (const Vector3f& unit : units) {
            const Vector3f axis = cross(unit, edge);
            if (axis.length_squared() <= 1e-20f) {
                continue; // parallel: the face tests above already cover it
            }
            const f32 p0 = axis.dot(v0);
            const f32 p1 = axis.dot(v1);
            const f32 p2 = axis.dot(v2);
            const f32 radius = half.x * std::abs(axis.x) + half.y * std::abs(axis.y) +
                               half.z * std::abs(axis.z);
            if (std::min({p0, p1, p2}) > radius || std::max({p0, p1, p2}) < -radius) {
                return false;
            }
        }
    }
    // The triangle's plane.
    const Vector3f normal = cross(edges[0], edges[1]);
    const f32 offset = normal.dot(v0);
    const f32 radius = half.x * std::abs(normal.x) + half.y * std::abs(normal.y) +
                       half.z * std::abs(normal.z);
    return std::abs(offset) <= radius;
}

/// The 26 steps a walk takes, and their lengths in cells.
struct Step {
    i32 dx, dy, dz;
    f32 length;
};

std::vector<Step> Steps() {
    std::vector<Step> steps;
    for (i32 dz = -1; dz <= 1; ++dz) {
        for (i32 dy = -1; dy <= 1; ++dy) {
            for (i32 dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                steps.push_back(
                    {dx, dy, dz, std::sqrt(static_cast<f32>(dx * dx + dy * dy + dz * dz))});
            }
        }
    }
    return steps;
}

} // namespace

bool SolidGrid::clip(const Vector3f& from, const Vector3f& to, f32& t0, f32& t1) const {
    t0 = 0.0f;
    t1 = 1.0f;
    for (i32 axis = 0; axis < 3; ++axis) {
        const f32 start = axis == 0 ? from.x : (axis == 1 ? from.y : from.z);
        const f32 end = axis == 0 ? to.x : (axis == 1 ? to.y : to.z);
        const f32 low = axis == 0 ? origin.x : (axis == 1 ? origin.y : origin.z);
        const f32 high = low + static_cast<f32>(size[static_cast<std::size_t>(axis)]) * cell;
        const f32 delta = end - start;
        if (std::abs(delta) < 1e-12f) {
            if (start < low || start > high) {
                return false;
            }
            continue;
        }
        f32 enter = (low - start) / delta;
        f32 leave = (high - start) / delta;
        if (enter > leave) {
            std::swap(enter, leave);
        }
        t0 = std::max(t0, enter);
        t1 = std::min(t1, leave);
        if (t0 > t1) {
            return false;
        }
    }
    return true;
}

SolidGrid MakeSolidGrid(const Vector3f& low, const Vector3f& high, u32 resolution) {
    SolidGrid grid;
    const f32 longest = std::max({high.x - low.x, high.y - low.y, high.z - low.z});
    resolution = std::max<u32>(resolution, 4);
    grid.cell = longest > 0.0f ? longest / static_cast<f32>(resolution) : 1.0f;
    grid.origin = Vector3f{low.x - kSolidPad * grid.cell, low.y - kSolidPad * grid.cell,
                           low.z - kSolidPad * grid.cell};
    const f32 extent[3]{high.x - low.x, high.y - low.y, high.z - low.z};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        grid.size[axis] = static_cast<i32>(std::ceil(extent[axis] / grid.cell)) + 1 + 2 * kSolidPad;
    }
    return grid;
}

std::vector<u8> BuildSolid(const SolidGrid& grid, std::span<const Mesh* const> meshes) {
    const std::size_t cells = grid.count();
    std::vector<u8> shell(cells, 0);
    // Per axis, per line: where the line crosses a triangle, along the axis.
    std::array<std::vector<std::vector<f32>>, 3> lines;
    for (i32 axis = 0; axis < 3; ++axis) {
        const i32 u = (axis + 1) % 3;
        const i32 v = (axis + 2) % 3;
        lines[static_cast<std::size_t>(axis)].resize(static_cast<std::size_t>(grid.size[static_cast<std::size_t>(u)]) *
                                                     static_cast<std::size_t>(grid.size[static_cast<std::size_t>(v)]));
    }
    // A hair over half a cell, so a triangle lying exactly on a cell face marks
    // the cells on both sides rather than neither.
    const f32 halfCell = grid.cell * 0.5f * 1.0001f;
    const Vector3f half{halfCell, halfCell, halfCell};
    const f32 originOf[3]{grid.origin.x, grid.origin.y, grid.origin.z};
    const auto sizeOf = [&](i32 axis) { return grid.size[static_cast<std::size_t>(axis)]; };

    for (const Mesh* mesh : meshes) {
        if (mesh == nullptr) {
            continue;
        }
        const std::span<const Vector3f> positions =
            mesh->attributes.get<Vector3f>(geom::names::kPosition, geom::Domain::Vertex);
        const std::vector<u32> triangles = detail::TrianglesOf(*mesh);
        for (std::size_t t = 0; t + 2 < triangles.size(); t += 3) {
            if (triangles[t] >= positions.size() || triangles[t + 1] >= positions.size() ||
                triangles[t + 2] >= positions.size()) {
                continue;
            }
            const Vector3f a = positions[triangles[t]];
            const Vector3f b = positions[triangles[t + 1]];
            const Vector3f c = positions[triangles[t + 2]];

            // The shell: every cell of the triangle's box it overlaps.
            const Vector3f low{std::min({a.x, b.x, c.x}), std::min({a.y, b.y, c.y}),
                               std::min({a.z, b.z, c.z})};
            const Vector3f high{std::max({a.x, b.x, c.x}), std::max({a.y, b.y, c.y}),
                                std::max({a.z, b.z, c.z})};
            std::array<i32, 3> from = grid.cellOf(low);
            std::array<i32, 3> to = grid.cellOf(high);
            for (std::size_t axis = 0; axis < 3; ++axis) {
                from[axis] = std::clamp(from[axis] - 1, 0, grid.size[axis] - 1);
                to[axis] = std::clamp(to[axis] + 1, 0, grid.size[axis] - 1);
            }
            for (i32 z = from[2]; z <= to[2]; ++z) {
                for (i32 y = from[1]; y <= to[1]; ++y) {
                    for (i32 x = from[0]; x <= to[0]; ++x) {
                        const std::size_t cell = grid.index(x, y, z);
                        if (shell[cell] != 0) {
                            continue;
                        }
                        const Vector3f centre = grid.centre(x, y, z);
                        if (TriangleOverlapsBox(half, a - centre, b - centre, c - centre)) {
                            shell[cell] = 1;
                        }
                    }
                }
            }

            // The parity lines: where each line through the triangle's shadow
            // crosses it.
            for (i32 axis = 0; axis < 3; ++axis) {
                const i32 u = (axis + 1) % 3;
                const i32 v = (axis + 2) % 3;
                const f32 au = Axis(a, u), av = Axis(a, v);
                const f32 bu = Axis(b, u), bv = Axis(b, v);
                const f32 cu = Axis(c, u), cv = Axis(c, v);
                const f32 area = (bu - au) * (cv - av) - (bv - av) * (cu - au);
                if (std::abs(area) <= 1e-20f) {
                    continue; // edge-on to this axis: no line crosses it
                }
                const auto lineOf = [&](f32 value, i32 along, f32 jitter) {
                    return static_cast<i32>(
                        std::floor((value - originOf[along]) / grid.cell - jitter));
                };
                const i32 u0 = std::max(lineOf(std::min({au, bu, cu}), u, kLineJitterU), 0);
                const i32 u1 = std::min(lineOf(std::max({au, bu, cu}), u, kLineJitterU), sizeOf(u) - 1);
                const i32 v0 = std::max(lineOf(std::min({av, bv, cv}), v, kLineJitterV), 0);
                const i32 v1 = std::min(lineOf(std::max({av, bv, cv}), v, kLineJitterV), sizeOf(v) - 1);
                for (i32 iv = v0; iv <= v1; ++iv) {
                    const f32 qv =
                        originOf[v] + (static_cast<f32>(iv) + 0.5f + kLineJitterV) * grid.cell;
                    for (i32 iu = u0; iu <= u1; ++iu) {
                        const f32 qu =
                            originOf[u] + (static_cast<f32>(iu) + 0.5f + kLineJitterU) * grid.cell;
                        // Barycentric weights of the line's foot in the shadow.
                        const f32 wa = ((bu - qu) * (cv - qv) - (bv - qv) * (cu - qu)) / area;
                        const f32 wb = ((cu - qu) * (av - qv) - (cv - qv) * (au - qu)) / area;
                        const f32 wc = 1.0f - wa - wb;
                        if (wa < 0.0f || wb < 0.0f || wc < 0.0f) {
                            continue;
                        }
                        const f32 along = wa * Axis(a, axis) + wb * Axis(b, axis) +
                                          wc * Axis(c, axis);
                        lines[static_cast<std::size_t>(axis)][static_cast<std::size_t>(iv) *
                                                                  static_cast<std::size_t>(sizeOf(u)) +
                                                              static_cast<std::size_t>(iu)]
                            .push_back(along);
                    }
                }
            }
        }
    }

    // The flood from outside, 6-connected through what is not shell. The pad
    // makes every border cell empty, so one seed in the corner reaches round.
    std::vector<u8> outside(cells, 0);
    std::deque<std::size_t> front;
    front.push_back(grid.index(0, 0, 0));
    outside[grid.index(0, 0, 0)] = 1;
    while (!front.empty()) {
        const std::size_t cell = front.front();
        front.pop_front();
        const std::array<i32, 3> at = grid.at(cell);
        static const i32 kSteps[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                         {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
        for (const auto& step : kSteps) {
            const i32 nx = at[0] + step[0];
            const i32 ny = at[1] + step[1];
            const i32 nz = at[2] + step[2];
            if (!grid.contains(nx, ny, nz)) {
                continue;
            }
            const std::size_t next = grid.index(nx, ny, nz);
            if (outside[next] != 0 || shell[next] != 0) {
                continue;
            }
            outside[next] = 1;
            front.push_back(next);
        }
    }

    // The vote: a cell is inside for an axis when its line crosses the shell
    // an odd number of times on BOTH sides of it. An open line -- one through a
    // hole, or past the edge of a cape -- has an odd total, so it can never
    // say "inside" on its own; two of the three lines have to agree.
    std::vector<u8> votes(cells, 0);
    for (i32 axis = 0; axis < 3; ++axis) {
        const i32 u = (axis + 1) % 3;
        const i32 v = (axis + 2) % 3;
        for (i32 iv = 0; iv < sizeOf(v); ++iv) {
            for (i32 iu = 0; iu < sizeOf(u); ++iu) {
                std::vector<f32>& hits =
                    lines[static_cast<std::size_t>(axis)][static_cast<std::size_t>(iv) *
                                                              static_cast<std::size_t>(sizeOf(u)) +
                                                          static_cast<std::size_t>(iu)];
                if (hits.size() < 2) {
                    continue;
                }
                std::sort(hits.begin(), hits.end());
                for (i32 k = 0; k < sizeOf(axis); ++k) {
                    const f32 centre =
                        originOf[axis] + (static_cast<f32>(k) + 0.5f) * grid.cell;
                    const std::size_t before = static_cast<std::size_t>(
                        std::lower_bound(hits.begin(), hits.end(), centre) - hits.begin());
                    const std::size_t after = hits.size() - before;
                    if ((before & 1u) == 0 || (after & 1u) == 0) {
                        continue;
                    }
                    i32 at[3];
                    at[axis] = k;
                    at[u] = iu;
                    at[v] = iv;
                    ++votes[grid.index(at[0], at[1], at[2])];
                }
            }
        }
    }

    std::vector<u8> solid(cells, 0);
    for (std::size_t cell = 0; cell < cells; ++cell) {
        solid[cell] = (outside[cell] == 0 || votes[cell] >= 2) ? 1 : 0;
    }
    return solid;
}

SolidWalk WalkSolid(const SolidGrid& grid, std::span<const u8> solid, std::span<const SolidSeed> seeds,
                    f32 cutoff) {
    SolidWalk walk;
    const std::size_t cells = grid.count();
    walk.distance.assign(cells, kUnreached);
    walk.low = {grid.size[0], grid.size[1], grid.size[2]};
    walk.high = {-1, -1, -1};
    using Entry = std::pair<f32, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    for (const SolidSeed& seed : seeds) {
        if (seed.cell < cells && seed.distance < walk.distance[seed.cell]) {
            walk.distance[seed.cell] = seed.distance;
            queue.push({seed.distance, seed.cell});
        }
    }
    // A walk's own: callers run walks side by side, and twenty-six steps cost
    // nothing beside one.
    const std::vector<Step> steps = Steps();
    while (!queue.empty()) {
        const auto [d, cell] = queue.top();
        queue.pop();
        if (d > walk.distance[cell]) {
            continue;
        }
        const std::array<i32, 3> at = grid.at(cell);
        walk.low = {std::min(walk.low[0], at[0]), std::min(walk.low[1], at[1]), std::min(walk.low[2], at[2])};
        walk.high = {std::max(walk.high[0], at[0]), std::max(walk.high[1], at[1]), std::max(walk.high[2], at[2])};
        for (const Step& step : steps) {
            const i32 nx = at[0] + step.dx;
            const i32 ny = at[1] + step.dy;
            const i32 nz = at[2] + step.dz;
            if (!grid.contains(nx, ny, nz)) {
                continue;
            }
            const std::size_t next = grid.index(nx, ny, nz);
            if (next >= solid.size() || solid[next] == 0) {
                continue; // a walk goes through the model, never round it
            }
            const f32 reached = d + step.length * grid.cell;
            if (reached < walk.distance[next] && reached <= cutoff) {
                walk.distance[next] = reached;
                queue.push({reached, next});
            }
        }
    }
    return walk;
}

bool ReadSolidWalk(const SolidGrid& grid, const SolidWalk& walk, const Vector3f& position, f32& distance) {
    if (!walk.reachedAny()) {
        return false;
    }
    const f32 gx = (position.x - grid.origin.x) / grid.cell - 0.5f;
    const f32 gy = (position.y - grid.origin.y) / grid.cell - 0.5f;
    const f32 gz = (position.z - grid.origin.z) / grid.cell - 0.5f;
    const i32 x0 = static_cast<i32>(std::floor(gx));
    const i32 y0 = static_cast<i32>(std::floor(gy));
    const i32 z0 = static_cast<i32>(std::floor(gz));
    if (x0 + 1 < walk.low[0] || y0 + 1 < walk.low[1] || z0 + 1 < walk.low[2] || x0 > walk.high[0] ||
        y0 > walk.high[1] || z0 > walk.high[2]) {
        return false;
    }
    const f32 fx = gx - static_cast<f32>(x0);
    const f32 fy = gy - static_cast<f32>(y0);
    const f32 fz = gz - static_cast<f32>(z0);
    f32 sum = 0.0f;
    f32 weight = 0.0f;
    for (i32 corner = 0; corner < 8; ++corner) {
        const i32 x = x0 + (corner & 1);
        const i32 y = y0 + ((corner >> 1) & 1);
        const i32 z = z0 + ((corner >> 2) & 1);
        if (!grid.contains(x, y, z)) {
            continue;
        }
        const f32 d = walk.distance[grid.index(x, y, z)];
        if (!std::isfinite(d)) {
            continue;
        }
        const f32 w = ((corner & 1) ? fx : 1.0f - fx) * (((corner >> 1) & 1) ? fy : 1.0f - fy) *
                      (((corner >> 2) & 1) ? fz : 1.0f - fz);
        sum += d * w;
        weight += w;
    }
    if (!(weight > 1e-6f)) {
        return false;
    }
    distance = sum / weight;
    return true;
}

} // namespace skinning
} // namespace wem
} // namespace models
} // namespace whiteout
