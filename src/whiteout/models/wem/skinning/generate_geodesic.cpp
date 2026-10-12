// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

/// Geodesic voxel binding (EDIT_MODE_SKIN_DESIGN.md §8.5): the grid, the shell,
/// the solid, and one Dijkstra walk per bone through it. `SolveGeodesic` in
/// `generate.h` says what is measured and why.

#include <whiteout/models/wem/skinning/generate.h>

#include <whiteout/models/wem/skinning/solid.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <utility>

#include "generate_common.h"

namespace whiteout {
namespace models {
namespace wem {
namespace skinning {

namespace {

using detail::PositionOf;

/// The grid and the solid are `solid.h`'s, shared with the soft bodies.
using Grid = SolidGrid;

/// A mesh of the run, flattened: each scope point's mesh, id and position.
struct RunPoint {
    u32 mesh = 0;
    u32 point = 0;
    Vector3f at{0, 0, 0};
};

/// One bone's walk (§8.5 step 3) and its reading at every point of the run
/// (step 4): the pairs (flat point, `d^-k`) for the points it reached.
std::vector<std::pair<u32, f32>> WalkBone(const Grid& grid, const std::vector<u8>& solid,
                                          const BoneSegments& segments, u32 bone,
                                          std::span<const RunPoint> run, f32 falloff) {
    std::vector<std::pair<u32, f32>> out;
    std::vector<SolidSeed> seeds;
    const auto seed = [&](i32 x, i32 y, i32 z, f32 d) {
        if (grid.contains(x, y, z)) {
            seeds.push_back({grid.index(x, y, z), d});
        }
    };

    // The seeds, and the bone's length for the cut-off.
    f32 length = 0.0f;
    for (const BoneSegments::Segment& segment : segments.segments) {
        if (segment.bone != bone) {
            continue;
        }
        const Vector3f along = segment.end - segment.start;
        const f32 span = along.length();
        if (segment.radius > 0.0f || span <= 0.0f) {
            // A leaf's sphere, at `distance - radius` as `NearestBone` has it,
            // so the hand is nearer than the forearm that ends on its joint.
            length = std::max(length, segment.radius * 2.0f);
            // Only the grid's own cells, which `seed` would refuse the rest of
            // anyway: the grid spans the run's meshes, the radius comes from the
            // whole skeleton, and a leaf of a big rig over a small selected mesh
            // would otherwise sweep a cube thousands of cells wide.
            const i32 widest = std::max({grid.size[0], grid.size[1], grid.size[2]});
            const i32 reach = static_cast<i32>(
                std::min(std::ceil(segment.radius / grid.cell) + 1.0f, static_cast<f32>(widest + 1)));
            const std::array<i32, 3> middle = grid.cellOf(segment.start);
            std::array<i32, 3> from{};
            std::array<i32, 3> to{};
            for (i32 axis = 0; axis < 3; ++axis) {
                from[axis] = std::max(middle[axis] - reach, 0);
                to[axis] = std::min(middle[axis] + reach, grid.size[axis] - 1);
            }
            for (i32 z = from[2]; z <= to[2]; ++z) {
                for (i32 y = from[1]; y <= to[1]; ++y) {
                    for (i32 x = from[0]; x <= to[0]; ++x) {
                        const f32 d = (grid.centre(x, y, z) - segment.start).length() -
                                      segment.radius;
                        if (d <= grid.cell) {
                            seed(x, y, z, std::max(d, 0.0f));
                        }
                    }
                }
            }
            continue;
        }
        length = std::max(length, span);
        // Half a cell apart, so no cell the segment crosses is stepped over --
        // over the part inside the grid only, for the reason the sphere above
        // is clipped: a long bone over a small grid is mostly outside it.
        f32 t0 = 0.0f;
        f32 t1 = 1.0f;
        if (!grid.clip(segment.start, segment.end, t0, t1)) {
            continue;
        }
        const u32 samples =
            static_cast<u32>(std::ceil(span * (t1 - t0) / (grid.cell * 0.5f))) + 1;
        for (u32 s = 0; s <= samples; ++s) {
            const f32 t = t0 + (t1 - t0) * (static_cast<f32>(s) / static_cast<f32>(samples));
            const Vector3f at = segment.start + along * t;
            const std::array<i32, 3> cell = grid.cellOf(at);
            seed(cell[0], cell[1], cell[2],
                 DistanceToSegment(grid.centre(cell[0], cell[1], cell[2]), segment.start,
                                   segment.end));
        }
    }
    // §8.5's cut-off, which is what bounds the cost; a bone of no length still
    // reaches a few cells, or a lone root would reach nothing at all.
    const f32 cutoff = 3.0f * std::max(length, grid.cell * 2.0f);

    const SolidWalk walk = WalkSolid(grid, solid, seeds, cutoff);
    if (!walk.reachedAny()) {
        return out;
    }

    // The reading, held at half a cell, the least the grid can tell apart.
    const f32 nearest = grid.cell * 0.5f;
    for (u32 index = 0; index < run.size(); ++index) {
        f32 read = 0.0f;
        if (!ReadSolidWalk(grid, walk, run[index].at, read)) {
            continue;
        }
        const f32 d = std::max(read, nearest);
        out.push_back({index, std::pow(d, -falloff)});
    }
    return out;
}

} // namespace

GeneratedRun SolveGeodesic(std::span<const MeshScope> meshes, const NodeTree& nodes,
                           const GenerateOptions& options, const GenerateControl& control) {
    GeneratedRun run;
    const BoneSegments segments = BuildBoneSegments(nodes, options.bones);

    // Every scope point of the run, flattened: a walk reads them all at once.
    std::vector<RunPoint> points;
    std::vector<std::vector<u32>> scopes(meshes.size());
    Vector3f low{std::numeric_limits<f32>::max(), std::numeric_limits<f32>::max(),
                 std::numeric_limits<f32>::max()};
    Vector3f high{std::numeric_limits<f32>::lowest(), std::numeric_limits<f32>::lowest(),
                  std::numeric_limits<f32>::lowest()};
    for (u32 m = 0; m < meshes.size(); ++m) {
        if (meshes[m].mesh == nullptr || meshes[m].points == nullptr) {
            continue;
        }
        const std::span<const Vector3f> positions = meshes[m].mesh->attributes.get<Vector3f>(
            geom::names::kPosition, geom::Domain::Vertex);
        // The grid holds the whole of every mesh, not only the scope: the
        // volume a distance is measured through is the model's.
        for (const Vector3f& at : positions) {
            low = {std::min(low.x, at.x), std::min(low.y, at.y), std::min(low.z, at.z)};
            high = {std::max(high.x, at.x), std::max(high.y, at.y), std::max(high.z, at.z)};
        }
        scopes[m] = detail::ScopeOrEvery(*meshes[m].points, meshes[m].scope);
        for (const u32 point : scopes[m]) {
            if (point < meshes[m].points->pointCount) {
                points.push_back({m, point, PositionOf(positions, *meshes[m].points, point)});
            }
        }
    }
    if (segments.empty() || points.empty()) {
        for (u32 m = 0; m < meshes.size(); ++m) {
            GeneratedWeights none;
            none.unreached = static_cast<u32>(scopes[m].size());
            run.meshes.push_back(std::move(none));
        }
        return run;
    }

    // ---- 1. The grid and the solid -----------------------------------------
    const Grid grid = MakeSolidGrid(low, high, options.voxelResolution);
    std::vector<const Mesh*> solidMeshes;
    for (const MeshScope& entry : meshes) {
        solidMeshes.push_back(entry.mesh);
    }
    const std::vector<u8> solid = BuildSolid(grid, solidMeshes);
    u32 done = 1;
    const u32 total = 1 + static_cast<u32>(segments.bones.size());
    if (control.progress) {
        control.progress(done, total);
    }
    if (control.stopped()) {
        run.cancelled = true;
        return run;
    }

    // ---- 2. One walk per bone ----------------------------------------------
    std::vector<std::vector<std::pair<u32, f32>>> reached(segments.bones.size());
    const f32 falloff = options.geodesicFalloff > 0.0f ? options.geodesicFalloff : 4.0f;
    if (!detail::RunJobs(static_cast<u32>(segments.bones.size()), control, done, total,
                         [&](u32 index) {
                             reached[index] =
                                 WalkBone(grid, solid, segments, segments.bones[index], points, falloff);
                         })) {
        run.cancelled = true;
        return run;
    }

    // ---- 3. Gather in bone order, and normalise ----------------------------
    std::vector<std::vector<geom::Influence>> given(points.size());
    for (std::size_t b = 0; b < reached.size(); ++b) {
        for (const auto& [index, weight] : reached[b]) {
            given[index].push_back({segments.bones[b], weight});
        }
    }
    run.meshes.resize(meshes.size());
    std::vector<std::vector<u32>> islandsSeen(meshes.size());
    for (u32 index = 0; index < points.size(); ++index) {
        const RunPoint& at = points[index];
        GeneratedWeights& out = run.meshes[at.mesh];
        std::vector<geom::Influence>& weights = given[index];
        f32 sum = 0.0f;
        for (const geom::Influence& influence : weights) {
            sum += influence.weight;
        }
        if (!(sum > 0.0f) || !std::isfinite(sum)) {
            // No walk reached it: §8.1's fallback, rigid to the nearest.
            weights.clear();
            ++out.unreached;
            const u32 bone = NearestBone(segments, at.at);
            if (bone == kInvalidNode) {
                continue;
            }
            weights.push_back({bone, 1.0f});
        } else {
            for (geom::Influence& influence : weights) {
                influence.weight /= sum;
            }
        }
        out.points.push_back(at.point);
        out.given.add(weights);
        const PointTable& table = *meshes[at.mesh].points;
        if (at.point < table.islandOf.size()) {
            islandsSeen[at.mesh].push_back(table.islandOf[at.point]);
        }
    }
    for (u32 m = 0; m < meshes.size(); ++m) {
        std::vector<u32>& seen = islandsSeen[m];
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        run.meshes[m].islands = static_cast<u32>(seen.size());
    }
    return run;
}

GenerateResult GeodesicWeights(Mesh& mesh, const NodeTree& nodes, const PointTable& points,
                               std::span<const u32> scope, const GenerateOptions& options,
                               const GenerateControl& control) {
    const MeshScope one[1]{{&mesh, &points, scope}};
    return detail::WriteOneMesh(mesh, nodes, points, SolveGeodesic(one, nodes, options, control),
                                options);
}

} // namespace skinning
} // namespace wem
} // namespace models
} // namespace whiteout
