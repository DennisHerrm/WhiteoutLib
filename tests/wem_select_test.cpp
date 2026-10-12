// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

/// The selection walkers (EDIT_MODE_MODELLING_DESIGN.md §2.5), over points.

#include <catch2/catch_test_macros.hpp>

#include <whiteout/models/wem/geometry/select.h>
#include <whiteout/models/wem/skinning/points.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

using namespace whiteout;
using namespace whiteout::models::wem;

namespace {

using geom::Domain;
using geom::EdgeId;
using geom::ElementLevel;
using geom::ElementSet;
using geom::HalfedgeId;
using geom::Topology;
using geom::VertexId;

Mesh fromFaces(const std::vector<std::vector<u32>>& corners, const std::vector<Vector3f>& positions) {
    geom::FaceSet faces;
    faces.vertexCount = static_cast<u32>(positions.size());
    for (const std::vector<u32>& face : corners) {
        faces.addFace(face);
    }
    Mesh mesh;
    mesh.setFaceSet(faces);
    const std::span<Vector3f> written = mesh.attributes.getOrCreate<Vector3f>(
        geom::names::kPosition, Domain::Vertex, geom::AttrType::F32x3);
    std::copy(positions.begin(), positions.end(), written.begin());
    mesh.attributes.getOrCreate<u32>(geom::names::kSection, Domain::Face, geom::AttrType::U32);
    mesh.sections.emplace_back();
    REQUIRE(mesh.ensureConnectivity().ok());
    return mesh;
}

/// @p w x @p h unit quads; with @p creaseAt, the column of vertices at that x is
/// held twice, once by each side, so the two halves meet only by position.
Mesh grid(u32 w, u32 h, u32 creaseAt = 0) {
    std::vector<Vector3f> positions;
    std::vector<std::vector<u32>> faces;
    std::vector<std::vector<u32>> id(w + 1, std::vector<u32>(h + 1));
    std::vector<std::vector<u32>> twin(w + 1, std::vector<u32>(h + 1));
    for (u32 x = 0; x <= w; ++x) {
        for (u32 y = 0; y <= h; ++y) {
            id[x][y] = static_cast<u32>(positions.size());
            twin[x][y] = id[x][y];
            positions.push_back(Vector3f{static_cast<f32>(x), static_cast<f32>(y), 0.0f});
            if (creaseAt != 0 && x == creaseAt) {
                twin[x][y] = static_cast<u32>(positions.size());
                positions.push_back(Vector3f{static_cast<f32>(x), static_cast<f32>(y), 0.0f});
            }
        }
    }
    for (u32 x = 0; x < w; ++x) {
        for (u32 y = 0; y < h; ++y) {
            // The cells right of the crease use its second copy.
            const auto at = [&](u32 cx, u32 cy) { return cx == creaseAt && x >= creaseAt ? twin[cx][cy] : id[cx][cy]; };
            faces.push_back({at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)});
        }
    }
    return fromFaces(faces, positions);
}

/// `grid`, each quad cut into two triangles, the cut leaning one way on the
/// cells where x + y is even and the other way on the rest.
Mesh triangles(u32 w, u32 h) {
    std::vector<Vector3f> positions;
    std::vector<std::vector<u32>> faces;
    const auto at = [&](u32 x, u32 y) { return x * (h + 1) + y; };
    for (u32 x = 0; x <= w; ++x) {
        for (u32 y = 0; y <= h; ++y) {
            positions.push_back(Vector3f{static_cast<f32>(x), static_cast<f32>(y), 0.0f});
        }
    }
    for (u32 x = 0; x < w; ++x) {
        for (u32 y = 0; y < h; ++y) {
            const u32 a = at(x, y);
            const u32 b = at(x + 1, y);
            const u32 c = at(x + 1, y + 1);
            const u32 d = at(x, y + 1);
            if ((x + y) % 2 == 0) {
                faces.push_back({a, b, c});
                faces.push_back({a, c, d});
            } else {
                faces.push_back({a, b, d});
                faces.push_back({b, c, d});
            }
        }
    }
    return fromFaces(faces, positions);
}

using Segment = std::pair<std::array<f32, 2>, std::array<f32, 2>>;

Segment segmentOf(const Mesh& mesh, u32 edge) {
    const Topology& topology = mesh.topology();
    const std::span<const Vector3f> p =
        mesh.attributes.get<const Vector3f>(geom::names::kPosition, Domain::Vertex);
    const HalfedgeId h = Topology::halfedge(EdgeId(edge), 0);
    std::array<f32, 2> a{p[topology.from(h).index()].x, p[topology.from(h).index()].y};
    std::array<f32, 2> b{p[topology.to(h).index()].x, p[topology.to(h).index()].y};
    if (b < a) {
        std::swap(a, b);
    }
    return {a, b};
}

std::set<Segment> segments(const Mesh& mesh, const std::vector<u32>& edges) {
    std::set<Segment> out;
    for (const u32 e : edges) {
        out.insert(segmentOf(mesh, e));
    }
    return out;
}

/// The (first) edge from (ax, ay) to (bx, by).
EdgeId edgeAt(const Mesh& mesh, f32 ax, f32 ay, f32 bx, f32 by) {
    const Topology& topology = mesh.topology();
    for (u32 e = 0; e < topology.edgeCount(); ++e) {
        const Segment s = segmentOf(mesh, e);
        std::array<f32, 2> a{ax, ay};
        std::array<f32, 2> b{bx, by};
        if (b < a) {
            std::swap(a, b);
        }
        if (s.first == a && s.second == b) {
            return EdgeId(e);
        }
    }
    FAIL("no such edge");
    return EdgeId();
}

std::vector<u32> verticesAt(const Mesh& mesh, std::initializer_list<std::array<f32, 2>> places) {
    const std::span<const Vector3f> p =
        mesh.attributes.get<const Vector3f>(geom::names::kPosition, Domain::Vertex);
    std::vector<u32> out;
    for (u32 v = 0; v < p.size(); ++v) {
        for (const auto& [x, y] : places) {
            if (p[v].x == x && p[v].y == y) {
                out.push_back(v);
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

/// The face whose lower-left corner is at (x, y).
u32 cellAt(const Mesh& mesh, f32 x, f32 y) {
    const Topology& topology = mesh.topology();
    const std::span<const Vector3f> p =
        mesh.attributes.get<const Vector3f>(geom::names::kPosition, Domain::Vertex);
    for (u32 f = 0; f < topology.faceCount(); ++f) {
        f32 lowX = 1e9f;
        f32 lowY = 1e9f;
        for (const VertexId v : topology.fv(geom::FaceId(f))) {
            lowX = std::min(lowX, p[v.index()].x);
            lowY = std::min(lowY, p[v.index()].y);
        }
        if (lowX == x && lowY == y) {
            return f;
        }
    }
    FAIL("no such cell");
    return 0;
}

} // namespace

TEST_CASE("wem select a loop runs straight on through points of four edges", "[wem][geometry][select]") {
    const Mesh mesh = grid(4, 4);
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    const ElementSet loop = geom::EdgeLoop(mesh, points, edgeAt(mesh, 1, 2, 2, 2));
    CHECK(segments(mesh, loop.edges) ==
          std::set<Segment>{{{0, 2}, {1, 2}}, {{1, 2}, {2, 2}}, {{2, 2}, {3, 2}}, {{3, 2}, {4, 2}}});
}

TEST_CASE("wem select a border edge walks its border loop", "[wem][geometry][select]") {
    const Mesh mesh = grid(4, 4);
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    const EdgeId border = edgeAt(mesh, 1, 0, 2, 0);
    CHECK(geom::EdgeLoop(mesh, points, border).edges.size() == 16u);
    CHECK(geom::BorderLoop(mesh, points, border).edges.size() == 16u);
    CHECK(geom::BorderLoop(mesh, points, edgeAt(mesh, 1, 1, 2, 1)).edges.empty());
}

TEST_CASE("wem select a loop across a kept-apart crease continues on the twin",
          "[wem][geometry][select]") {
    // The column at x = 2 is held twice: the two halves share no edge, only
    // points, as the weld leaves a binding crease.
    const Mesh mesh = grid(4, 3, 2);
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    REQUIRE(points.pointCount == 5u * 4u);
    const ElementSet loop = geom::EdgeLoop(mesh, points, edgeAt(mesh, 0, 1, 1, 1));
    CHECK(segments(mesh, loop.edges) ==
          std::set<Segment>{{{0, 1}, {1, 1}}, {{1, 1}, {2, 1}}, {{2, 1}, {3, 1}}, {{3, 1}, {4, 1}}});
    // Along the crease, both sides' edges come as one pair of points.
    const ElementSet along = geom::EdgeLoop(mesh, points, edgeAt(mesh, 2, 1, 2, 2));
    CHECK(segments(mesh, along.edges) ==
          std::set<Segment>{{{2, 0}, {2, 1}}, {{2, 1}, {2, 2}}, {{2, 2}, {2, 3}}});
    CHECK(along.edges.size() == 6u);
    // The crease is no border: the border loop is the outline alone.
    const ElementSet outline = geom::BorderLoop(mesh, points, edgeAt(mesh, 0, 0, 1, 0));
    CHECK(outline.edges.size() == 14u);
    CHECK(geom::BorderLoop(mesh, points, edgeAt(mesh, 2, 1, 2, 2)).edges.empty());
}

TEST_CASE("wem select a ring crosses quads and a face loop is its faces", "[wem][geometry][select]") {
    const Mesh mesh = grid(4, 4);
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    const ElementSet ring = geom::EdgeRing(mesh, points, edgeAt(mesh, 1, 2, 2, 2));
    CHECK(ring.edges.size() == 5u);
    for (const Segment& s : segments(mesh, ring.edges)) {
        CHECK(s.first[0] == 1.0f);
        CHECK(s.second[0] == 2.0f);
    }
    const ElementSet faces = geom::FaceLoop(mesh, points, edgeAt(mesh, 1, 2, 2, 2));
    CHECK(faces.faces.size() == 4u);
    CHECK(faces.edges.empty());
}

TEST_CASE("wem select the diagonals of a triangulated grid are its cuts", "[wem][geometry][select]") {
    const Mesh mesh = triangles(4, 4);
    const std::vector<u8> diagonals = geom::JoinDiagonals(mesh);
    REQUIRE(diagonals.size() == mesh.topology().edgeCount());
    u32 marked = 0;
    for (u32 e = 0; e < diagonals.size(); ++e) {
        const Segment s = segmentOf(mesh, e);
        const bool cut = s.first[0] != s.second[0] && s.first[1] != s.second[1];
        CHECK((diagonals[e] != 0) == cut);
        marked += diagonals[e];
    }
    CHECK(marked == 16u);
    // A mesh of quads has none to step over.
    CHECK(geom::JoinDiagonals(grid(4, 4)).empty());
}

TEST_CASE("wem select a loop and a ring walk the quads a triangulation cut", "[wem][geometry][select]") {
    const Mesh mesh = triangles(4, 4);
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    const std::vector<u8> diagonals = geom::JoinDiagonals(mesh);
    const EdgeId edge = edgeAt(mesh, 1, 2, 2, 2);
    // Without them the loop stops at the first point a cut reaches, and the
    // ring has no quad to cross.
    CHECK(geom::EdgeLoop(mesh, points, edge).edges.size() == 2u);
    CHECK(geom::EdgeRing(mesh, points, edge).edges.size() == 1u);
    // With them it is the grid's own, as on the quads.
    const ElementSet loop = geom::EdgeLoop(mesh, points, edge, diagonals);
    CHECK(segments(mesh, loop.edges) ==
          std::set<Segment>{{{0, 2}, {1, 2}}, {{1, 2}, {2, 2}}, {{2, 2}, {3, 2}}, {{3, 2}, {4, 2}}});
    const ElementSet ring = geom::EdgeRing(mesh, points, edge, diagonals);
    CHECK(segments(mesh, ring.edges) == std::set<Segment>{{{1, 0}, {2, 0}},
                                                          {{1, 1}, {2, 1}},
                                                          {{1, 2}, {2, 2}},
                                                          {{1, 3}, {2, 3}},
                                                          {{1, 4}, {2, 4}}});
    // The face loop is both triangles of each quad crossed.
    const ElementSet faces = geom::FaceLoop(mesh, points, edge, diagonals);
    CHECK(faces.faces.size() == 8u);
    CHECK(faces.edges.empty());
    // The border still walks its border. A cut has no ring; its loop is the
    // straight line of the cuts that carry it on, here the grid's long diagonal.
    CHECK(geom::EdgeLoop(mesh, points, edgeAt(mesh, 1, 0, 2, 0), diagonals).edges.size() == 16u);
    const EdgeId cut = edgeAt(mesh, 0, 0, 1, 1);
    CHECK(geom::EdgeRing(mesh, points, cut, diagonals).edges == std::vector<u32>{cut.value()});
    CHECK(geom::FaceLoop(mesh, points, cut, diagonals).faces.empty());
    CHECK(segments(mesh, geom::EdgeLoop(mesh, points, cut, diagonals).edges) ==
          std::set<Segment>{{{0, 0}, {1, 1}}, {{1, 1}, {2, 2}}, {{2, 2}, {3, 3}}, {{3, 3}, {4, 4}}});
}

/// A tube of @p segments quads round the z axis and @p rings high, each quad
/// two triangles, with every vertex on a ring pushed off the grid by
/// @p jitter so that no two triangles pair by shape alone.
Mesh roughTube(u32 segments, u32 rings, f32 jitter) {
    std::vector<Vector3f> positions;
    std::vector<std::vector<u32>> faces;
    for (u32 r = 0; r <= rings; ++r) {
        for (u32 s = 0; s < segments; ++s) {
            const f32 round = static_cast<f32>((r * 7 + s * 3) % 5) - 2.0f;
            const f32 along = static_cast<f32>((r + s * 5) % 3) - 1.0f;
            const f32 a = 6.2831853f * (static_cast<f32>(s) + jitter * round) / static_cast<f32>(segments);
            positions.push_back(Vector3f{std::cos(a), std::sin(a), static_cast<f32>(r) * 0.6f + jitter * along});
        }
    }
    for (u32 r = 0; r < rings; ++r) {
        for (u32 s = 0; s < segments; ++s) {
            const u32 n = (s + 1) % segments;
            const u32 a = r * segments + s;
            const u32 b = r * segments + n;
            const u32 c = (r + 1) * segments + n;
            const u32 d = (r + 1) * segments + s;
            faces.push_back({a, b, c});
            faces.push_back({a, c, d});
        }
    }
    return fromFaces(faces, positions);
}

TEST_CASE("wem select a loop carries straight on where triangles do not pair", "[wem][geometry][select]") {
    // No diagonal is marked, so nothing is walked as a quad: the rings round
    // the tube are found by going straight on over its surface.
    const Mesh mesh = roughTube(12, 6, 0.03f);
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    const std::vector<u8> none(mesh.topology().edgeCount(), 0);
    const Topology& topology = mesh.topology();
    const HalfedgeId h = topology.findHalfedge(VertexId(3 * 12 + 4), VertexId(3 * 12 + 5));
    REQUIRE(h.valid());
    const ElementSet loop = geom::EdgeLoop(mesh, points, Topology::edge(h), none);
    // All the way round ring 3, and nothing off it.
    CHECK(loop.edges.size() == 12u);
    for (const u32 e : loop.edges) {
        const HalfedgeId side = Topology::halfedge(EdgeId(e), 0);
        CHECK(topology.from(side).value() / 12 == 3u);
        CHECK(topology.to(side).value() / 12 == 3u);
    }
    // Up the tube from an edge between two rings, border to border.
    const HalfedgeId up = topology.findHalfedge(VertexId(2 * 12 + 4), VertexId(3 * 12 + 4));
    REQUIRE(up.valid());
    const ElementSet column = geom::EdgeLoop(mesh, points, Topology::edge(up), none);
    CHECK(column.edges.size() == 6u);
    for (const u32 e : column.edges) {
        const HalfedgeId side = Topology::halfedge(EdgeId(e), 0);
        CHECK(topology.from(side).value() % 12 == 4u);
        CHECK(topology.to(side).value() % 12 == 4u);
    }
    // Without a mask it is the quad rule alone, and stops where it stood.
    CHECK(geom::EdgeLoop(mesh, points, Topology::edge(h)).edges.size() == 1u);
}

TEST_CASE("wem select grow and shrink by face steps", "[wem][geometry][select]") {
    const Mesh mesh = grid(4, 4);
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    ElementSet centre;
    centre.vertices = verticesAt(mesh, {{2, 2}});
    const ElementSet grown = geom::Grow(mesh, points, centre);
    CHECK(grown.vertices.size() == 9u);
    CHECK(geom::Shrink(mesh, points, grown).vertices == centre.vertices);
    ElementSet cell;
    cell.faces = {cellAt(mesh, 1, 1)};
    CHECK(geom::Grow(mesh, points, cell).faces.size() == 9u);
}

TEST_CASE("wem select convert touching and enclosed", "[wem][geometry][select]") {
    const Mesh mesh = grid(4, 4);
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    ElementSet block;
    block.vertices = verticesAt(mesh, {{1, 1}, {2, 1}, {1, 2}, {2, 2}});
    using geom::ConvertRule;
    CHECK(geom::Convert(mesh, points, block, ElementLevel::Vertex, ElementLevel::Face,
                        ConvertRule::Touching).faces.size() == 9u);
    CHECK(geom::Convert(mesh, points, block, ElementLevel::Vertex, ElementLevel::Face,
                        ConvertRule::Enclosed).faces == std::vector<u32>{cellAt(mesh, 1, 1)});
    CHECK(geom::Convert(mesh, points, block, ElementLevel::Vertex, ElementLevel::Edge,
                        ConvertRule::Enclosed).edges.size() == 4u);
    CHECK(geom::Convert(mesh, points, block, ElementLevel::Vertex, ElementLevel::Edge,
                        ConvertRule::Touching).edges.size() == 12u);
    ElementSet face;
    face.faces = {cellAt(mesh, 1, 1)};
    CHECK(geom::Convert(mesh, points, face, ElementLevel::Face, ElementLevel::Edge,
                        ConvertRule::Touching).edges.size() == 4u);
    CHECK(geom::Convert(mesh, points, face, ElementLevel::Face, ElementLevel::Edge,
                        ConvertRule::Enclosed).edges.empty());
    CHECK(geom::Convert(mesh, points, face, ElementLevel::Face, ElementLevel::Vertex,
                        ConvertRule::Touching).vertices.size() == 4u);
    CHECK(geom::Convert(mesh, points, face, ElementLevel::Face, ElementLevel::Border,
                        ConvertRule::Touching).edges.empty());
    ElementSet corner;
    corner.vertices = verticesAt(mesh, {{0, 0}});
    CHECK(geom::Convert(mesh, points, corner, ElementLevel::Vertex, ElementLevel::Border,
                        ConvertRule::Touching).edges.size() == 16u);
}

TEST_CASE("wem select linked stops at the delimits asked for", "[wem][geometry][select]") {
    Mesh mesh = grid(4, 4);
    // A seam and a section border down x = 2.
    const std::span<u8> seam = mesh.attributes.getOrCreate<u8>(geom::names::kSeam, Domain::Edge,
                                                               geom::AttrType::Bool);
    for (u32 y = 0; y < 4; ++y) {
        seam[edgeAt(mesh, 2, static_cast<f32>(y), 2, static_cast<f32>(y + 1)).value()] = 1;
    }
    mesh.sections.emplace_back();
    const std::span<u32> sections = mesh.faceSections();
    for (u32 y = 0; y < 4; ++y) {
        sections[cellAt(mesh, 2, static_cast<f32>(y))] = 1;
        sections[cellAt(mesh, 3, static_cast<f32>(y))] = 1;
    }
    const skinning::PointTable points = skinning::BuildPointTable(mesh);
    ElementSet start;
    start.faces = {cellAt(mesh, 0, 0)};
    using geom::LinkDelimit;
    CHECK(geom::Linked(mesh, points, start, LinkDelimit::None).faces.size() == 16u);
    CHECK(geom::Linked(mesh, points, start, LinkDelimit::Seam).faces.size() == 8u);
    CHECK(geom::Linked(mesh, points, start, LinkDelimit::Section).faces.size() == 8u);
    CHECK(geom::Linked(mesh, points, start, LinkDelimit::Sharp).faces.size() == 16u);
    ElementSet vertex;
    vertex.vertices = verticesAt(mesh, {{0, 0}});
    CHECK(geom::Linked(mesh, points, vertex, LinkDelimit::Seam).vertices.size() == 15u);
}

TEST_CASE("wem select by material takes the faces of every section that draws with it",
          "[wem][geometry][select]") {
    // Three sections: columns 0-1 and column 3 draw with slot 5, column 2 with slot 2.
    Mesh mesh = grid(4, 4);
    mesh.sections[0].materialSlot = 5;
    mesh.sections.emplace_back().materialSlot = 2;
    mesh.sections.emplace_back().materialSlot = 5;
    const std::span<u32> sections = mesh.faceSections();
    for (u32 y = 0; y < 4; ++y) {
        sections[cellAt(mesh, 2, static_cast<f32>(y))] = 1;
        sections[cellAt(mesh, 3, static_cast<f32>(y))] = 2;
    }
    const u32 five[] = {5};
    const u32 two[] = {2};
    const u32 both[] = {2, 5};
    const u32 unused[] = {7};
    const ElementSet onFive = geom::FacesOfMaterials(mesh, five);
    CHECK(onFive.faces.size() == 12u);
    CHECK(std::is_sorted(onFive.faces.begin(), onFive.faces.end()));
    CHECK(onFive.vertices.empty());
    for (u32 y = 0; y < 4; ++y) {
        CHECK(std::binary_search(onFive.faces.begin(), onFive.faces.end(), cellAt(mesh, 3, static_cast<f32>(y))));
        CHECK(!std::binary_search(onFive.faces.begin(), onFive.faces.end(), cellAt(mesh, 2, static_cast<f32>(y))));
    }
    CHECK(geom::FacesOfMaterials(mesh, two).faces.size() == 4u);
    CHECK(geom::FacesOfMaterials(mesh, both).faces.size() == 16u);
    // A slot no section uses, and no slot at all.
    CHECK(geom::FacesOfMaterials(mesh, unused).empty());
    CHECK(geom::FacesOfMaterials(mesh, {}).empty());

    // A mesh of one section is all of it or none; no connectivity is needed.
    Mesh one = grid(2, 2);
    one.sections[0].materialSlot = 3;
    one.invalidateConnectivity();
    const u32 three[] = {3};
    CHECK(geom::FacesOfMaterials(one, three).faces.size() == 4u);
    CHECK(geom::FacesOfMaterials(one, five).empty());
}

TEST_CASE("wem select sharp takes the edges whose faces fold", "[wem][geometry][select]") {
    // A box folds at a right angle everywhere, so every edge is sharp above 60
    // degrees and none of them is above 100.
    const std::vector<Vector3f> corners = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                           {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    Mesh box = fromFaces({{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}},
                         corners);
    CHECK(geom::SharpEdges(box, 1.0471976f).edges.size() == 12u); // 60 degrees
    CHECK(geom::SharpEdges(box, 1.7453293f).edges.empty());       // 100 degrees
    // A flat grid folds nowhere.
    Mesh flat = grid(2, 2);
    CHECK(geom::SharpEdges(flat, 0.1745329f).edges.empty()); // 10 degrees
}

TEST_CASE("wem select linked by angle stops at a fold", "[wem][geometry][select]") {
    const std::vector<Vector3f> corners = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                           {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    Mesh box = fromFaces({{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}},
                         corners);
    using geom::LinkDelimit;
    const skinning::PointTable points = skinning::BuildPointTable(box);
    ElementSet one;
    one.faces = {0};
    // Within 45 degrees the grow cannot leave the face it started on; past 100
    // it takes the whole box.
    CHECK(geom::LinkedByAngle(box, points, one, LinkDelimit::Angle, 0.7853982f).faces.size() == 1u);
    CHECK(geom::LinkedByAngle(box, points, one, LinkDelimit::Angle, 1.7453293f).faces.size() == 6u);
    // And without the flag the angle is not read at all.
    CHECK(geom::Linked(box, points, one, LinkDelimit::Seam).faces.size() == 6u);
}
