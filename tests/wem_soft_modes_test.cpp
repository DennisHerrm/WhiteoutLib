// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Fernando Sahmkow

// Jiggle's ground (EDIT_MODE_SOFTBODY_PLAN.md S5, gate G-S3): a body's modes,
// the exact step, the loop's solved state, the push's ceiling and the limits on
// what comes out.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <whiteout/models/wem/anim/animator.h>
#include <whiteout/models/wem/physics/bake.h>
#include <whiteout/models/wem/physics/soft_fields.h>
#include <whiteout/models/wem/physics/soft_fit.h>
#include <whiteout/models/wem/physics/references.h>
#include <whiteout/models/wem/physics/soft_modes.h>
#include <whiteout/models/wem/validate.h>
#include <whiteout/models/wem/physics/soft_run.h>
#include <whiteout/models/wem/physics/soft_step.h>

#include "whiteout/models/wem/physics/soft_linalg.h"
#include "wem_soft_figures.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace whiteout;
using namespace whiteout::models::wem;
using namespace softfig;
using Catch::Approx;

namespace {

constexpr f32 kTau = 6.2831853f;

/// The tube's part above z = 1, soft throughout, on copies and @p detail
/// surface lumps: every lump moves, turns and scales.
Figure MakeJelly(u32 detail = 12) {
    Figure f = MakeTube(Skinned::Two);
    Soften(f, 1.0f, detail, true);
    return f;
}

/// A drive in which the whole skeleton is carried as one by `motion(t)`.
template <class Motion>
SoftJiggleDrive Carried(const SoftRig& rig, u32 steps, f32 dt, bool loops, Motion&& motion) {
    SoftJiggleDrive drive;
    drive.steps = steps;
    drive.dt = dt;
    drive.loops = loops;
    for (u32 i = 0; i <= steps; ++i) {
        const Matrix44f carrier = ToMatrix(motion(static_cast<f32>(loops ? i % steps : i) * dt));
        for (const SoftPoint& point : rig.points().points) {
            drive.driven.push_back(transform_point(point.at, carrier));
        }
        drive.carriers.insert(drive.carriers.end(), rig.lumpCount(), carrier);
    }
    return drive;
}

Transform Still(f32) {
    return Transform{};
}

/// Step @p step of @p result, posed.
std::vector<Vector3f> PosedAt(const Figure& f, const SoftRig& rig, const SoftJiggleDrive& drive,
                              const SoftJiggleResult& result, u32 step) {
    const std::size_t lumps = rig.lumpCount();
    const std::vector<Matrix44f> skin(f.model().nodes.size(), drive.carriers[step * lumps]);
    std::vector<Vector3f> out;
    rig.pose(skin, std::span<const Transform>(result.swing).subspan(step * lumps, lumps), out);
    return out;
}

/// The farthest any point is from where it is driven, over the clip.
f32 Farthest(const Figure& f, const SoftRig& rig, const SoftJiggleDrive& drive, const SoftJiggleResult& result,
             bool* finite = nullptr) {
    f32 farthest = 0.0f;
    const std::size_t count = rig.pointCount();
    for (u32 i = 0; i <= drive.steps; ++i) {
        const std::vector<Vector3f> posed = PosedAt(f, rig, drive, result, i);
        for (std::size_t p = 0; p < count; ++p) {
            const f32 d = (posed[p] - drive.driven[i * count + p]).length();
            if (finite != nullptr && !std::isfinite(d)) {
                *finite = false;
            }
            farthest = std::max(farthest, d);
        }
    }
    return farthest;
}

/// A hard shake with nothing of a loop in it.
Transform Shaken(f32 t, f32 size) {
    Transform out;
    out.translation = Vector3f{0.20f * std::sin(kTau * 1.3f * t) + 0.07f * std::sin(kTau * 4.1f * t),
                               0.15f * std::sin(kTau * 0.7f * t + 1.0f), 0.10f * std::sin(kTau * 2.3f * t)} *
                      size;
    out.rotation = Quaternion::from_axis_angle(Vector3f{0.6f, 0.0f, 0.8f}, 0.5f * std::sin(kTau * 0.9f * t));
    return out;
}

/// `x'' + 2 zeta omega x' + omega^2 x = 0` from (1, 0), by small steps of the
/// textbook fourth-order rule: an answer that owes nothing to the closed form.
std::array<f64, 2> Integrated(f64 omega, f64 zeta, f64 seconds) {
    const auto rate = [&](const std::array<f64, 2>& s) {
        return std::array<f64, 2>{s[1], -2.0 * zeta * omega * s[1] - omega * omega * s[0]};
    };
    std::array<f64, 2> s{1.0, 0.0};
    const int steps = 200000;
    const f64 h = seconds / steps;
    for (int i = 0; i < steps; ++i) {
        const auto k1 = rate(s);
        const auto k2 = rate({s[0] + 0.5 * h * k1[0], s[1] + 0.5 * h * k1[1]});
        const auto k3 = rate({s[0] + 0.5 * h * k2[0], s[1] + 0.5 * h * k2[1]});
        const auto k4 = rate({s[0] + h * k3[0], s[1] + h * k3[1]});
        s[0] += h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
        s[1] += h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
    }
    return s;
}

} // namespace

TEST_CASE("G-S3 the symmetric eigen solver", "[wem][soft][modes]") {
    const std::size_t n = 37;
    std::vector<f64> a(n * n);
    u32 seed = 12345u;
    const auto next = [&] {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<f64>(seed >> 8) / 16777216.0 - 0.5;
    };
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            a[i * n + j] = a[j * n + i] = next();
        }
    }
    std::vector<f64> vectors = a, values;
    REQUIRE(softdetail::SymmetricEigen(vectors, n, values));
    f64 worst = 0.0, skew = 0.0;
    for (std::size_t m = 0; m < n; ++m) {
        if (m > 0) {
            CHECK(values[m] >= values[m - 1]);
        }
        for (std::size_t i = 0; i < n; ++i) {
            f64 s = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                s += a[i * n + j] * vectors[m * n + j];
            }
            worst = std::max(worst, std::abs(s - values[m] * vectors[m * n + i]));
        }
        for (std::size_t k = 0; k <= m; ++k) {
            f64 dot = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                dot += vectors[m * n + j] * vectors[k * n + j];
            }
            skew = std::max(skew, std::abs(dot - (k == m ? 1.0 : 0.0)));
        }
    }
    CHECK(worst < 1e-11);
    CHECK(skew < 1e-12);

    // Equal eigenvalues, and a matrix that is already diagonal.
    std::vector<f64> diagonal{2, 0, 0, 0, 0, 5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
    REQUIRE(softdetail::SymmetricEigen(diagonal, 4, values));
    CHECK(values[0] == Approx(1.0));
    CHECK(values[1] == Approx(2.0));
    CHECK(values[2] == Approx(2.0));
    CHECK(values[3] == Approx(5.0));
    CHECK(std::abs(diagonal[0 * 4 + 3]) == Approx(1.0));
    CHECK(std::abs(diagonal[3 * 4 + 1]) == Approx(1.0));
}

TEST_CASE("G-S3 the eigenvectors are the matrix's own, whatever basis the solver hands back", "[wem][soft][modes]") {
    // A matrix with one eigenvalue three times over, said in two frames that
    // are a turn apart: a solver's basis of that space differs between them.
    const std::size_t n = 6;
    const std::array<f64, 6> wanted{1.0, 1.0, 1.0, 2.5, 4.0, 4.0};
    u32 seed = 2024u;
    const auto next = [&] {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<f64>(seed >> 8) / 16777216.0 - 0.5;
    };
    // An orthonormal frame, by Gram-Schmidt of random rows.
    std::vector<f64> frame(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < n; ++c) {
            frame[i * n + c] = next();
        }
        for (std::size_t k = 0; k <= i; ++k) {
            f64 dot = 0.0;
            for (std::size_t c = 0; c < n; ++c) {
                dot += frame[i * n + c] * frame[k * n + c];
            }
            for (std::size_t c = 0; c < n; ++c) {
                frame[i * n + c] = k < i ? frame[i * n + c] - dot * frame[k * n + c] : frame[i * n + c] / std::sqrt(dot);
            }
        }
    }
    const auto matrixOf = [&](const std::vector<f64>& rows) {
        std::vector<f64> a(n * n, 0.0);
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    a[i * n + j] += wanted[k] * rows[k * n + i] * rows[k * n + j];
                }
            }
        }
        return a;
    };
    // The same matrix from another basis of the equal spaces, one vector of
    // them turned about: rounding alone tells the two apart.
    std::vector<f64> mixed = frame;
    for (std::size_t c = 0; c < n; ++c) {
        const f64 a = frame[0 * n + c], b = frame[1 * n + c], d = frame[2 * n + c];
        mixed[0 * n + c] = 0.6 * a + 0.8 * b;
        mixed[1 * n + c] = -(0.8 * a - 0.6 * b);
        mixed[2 * n + c] = -d;
        mixed[3 * n + c] = -frame[3 * n + c];
        const f64 e = frame[4 * n + c], g = frame[5 * n + c];
        mixed[4 * n + c] = g;
        mixed[5 * n + c] = e;
    }
    std::vector<f64> one = matrixOf(frame), two = matrixOf(mixed), values, again;
    const std::vector<f64> matrix = one;
    REQUIRE(softdetail::SymmetricEigen(one, n, values));
    softdetail::SettleEigenBasis(one, n, values);
    REQUIRE(softdetail::SymmetricEigen(two, n, again));
    softdetail::SettleEigenBasis(two, n, again);
    f64 apart = 0.0, worst = 0.0, skew = 0.0;
    for (std::size_t m = 0; m < n; ++m) {
        CHECK(values[m] == Approx(wanted[m]).epsilon(1e-12));
        for (std::size_t c = 0; c < n; ++c) {
            apart = std::max(apart, std::abs(one[m * n + c] - two[m * n + c]));
            f64 s = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                s += matrix[c * n + j] * one[m * n + j];
            }
            worst = std::max(worst, std::abs(s - values[m] * one[m * n + c]));
        }
        for (std::size_t k = 0; k <= m; ++k) {
            f64 dot = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                dot += one[m * n + j] * one[k * n + j];
            }
            skew = std::max(skew, std::abs(dot - (k == m ? 1.0 : 0.0)));
        }
    }
    // One basis from both, still orthonormal eigenvectors.
    CHECK(apart < 1e-9);
    CHECK(worst < 1e-11);
    CHECK(skew < 1e-12);
}

TEST_CASE("G-S3 the nearest rotation of a matrix flattened along an axis", "[wem][soft][modes]") {
    const softdetail::Mat3 turn = softdetail::TurnOf({0.3, -0.7, 0.5});
    for (const f64 thin : {1.0, 1e-3, 1e-6, 1e-8, 1e-10, 3e-12}) {
        softdetail::Mat3 stretch{};
        stretch[0][0] = 2.5, stretch[1][1] = 2.5, stretch[2][2] = thin;
        const softdetail::Mat3 found = softdetail::NearestRotation(softdetail::Mul(stretch, turn));
        f64 off = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                off = std::max(off, std::abs(found[i][j] - turn[i][j]));
            }
        }
        INFO("flattened to " << thin);
        CHECK(off < 1e-6);
    }
}

TEST_CASE("G-S3 a body's modes, and what a unit of one does", "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    SoftJiggle settings;
    const SoftModes modes = MakeSoftModes(rig, volume, settings);
    REQUIRE(!modes.empty());
    CHECK(modes.count() == modes.coordinates.size());
    CHECK(modes.count() <= 9u * rig.lumpCount());
    CHECK(modes.count() >= 9u * 12u);
    CHECK(modes.stiffness.front() == Approx(1.0));
    CHECK(std::is_sorted(modes.stiffness.begin(), modes.stiffness.end()));
    CHECK(modes.rate(0, 2.2f) == Approx(2.2f));
    CHECK(modes.size == Approx(rig.points().size));

    // The same twice, to the bit.
    const SoftModes again = MakeSoftModes(rig, volume, settings);
    CHECK((modes.shapes == again.shapes));
    CHECK((modes.stiffness == again.stiffness));

    // The body lagging as one is the slowest swing: three of them, and the
    // next is stiffer, since its points do not move alike.
    CHECK(modes.stiffness[2] == Approx(1.0).epsilon(1e-6));
    CHECK(modes.stiffness[3] > 1.01);

    // What the modes say a unit moves the farthest point is what the lumps'
    // departures then do: the model's move, turn and scale are the rig's own.
    SoftJiggle still = settings;
    still.weight = 0.0f;
    still.limit = 4.0f;
    const SoftJiggleDrive drive = Carried(rig, 1, 1.0f / 60.0f, false, Still);
    const f32 reach = 1e-3f * modes.size;
    for (u32 mode = 0; mode < 24; ++mode) {
        REQUIRE(modes.span[mode] > 0.0);
        std::vector<f64> start(2 * modes.count(), 0.0);
        start[2 * mode] = static_cast<f64>(reach) / modes.span[mode];
        const SoftJiggleResult out = RunSoftJiggle(rig, modes, still, drive, SoftCeiling::None, start);
        CHECK(Farthest(f, rig, drive, out) == Approx(reach).epsilon(0.02));
    }

    // A body none of whose lumps swings has none.
    Figure stiff = MakeJelly();
    for (SoftLump& lump : stiff.model().physics.soft[0].lumps) {
        lump.swings = false;
    }
    const SoftRig stiffRig(stiff.model(), stiff.model().physics.soft[0].id);
    CHECK(MakeSoftModes(stiffRig, MakeSoftVolume(stiffRig), settings).empty());
}

TEST_CASE("G-S3 a free swing is its closed form at any step rate", "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    for (const SoftJigglePreset preset : {SoftJigglePreset::Fat, SoftJigglePreset::Rubber, SoftJigglePreset::Slime}) {
        SoftJiggle settings = SoftJigglePresetOf(preset);
        settings.weight = 0.0f;
        settings.limit = 4.0f;
        const SoftModes modes = MakeSoftModes(rig, volume, settings);
        REQUIRE(modes.count() > 4);
        // The slowest, under its damping; and a stiffer one, damped more.
        for (const u32 mode : {0u, 3u}) {
            const f64 ratio = std::sqrt(modes.stiffness[mode]);
            const f64 omega = 6.283185307179586 * settings.looseness * ratio;
            const f64 zeta = std::min(static_cast<f64>(SoftDampingOf(settings.bounce)) * ratio, 4.0);
            const f64 seconds = 0.5;
            const std::array<f64, 2> expect = Integrated(omega, zeta, seconds);
            for (const u32 rate : {64u, 128u, 512u}) {
                // Powers of two, so the step is the same number in both widths.
                const SoftJiggleDrive drive = Carried(rig, rate / 2, 1.0f / static_cast<f32>(rate), false, Still);
                std::vector<f64> start(2 * modes.count(), 0.0), end;
                start[2 * mode] = 1.0;
                RunSoftJiggle(rig, modes, settings, drive, SoftCeiling::Swing, start, &end);
                REQUIRE(end.size() == start.size());
                CHECK(end[2 * mode] == Approx(expect[0]).margin(1e-8));
                CHECK(end[2 * mode + 1] == Approx(expect[1]).margin(1e-7 * omega));
            }
        }
    }
}

TEST_CASE("G-S3 a loop ends where it began, with no pass to warm it", "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const f32 size = rig.points().size;
    // Rubber is the one a warm-up serves worst: a second on, a third of a
    // swing is still there.
    const SoftJiggle settings = SoftJigglePresetOf(SoftJigglePreset::Rubber);
    const SoftModes modes = MakeSoftModes(rig, volume, settings);
    const u32 steps = 60;
    const std::size_t lumps = rig.lumpCount();
    // Gently, and then hard enough that the ceiling and the stop both act.
    for (const f32 hard : {1.0f, 12.0f}) {
        const SoftJiggleDrive drive = Carried(rig, steps, 1.0f / 60.0f, true, [&](f32 t) {
            Transform out;
            out.translation = Vector3f{0.2f * std::sin(kTau * t), 0.1f * std::sin(kTau * 2.0f * t + 1.0f),
                                       0.05f * std::sin(kTau * 3.0f * t)} *
                              (size * hard);
            out.rotation = Quaternion::from_axis_angle(Vector3f{0.6f, 0.0f, 0.8f}, 0.4f * std::sin(kTau * t));
            return out;
        });
        const SoftJiggleResult out = RunSoftJiggle(rig, modes, settings, drive);
        CHECK(out.seam < 1e-5f);
        f32 first = 0.0f, apart = 0.0f;
        for (std::size_t k = 0; k < lumps; ++k) {
            first = std::max(first, out.swing[k].translation.length());
            apart = std::max(apart, (out.swing[steps * lumps + k].translation - out.swing[k].translation).length());
        }
        // Its first frame is mid-swing, which a clip run from rest is not.
        CHECK(first > 1e-3f * size);
        CHECK(apart < 1e-5f * size);
        if (hard > 1.0f) {
            // A push at the body's own rate, which rings up: the ceiling holds
            // what no single push would have passed. Gravity's sag is on top.
            SoftJiggle weightless = settings;
            weightless.weight = 0.0f;
            CHECK(RunSoftJiggle(rig, modes, weightless, drive).peak <= 1.0f + 1e-5f);
            CHECK(RunSoftJiggle(rig, modes, weightless, drive, SoftCeiling::None).peak > 1.5f);
        }

        // And the same bytes a second time.
        const SoftJiggleResult again = RunSoftJiggle(rig, modes, settings, drive);
        REQUIRE(again.swing.size() == out.swing.size());
        CHECK(std::memcmp(again.swing.data(), out.swing.data(), out.swing.size() * sizeof(Transform)) == 0);
    }
}

TEST_CASE("G-S3 the answer does not depend on the step rate", "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const f32 size = rig.points().size;
    const SoftJiggle settings;
    const SoftModes modes = MakeSoftModes(rig, volume, settings);
    // There and back in a second, then a second to settle.
    const auto motion = [&](f32 t) {
        const f32 s = std::sin(3.14159265f * std::min(t, 1.0f));
        const f32 ease = s * s * s * s;
        Transform out;
        out.translation = Vector3f{0.5f, 0.2f, 0.1f} * (size * ease);
        out.rotation = Quaternion::from_axis_angle(Vector3f{0, 1, 0}, 0.6f * ease);
        return out;
    };
    const SoftJiggleDrive coarse = Carried(rig, 120, 1.0f / 60.0f, false, motion);
    const SoftJiggleDrive fine = Carried(rig, 240, 1.0f / 120.0f, false, motion);
    const SoftJiggleResult a = RunSoftJiggle(rig, modes, settings, coarse);
    const SoftJiggleResult b = RunSoftJiggle(rig, modes, settings, fine);
    f32 apart = 0.0f;
    for (u32 i = 0; i <= 120; ++i) {
        const std::vector<Vector3f> one = PosedAt(f, rig, coarse, a, i);
        const std::vector<Vector3f> two = PosedAt(f, rig, fine, b, 2 * i);
        for (std::size_t p = 0; p < one.size(); ++p) {
            apart = std::max(apart, (one[p] - two[p]).length());
        }
    }
    const f32 swung = Farthest(f, rig, coarse, a);
    WARN("60 against 120 a second: " << apart / size << " of the region, on a swing of " << swung / size);
    CHECK(swung > 0.03f * size);
    CHECK(apart < 0.01f * size);
}

TEST_CASE("G-S3 every preset, shaken for ten seconds, stays inside its Limit", "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const f32 size = rig.points().size;
    for (u8 preset = 1; preset < static_cast<u8>(SoftJigglePreset::Count); ++preset) {
        const SoftJiggle settings = SoftJigglePresetOf(static_cast<SoftJigglePreset>(preset));
        const SoftModes modes = MakeSoftModes(rig, volume, settings);
        REQUIRE(!modes.empty());
        const SoftJiggleDrive drive = Carried(rig, 600, 1.0f / 60.0f, false, [&](f32 t) { return Shaken(t, size); });
        const SoftJiggleResult out = RunSoftJiggle(rig, modes, settings, drive);
        bool finite = true;
        const f32 farthest = Farthest(f, rig, drive, out, &finite);
        WARN(ToString(static_cast<SoftJigglePreset>(preset))
             << ": farthest " << farthest / size << " of the region, peak " << out.peak);
        CHECK(finite);
        CHECK(farthest <= settings.limit * size * 1.001f);
        CHECK(farthest > 0.02f * size);
        SoftJiggle weightless = settings;
        weightless.weight = 0.0f;
        CHECK(RunSoftJiggle(rig, modes, weightless, drive).peak <= 1.0f + 1e-5f);
    }
}

TEST_CASE("G-S3 nothing the swing moves goes under the floor", "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const f32 size = rig.points().size;
    SoftJiggle settings = SoftJigglePresetOf(SoftJigglePreset::WaterBalloon);
    const SoftModes modes = MakeSoftModes(rig, volume, settings);
    // Shaken sideways on a floor at its lowest point, sagging under its weight.
    SoftJiggleDrive drive = Carried(rig, 240, 1.0f / 60.0f, false, [&](f32 t) {
        Transform out;
        out.translation = Vector3f{0.2f * std::sin(kTau * 1.3f * t), 0.1f * std::sin(kTau * 2.1f * t), 0.0f} * size;
        out.rotation = Quaternion::from_axis_angle(Vector3f{0, 0, 1}, 0.5f * std::sin(kTau * 0.9f * t));
        return out;
    });
    drive.floor = rig.points().low.z;
    const std::size_t count = rig.pointCount();
    const auto lowest = [&](const SoftJiggleResult& out) {
        f32 under = 0.0f;
        for (u32 i = 0; i <= drive.steps; ++i) {
            const std::vector<Vector3f> posed = PosedAt(f, rig, drive, out, i);
            for (std::size_t p = 0; p < count; ++p) {
                under = std::max(under, std::min(drive.floor, drive.driven[i * count + p].z) - posed[p].z);
            }
        }
        return under;
    };
    CHECK(lowest(RunSoftJiggle(rig, modes, settings, drive)) < 1e-4f * size);
    settings.meetsFloor = false;
    CHECK(lowest(RunSoftJiggle(rig, modes, settings, drive)) > 2e-3f * size);
}

TEST_CASE("G-S3 a dead stop from ten regions a second", "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const f32 size = rig.points().size;
    SoftJiggle settings;
    settings.weight = 0.0f;
    const SoftModes modes = MakeSoftModes(rig, volume, settings);
    const SoftJiggleDrive drive = Carried(rig, 240, 1.0f / 60.0f, false, [&](f32 t) {
        Transform out;
        out.translation = Vector3f{10.0f * size * std::min(t, 0.5f), 0, 0};
        return out;
    });
    const SoftJiggleResult held = RunSoftJiggle(rig, modes, settings, drive, SoftCeiling::Swing);
    const SoftJiggleResult clipped = RunSoftJiggle(rig, modes, settings, drive, SoftCeiling::Clip);
    const SoftJiggleResult loose = RunSoftJiggle(rig, modes, settings, drive, SoftCeiling::None);
    WARN("dead stop: peak " << held.peak << " held, " << clipped.peak << " clipped at gain " << clipped.gain << ", "
                            << loose.peak << " with no ceiling; settles in " << held.settles << " s");
    // The ceiling reads a swing's size whatever its phase, so a damped one
    // lands a little inside.
    CHECK(held.peak <= 1.0f + 1e-5f);
    CHECK(held.peak > 0.7f);
    CHECK(loose.peak > 1.5f);
    CHECK(clipped.gain < 0.5f);
    CHECK(clipped.peak == Approx(1.0f).epsilon(1e-3));
    for (const SoftJiggleResult* out : {&held, &clipped, &loose}) {
        CHECK(Farthest(f, rig, drive, *out) <= settings.limit * size * 1.001f);
    }
    // The stop is at half a second; the body is still within a second and a half.
    CHECK(held.settles > 0.0f);
    CHECK(held.settles < 2.0f);

    // Slime settles at once, Rubber rings on.
    const auto settles = [&](SoftJigglePreset preset) {
        SoftJiggle of = SoftJigglePresetOf(preset);
        of.weight = 0.0f;
        const SoftJiggleDrive longer = Carried(rig, 600, 1.0f / 60.0f, false, [&](f32 t) {
            Transform out;
            out.translation = Vector3f{10.0f * size * std::min(t, 0.5f), 0, 0};
            return out;
        });
        return RunSoftJiggle(rig, MakeSoftModes(rig, volume, of), of, longer).settles;
    };
    const f32 slime = settles(SoftJigglePreset::Slime), rubber = settles(SoftJigglePreset::Rubber);
    WARN("settles: slime " << slime << " s, rubber " << rubber << " s");
    CHECK(slime > 0.0f);
    CHECK(rubber > 2.0f * slime);
}

TEST_CASE("G-S3 Keep volume stiffens the one direction that changes it", "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    REQUIRE(volume.encloses);
    {
        // The gradient is the cubic's linear term, fans and all: the body is
        // the tube above z = 1, open below.
        REQUIRE(!volume.fans.empty());
        std::vector<Vector3f> at, moves;
        u32 seed = 7u;
        const auto next = [&] {
            seed = seed * 1664525u + 1013904223u;
            return static_cast<f32>(seed >> 8) / 16777216.0f - 0.5f;
        };
        for (const SoftPoint& point : rig.points().points) {
            at.push_back(point.at);
            moves.push_back(Vector3f{next(), next(), next()});
        }
        const std::vector<Vector3f> gradient = SoftVolumeGradient(volume, at);
        f64 sum = 0.0;
        for (std::size_t p = 0; p < at.size(); ++p) {
            sum += static_cast<f64>(gradient[p].dot(moves[p]));
        }
        CHECK(sum == Approx(OffsetCubic(volume, at, moves)[1]).epsilon(1e-4));
    }
    // Driven by a swell about the tube's axis, as an Inflate that drives the
    // Jiggle would: the push is the one that changes the volume.
    SoftJiggleDrive drive;
    drive.steps = 120;
    drive.dt = 1.0f / 60.0f;
    for (u32 i = 0; i <= drive.steps; ++i) {
        const f32 swell = 1.0f + 0.3f * std::sin(kTau * 2.0f * static_cast<f32>(i) * drive.dt);
        for (const SoftPoint& point : rig.points().points) {
            drive.driven.push_back(Vector3f{point.at.x * swell, point.at.y * swell, point.at.z});
        }
        drive.carriers.insert(drive.carriers.end(), rig.lumpCount(), Matrix44f::identity());
    }
    const std::size_t lumps = rig.lumpCount();
    const std::vector<Matrix44f> bind(f.model().nodes.size(), Matrix44f::identity());
    const auto changes = [&](f32 keep) {
        SoftJiggle settings;
        settings.weight = 0.0f;
        settings.keepVolume = keep;
        const SoftJiggleResult out = RunSoftJiggle(rig, MakeSoftModes(rig, volume, settings), settings, drive);
        f64 most = 0.0;
        std::vector<Vector3f> posed;
        for (u32 i = 0; i <= drive.steps; ++i) {
            rig.pose(bind, std::span<const Transform>(out.swing).subspan(i * lumps, lumps), posed);
            most = std::max(most, std::abs(SoftVolumeAt(volume, posed) / static_cast<f64>(volume.rest) - 1.0));
        }
        return most;
    };
    const f64 kept = changes(1.0f), free = changes(0.0f);
    WARN("volume changed by " << free << " free, " << kept << " kept");
    CHECK(free > 0.02);
    CHECK(kept < 0.2 * free);
}

TEST_CASE("G-S3 a body rests under its weight, and a clip goes on from where the last one ended",
          "[wem][soft][modes]") {
    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const f32 size = rig.points().size;
    const SoftJiggle settings = SoftJigglePresetOf(SoftJigglePreset::WaterBalloon);
    const SoftModes modes = MakeSoftModes(rig, volume, settings);
    const std::size_t lumps = rig.lumpCount();

    // Standing still it sags, from its first frame, and does not move.
    const SoftJiggleDrive still = Carried(rig, 60, 1.0f / 60.0f, false, Still);
    const SoftJiggleResult rested = RunSoftJiggle(rig, modes, settings, still);
    f32 sag = 0.0f, moved = 0.0f;
    for (std::size_t k = 0; k < lumps; ++k) {
        sag = std::min(sag, rested.swing[k].translation.z);
        for (u32 i = 1; i <= 60; ++i) {
            moved = std::max(moved, (rested.swing[i * lumps + k].translation - rested.swing[k].translation).length());
        }
    }
    CHECK(sag < -0.01f * size);
    CHECK(moved < 1e-6f * size);

    // Two seconds at once, and as a second then a second from its end.
    const auto motion = [&](f32 t) { return Shaken(t, size); };
    const SoftJiggleDrive whole = Carried(rig, 120, 1.0f / 60.0f, false, motion);
    const SoftJiggleDrive first = Carried(rig, 60, 1.0f / 60.0f, false, motion);
    const SoftJiggleDrive second = Carried(rig, 60, 1.0f / 60.0f, false, [&](f32 t) { return Shaken(t + 1.0f, size); });
    std::vector<f64> middle, endOfWhole, endOfSecond;
    RunSoftJiggle(rig, modes, settings, whole, SoftCeiling::None, {}, &endOfWhole);
    RunSoftJiggle(rig, modes, settings, first, SoftCeiling::None, {}, &middle);
    const SoftJiggleResult onward = RunSoftJiggle(rig, modes, settings, second, SoftCeiling::None, middle, &endOfSecond);
    REQUIRE(endOfWhole.size() == endOfSecond.size());
    // A clip's ends are taken as still, so the join is not the middle of the
    // whole; what is checked is that the second begins where the first ended.
    const SoftJiggleResult before = RunSoftJiggle(rig, modes, settings, first, SoftCeiling::None);
    f32 jump = 0.0f, swung = 0.0f;
    for (std::size_t k = 0; k < lumps; ++k) {
        jump = std::max(jump, (onward.swing[k].translation - before.swing[60 * lumps + k].translation).length());
        swung = std::max(swung, before.swing[60 * lumps + k].translation.length());
    }
    CHECK(swung > 0.01f * size);
    CHECK(jump < 1e-5f * size);
}

TEST_CASE("G-S3 two layers that share no edge still move alike", "[wem][soft][modes]") {
    // A box of side 2 turned inside out round a box of side 1, as the cat
    // slime's two layers are: neighbours are found through space.
    Figure f;
    f.document.profiles = {ProfileId::Wc3Reforged};
    f.document.defaultProfile = ProfileId::Wc3Reforged;
    Model& model = f.document.models.emplace_back();
    f.bones.push_back(AddNode(model.nodes, "root", kInvalidNode, {0, 0, 0}));
    geom::MeshBuilder builder = StartMesh("layers");
    AddBox(builder, {-1, -1, -1}, 2.0f, {{0, 1.0f}}, /*insideOut=*/true);
    AddBox(builder, {-0.5f, -0.5f, -0.5f}, 1.0f, {{0, 1.0f}});
    model.meshes.push_back(builder.build().mesh);
    const u32 id = SoftenAll(f, 12, true).id;
    const SoftRig rig(f.model(), id);
    const SoftVolume volume = MakeSoftVolume(rig);
    SoftJiggle settings;
    settings.weight = 0.0f;
    settings.holdShape = 1.0f;
    settings.keepVolume = 0.0f;
    settings.limit = 4.0f;
    const SoftModes modes = MakeSoftModes(rig, volume, settings);
    REQUIRE(!modes.empty());
    // The inner layer alone is shaken, as an effect on it that drives the
    // Jiggle would: how far the outer one is carried with it.
    SoftJiggleDrive drive;
    drive.steps = 60;
    drive.dt = 1.0f / 60.0f;
    drive.loops = true;
    const auto inner = [](const Vector3f& at) { return std::abs(at.x) < 0.75f; };
    for (u32 i = 0; i <= drive.steps; ++i) {
        const f32 shake = 0.2f * std::sin(kTau * 2.0f * static_cast<f32>(i % drive.steps) * drive.dt);
        for (const SoftPoint& point : rig.points().points) {
            drive.driven.push_back(inner(point.at) ? point.at + Vector3f{shake, 0, 0} : point.at);
        }
        drive.carriers.insert(drive.carriers.end(), rig.lumpCount(), Matrix44f::identity());
    }
    const SoftJiggleResult out = RunSoftJiggle(rig, modes, settings, drive, SoftCeiling::None);
    const std::vector<Matrix44f> bind(f.model().nodes.size(), Matrix44f::identity());
    const std::size_t lumps = rig.lumpCount();
    f32 within = 0.0f, without = 0.0f, apart = 0.0f;
    std::vector<Vector3f> posed;
    for (u32 i = 0; i <= drive.steps; ++i) {
        rig.pose(bind, std::span<const Transform>(out.swing).subspan(i * lumps, lumps), posed);
        f32 in = 0.0f, far = 0.0f;
        for (std::size_t p = 0; p < posed.size(); ++p) {
            const Vector3f& at = rig.points().points[p].at;
            (inner(at) ? in : far) += (posed[p] - at).x / 8.0f;
        }
        within = std::max(within, std::abs(in));
        without = std::max(without, std::abs(far));
        apart = std::max(apart, std::abs(in - far));
    }
    WARN("two layers: the shaken one swings " << within << ", the other " << without << ", apart by " << apart);
    CHECK(within > 1e-3f);
    // Held to each other, they part by less than either moves.
    CHECK(apart < 0.5f * within);
}

TEST_CASE("G-S3 two lumps that cancel at the bind do not swing free once a pose turns them apart",
          "[wem][soft][modes]") {
    // A box every point of which rides two bones by halves: their two copies
    // can move against each other and leave every point where it is, so the
    // pair weighs nothing at the bind. The giant slime's bones do it.
    Figure f;
    f.document.profiles = {ProfileId::Wc3Reforged};
    f.document.defaultProfile = ProfileId::Wc3Reforged;
    Model& model = f.document.models.emplace_back();
    f.bones.push_back(AddNode(model.nodes, "a", kInvalidNode, {0, 0, 0}));
    f.bones.push_back(AddNode(model.nodes, "b", 0, {0, 0, 1}));
    geom::MeshBuilder builder = StartMesh("box");
    AddBox(builder, {-1, -1, 0}, 2.0f, {{0, 0.5f}, {1, 0.5f}});
    model.meshes.push_back(builder.build().mesh);
    const u32 id = SoftenAll(f, 0, true).id;
    const SoftRig rig(f.model(), id);
    REQUIRE(rig.lumpCount() == 2u);
    SoftJiggle settings;
    settings.weight = 0.0f;
    const SoftModes modes = MakeSoftModes(rig, MakeSoftVolume(rig), settings);
    REQUIRE(modes.count() == 18u);
    // Nine ways the body moves, and nine in which the two cancel: those are
    // stiff, not free.
    CHECK(modes.stiffness[8] < 100.0);
    CHECK(modes.stiffness[9] > 1000.0);

    // Shaken as one, with the second bone turned a quarter round the first:
    // each lump feels the shake along a different axis of its own.
    SoftJiggleDrive drive;
    drive.steps = 120;
    drive.dt = 1.0f / 60.0f;
    const f32 size = rig.points().size;
    const Matrix44f turned = ToMatrix(Transform{Vector3f{0, 0, 0}, Quaternion::from_axis_angle(Vector3f{1, 0, 0}, 1.5707963f)});
    for (u32 i = 0; i <= drive.steps; ++i) {
        const Vector3f shake{0, 0, 0.1f * size * std::sin(kTau * 1.5f * static_cast<f32>(i) * drive.dt)};
        for (const SoftPoint& point : rig.points().points) {
            drive.driven.push_back(point.at + shake);
        }
        for (u32 k = 0; k < rig.lumpCount(); ++k) {
            drive.carriers.push_back(rig.body().lumps[k].family() == 1 ? turned : Matrix44f::identity());
        }
    }
    const SoftJiggleResult out = RunSoftJiggle(rig, modes, settings, drive, SoftCeiling::None);
    WARN("lumps that cancel: push " << out.over << " of the limit");
    // The shake alone throws the body about a tenth of its size at most.
    CHECK(out.over < 3.0f);
}

TEST_CASE("G-S3 the presets and the damping", "[wem][soft][modes]") {
    const SoftJiggle fat = SoftJigglePresetOf(SoftJigglePreset::Fat), plain;
    CHECK(fat.looseness == plain.looseness);
    CHECK(fat.bounce == plain.bounce);
    CHECK(fat.weight == plain.weight);
    CHECK(fat.holdShape == plain.holdShape);
    CHECK(fat.keepVolume == plain.keepVolume);
    CHECK(SoftJigglePresetOf(SoftJigglePreset::Slime).preset == SoftJigglePreset::Slime);
    CHECK(SoftDampingOf(0.0f) == Approx(1.0f));
    CHECK(SoftDampingOf(1.0f) == Approx(0.06f));
    for (f32 bounce = 0.0f; bounce < 1.0f; bounce += 0.1f) {
        CHECK(SoftDampingOf(bounce + 0.1f) < SoftDampingOf(bounce));
    }
}

// ============================================================================
// Over a clip: the run, and the bake on top of it.
// ============================================================================

namespace {

SoftEffect& AddEffect(Figure& f, SoftEffectKind kind) {
    SoftBody& body = f.model().physics.soft[0];
    SoftEffect effect = DefaultSoftEffect(kind);
    effect.id = body.nextEffect++;
    effect.everywhere = true;
    body.effects.push_back(effect);
    return body.effects.back();
}

std::vector<Matrix44f> SkinAt(const Document& document, u32 clip, f32 seconds) {
    const Animator animator(document, 0);
    Mix mix;
    mix.plays.push_back(Play{clip, seconds, 1.0f, false});
    Pose pose;
    animator.evaluate(mix, pose);
    return pose.skinning;
}

SoftPoseAt PoseOf(const Document& document, u32 clip) {
    return [&document, clip](f32 at, std::vector<Matrix44f>& skin) { skin = SkinAt(document, clip, at); };
}

bool SameBytes(const std::vector<Transform>& a, const std::vector<Transform>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(Transform)) == 0;
}

} // namespace

TEST_CASE("G-S3 a Jiggle over a looping clip: it swings, and its last frame is its first", "[wem][soft][modes]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = Soften(f, 1.0f, 8, true).id;
    AddEffect(f, SoftEffectKind::Jiggle);
    const u32 clip = 0; // Bend: a second, looping.
    const u32 steps = 30;
    const f32 dt = 1.0f / 30.0f;
    const SoftRun run = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    REQUIRE(run.swings);
    REQUIRE(run.lumps > 0);
    CHECK(run.moves());
    CHECK(run.swing.seam < 1e-5f);
    const f32 size = SoftRig(f.model(), id).points().size;
    f32 apart = 0.0f, most = 0.0f;
    u32 swinging = 0;
    for (u32 k = 0; k < run.lumps; ++k) {
        apart = std::max(apart, (run.at(steps)[k].translation - run.at(0)[k].translation).length());
        f32 low = 1e30f, high = 0.0f;
        for (u32 i = 0; i <= steps; ++i) {
            low = std::min(low, run.at(i)[k].translation.length());
            high = std::max(high, run.at(i)[k].translation.length());
        }
        most = std::max(most, high);
        // Its swing changes over the clip: it is not only its sag.
        swinging += high - low > 1e-4f * size ? 1 : 0;
    }
    CHECK(apart < 1e-5f * size);
    CHECK(most > 5e-3f * size);
    CHECK(swinging * 2 > run.lumps);

    // The clip's keys are straight lines, so its bones turn round in a step.
    // Recorded twice and four times as finely the swing is the same one.
    const SoftRig rig(f.model(), id);
    const auto apartFrom = [&](u32 finer) {
        const SoftRun other =
            RunSoft(f.document, 0, id, clip, steps * finer, dt / static_cast<f32>(finer), PoseOf(f.document, clip));
        f32 far = 0.0f;
        std::vector<Vector3f> one, two;
        for (u32 i = 0; i <= steps; ++i) {
            const std::vector<Matrix44f> skin = SkinAt(f.document, clip, static_cast<f32>(i) * dt);
            rig.pose(skin, std::span<const Transform>(run.at(i), run.lumps), one);
            rig.pose(skin, std::span<const Transform>(other.at(i * finer), run.lumps), two);
            for (std::size_t p = 0; p < one.size(); ++p) {
                far = std::max(far, (one[p] - two[p]).length());
            }
        }
        return far;
    };
    const f32 twice = apartFrom(2), fourfold = apartFrom(4);
    WARN("30 a second against 60: " << twice / size << " of the region; against 120: " << fourfold / size
                                    << "; on a swing of " << most / size);
    CHECK(twice < 0.01f * size);
    CHECK(fourfold < 0.01f * size);

    // The goal alone asks for nothing; a step alone does not swing.
    const SoftGoalRun goal = RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    CHECK(goal.jiggle != 0u);
    CHECK(!goal.run.moves());
    CHECK(goal.drive.driven.size() == static_cast<std::size_t>(steps + 1) * SoftRig(f.model(), id).pointCount());

    // The same bytes again, and with the modes kept.
    SoftModesCache cache;
    CHECK(SameBytes(run.departures, RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).departures));
    CHECK(SameBytes(run.departures, SwingSoft(f.document, 0, id, goal, &cache).departures));
    CHECK(SameBytes(run.departures, SwingSoft(f.document, 0, id, goal, &cache).departures));
    CHECK(cache.entries.size() == 1u);

    // A slider that is not in the modes runs on the ones kept; one that is,
    // finds them again.
    f.model().physics.soft[0].effects[0].jiggle->looseness = 4.0f;
    CHECK(!SameBytes(run.departures, SwingSoft(f.document, 0, id, goal, &cache).departures));
    CHECK(cache.entries.size() == 1u);
    f.model().physics.soft[0].effects[0].jiggle->holdShape = 0.1f;
    SwingSoft(f.document, 0, id, goal, &cache);
    CHECK(cache.entries.size() == 1u);
    CHECK(cache.entries[0].holdShape == 0.1f);
}

TEST_CASE("G-S3 the bake keys a Jiggle, and the keys play what the run gave", "[wem][soft][modes]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = Soften(f, 1.0f, 8, true).id;
    AddEffect(f, SoftEffectKind::Jiggle);
    const u32 clip = 0;
    BakeHooks hooks;
    hooks.reset = [](const Document&, u32, std::span<const u8>) {};
    hooks.step = [](const Document&, u32, const BakeStep&, Pose&) {};
    const BakeRun recorded = SimulateClip(f.document, clip, hooks);
    INFO(recorded.error);
    REQUIRE(recorded.complete);
    const BakeReport report = BakeClip(f.document, recorded);
    REQUIRE(report.soft.size() == 1u);
    CHECK(report.soft[0].lumps > 0u);
    CHECK(report.soft[0].keys > 0u);
    REQUIRE(ClipKeysSoftLumps(f.document, clip));

    // The run on the baked clip's own pose is what the keys were made from.
    const u32 steps = recorded.steps;
    const f32 dt = f.document.clips[clip].duration / static_cast<f32>(steps);
    const SoftRun run = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    const SoftDeformer deformer(f.model());
    const f32 size = SoftRig(f.model(), id).points().size;
    f32 apart = 0.0f, swung = 0.0f;
    for (u32 i = 0; i <= steps; ++i) {
        const f32 seconds = static_cast<f32>(i) * dt;
        const std::vector<Matrix44f> skin = SkinAt(f.document, clip, seconds);
        const std::vector<Vector3f> keyed =
            deformer.positions(0, deformer.palette(skin, SoftDeparturesAt(f.document, 0, clip, seconds)));
        const std::vector<Vector3f> stepped =
            deformer.positions(0, deformer.palette(skin, std::span<const Transform>(run.at(i), run.lumps)));
        const std::vector<Vector3f> plain = deformer.positions(0, deformer.palette(skin, {}));
        for (std::size_t v = 0; v < keyed.size(); ++v) {
            apart = std::max(apart, (keyed[v] - stepped[v]).length());
            swung = std::max(swung, (stepped[v] - plain[v]).length());
        }
    }
    WARN("baked Jiggle: " << report.soft[0].lumps << " lumps, " << report.soft[0].keys << " keys, within "
                          << report.soft[0].within / size << " of the region; played " << apart / size
                          << " from the run, on a swing of " << swung / size);
    CHECK(swung > 5e-3f * size);
    CHECK(apart < 0.2f * swung);
}

TEST_CASE("G-S3 an effect that does not drive the Jiggle is left out of what throws it", "[wem][soft][modes]") {
    const u32 clip = 0;
    const u32 steps = 30;
    const f32 dt = 1.0f / 30.0f;
    const auto driven = [&](bool wobbles, bool drives) {
        Figure f = MakeTube(Skinned::Two, 19, 16);
        const u32 id = Soften(f, 1.0f, 8, true).id;
        if (wobbles) {
            SoftEffect& wobble = AddEffect(f, SoftEffectKind::Wobble);
            wobble.wobble->pattern = SoftWobblePattern::Sway;
            wobble.amount = 0.1f;
            wobble.drivesJiggle = drives;
        }
        AddEffect(f, SoftEffectKind::Jiggle);
        return RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    };
    const SoftGoalRun alone = driven(false, false), beside = driven(true, false), under = driven(true, true);
    REQUIRE(alone.drive.driven.size() == beside.drive.driven.size());
    REQUIRE(alone.drive.driven.size() == under.drive.driven.size());
    f32 left = 0.0f, thrown = 0.0f;
    for (std::size_t p = 0; p < alone.drive.driven.size(); ++p) {
        left = std::max(left, (beside.drive.driven[p] - alone.drive.driven[p]).length());
        thrown = std::max(thrown, (under.drive.driven[p] - alone.drive.driven[p]).length());
    }
    CHECK(left < 1e-5f);
    CHECK(thrown > 0.05f);
    // The Wobble is in the goal either way.
    CHECK(beside.run.moves());
    CHECK(!alone.run.moves());
}

TEST_CASE("G-S3 a Jiggle acts where its span is, by its level", "[wem][soft][modes]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = Soften(f, 1.0f, 8, true).id;
    SoftEffect& jiggle = AddEffect(f, SoftEffectKind::Jiggle);
    jiggle.everywhere = false;
    const u32 effect = jiggle.id;
    const u32 clip = 0;
    f.document.clips[clip].looping = false;
    const SoftSpan span{0.4f, 0.8f, 0.1f, 0.1f};
    SetSoftSpans(f.document, 0, clip, id, effect, std::span<const SoftSpan>(&span, 1));
    const u32 steps = 30;
    const f32 dt = 1.0f / 30.0f;
    const SoftRun run = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    REQUIRE(run.swings);
    const f32 size = SoftRig(f.model(), id).points().size;
    f32 outside = 0.0f, inside = 0.0f;
    for (u32 i = 0; i <= steps; ++i) {
        const f32 seconds = static_cast<f32>(i) * dt;
        for (u32 k = 0; k < run.lumps; ++k) {
            f32& into = seconds < 0.39f || seconds > 0.81f ? outside : inside;
            into = std::max(into, run.at(i)[k].translation.length());
        }
        if (seconds < 0.39f || seconds > 0.81f) {
            CHECK(run.acts[i] == 0);
        }
    }
    CHECK(outside == 0.0f);
    CHECK(inside > 2e-3f * size);

    // A clip none of whose steps the Jiggle acts in is not swung at all.
    SetSoftSpans(f.document, 0, clip, id, effect, {});
    const SoftRun none = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    CHECK(!none.swings);
    CHECK(!none.moves());
}

TEST_CASE("G-S3 the clip's winds and blasts push a Jiggle that asks for them", "[wem][soft][modes]") {
    // A wind: by its strength in the body's own g, its ramps, down its heading.
    BakeWorldForce wind;
    wind.kind = BakeWorldForce::Kind::Wind;
    wind.start = 0.5f;
    wind.end = 2.5f;
    wind.rampIn = 0.5f;
    wind.rampOut = 1.0f;
    wind.strength = 0.2f;
    wind.heading = 90.0f;
    wind.gusts = 0.0f;
    SoftWorld world;
    world.forces = std::span<const BakeWorldForce>(&wind, 1);
    world.size = 2.0f; // Its g is five sizes a second squared: 10.
    world.duration = 3.0f;
    CHECK(SoftWindAt(world, 0.4f).length() == 0.0f);
    CHECK(SoftWindAt(world, 0.75f).y == Approx(1.0f));
    CHECK(std::abs(SoftWindAt(world, 0.75f).x) < 1e-6f);
    CHECK(SoftWindAt(world, 1.5f).y == Approx(2.0f));
    CHECK(SoftWindAt(world, 2.0f).y == Approx(1.0f));
    CHECK(SoftWindAt(world, 2.6f).length() == 0.0f);
    wind.rise = 30.0f;
    CHECK(SoftWindAt(world, 1.5f).z == Approx(1.0f));
    CHECK(SoftWindAt(world, 1.5f).y == Approx(2.0f * std::cos(kTau / 12.0f)));
    wind.rise = 0.0f;
    // Gusts, at their own rate; in a loop, in whole cycles of it.
    wind.gusts = 0.5f;
    wind.gustSpeed = 0.7f;
    CHECK(SoftWindAt(world, 1.5f).y == Approx(2.0f * (1.0f + 0.5f * std::sin(kTau * 0.7f * 1.0f))));
    world.looping = true;
    CHECK(SoftWindAt(world, 1.5f).y == Approx(2.0f * (1.0f + 0.5f * std::sin(kTau * (2.0f / 3.0f) * 1.0f))));

    // A blast: one step, away from its centre, nothing at its radius or past it.
    BakeWorldForce blast;
    blast.kind = BakeWorldForce::Kind::Blast;
    blast.start = 1.0f;
    blast.strength = 2.0f;
    blast.radius = 1.0f;
    blast.height = 0.5f;
    world.forces = std::span<const BakeWorldForce>(&blast, 1);
    world.lower = Vector3f{-1, -1, 0};
    world.upper = Vector3f{1, 1, 4}; // Four high: it goes off at (0, 0, 2), four wide.
    const f32 dt = 1.0f / 60.0f;
    const Vector3f kicked = SoftBlastAt(world, 1.0f, dt, Vector3f{2, 0, 2});
    // Half way out: half of 10 x 2 x 0.1 a second, over the one step.
    CHECK(kicked.x == Approx(1.0f / dt));
    CHECK(std::abs(kicked.y) + std::abs(kicked.z) < 1e-4f);
    CHECK(SoftBlastAt(world, 1.0f + dt, dt, Vector3f{2, 0, 2}).length() == 0.0f);
    CHECK(SoftBlastAt(world, 1.0f - dt, dt, Vector3f{2, 0, 2}).length() == 0.0f);
    CHECK(SoftBlastAt(world, 1.0f, dt, Vector3f{4.5f, 0, 2}).length() == 0.0f);
    // The same change of speed whatever the step.
    CHECK(SoftBlastAt(world, 1.0f, 0.5f * dt, Vector3f{2, 0, 2}).x * 0.5f * dt == Approx(kicked.x * dt));
    // From a node, where the pose has it.
    blast.centreNode = 0;
    world.centres = {Vector3f{4, 0, 2}};
    CHECK(SoftBlastAt(world, 1.0f, dt, Vector3f{2, 0, 2}).x == Approx(-1.0f / dt));
    // It goes off in the step whose number holds its moment, once. Two times
    // compared gave a blast at 0.5 s of a 0.65 s clip in 39 steps to no step,
    // and one at 2.15 s of a 2.58 s clip in 78 to two.
    blast.centreNode = kInvalidNode;
    const auto fires = [&](f32 start, f32 duration, u32 count, bool loops, u32* where = nullptr) {
        blast.start = start;
        world.duration = duration;
        world.looping = loops;
        const f32 each = duration / static_cast<f32>(count);
        u32 hit = 0;
        for (u32 i = 0; i <= count; ++i) {
            if (SoftBlastAt(world, static_cast<f32>(i) * each, each, Vector3f{2, 0, 2}).length() > 0.0f) {
                ++hit;
                if (where != nullptr) {
                    *where = i;
                }
            }
        }
        return hit;
    };
    CHECK(fires(0.5f, 0.65f, 39, false) == 1u);
    CHECK(fires(2.15f, 2.58f, 78, false) == 1u);
    // In a loop the last frame is the first, which is the one a swing reads: a
    // blast in the last step stays in it, and one on the last frame is the
    // first step's.
    u32 where = 99;
    CHECK(fires(0.645f, 0.65f, 39, true, &where) == 1u);
    CHECK(where == 38u);
    CHECK(fires(0.65f, 0.65f, 39, true, &where) == 1u);
    CHECK(where == 0u);

    // Through a run: a steady wind leans the body downwind, the loop still
    // closes, and it is the Jiggle's to ask for.
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = Soften(f, 1.0f, 8, true).id;
    AddEffect(f, SoftEffectKind::Jiggle);
    const u32 clip = 0;
    const u32 steps = 30;
    const f32 step = 1.0f / 30.0f;
    const SoftRig rig(f.model(), id);
    const f32 size = rig.points().size;
    const auto meanX = [&](const SoftRun& run) {
        f64 sum = 0.0;
        std::vector<Vector3f> posed;
        for (u32 i = 0; i < steps; ++i) {
            rig.pose(SkinAt(f.document, clip, static_cast<f32>(i) * step), std::span<const Transform>(run.at(i), run.lumps),
                     posed);
            for (const Vector3f& p : posed) {
                sum += p.x;
            }
        }
        return static_cast<f32>(sum / (static_cast<f64>(steps) * static_cast<f64>(rig.pointCount())));
    };
    const SoftRun calm = RunSoft(f.document, 0, id, clip, steps, step, PoseOf(f.document, clip));
    BakeWorldForce steady;
    steady.kind = BakeWorldForce::Kind::Wind;
    steady.start = 0.0f;
    steady.end = f.document.clips[clip].duration;
    steady.strength = 0.4f;
    steady.heading = 0.0f;
    steady.gusts = 0.4f;
    steady.gustSpeed = 1.3f;
    f.document.clips[clip].physics.emplace();
    f.document.clips[clip].physics->world.push_back(steady);
    const SoftGoalRun goal = RunSoftGoal(f.document, 0, id, clip, steps, step, PoseOf(f.document, clip));
    CHECK(goal.drive.pushed.size() == static_cast<std::size_t>(steps + 1) * rig.pointCount());
    const SoftRun windy = RunSoft(f.document, 0, id, clip, steps, step, PoseOf(f.document, clip));
    REQUIRE(windy.swings);
    CHECK(windy.swing.seam < 1e-5f);
    CHECK(meanX(windy) - meanX(calm) > 2e-3f * size);
    WARN("a wind of 0.4 g leans the tube " << (meanX(windy) - meanX(calm)) / size << " of its size downwind");
    // Not asked for, the wind is not felt: the run is the calm one's, to the byte.
    f.model().physics.soft[0].effects[0].jiggle->worldForces = false;
    CHECK(RunSoftGoal(f.document, 0, id, clip, steps, step, PoseOf(f.document, clip)).drive.pushed.empty());
    CHECK(SameBytes(calm.departures, RunSoft(f.document, 0, id, clip, steps, step, PoseOf(f.document, clip)).departures));
    f.model().physics.soft[0].effects[0].jiggle->worldForces = true;

    // A blast in a clip played once throws the body away from where it goes off.
    f.document.clips[clip].looping = false;
    f.document.clips[clip].physics->world.clear();
    // Under no ceiling, so what a push does is the push's alone.
    f.model().physics.soft[0].effects[0].jiggle->limit = 1.0f;
    const SoftRun quiet = RunSoft(f.document, 0, id, clip, steps, step, PoseOf(f.document, clip));
    BakeWorldForce bang;
    bang.kind = BakeWorldForce::Kind::Blast;
    bang.start = 0.2f;
    bang.strength = 3.0f;
    bang.radius = 2.0f;
    bang.height = 0.5f;
    f.document.clips[clip].physics->world.push_back(bang);
    const SoftRun thrown = RunSoft(f.document, 0, id, clip, steps, step, PoseOf(f.document, clip));
    // Before it goes off the two runs are one; after it they are not.
    f32 before = 0.0f, after = 0.0f;
    for (u32 k = 0; k < thrown.lumps; ++k) {
        before = std::max(before, (thrown.at(5)[k].translation - quiet.at(5)[k].translation).length());
        after = std::max(after, (thrown.at(12)[k].translation - quiet.at(12)[k].translation).length());
    }
    CHECK(before < 1e-6f * size);
    CHECK(after > 1e-3f * size);
}

TEST_CASE("G-S2 a Wobble on Random is the body's own ways of swinging, in whole cycles a loop", "[wem][soft][modes]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = Soften(f, 1.0f, 8, true).id;
    const SoftRig rig(f.model(), id);
    const f32 size = rig.points().size;
    const SoftStepper stepper(f.document, 0, id);
    const SoftModes modes = MakeSoftModes(rig, stepper.volume(), SoftJiggle{});
    REQUIRE(modes.count() >= kSoftRandomModes);

    // The swing itself: none where the amount is none, and never farther than
    // `kSoftRandomFar` amounts times the root of what is mixed.
    const u32 steps = 60;
    const f32 dt = 1.0f / 60.0f;
    std::vector<f32> amount(steps + 1, 0.06f * size);
    amount[10] = 0.0f;
    const std::vector<Transform> swing = SoftRandomWobble(rig, modes, 1.0f, 7, amount, steps, dt, 1.0f, true);
    REQUIRE(swing.size() == static_cast<std::size_t>(steps + 1) * rig.lumpCount());
    const std::vector<Matrix44f> rest(f.model().nodes.size(), Matrix44f::identity());
    std::vector<Vector3f> still, posed;
    rig.carry(rest, still);
    const auto farthest = [&](const std::vector<Transform>& from, u32 step) {
        rig.pose(rest, std::span<const Transform>(from).subspan(step * rig.lumpCount(), rig.lumpCount()), posed);
        f32 far = 0.0f;
        for (std::size_t p = 0; p < posed.size(); ++p) {
            far = std::max(far, (posed[p] - still[p]).length());
        }
        return far;
    };
    f32 most = 0.0f;
    for (u32 i = 0; i <= steps; ++i) {
        most = std::max(most, farthest(swing, i));
    }
    CHECK(farthest(swing, 10) < 1e-6f * size);
    CHECK(most > 0.01f * size);
    CHECK(most < 0.06f * size * kSoftRandomFar * std::sqrt(static_cast<f32>(kSoftRandomModes)) * 1.05f);
    // A loop's last frame is its first.
    for (u32 k = 0; k < rig.lumpCount(); ++k) {
        CHECK((swing[k].translation - swing[steps * rig.lumpCount() + k].translation).length() < 1e-5f * size);
    }
    // And it comes round to it: at a speed that is no whole number of swings
    // a loop, the step over the seam is as small as any other.
    {
        const u32 fine = 240;
        const std::vector<f32> level(fine + 1, 0.06f * size);
        const std::vector<Transform> round =
            SoftRandomWobble(rig, modes, 1.3f, 7, level, fine, 1.0f / static_cast<f32>(fine), 1.0f, true);
        f32 inside = 0.0f, seam = 0.0f;
        for (u32 k = 0; k < rig.lumpCount(); ++k) {
            for (u32 i = 0; i + 2 <= fine; ++i) {
                inside = std::max(inside, (round[(i + 1) * rig.lumpCount() + k].translation -
                                           round[i * rig.lumpCount() + k].translation)
                                              .length());
            }
            seam = std::max(seam, (round[fine * rig.lumpCount() + k].translation -
                                   round[(fine - 1) * rig.lumpCount() + k].translation)
                                      .length());
        }
        CHECK(inside > 0.0f);
        CHECK(seam < 1.5f * inside);
    }
    // Another seed is another mix; no amount is no swing at all.
    const std::vector<Transform> other = SoftRandomWobble(rig, modes, 1.0f, 8, amount, steps, dt, 1.0f, true);
    f32 apart = 0.0f;
    for (std::size_t s = 0; s < swing.size(); ++s) {
        apart = std::max(apart, (swing[s].translation - other[s].translation).length());
    }
    CHECK(apart > 1e-3f * size);
    const std::vector<f32> none(steps + 1, 0.0f);
    for (const Transform& each : SoftRandomWobble(rig, modes, 1.0f, 7, none, steps, dt, 1.0f, true)) {
        CHECK(each.translation.length() == 0.0f);
    }
    // Played once it swings at its own rate: not the loop's whole cycles.
    const std::vector<Transform> once = SoftRandomWobble(rig, modes, 1.3f, 7, amount, steps, dt, 1.0f, false);
    f32 seam = 0.0f;
    for (u32 k = 0; k < rig.lumpCount(); ++k) {
        seam = std::max(seam, (once[k].translation - once[steps * rig.lumpCount() + k].translation).length());
    }
    CHECK(seam > 1e-4f * size);

    // Through a run, with no Jiggle: the body is found over the whole clip,
    // moves, and ends where it began.
    SoftEffect& wobble = AddEffect(f, SoftEffectKind::Wobble);
    wobble.wobble->pattern = SoftWobblePattern::Random;
    wobble.amount = 0.06f;
    CHECK(SoftRunsWhole(f.model().physics.soft[0]));
    const u32 clip = 0;
    const SoftGoalRun goal = RunSoftGoal(f.document, 0, id, clip, 30, 1.0f / 30.0f, PoseOf(f.document, clip));
    CHECK(goal.jiggle == 0u);
    REQUIRE(goal.random.size() == 1u);
    CHECK(goal.swings());
    CHECK(!goal.run.moves());
    const SoftRun run = RunSoft(f.document, 0, id, clip, 30, 1.0f / 30.0f, PoseOf(f.document, clip));
    CHECK(run.swings);
    CHECK(run.moves());
    f32 ends = 0.0f, moves = 0.0f;
    for (u32 k = 0; k < run.lumps; ++k) {
        ends = std::max(ends, (run.at(30)[k].translation - run.at(0)[k].translation).length());
        moves = std::max(moves, (run.at(8)[k].translation - run.at(0)[k].translation).length());
    }
    CHECK(ends < 1e-5f * size);
    CHECK(moves > 1e-3f * size);
    // Off, or on another pattern, nothing is found over the clip.
    f.model().physics.soft[0].effects[0].enabled = false;
    CHECK(!SoftRunsWhole(f.model().physics.soft[0]));
    CHECK(!RunSoft(f.document, 0, id, clip, 30, 1.0f / 30.0f, PoseOf(f.document, clip)).swings);
    f.model().physics.soft[0].effects[0].enabled = true;
    f.model().physics.soft[0].effects[0].wobble->pattern = SoftWobblePattern::Jelly;
    CHECK(!SoftRunsWhole(f.model().physics.soft[0]));

    // Beside a Jiggle it is added to the swing, and the loop still closes.
    f.model().physics.soft[0].effects[0].wobble->pattern = SoftWobblePattern::Random;
    AddEffect(f, SoftEffectKind::Jiggle);
    const SoftRun both = RunSoft(f.document, 0, id, clip, 30, 1.0f / 30.0f, PoseOf(f.document, clip));
    CHECK(both.swings);
    CHECK(both.swing.seam < 1e-5f);
    f.model().physics.soft[0].effects[0].enabled = false;
    const SoftRun jiggled = RunSoft(f.document, 0, id, clip, 30, 1.0f / 30.0f, PoseOf(f.document, clip));
    f32 added = 0.0f, closes = 0.0f;
    for (u32 k = 0; k < both.lumps; ++k) {
        added = std::max(added, (both.at(8)[k].translation - jiggled.at(8)[k].translation).length());
        closes = std::max(closes, (both.at(30)[k].translation - both.at(0)[k].translation).length());
    }
    CHECK(added > 1e-3f * size);
    CHECK(closes < 1e-5f * size);
}

TEST_CASE("G-S2 a Wobble on Random acts where its span is, by its level", "[wem][soft][modes]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = Soften(f, 1.0f, 8, true).id;
    SoftEffect& wobble = AddEffect(f, SoftEffectKind::Wobble);
    wobble.wobble->pattern = SoftWobblePattern::Random;
    wobble.amount = 0.06f;
    wobble.everywhere = false;
    const u32 effect = wobble.id;
    const u32 clip = 0;
    f.document.clips[clip].looping = false;
    const SoftSpan span{0.4f, 0.8f, 0.1f, 0.1f};
    SetSoftSpans(f.document, 0, clip, id, effect, std::span<const SoftSpan>(&span, 1));
    const u32 steps = 30;
    const f32 dt = 1.0f / 30.0f;
    const SoftRun run = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    REQUIRE(run.swings);
    const f32 size = SoftRig(f.model(), id).points().size;
    f32 outside = 0.0f, inside = 0.0f;
    for (u32 i = 0; i <= steps; ++i) {
        const f32 seconds = static_cast<f32>(i) * dt;
        for (u32 k = 0; k < run.lumps; ++k) {
            f32& into = seconds < 0.39f || seconds > 0.81f ? outside : inside;
            into = std::max(into, run.at(i)[k].translation.length());
        }
        if (seconds < 0.39f || seconds > 0.81f) {
            CHECK(run.acts[i] == 0);
        }
    }
    CHECK(outside == 0.0f);
    CHECK(inside > 1e-3f * size);
    // Placed nowhere, nothing is run over the clip.
    SetSoftSpans(f.document, 0, clip, id, effect, {});
    const SoftGoalRun goal = RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    CHECK(goal.random.empty());
    CHECK(!RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).swings);
}

TEST_CASE("G-S3 a kick is an instant change of speed round its place, at each of its marks", "[wem][soft][modes]") {
    // All of it at its place, none half the region away; the same change of speed whatever the step.
    CHECK(SoftKickShare(0.0f, 4.0f) == 1.0f);
    CHECK(SoftKickShare(1.0f, 4.0f) == Approx(0.25f));
    CHECK(SoftKickShare(kSoftKickReach * 4.0f, 4.0f) == 0.0f);
    CHECK(SoftKickShare(9.0f, 4.0f) == 0.0f);
    const f32 dt = 1.0f / 30.0f;
    // 2 g on a body two in size: 5 x 2 x 2 x 0.1 a second, over the one step.
    CHECK(SoftKickPush(2.0f, 2.0f, dt) == Approx(2.0f / dt));
    CHECK(SoftKickPush(2.0f, 2.0f, 0.5f * dt) * 0.5f * dt == Approx(SoftKickPush(2.0f, 2.0f, dt) * dt));

    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = Soften(f, 1.0f, 8, true).id;
    SoftEffect& jiggle = AddEffect(f, SoftEffectKind::Jiggle);
    // Under no ceiling, so what a push does is the push's alone.
    jiggle.jiggle->limit = 1.0f;
    const u32 clip = 0;
    f.document.clips[clip].looping = false;
    const u32 steps = 30;
    const SoftRig rig(f.model(), id);
    const f32 size = rig.points().size;
    const std::size_t count = rig.pointCount();
    const SoftRun quiet = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    CHECK(RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).drive.pushed.empty());

    // A kick near the tube's top, along +X, 3 g. A place is a face and weights on it.
    const u32 face = f.model().meshes[0].faceCount() - 1;
    SoftPlace place;
    place.mesh = 0;
    place.face = face;
    place.weights = Vector3f{1, 0, 0};
    const SoftPlaceAt at = ResolveSoftPlace(f.model(), rig, place);
    REQUIRE(at.valid);
    {
        SoftBody& body = f.model().physics.soft[0];
        SoftJiggle& settings = *body.effects[0].jiggle;
        settings.kickIds = {body.nextEffect++};
        settings.kickMesh = {0};
        settings.kickFace = {face};
        settings.kickWeights = {Vector3f{1, 0, 0}};
        settings.kickDirection = {Vector3f{2, 0, 0}}; // Its length is not read: a direction.
        settings.kickStrength = {3.0f};
    }
    const u32 kick = f.model().physics.soft[0].effects[0].jiggle->kickIds[0];
    const std::size_t issues = Validate(f.document, ValidateLevel::Structural).size();
    // With no mark in the clip it does nothing: the run is the quiet one's, to the byte.
    CHECK(RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).drive.pushed.empty());
    CHECK(SameBytes(quiet.departures, RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).departures));

    // A mark is the start of a span of the kick's own.
    const SoftSpan marks[] = {SoftSpan{0.2f, 0.25f, 0.0f, 0.0f}, SoftSpan{0.7f, 0.75f, 0.0f, 0.0f}};
    SetSoftSpans(f.document, 0, clip, id, kick, std::span<const SoftSpan>(marks, 1));
    // Its channel names a kick of the body's Jiggle: nothing the document was not before.
    CHECK(Validate(f.document, ValidateLevel::Structural).size() == issues);
    const SoftGoalRun goal = RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    REQUIRE(goal.drive.pushed.size() == static_cast<std::size_t>(steps + 1) * count);
    // On the one step that holds the mark, along its direction, by each point's share.
    u32 landed = 0;
    const f32 whole = SoftKickPush(3.0f, size, dt);
    for (u32 i = 0; i <= steps; ++i) {
        f32 most = 0.0f;
        for (std::size_t p = 0; p < count; ++p) {
            const Vector3f& push = goal.drive.pushed[i * count + p];
            most = std::max(most, push.length());
            if (push.length() > 0.0f) {
                const f32 share = SoftKickShare((rig.points().points[p].at - at.at).length(), size);
                CHECK(push.x == Approx(whole * share).epsilon(1e-4));
                CHECK(std::abs(push.y) + std::abs(push.z) < 1e-4f * whole);
            }
        }
        if (most > 0.0f) {
            ++landed;
            CHECK(static_cast<f32>(i) * dt <= 0.2f + 1e-5f);
            CHECK(static_cast<f32>(i + 1) * dt > 0.2f - 1e-5f);
            CHECK(most == Approx(whole).epsilon(0.02));
        }
    }
    CHECK(landed == 1u);
    // The body's far end is out of its reach.
    {
        bool none = false;
        for (std::size_t p = 0; p < count; ++p) {
            none = none || (rig.points().points[p].at - at.at).length() > kSoftKickReach * size;
        }
        CHECK(none);
    }
    // Before it lands the run is the quiet one's; after, the body is thrown its way.
    const SoftRun kicked = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    const auto meanX = [&](const SoftRun& run, u32 step) {
        std::vector<Vector3f> posed;
        rig.pose(SkinAt(f.document, clip, static_cast<f32>(step) * dt), std::span<const Transform>(run.at(step), run.lumps),
                 posed);
        f64 sum = 0.0;
        for (const Vector3f& p : posed) {
            sum += p.x;
        }
        return static_cast<f32>(sum / static_cast<f64>(posed.size()));
    };
    f32 before = 0.0f;
    for (u32 k = 0; k < kicked.lumps; ++k) {
        before = std::max(before, (kicked.at(5)[k].translation - quiet.at(5)[k].translation).length());
    }
    CHECK(before < 1e-6f * size);
    CHECK(meanX(kicked, 10) - meanX(quiet, 10) > 1e-3f * size);
    WARN("a kick of 3 g throws the tube " << (meanX(kicked, 10) - meanX(quiet, 10)) / size << " of its size");
    // A stronger one, farther.
    f.model().physics.soft[0].effects[0].jiggle->kickStrength = {6.0f};
    const SoftRun harder = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    CHECK(meanX(harder, 10) - meanX(quiet, 10) > 1.5f * (meanX(kicked, 10) - meanX(quiet, 10)));
    // With no direction said it goes into the body where it lands: against the surface's way out there.
    f.model().physics.soft[0].effects[0].jiggle->kickDirection = {Vector3f{0, 0, 0}};
    {
        const SoftGoalRun into = RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
        REQUIRE(into.drive.pushed.size() == static_cast<std::size_t>(steps + 1) * count);
        u32 pushed = 0;
        for (const Vector3f& push : into.drive.pushed) {
            if (push.length() > 0.0f) {
                ++pushed;
                CHECK(cross(push, at.normal).length() < 1e-3f * push.length());
                CHECK(push.dot(at.normal) < 0.0f);
            }
        }
        CHECK(pushed > 0u);
    }
    f.model().physics.soft[0].effects[0].jiggle->kickDirection = {Vector3f{2, 0, 0}};
    // Two marks, two steps.
    SetSoftSpans(f.document, 0, clip, id, kick, std::span<const SoftSpan>(marks, 2));
    {
        const SoftGoalRun twice = RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
        u32 hits = 0;
        for (u32 i = 0; i <= steps; ++i) {
            f32 most = 0.0f;
            for (std::size_t p = 0; p < count; ++p) {
                most = std::max(most, twice.drive.pushed[i * count + p].length());
            }
            hits += most > 0.0f ? 1 : 0;
        }
        CHECK(hits == 2u);
    }
    // In a loop the kick is one more push at a time of it: the loop still closes.
    f.document.clips[clip].looping = true;
    const SoftRun looped = RunSoft(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    REQUIRE(looped.swings);
    CHECK(looped.swing.seam < 1e-5f);
    // It is its Jiggle's: with the Jiggle off, nothing is thrown.
    f.model().physics.soft[0].effects[0].enabled = false;
    CHECK(RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).drive.pushed.empty());
}

TEST_CASE("G-S3 nothing the swing moves is left in a collider that what drives the body had it out of",
          "[wem][soft][modes]") {
    // The shapes: a sphere, a capsule and a plane, each with its distance.
    {
        Vector3f way{0, 0, 0};
        const SoftCollider ball{false, Vector3f{1, 2, 3}, Vector3f{1, 2, 3}, 0.5f};
        CHECK(SoftColliderDepth(ball, Vector3f{1.3f, 2, 3}, way) == Approx(0.2f));
        CHECK(way.x == Approx(1.0f));
        CHECK(SoftColliderDepth(ball, Vector3f{1, 2, 5}, way) == Approx(-1.5f));
        CHECK(way.z == Approx(1.0f));
        // At its very middle any way out is as short.
        CHECK(SoftColliderDepth(ball, Vector3f{1, 2, 3}, way) == Approx(0.5f));
        CHECK(way.length() == Approx(1.0f));
        const SoftCollider rod{false, Vector3f{0, 0, 0}, Vector3f{0, 0, 2}, 0.25f};
        CHECK(SoftColliderDepth(rod, Vector3f{0.1f, 0, 1}, way) == Approx(0.15f));
        CHECK(way.x == Approx(1.0f));
        CHECK(std::abs(way.z) < 1e-6f);
        // Past an end it is that end's ball.
        CHECK(SoftColliderDepth(rod, Vector3f{0, 0, 2.1f}, way) == Approx(0.15f));
        CHECK(way.z == Approx(1.0f));
        const SoftCollider wall{true, Vector3f{0, 0, 1}, Vector3f{0, 0, 1}, 0.0f};
        CHECK(SoftColliderDepth(wall, Vector3f{5, 5, 0.6f}, way) == Approx(0.4f));
        CHECK(SoftColliderDepth(wall, Vector3f{5, 5, 1.5f}, way) == Approx(-0.5f));
        CHECK(way.z == 1.0f);
    }
    // A body's colliders, where a pose has them: a capsule on a bone, a sphere and a plane on the model.
    {
        Figure f = MakeJelly();
        Model& model = f.model();
        ClothCollider capsule;
        capsule.id = model.physics.allocateId();
        capsule.node = f.bones[1]; // Its pivot is (0, 0, 2).
        capsule.transform.data[3][0] = 3.0f;
        capsule.radius = 0.25f;
        capsule.length = 2.0f;
        ClothCollider sphere;
        sphere.id = model.physics.allocateId();
        sphere.transform.data[3][1] = 4.0f;
        sphere.radius = 0.5f;
        ClothCollider plane;
        plane.id = model.physics.allocateId();
        plane.kind = ClothColliderKind::Plane;
        plane.transform.data[3][2] = -1.0f;
        model.physics.colliders = {capsule, sphere, plane};
        SoftBody& body = model.physics.soft[0];
        std::vector<Matrix44f> skin(model.nodes.size(), Matrix44f::identity());
        CHECK(SoftCollidersAt(model, body, skin).empty());
        // In the order it lists them; one the model no longer has is left out.
        body.colliders = {plane.id, 9999u, capsule.id, sphere.id};
        std::vector<SoftCollider> at = SoftCollidersAt(model, body, skin);
        REQUIRE(at.size() == 3u);
        CHECK(at[0].plane);
        CHECK(at[0].a.z == Approx(-1.0f));
        CHECK(at[0].b.z == Approx(1.0f));
        CHECK(!at[1].plane);
        CHECK((at[1].a - Vector3f{3, 0, 1}).length() < 1e-5f);
        CHECK((at[1].b - Vector3f{3, 0, 3}).length() < 1e-5f);
        CHECK(at[1].radius == Approx(0.25f));
        CHECK((at[2].a - Vector3f{0, 4, 0}).length() < 1e-5f);
        CHECK((at[2].b - at[2].a).length() < 1e-6f);
        // Its bone moved, the capsule goes with it; the model's own stay.
        skin[f.bones[1]].data[3][1] = 5.0f;
        at = SoftCollidersAt(model, body, skin);
        CHECK((at[1].a - Vector3f{3, 5, 1}).length() < 1e-5f);
        CHECK((at[2].a - Vector3f{0, 4, 0}).length() < 1e-5f);
    }

    const Figure f = MakeJelly();
    const SoftRig rig(f.model(), f.model().physics.soft[0].id);
    const SoftVolume volume = MakeSoftVolume(rig);
    const f32 size = rig.points().size;
    const SoftJiggle settings = SoftJigglePresetOf(SoftJigglePreset::WaterBalloon);
    const SoftModes modes = MakeSoftModes(rig, volume, settings);
    const u32 steps = 240;
    const f32 dt = 1.0f / 60.0f;
    const auto shaken = [&](f32 t) {
        Transform out;
        out.translation = Vector3f{0.2f * std::sin(kTau * 1.3f * t), 0.1f * std::sin(kTau * 2.1f * t), 0.0f} * size;
        out.rotation = Quaternion::from_axis_angle(Vector3f{0, 0, 1}, 0.5f * std::sin(kTau * 0.9f * t));
        return out;
    };
    const std::size_t count = rig.pointCount();
    // How much deeper in a collider the swing leaves any point than what drives the body had it.
    const auto deeper = [&](const SoftJiggleDrive& drive, const std::vector<SoftCollider>& shapes, u32 each) {
        const SoftJiggleResult out = RunSoftJiggle(rig, modes, settings, drive);
        f32 most = 0.0f;
        for (u32 i = 0; i <= drive.steps; ++i) {
            const std::vector<Vector3f> posed = PosedAt(f, rig, drive, out, i);
            for (std::size_t p = 0; p < count; ++p) {
                for (u32 c = 0; c < each; ++c) {
                    Vector3f way{0, 0, 0};
                    const SoftCollider& shape = shapes[i * each + c];
                    const f32 was = std::max(SoftColliderDepth(shape, drive.driven[i * count + p], way), 0.0f);
                    most = std::max(most, SoftColliderDepth(shape, posed[p], way) - was);
                }
            }
        }
        return most;
    };
    // A wall beside the body and a ball beside it, each carried with it as
    // one on its bones would be, a hair off its surface.
    for (const bool plane : {true, false}) {
        SoftJiggleDrive drive = Carried(rig, steps, dt, false, shaken);
        std::vector<SoftCollider> shapes;
        for (u32 i = 0; i <= steps; ++i) {
            const Matrix44f carrier = ToMatrix(shaken(static_cast<f32>(i) * dt));
            SoftCollider shape;
            if (plane) {
                // Everything past x = 1.02 is inside it.
                shape.plane = true;
                shape.a = transform_point(Vector3f{1.02f, 0, 0}, carrier);
                shape.b = transform_normal(Vector3f{-1, 0, 0}, carrier);
            } else {
                shape.a = shape.b = transform_point(Vector3f{1.52f, 0, 4.0f}, carrier);
                shape.radius = 0.5f;
            }
            shapes.push_back(shape);
        }
        const f32 free = deeper(drive, shapes, 1);
        drive.colliderCount = 1;
        drive.colliders = shapes;
        const f32 kept = deeper(drive, shapes, 1);
        INFO((plane ? "plane" : "ball") << ": free " << free / size << " kept " << kept / size);
        CHECK(free > 2e-3f * size);
        CHECK(kept < (plane ? 1e-4f : 0.1f * free / size) * size);
        // One step short of what the frames need, it is not read: no half a clip of them.
        drive.colliders.pop_back();
        CHECK(deeper(drive, shapes, 1) == Approx(free));
    }
    // What the animation itself has inside a collider is left there: with no swing, nothing is moved.
    {
        SoftJiggleDrive drive = Carried(rig, 60, dt, false, shaken);
        drive.level.assign(61, 0.0f);
        for (u32 i = 0; i <= 60; ++i) {
            const Matrix44f carrier = ToMatrix(shaken(static_cast<f32>(i) * dt));
            SoftCollider shape;
            // A ball that reaches a tenth into the body's side.
            shape.a = shape.b = transform_point(Vector3f{1.4f, 0, 4.0f}, carrier);
            shape.radius = 0.5f;
            drive.colliders.push_back(shape);
        }
        drive.colliderCount = 1;
        const SoftJiggleResult out = RunSoftJiggle(rig, modes, settings, drive);
        CHECK(Farthest(f, rig, drive, out) < 1e-6f * size);
    }
    // In a loop the stop has no memory, so the loop still closes on what is posed.
    {
        SoftJiggleDrive drive = Carried(rig, 120, dt, true, [&](f32 t) {
            Transform out;
            out.translation = Vector3f{0.2f * std::sin(kTau * 1.0f * t), 0.0f, 0.0f} * size;
            return out;
        });
        for (u32 i = 0; i <= 120; ++i) {
            SoftCollider shape;
            shape.plane = true;
            // Carried as the body is: by how far its first point has gone.
            shape.a = (drive.driven[i * count] - rig.points().points[0].at) + Vector3f{1.02f, 0, 0};
            shape.b = Vector3f{-1, 0, 0};
            drive.colliders.push_back(shape);
        }
        drive.colliderCount = 1;
        const SoftJiggleResult out = RunSoftJiggle(rig, modes, settings, drive);
        CHECK(out.seam < 1e-5f);
        const std::vector<Vector3f> first = PosedAt(f, rig, drive, out, 0), last = PosedAt(f, rig, drive, out, 120);
        f32 apart = 0.0f;
        for (std::size_t p = 0; p < count; ++p) {
            apart = std::max(apart, (first[p] - last[p]).length());
        }
        CHECK(apart < 1e-4f * size);
    }
}

TEST_CASE("G-S3 a body's colliders reach its Jiggle through a run", "[wem][soft][modes]") {
    Figure f = MakeTube(Skinned::Two, 19, 16);
    const u32 id = Soften(f, 1.0f, 8, true).id;
    AddEffect(f, SoftEffectKind::Jiggle);
    const u32 clip = 0, steps = 30;
    const f32 dt = 1.0f / 30.0f;
    CHECK(RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).drive.colliderCount == 0u);
    ClothCollider sphere;
    sphere.id = f.model().physics.allocateId();
    sphere.node = f.bones[2];
    sphere.transform.data[3][0] = 1.6f;
    sphere.radius = 0.5f;
    f.model().physics.colliders.push_back(sphere);
    // Not listed by the body, it is not met.
    CHECK(RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).drive.colliderCount == 0u);
    f.model().physics.soft[0].colliders = {sphere.id};
    const SoftGoalRun goal = RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
    CHECK(goal.drive.colliderCount == 1u);
    REQUIRE(goal.drive.colliders.size() == static_cast<std::size_t>(steps + 1));
    // Where its bone has it at each step.
    for (const u32 i : {0u, 7u, 15u}) {
        const std::vector<Matrix44f> skin = SkinAt(f.document, clip, static_cast<f32>(i) * dt);
        const Vector3f at = transform_point(Vector3f{1.6f, 0.0f, 4.0f}, skin[f.bones[2]]);
        CHECK((goal.drive.colliders[i].a - at).length() < 1e-4f);
    }
    // With no Jiggle acting there is nothing to keep out.
    f.model().physics.soft[0].effects[0].enabled = false;
    CHECK(RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip)).drive.colliders.empty());

    // The bone a collider rides gone, the collider goes with it, and leaves the body's list.
    Model& model = f.model();
    const u32 spare = AddNode(model.nodes, "spare", f.bones[2], Vector3f{0, 0, 5});
    ClothCollider other;
    other.id = model.physics.allocateId();
    other.node = spare;
    other.radius = 0.25f;
    model.physics.colliders.push_back(other);
    model.physics.soft[0].colliders = {sphere.id, other.id};
    std::vector<u32> remap(model.nodes.size());
    for (u32 n = 0; n < remap.size(); ++n) {
        remap[n] = n;
    }
    remap[spare] = kInvalidNode;
    RemapPhysicsNodes(model.physics, remap);
    CHECK(model.physics.collider(other.id) == nullptr);
    CHECK(model.physics.soft[0].colliders == std::vector<u32>{sphere.id});
}

TEST_CASE("G-S3 what a Jiggle's swing changes of the volume is taken back, by its Keep volume",
          "[wem][soft][modes]") {
    const u32 clip = 0, steps = 30;
    const f32 dt = 1.0f / 30.0f;
    // A run with the trim and the same swing without it, and the volume of
    // each beside the goal's, a step at a time.
    struct Read {
        SoftRun trimmed, bare;
        std::vector<f64> asked, kept, left;
        std::vector<u8> scales;
        f64 rest = 0.0;
        f32 size = 0.0f;
        /// The farthest a volume is from the goal's, of the goal's.
        f64 worst(const std::vector<f64>& volume) const {
            f64 most = 0.0;
            for (std::size_t i = 0; i < volume.size(); ++i) {
                most = std::max(most, std::abs(volume[i] - asked[i]) / asked[i]);
            }
            return most;
        }
    };
    const auto read = [&](f32 keepVolume, bool inflated) {
        Figure f = MakeTube(Skinned::Two, 19, 16);
        const u32 id = Soften(f, 1.0f, 8, true).id;
        if (inflated) {
            // The goal's volume is then not the bind's.
            AddEffect(f, SoftEffectKind::Inflate).amount = 0.3f;
        }
        SoftJiggle& jiggle = *AddEffect(f, SoftEffectKind::Jiggle).jiggle;
        jiggle.keepVolume = keepVolume;
        // Loose, and with room: a swing that changes the volume.
        jiggle.looseness = 1.0f;
        jiggle.limit = 0.6f;
        Read out;
        SoftGoalRun goal = RunSoftGoal(f.document, 0, id, clip, steps, dt, PoseOf(f.document, clip));
        REQUIRE(goal.volume.size() == static_cast<std::size_t>(steps + 1));
        REQUIRE(!goal.ridden.empty());
        REQUIRE(goal.skins.size() == static_cast<std::size_t>(steps + 1) * goal.ridden.size());
        out.trimmed = SwingSoft(f.document, 0, id, goal, nullptr);
        // With nothing to pose the body by again, the swing comes out as it is.
        const std::vector<f64> recorded = goal.volume;
        goal.ridden.clear();
        goal.skins.clear();
        out.bare = SwingSoft(f.document, 0, id, goal, nullptr);
        const SoftStepper stepper(f.document, 0, id);
        const SoftRig& rig = stepper.rig();
        out.rest = stepper.volume().rest;
        out.size = rig.points().size;
        for (u32 k = 0; k < rig.lumpCount(); ++k) {
            out.scales.push_back(rig.scales(k) ? 1 : 0);
        }
        std::vector<Vector3f> posed;
        for (u32 i = 0; i <= steps; ++i) {
            const std::vector<Matrix44f> skin = SkinAt(f.document, clip, static_cast<f32>(i) * dt);
            const auto volumeOf = [&](const SoftRun& run) {
                rig.pose(skin, std::span<const Transform>(run.at(i), run.lumps), posed);
                return SoftVolumeAt(stepper.volume(), posed);
            };
            out.asked.push_back(volumeOf(goal.run));
            out.kept.push_back(volumeOf(out.trimmed));
            out.left.push_back(volumeOf(out.bare));
            // What the goal recorded is the goal's volume at that step.
            CHECK(recorded[i] == Approx(out.asked.back()).epsilon(1e-6));
        }
        return out;
    };

    // At full, the volume is the goal's at every step; the swing alone left it elsewhere.
    const Read full = read(1.0f, false);
    REQUIRE(full.trimmed.swings);
    WARN("Keep volume 1: the swing alone leaves the volume " << full.worst(full.left) << " of itself away; trimmed, "
                                                             << full.worst(full.kept));
    CHECK(full.worst(full.left) > 2e-3);
    CHECK(full.worst(full.kept) < 1e-5);
    // It is an even change of the scales of the lumps that scale, and nothing else.
    REQUIRE(full.trimmed.departures.size() == full.bare.departures.size());
    u32 scaled = 0;
    for (u32 i = 0; i <= steps; ++i) {
        f32 factor = 0.0f;
        for (u32 k = 0; k < full.trimmed.lumps; ++k) {
            const Transform& a = full.trimmed.at(i)[k];
            const Transform& b = full.bare.at(i)[k];
            CHECK(std::memcmp(&a.translation, &b.translation, sizeof a.translation) == 0);
            CHECK(std::memcmp(&a.rotation, &b.rotation, sizeof a.rotation) == 0);
            if (full.scales[k] == 0) {
                CHECK(std::memcmp(&a.scale, &b.scale, sizeof a.scale) == 0);
                continue;
            }
            const f32 by = a.scale.x / b.scale.x;
            CHECK(a.scale.y / b.scale.y == Approx(by).epsilon(1e-5));
            CHECK(a.scale.z / b.scale.z == Approx(by).epsilon(1e-5));
            if (factor != 0.0f) {
                CHECK(by == Approx(factor).epsilon(1e-5));
            }
            factor = by;
            scaled += std::abs(by - 1.0f) > 1e-6f ? 1 : 0;
        }
    }
    CHECK(scaled > 0u);
    // The loop is still closed.
    for (u32 k = 0; k < full.trimmed.lumps; ++k) {
        CHECK((full.trimmed.at(steps)[k].scale - full.trimmed.at(0)[k].scale).length() < 1e-5f);
        CHECK((full.trimmed.at(steps)[k].translation - full.trimmed.at(0)[k].translation).length() < 1e-5f * full.size);
    }

    // It is the goal's volume that is kept, where an effect swelled it: not the bind's.
    const Read swollen = read(1.0f, true);
    f64 least = 1e30;
    for (const f64 asked : swollen.asked) {
        least = std::min(least, asked / swollen.rest);
    }
    CHECK(least > 1.2);
    CHECK(swollen.worst(swollen.left) > 2e-3);
    WARN("under an Inflate of 0.3: " << swollen.worst(swollen.left) << " away, trimmed " << swollen.worst(swollen.kept));
    CHECK(swollen.worst(swollen.kept) < 1e-5);

    // At half, half of what the swing changed is taken back.
    const Read half = read(0.5f, false);
    u32 measured = 0;
    for (u32 i = 0; i <= steps; ++i) {
        const f64 changed = half.left[i] - half.asked[i];
        if (std::abs(changed) < 1e-3 * half.asked[i]) {
            continue;
        }
        CHECK((half.kept[i] - half.asked[i]) / changed == Approx(0.5).margin(0.02));
        ++measured;
    }
    CHECK(measured > 5u);

    // At none, the swing is as it comes.
    const Read none = read(0.0f, false);
    CHECK(SameBytes(none.trimmed.departures, none.bare.departures));
    CHECK(none.worst(none.left) > 2e-3);
}
