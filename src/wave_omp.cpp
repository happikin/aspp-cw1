// wave_omp.cpp
// OpenMP offload implementation for the 3D wave equation.


#include "wave_omp.h"
#include <omp.h>
#include <stdexcept>
#include <filesystem>
#include <memory>

struct OmpImplementationData {
    // Device buffers
    double *d_cs2 = nullptr;    // interior coefficients (no halo)
    double *d_damp = nullptr;   // interior damping (no halo)
    double *d_u_prev = nullptr; // padded field (t-1)
    double *d_u_now  = nullptr; // padded field (t)
    double *d_u_next = nullptr; // padded field (t+1)

    // Domain sizes (interior + padded)
    std::size_t nx = 0, ny = 0, nz = 0;
    std::size_t pnx = 0, pny = 0, pnz = 0;

    double dx = 1.0, dt = 1.0;
    int device = -1;

    OmpImplementationData() = default;

    ~OmpImplementationData() {
        // Free target allocations (RAII cleanup).
        if (device >= 0) {
            if (d_cs2)    omp_target_free(d_cs2, device);
            if (d_damp)   omp_target_free(d_damp, device);
            if (d_u_prev) omp_target_free(d_u_prev, device);
            if (d_u_now)  omp_target_free(d_u_now, device);
            if (d_u_next) omp_target_free(d_u_next, device);
        }
    }
};

static void device_step(OmpImplementationData &impl) {
    const std::size_t nx = impl.nx, ny = impl.ny, nz = impl.nz;
    const std::size_t pny = impl.pny, pnz = impl.pnz;
    const double dt = impl.dt, dx = impl.dx;
    const double factor = (dt * dt) / (dx * dx);

    // Strides for padded 3D layout (neighbor access in x/y/z).
    const std::size_t stride_x = pny * pnz;
    const std::size_t stride_y = pnz;

    double *const d_cs2    = impl.d_cs2;
    double *const d_damp   = impl.d_damp;
    double *const d_u_prev = impl.d_u_prev;
    double *const d_u_now  = impl.d_u_now;
    double *const d_u_next = impl.d_u_next;

    // Offload full 3D interior update to the device.
    #pragma omp target teams distribute parallel for collapse(3) \
        is_device_ptr(d_cs2, d_damp, d_u_prev, d_u_now, d_u_next)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nx); ++i) {
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(ny); ++j) {
            for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(nz); ++k) {
                const std::size_t idx_cs =
                    (static_cast<std::size_t>(i) * ny + static_cast<std::size_t>(j)) * nz +
                    static_cast<std::size_t>(k);

                // Wave fields are padded, so interior indices are offset by +1.
                const std::size_t idx =
                    ((static_cast<std::size_t>(i) + 1) * pny +
                     (static_cast<std::size_t>(j) + 1)) * pnz +
                     (static_cast<std::size_t>(k) + 1);

                const double center = d_u_now[idx];

                // 7-point stencil Laplacian contribution.
                double value = factor * d_cs2[idx_cs] * (
                    d_u_now[idx - stride_x] + d_u_now[idx + stride_x] +
                    d_u_now[idx - stride_y] + d_u_now[idx + stride_y] +
                    d_u_now[idx - 1]        + d_u_now[idx + 1]        -
                    6.0 * center);

                const double d = d_damp[idx_cs];
                if (d == 0.0) {
                    // Fast path for undamped cells.
                    d_u_next[idx] = 2.0 * center - d_u_prev[idx] + value;
                } else {
                    // Damped update (one reciprocal reused).
                    const double inv_denominator = 1.0 / (1.0 + d * dt);
                    const double numerator = 1.0 - d * dt;
                    value *= inv_denominator;
                    d_u_next[idx] = 2.0 * inv_denominator * center -
                                    numerator * inv_denominator * d_u_prev[idx] + value;
                }
            }
        }
    }
}

OmpWaveSimulation::OmpWaveSimulation() = default;
OmpWaveSimulation::OmpWaveSimulation(OmpWaveSimulation&&) noexcept = default;
OmpWaveSimulation& OmpWaveSimulation::operator=(OmpWaveSimulation&&) = default;
OmpWaveSimulation::~OmpWaveSimulation() = default;

OmpWaveSimulation OmpWaveSimulation::from_cpu_sim(const fs::path& cp, const WaveSimulation& source) {
    OmpWaveSimulation ans;
    out("Initialising {} simulation as copy of {}...", ans.ID(), source.ID());

    // Clone host-side state (used for output/checkpointing and metadata).
    ans.params = source.params;
    ans.u    = source.u.clone();
    ans.sos  = source.sos.clone();
    ans.cs2  = source.cs2.clone();
    ans.damp = source.damp.clone();
    ans.checkpoint = cp;
    ans.h5 = H5IO::from_params(cp, ans.params);

    out("Write initial conditions to {}", ans.checkpoint.c_str());
    ans.h5.put_params(ans.params);
    ans.h5.put_damp(ans.damp);
    ans.h5.put_sos(ans.sos);
    ans.append_u_fields();

    ans.impl = std::make_unique<OmpImplementationData>();
    auto &im = *ans.impl;
    auto [nx, ny, nz] = ans.params.shape;

    im.nx = nx; im.ny = ny; im.nz = nz;
    im.pnx = nx + 2; im.pny = ny + 2; im.pnz = nz + 2; // 1-cell halo on each side
    im.dx = ans.params.dx; im.dt = ans.params.dt;
    im.device = omp_get_default_device();
    const int host_dev = omp_get_initial_device();

    const std::size_t n_cs = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t n_u  = static_cast<std::size_t>(im.pnx) * im.pny * im.pnz;

    // Persistent device allocations (avoid per-step map/unmap overhead).
    im.d_cs2    = static_cast<double*>(omp_target_alloc(n_cs * sizeof(double), im.device));
    im.d_damp   = static_cast<double*>(omp_target_alloc(n_cs * sizeof(double), im.device));
    im.d_u_prev = static_cast<double*>(omp_target_alloc(n_u  * sizeof(double), im.device));
    im.d_u_now  = static_cast<double*>(omp_target_alloc(n_u  * sizeof(double), im.device));
    im.d_u_next = static_cast<double*>(omp_target_alloc(n_u  * sizeof(double), im.device));

    if (!im.d_cs2 || !im.d_damp || !im.d_u_prev || !im.d_u_now || !im.d_u_next)
        throw std::runtime_error("OpenMP target allocation failed");

    // One-time host -> device initialization.
    omp_target_memcpy(im.d_cs2,    ans.cs2.data(),      n_cs * sizeof(double), 0, 0, im.device, host_dev);
    omp_target_memcpy(im.d_damp,   ans.damp.data(),     n_cs * sizeof(double), 0, 0, im.device, host_dev);
    omp_target_memcpy(im.d_u_prev, ans.u.prev().data(), n_u  * sizeof(double), 0, 0, im.device, host_dev);
    omp_target_memcpy(im.d_u_now,  ans.u.now().data(),  n_u  * sizeof(double), 0, 0, im.device, host_dev);
    omp_target_memcpy(im.d_u_next, ans.u.next().data(), n_u  * sizeof(double), 0, 0, im.device, host_dev);

    return ans;
}

void OmpWaveSimulation::append_u_fields() {
    if (impl) {
        auto &im = *impl;
        const std::size_t n_u = im.pnx * im.pny * im.pnz;
        const int host_dev = omp_get_initial_device();

        // Copy back only when writing output/checkpoints.
        omp_target_memcpy(u.prev().data(), im.d_u_prev, n_u * sizeof(double), 0, 0, host_dev, im.device);
        omp_target_memcpy(u.now().data(),  im.d_u_now,  n_u * sizeof(double), 0, 0, host_dev, im.device);
    }

    h5.append_u(u);
}

void OmpWaveSimulation::run(int n) {
    auto &im = *impl;
    for (int step = 0; step < n; ++step) {
        device_step(im);

        // Rotate time buffers (O(1)) instead of copying arrays.
        double *temp = im.d_u_prev;
        im.d_u_prev = im.d_u_now;
        im.d_u_now  = im.d_u_next;
        im.d_u_next = temp;

        // Advance host-side time indexing only (no host data copy here).
        u.advance();
    }
}