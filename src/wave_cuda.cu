// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_cuda.h"

#include <nvtx3/nvtx3.hpp>

// ---- CUDA Step Kernel ---- //
__global__
void step_kernel(
    double* now_ptr, double* prev_ptr, double* next_ptr,
    const double* cs2_ptr, const double* damp_ptr,
    size_t nx, size_t ny, size_t nz,
    size_t ny_tot, size_t nz_tot,
    double factor, double dt
) {
    size_t idx_c = blockIdx.x * blockDim.x + threadIdx.x;

    size_t total = nx * ny * nz;
    if (idx_c >= total) return;

    size_t i = idx_c / (ny * nz);
    size_t rem = idx_c % (ny * nz);
    size_t j = rem / nz;
    size_t k = rem % nz;

    size_t ii = i + 1;
    size_t jj = j + 1;
    size_t kk = k + 1;

    size_t stride_x = ny_tot * nz_tot;
    size_t stride_y = nz_tot;

    size_t idx  = ii*stride_x + jj*stride_y + kk;
    size_t idx_xm = idx - stride_x;
    size_t idx_xp = idx + stride_x;
    size_t idx_ym = idx - stride_y;
    size_t idx_yp = idx + stride_y;
    size_t idx_zm = idx - 1;
    size_t idx_zp = idx + 1;

    double value = factor * cs2_ptr[idx_c] * (
        now_ptr[idx_xm] + now_ptr[idx_xp] +
        now_ptr[idx_ym] + now_ptr[idx_yp] +
        now_ptr[idx_zm] + now_ptr[idx_zp]
        - 6.0 * now_ptr[idx]
    );

    double d = damp_ptr[idx_c];

    if (d == 0.0) {
        next_ptr[idx] =
            2.0 * now_ptr[idx]
            - prev_ptr[idx]
            + value;
    } else {
        double inv_den = 1.0 / (1.0 + d * dt);
        double numerator = 1.0 - d * dt;
        value *= inv_den;

        next_ptr[idx] =
            2.0 * inv_den * now_ptr[idx]
            - numerator * inv_den * prev_ptr[idx]
            + value;
    }
}
// -------------------------- //

// Free helper macro to check for CUDA errors!
#define CUDA_CHECK(expr) do { \
    cudaError_t res = expr; \
    if (res != cudaSuccess) \
      throw std::runtime_error(std::format(__FILE__ ":{} CUDA error: {}", __LINE__, cudaGetErrorString(res))); \
  } while (0)


// This struct can hold any data you need to manage running on the device
//
// Allocate with std::make_unique when you create the simulation
// object below, in `from_cpu_sim`.
struct CudaImplementationData {
    // Add any data members you need here, such as device arrays, streams, etc

    CudaImplementationData() {
        nvtx3::scoped_range r{"initialise"};
        // allocate them here?
    }
    ~CudaImplementationData() {
        // free here probably
   }
};

CudaWaveSimulation::CudaWaveSimulation() = default;
CudaWaveSimulation::CudaWaveSimulation(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation& CudaWaveSimulation::operator=(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation::~CudaWaveSimulation() = default;

CudaWaveSimulation CudaWaveSimulation::from_cpu_sim(const fs::path& cp, const WaveSimulation& source) {
    CudaWaveSimulation ans;
    out("Initialising {} simulation as copy of {}...", ans.ID(), source.ID());
    ans.params = source.params;
    ans.u = source.u.clone();
    ans.sos = source.sos.clone();
    ans.cs2 = source.cs2.clone();
    ans.damp = source.damp.clone();

    ans.checkpoint = cp;
    ans.h5 = H5IO::from_params(cp, ans.params);

    out("Write initial conditions to {}", ans.checkpoint.c_str());
    ans.h5.put_params(ans.params);
    ans.h5.put_damp(ans.damp);
    ans.h5.put_sos(ans.sos);
    ans.append_u_fields();

    // Perhaps you want to do some device set up now?
    // ans.impl = std::make_unique<CudaImplementationData>();

    return ans;
}

void CudaWaveSimulation::append_u_fields() {
    h5.append_u(u);
}

// static void step(Params const& params, array3d const& cs2, array3d const& damp, uField& u) {
//     nvtx3::scoped_range r{"step"};
//     auto d2 = params.dx * params.dx;
//     auto dt = params.dt;
//     auto factor = dt*dt / d2;
//     auto [nx, ny, nz] = params.shape;
//     for (unsigned i = 0; i < nx; ++i) {
//         auto ii = i + 1;
//         for (unsigned j = 0; j < ny; ++j) {
//             auto jj = j + 1;
//             for (unsigned k = 0; k < nz; ++k) {
//                 auto kk = k + 1;
//                 // Simple approximation of Laplacian
//                 auto value = factor * cs2(i, j, k) * (
//                         u.now()(ii - 1, jj, kk) + u.now()(ii + 1, jj, kk) +
//                         u.now()(ii, jj - 1, kk) + u.now()(ii, jj + 1, kk) +
//                         u.now()(ii, jj, kk - 1) + u.now()(ii, jj, kk + 1)
//                         - 6.0 * u.now()(ii, jj, kk)
//                 );
//                 // Deal with the damping field
//                 auto& d = damp(i, j, k);
//                 if (d == 0.0) {
//                     u.next()(ii, jj, kk) = 2.0 * u.now()(ii, jj, kk) - u.prev()(ii, jj, kk) + value;
//                 } else {
//                     auto inv_denominator = 1.0 / (1.0 + d * dt);
//                     auto numerator = 1.0 - d * dt;
//                     value *= inv_denominator;
//                     u.next()(ii, jj, kk) = 2.0 * inv_denominator * u.now()(ii, jj, kk) -
//                                            numerator * inv_denominator * u.prev()(ii, jj, kk) + value;
//                 }
//             }
//         }
//     }
//     u.advance();
// }

void CudaWaveSimulation::run(int n)
{
    auto [nx, ny, nz] = params.shape;

    size_t interior_size = nx * ny * nz;

    size_t nx_tot = nx + 2;
    size_t ny_tot = ny + 2;
    size_t nz_tot = nz + 2;

    size_t total_size = nx_tot * ny_tot * nz_tot;

    double d2 = params.dx * params.dx;
    double dt = params.dt;
    double factor = (dt * dt) / d2;

    // ---- Allocate fixed device buffers ---- //
    double* d_buf[3];
    cudaMalloc(&d_buf[0], total_size * sizeof(double));
    cudaMalloc(&d_buf[1], total_size * sizeof(double));
    cudaMalloc(&d_buf[2], total_size * sizeof(double));

    double* d_cs2;
    double* d_damp;

    cudaMalloc(&d_cs2, interior_size * sizeof(double));
    cudaMalloc(&d_damp, interior_size * sizeof(double));

    // ---- Copy initial data ---- //
    cudaMemcpy(d_buf[0], u.now().data(),  total_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf[1], u.prev().data(), total_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buf[2], u.next().data(), total_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_cs2,  cs2.data(),  interior_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_damp, damp.data(), interior_size * sizeof(double), cudaMemcpyHostToDevice);

    // ---- Role indices ---- //
    int cur  = 0;
    int prev = 1;
    int next = 2;

    int threads = 256;
    int blocks = (interior_size + threads - 1) / threads;

    for (int t = 0; t < n; ++t)
    {
        step_kernel<<<blocks, threads>>>(
            d_buf[cur],
            d_buf[prev],
            d_buf[next],
            d_cs2,
            d_damp,
            nx,
            ny,
            nz,
            ny_tot,
            nz_tot,
            factor,
            dt
        );

        cudaDeviceSynchronize();

        // Rotate roles cyclically //
        int tmp = prev;
        prev = cur;
        cur  = next;
        next = tmp;

        // Rotate host ring buffer //
        u.advance();
    }

    // Copy final result back (cur matches u.now()) //
    cudaMemcpy(u.now().data(),
               d_buf[cur],
               total_size * sizeof(double),
               cudaMemcpyDeviceToHost);

    // ---- copy back all three buffers to make them persist between run calls ---- //
    cudaMemcpy(u.now().data(),  d_buf[cur],  total_size*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u.prev().data(), d_buf[prev], total_size*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(u.next().data(), d_buf[next], total_size*sizeof(double), cudaMemcpyDeviceToHost);
    
    // ---- Free memory ---- //
    cudaFree(d_buf[0]);
    cudaFree(d_buf[1]);
    cudaFree(d_buf[2]);
    cudaFree(d_cs2);
    cudaFree(d_damp);
}
