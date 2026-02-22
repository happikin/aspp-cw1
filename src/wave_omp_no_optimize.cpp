//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_omp.h"

// This struct can hold any data you need to manage running on the device
//
// Allocate with std::make_unique when you create the simulation
// object below, in `from_cpu_sim`.
struct OmpImplementationData {
    // Add any data members you need here

    OmpImplementationData() {
    }
    ~OmpImplementationData() {
   }
};

OmpWaveSimulation::OmpWaveSimulation() = default;
OmpWaveSimulation::OmpWaveSimulation(OmpWaveSimulation&&)  noexcept = default;
OmpWaveSimulation& OmpWaveSimulation::operator=(OmpWaveSimulation&&) = default;
OmpWaveSimulation::~OmpWaveSimulation() = default;

OmpWaveSimulation OmpWaveSimulation::from_cpu_sim(const fs::path& cp, const WaveSimulation& source) {
    OmpWaveSimulation ans;
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
    // ans.impl = std::make_unique<OmpImplementationData>();

    return ans;
}


void OmpWaveSimulation::append_u_fields() {
    h5.append_u(u);
}

static void step(Params const& params, array3d const& cs2, array3d const& damp, uField& u) {

    auto d2 = params.dx * params.dx;
    auto dt = params.dt;
    auto factor = dt*dt / d2;
    auto [nx, ny, nz] = params.shape;

    auto* cs2_ptr  = cs2.data();
    auto* damp_ptr = damp.data();
    auto* now_ptr  = u.now().data();
    auto* prev_ptr = u.prev().data();
    auto* next_ptr = u.next().data();

    auto nx_tot = nx + 2;
    auto ny_tot = ny + 2;
    auto nz_tot = nz + 2;

    #pragma omp target teams distribute parallel for collapse(3) \
        map(to: cs2_ptr[0:nx*ny*nz], damp_ptr[0:nx*ny*nz], \
                now_ptr[0:nx_tot*ny_tot*nz_tot], \
                prev_ptr[0:nx_tot*ny_tot*nz_tot]) \
        map(from: next_ptr[0:nx_tot*ny_tot*nz_tot])
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < ny; ++j) {
            for (unsigned k = 0; k < nz; ++k) {

                unsigned ii = i + 1;
                unsigned jj = j + 1;
                unsigned kk = k + 1;
                
                size_t idx  = ii*ny_tot*nz_tot + jj*nz_tot + kk;
                size_t idx_xm = (ii-1)*ny_tot*nz_tot + jj*nz_tot + kk;
                size_t idx_xp = (ii+1)*ny_tot*nz_tot + jj*nz_tot + kk;
                size_t idx_ym = ii*ny_tot*nz_tot + (jj-1)*nz_tot + kk;
                size_t idx_yp = ii*ny_tot*nz_tot + (jj+1)*nz_tot + kk;
                size_t idx_zm = ii*ny_tot*nz_tot + jj*nz_tot + (kk-1);
                size_t idx_zp = ii*ny_tot*nz_tot + jj*nz_tot + (kk+1);

                size_t idx_c = i*ny*nz + j*nz + k;

                auto value = factor * cs2_ptr[idx_c] * (
                    now_ptr[idx_xm] + now_ptr[idx_xp] +
                    now_ptr[idx_ym] + now_ptr[idx_yp] +
                    now_ptr[idx_zm] + now_ptr[idx_zp]
                    - 6.0 * now_ptr[idx]
                );

                auto d = damp_ptr[idx_c];

                if (d == 0.0) {
                    next_ptr[idx] = 2.0 * now_ptr[idx] - prev_ptr[idx] + value;
                } else {
                    auto inv_den = 1.0 / (1.0 + d * dt);
                    auto numerator = 1.0 - d * dt;
                    value *= inv_den;
                    next_ptr[idx] =
                        2.0 * inv_den * now_ptr[idx]
                        - numerator * inv_den * prev_ptr[idx]
                        + value;
                }
            }
        }
    }

    u.advance();

}

void OmpWaveSimulation::run(int n) {
    for (int i = 0; i < n; ++i) {
        step(params, cs2, damp, u);
    }
}
