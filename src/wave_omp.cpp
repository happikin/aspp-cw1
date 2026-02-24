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
    std::size_t m_nx;
    std::size_t m_ny;
    std::size_t m_nz;
    double      m_factor;
    double      m_d2;
    double      m_dt;
    double*     m_now_ptr;
    double*     m_prev_ptr;
    double*     m_next_ptr;
    double*     m_cs2_ptr;
    double*     m_damp_ptr;
    OmpImplementationData(
        OmpWaveSimulation& _omp_wave
    ) {
        auto [nx, ny, nz] = _omp_wave.params.shape;
        m_nx = nx; m_ny = ny; m_nz = nz;

        m_cs2_ptr  = _omp_wave.cs2.data();
        m_damp_ptr = _omp_wave.damp.data();

    }
    ~OmpImplementationData() {
    }
    std::size_t interior_size() const { return ( m_nx * m_ny * m_nz ); }
    std::size_t total_size() const { return ( (m_nx + 2) * (m_ny + 2) * (m_nz + 2) ); }
    void pack_params(
        double* _now, double* _prev, 
        double* _next, Params& _params
    ) {
        m_now_ptr   = _now;
        m_prev_ptr  = _prev;
        m_next_ptr  = _next;

        m_d2 = _params.dx * _params.dx;
        m_dt = _params.dt;
        m_factor    = m_dt * m_dt / m_d2;
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
    ans.impl
        = std::make_unique<OmpImplementationData>(ans);

    return ans;
}


void OmpWaveSimulation::append_u_fields() {
    h5.append_u(u);
}

static
void step(
    OmpImplementationData* _impl
) {
    auto nx         = _impl->m_nx;
    auto ny         = _impl->m_ny;
    auto nz         = _impl->m_nz;

    auto nx_tot     = _impl->m_nx + 2;
    auto ny_tot     = _impl->m_ny + 2;
    auto nz_tot     = _impl->m_nz + 2;

    auto* now_ptr   = _impl->m_now_ptr;
    auto* prev_ptr  = _impl->m_prev_ptr;
    auto* next_ptr  = _impl->m_next_ptr;
    auto* cs2_ptr   = _impl->m_cs2_ptr;
    auto* damp_ptr  = _impl->m_damp_ptr;

    auto factor     = _impl->m_factor;
    auto dt         = _impl->m_dt;
    
    auto stride_x = ny_tot*nz_tot;
    auto stride_y = nz_tot;
    int index_max = (int)(nx * ny * nz);

    #pragma omp target teams distribute parallel for
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(index_max); ++index) {

        // Recover 3D indices
        std::size_t i = index / (ny * nz);
        std::size_t rem = index % (ny * nz);
        std::size_t j = rem / nz;
        std::size_t k = rem % nz;

        unsigned ii = i + 1;
        unsigned jj = j + 1;
        unsigned kk = k + 1;

        std::size_t idx  = ii*stride_x + jj*stride_y + kk;
        std::size_t idx_xm = (ii-1)*stride_x + jj*stride_y + kk;
        std::size_t idx_xp = (ii+1)*stride_x + jj*stride_y + kk;
        std::size_t idx_ym = ii*stride_x + (jj-1)*stride_y + kk;
        std::size_t idx_yp = ii*stride_x + (jj+1)*stride_y + kk;
        std::size_t idx_zm = ii*stride_x + jj*stride_y + (kk-1);
        std::size_t idx_zp = ii*stride_x + jj*stride_y + (kk+1);

        auto value = factor * cs2_ptr[index] * (
            now_ptr[idx_xm] + now_ptr[idx_xp] +
            now_ptr[idx_ym] + now_ptr[idx_yp] +
            now_ptr[idx_zm] + now_ptr[idx_zp]
            - 6.0 * now_ptr[idx]
        );

        auto d = damp_ptr[index];

        if (d == 0.0) {
            next_ptr[idx] =
                2.0 * now_ptr[idx]
                - prev_ptr[idx]
                + value;
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

void OmpWaveSimulation::run(int n) {

    size_t interior_size = impl->interior_size();
    size_t total_size = impl->total_size();

    auto* cs2_ptr   = cs2.data();
    auto* damp_ptr  = damp.data();
    /* Capture all 3 buffers ONCE (underlying memory never moves) */
    auto* buf0 = u.now().data();
    auto* buf1 = u.prev().data();
    auto* buf2 = u.next().data();

    // Local rotating pointers (these will change)
    double* p_now  = buf0;
    double* p_prev = buf1;
    double* p_next = buf2;
    OmpImplementationData* local_impl = impl.get();
    // ---- OMP Data Movement ---- //
    #pragma omp target data                 \
    map(                                    \
        to:                                 \
            cs2_ptr[0:interior_size],       \
            damp_ptr[0:interior_size]       \
    )                                       \
    map(                                    \
        tofrom:                             \
            buf0[0:total_size],             \
            buf1[0:total_size],             \
            buf2[0:total_size]              \
    )
    // --------------------------- //
    {
        for (int t = 0; t < n; ++t) {

            // ---- Prep Args ---- //
            local_impl->pack_params(
                p_now,
                p_prev,
                p_next,
                params
            );
            // ------------------- //

            // ---- OMP GPU Offloading ---- //
            step(local_impl);
            // ---------------------------- //

            // Rotate pointers locally (do NOT re-query u.now())
            double* tmp = p_prev;
            p_prev = p_now;
            p_now  = p_next;
            p_next = tmp;

            u.advance();
        }
    }
}
