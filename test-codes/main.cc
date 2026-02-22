#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

int main() {
    const int N = 1 << 20;  // 1M elements

    std::vector<double> a(N, 1.0);
    std::vector<double> b(N, 2.0);
    std::vector<double> c(N, 0.0);

    double* a_ptr = a.data();
    double* b_ptr = b.data();
    double* c_ptr = c.data();

    // -----------------------------
    // Check if device exists
    // -----------------------------
    int num_devices = omp_get_num_devices();
    std::cout << "Number of OpenMP target devices: "
              << num_devices << "\n";

    // -----------------------------
    // Offload computation to GPU
    // -----------------------------
    #pragma omp target teams distribute parallel for \
        map(to: a_ptr[0:N], b_ptr[0:N]) \
        map(from: c_ptr[0:N])
    for (int i = 0; i < N; ++i) {

        // Check inside device
        if (i == 0) {
            if (!omp_is_initial_device()) {
                printf("Running on GPU device\n");
            } else {
                printf("Running on CPU (fallback)\n");
            }
        }

        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }

    // -----------------------------
    // Verification
    // -----------------------------
    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double expected = 3.0;
        max_error = std::max(max_error, std::abs(c[i] - expected));
    }

    std::cout << "Max error: " << max_error << "\n";

    if (max_error < 1e-12)
        std::cout << "Verification PASSED\n";
    else
        std::cout << "Verification FAILED\n";

    return 0;
}