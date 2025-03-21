#include "../volume.h"

__host__ __device__ Spectrum get_majorant_op::operator()(const HeterogeneousMedium &m) {
    if (intersect(m.density, ray)) {
        return get_max_value(m.density);
    } else {
        return make_zero_spectrum();
    }
}

__host__ __device__ Spectrum get_sigma_s_op::operator()(const HeterogeneousMedium &m) {
    Spectrum density = lookup(m.density, p);
    Spectrum albedo = lookup(m.albedo, p);
    return density * albedo;
}

__host__ __device__ Spectrum get_sigma_a_op::operator()(const HeterogeneousMedium &m) {
    Spectrum density = lookup(m.density, p);
    Spectrum albedo = lookup(m.albedo, p);
    return density * (Real(1) - albedo);
}
