#include "medium.h"

struct get_majorant_op {
    __host__ __device__ Spectrum operator()(const HomogeneousMedium &m);
    __host__ __device__ Spectrum operator()(const HeterogeneousMedium &m);

    const Ray &ray;
};

struct get_sigma_s_op {
    __host__ __device__ Spectrum operator()(const HomogeneousMedium &m);
    __host__ __device__ Spectrum operator()(const HeterogeneousMedium &m);

    const Vector3 &p;
};

struct get_sigma_a_op {
    __host__ __device__ Spectrum operator()(const HomogeneousMedium &m);
    __host__ __device__ Spectrum operator()(const HeterogeneousMedium &m);

    const Vector3 &p;
};

#include "media/homogeneous.inl"
#include "media/heterogeneous.inl"

__host__ __device__ Spectrum get_majorant(const Medium &medium, const Ray &ray) {
    return std::visit(get_majorant_op{ray}, medium);
}

__host__ __device__ Spectrum get_sigma_s(const Medium &medium, const Vector3 &p) {
    return std::visit(get_sigma_s_op{p}, medium);
}

__host__ __device__ Spectrum get_sigma_a(const Medium &medium, const Vector3 &p) {
    return std::visit(get_sigma_a_op{p}, medium);
}
