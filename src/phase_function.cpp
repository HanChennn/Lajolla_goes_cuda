#include "phase_function.h"

struct eval_op {
    __host__ __device__ Spectrum operator()(const IsotropicPhase &p) const;
    __host__ __device__ Spectrum operator()(const HenyeyGreenstein &p) const;

    const Vector3 &dir_in;
    const Vector3 &dir_out;
};

struct sample_phase_function_op {
    __host__ __device__ std::optional<Vector3> operator()(const IsotropicPhase &p) const;
    __host__ __device__ std::optional<Vector3> operator()(const HenyeyGreenstein &p) const;

    const Vector3 &dir_in;
    const Vector2 &rnd_param;
};

struct pdf_sample_phase_op {
    __host__ __device__ Real operator()(const IsotropicPhase &p) const;
    __host__ __device__ Real operator()(const HenyeyGreenstein &p) const;

    const Vector3 &dir_in;
    const Vector3 &dir_out;
};

#include "phase_functions/isotropic.inl"
#include "phase_functions/henyeygreenstein.inl"

__host__ __device__ Spectrum eval(const PhaseFunction &phase_function,
              const Vector3 &dir_in,
              const Vector3 &dir_out) {
    return std::visit(eval_op{dir_in, dir_out}, phase_function);
}

__host__ __device__ std::optional<Vector3> sample_phase_function(
        const PhaseFunction &phase_function,
        const Vector3 &dir_in,
        const Vector2 &rnd_param) {
    return std::visit(sample_phase_function_op{dir_in, rnd_param}, phase_function);
}

__host__ __device__ Real pdf_sample_phase(const PhaseFunction &phase_function,
                      const Vector3 &dir_in,
                      const Vector3 &dir_out) {
    return std::visit(pdf_sample_phase_op{dir_in, dir_out}, phase_function);
}
