#pragma once

#include "lajolla.h"
#include "spectrum.h"
#include "vector.h"
#include <optional>
#include <variant>

struct IsotropicPhase {
};

struct HenyeyGreenstein {
    Real g;
};

using PhaseFunction = std::variant<IsotropicPhase, HenyeyGreenstein>;

__host__ __device__ Spectrum eval(const PhaseFunction &phase_function,
              const Vector3 &dir_in,
              const Vector3 &dir_out);

__host__ __device__ std::optional<Vector3> sample_phase_function(
    const PhaseFunction &phase_function,
    const Vector3 &dir_in,
    const Vector2 &rnd_param);

__host__ __device__ Real pdf_sample_phase(const PhaseFunction &phase_function,
                      const Vector3 &dir_in,
                      const Vector3 &dir_out);
