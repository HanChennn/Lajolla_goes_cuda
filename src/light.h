#pragma once

#include "lajolla.h"
#include "matrix.h"
#include "point_and_normal.h"
#include "shape.h"
#include "spectrum.h"
#include "texture.h"
#include "vector.h"
#include "transform.h"
#include <variant>

/// An area light attached on a shape to make it emit lights.
struct DiffuseAreaLight {
    int shape_id;
    Vector3 intensity;
};

/// An environment map (Envmap) is an infinitely far area light source
/// that covers the whole bounding spherical domain of the scene.
/// A texture is used to represent light coming from each direction.

struct Envmap {
    Texture<Spectrum> values;
    Matrix4x4 to_world, to_local;
    Real scale;

    // For sampling a point on the envmap
    TableDist2D sampling_dist;
};

// To add more lights, first create a struct for the light, add it to the variant type below,
// then implement all the relevant function below with the Light type.
using Light = std::variant<DiffuseAreaLight, Envmap>;

#include "lights/diffuse_area_light.inl"
#include "lights/envmap.inl"

__device__ inline PointAndNormal sample_point_on_light(const Light &light,
                                     const Vector3 &ref_point,
                                     const Vector2 &rnd_param_uv,
                                     Real rnd_param_w,
                                     const CUArray<Shape> &shapes) {
    if (auto *l = std::get_if<DiffuseAreaLight>(&light)) 
        return sample_point_on_light_diffuse_area_light(*l, ref_point, rnd_param_uv, rnd_param_w, shapes);
    else if (auto *l = std::get_if<Envmap>(&light))
        return sample_point_on_light_envmap(*l, rnd_param_uv);
    else
        return PointAndNormal{ Vector3{0.0, 0.0, 0.0}, Vector3{0.0, 0.0, 0.0} };
}

__device__ inline Real pdf_point_on_light(const Light &light,
                        const PointAndNormal &point_on_light,
                        const Vector3 &ref_point,
                        const CUArray<Shape> &shapes) {
    if (auto *l = std::get_if<DiffuseAreaLight>(&light)) 
        return pdf_point_on_light_diffuse_area_light(*l, point_on_light, ref_point, shapes);
    else if (auto *l = std::get_if<Envmap>(&light))
        return pdf_point_on_light_envmap(*l, point_on_light);
    else
        return Real(0.0);
}

__device__ inline Spectrum emission(const Light &light,
                  const Vector3 &view_dir,
                  Real view_footprint,
                  const PointAndNormal &point_on_light,
                  const TexturePool& texture_pool) {
    if (auto *l = std::get_if<DiffuseAreaLight>(&light)) 
        return emission_diffuse_area_light(*l, view_dir, point_on_light);
    else if (auto *l = std::get_if<Envmap>(&light))
        return emission_envmap(*l, view_dir, view_footprint, texture_pool);
    else
        return Spectrum{0.0, 0.0, 0.0};
}

__device__ inline bool is_envmap(const Light &light) {
    return std::get_if<Envmap>(&light) != nullptr;
}