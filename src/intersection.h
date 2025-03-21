#pragma once

#include "lajolla.h"
#include "ray.h"
#include "spectrum.h"
#include "vector.h"
#include "point_and_normal.h"
#include "shape.h"
#include "light.h"

#include <optional>
#include <variant>

/// Intersect a ray with a scene. If the ray doesn't hit anything,
/// returns an invalid optional output. 
__device__ inline std::optional<PathVertex> intersect(const CUArray<Shape> &shapes,
                                                      const Ray &ray,
                                                      const RayDifferential &ray_diff = RayDifferential{}) {
    
    PathVertex vertex{};
    vertex.shape_id = -1;
    Ray r = ray;
    for(int i = 0; i < shapes.size(); ++i){
        std::optional<PathVertex> v_ = std::visit(intersect_op{r, ray_diff}, shapes[i]);
        if(!v_) continue;
        PathVertex v = *v_;
        Real t = length(v.position - ray.org);
        if (t < r.tfar){
            r.tfar = t;
            vertex = v;
        }
    }
    if(vertex.shape_id >= 0){
        return vertex;
    }else{
        return {};
    }
}

/// Test is a ray segment intersect with anything in a scene.
__device__ inline bool occluded(const CUArray<Shape> &shapes, const Ray &ray) {
    Ray r = ray;
    for(int i = 0; i < shapes.size(); ++i){
        if(std::visit(occluded_op{r}, shapes[i])){
            return true;
        }
    }
    return false;
}

/// Computes the emission at a path vertex v, with the viewing direction
/// pointing outwards of the intersection.
__device__ inline Spectrum emission(const PathVertex &v,
                                    const Vector3 &view_dir,
                                    const CUArray<Shape> &shapes,
                                    const CUArray<Light> &lights,
                                    const TexturePool& texture_pool) {
    int light_id = get_area_light_id(shapes[v.shape_id]);
    assert(light_id >= 0);
    const Light &light = lights[light_id];
    return emission(light,
                    view_dir,
                    v.uv_screen_size,
                    PointAndNormal{v.position, v.geometric_normal},
                    texture_pool);
}
