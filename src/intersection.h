#pragma once

#include "lajolla.h"
#include "frame.h"
#include "ray.h"
#include "spectrum.h"
#include "vector.h"
#include "bvh.h"

#include <optional>

struct Scene;
struct cudaScene;

/// An "PathVertex" represents a vertex of a light path.
/// We store the information we need for computing any sort of path contribution & sampling density.
struct PathVertex {
    Vector3 position;
    Vector3 geometric_normal; // always face at the same direction at shading_frame.n
    Frame shading_frame;
    Vector2 st; // A 2D parametrization of the surface. Irrelavant to UV mapping.
                // for triangle this is the barycentric coordinates, which we use
                // for interpolating the uv map.
    Vector2 uv; // The actual UV we use for texture fetching.
    // For texture filtering, stores approximatedly min(abs(du/dx), abs(dv/dx), abs(du/dy), abs(dv/dy))
    Real uv_screen_size;
    Real mean_curvature; // For ray differential propagation.
    Real ray_radius; // For ray differential propagation.
    int shape_id = -1;
    int primitive_id = -1; // For triangle meshes. This indicates which triangle it hits.
    int material_id = -1;

    // If the path vertex is inside a medium, these two IDs
    // are the same.
    int interior_medium_id = -1;
    int exterior_medium_id = -1;
};

/// Intersect a ray with a scene. If the ray doesn't hit anything,
/// returns an invalid optional output. 
// std::optional<PathVertex> intersect(const Scene &scene,
//                                     const Ray &ray,
//                                     const RayDifferential &ray_diff = RayDifferential{});
// __host__ __device__ std::optional<PathVertex> intersect(const cuda_bvh &bvh, 
//                                     const Shape *shapes, 
//                                     const Ray &ray, 
//                                     const RayDifferential &ray_diff = RayDifferential{});


// /// Test is a ray segment intersect with anything in a scene.
// // bool occluded(const Scene &scene, const Ray &ray);
// __host__ __device__ bool occluded(const cuda_bvh &bvh, const Shape *shapes, const Ray &ray);

// /// Computes the emission at a path vertex v, with the viewing direction
// /// pointing outwards of the intersection.
// __host__ __device__ Spectrum emission(const PathVertex &v,
//                   const Vector3 &view_dir,
//                   const cudaScene &scene);

#include "material.h"
#include "ray.h"
#include "scene.h"
#include "shape.h"

__host__ __device__ inline std::optional<PathVertex> intersect(const cuda_bvh &bvh, 
                                                        const Shape *shapes, 
                                                        const Ray &ray, 
                                                        const RayDifferential &ray_diff){
    double t_min = INFINITY;
    int hit_shape_idx = -1;
    int hit_primitive = -1;
    Real u = 0, v = 0;
    
    if (!bvh.intersect(ray, shapes, t_min, hit_shape_idx, hit_primitive, u, v)) {
        return {};
    }

    PathVertex vertex;
    vertex.position = ray.org + t_min * ray.dir ;
    vertex.shape_id = hit_shape_idx;
    vertex.primitive_id = hit_primitive;
    const Shape &shape = shapes[vertex.shape_id];
    vertex.material_id = get_material_id(shape);
    vertex.interior_medium_id = get_interior_medium_id(shape);
    vertex.exterior_medium_id = get_exterior_medium_id(shape);
    vertex.st = Vector2{u, v};
    
    ShadingInfo shading_info = compute_shading_info(shape, vertex.geometric_normal, vertex.st, vertex.primitive_id);
    vertex.shading_frame = shading_info.shading_frame;
    vertex.uv = shading_info.uv;
    vertex.mean_curvature = shading_info.mean_curvature;
    vertex.ray_radius = transfer(ray_diff, distance(ray.org, vertex.position));
    vertex.uv_screen_size = vertex.ray_radius / shading_info.inv_uv_size;

    if (dot(vertex.geometric_normal, vertex.shading_frame.n) < 0) {
        vertex.geometric_normal = -vertex.geometric_normal;
    }

    return vertex;
}
__host__ __device__ inline std::optional<PathVertex> intersect(const cuda_bvh &bvh, 
    const Shape *shapes, 
    const Ray &ray){
    return intersect(bvh, shapes, ray, RayDifferential{});
}

__host__ __device__ inline bool occluded(const cuda_bvh &bvh, const Shape *shapes, const Ray &ray) {
    Real t_min = INFINITY;
    int hit_shape_idx = -1;
    int hit_primitive = -1;
    Real u = 0, v = 0;
    return bvh.intersect(ray, shapes, t_min, hit_shape_idx, hit_primitive, u, v);
}

__host__ __device__ inline Spectrum emission(const PathVertex &v,
                  const Vector3 &view_dir,
                  const Shape* shapes,
                  const Light* lights,
                  const TexturePool texture_pool) {
    int light_id = get_area_light_id(shapes[v.shape_id]);
    assert(light_id >= 0);
    const Light &light = lights[light_id];
    return emission(light,
                    view_dir,
                    v.uv_screen_size,
                    PointAndNormal{v.position, v.geometric_normal},
                    texture_pool);
}

