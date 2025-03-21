// #include "intersection.h"
// #include "material.h"
// #include "ray.h"
// #include "scene.h"
// #include "shape.h"

// __host__ __device__ std::optional<PathVertex> intersect(const cuda_bvh &bvh, 
//                                                         const Shape *shapes, 
//                                                         const Ray &ray, 
//                                                         const RayDifferential &ray_diff){
//     double t_min = INFINITY;
//     int hit_shape_idx = -1;
//     int hit_primitive = -1;
//     Real u = 0, v = 0;
    
//     if (!bvh.intersect(ray, shapes, t_min, hit_shape_idx, hit_primitive, u, v)) {
//         return {};
//     }

//     PathVertex vertex;
//     vertex.position = ray.org + t_min * ray.dir ;
//     vertex.shape_id = hit_shape_idx;
//     vertex.primitive_id = hit_primitive;
//     const Shape &shape = shapes[vertex.shape_id];
//     vertex.material_id = get_material_id(shape);
//     vertex.interior_medium_id = get_interior_medium_id(shape);
//     vertex.exterior_medium_id = get_exterior_medium_id(shape);
//     vertex.st = Vector2{u, v};
    
//     ShadingInfo shading_info = compute_shading_info(shape, vertex);
//     vertex.shading_frame = shading_info.shading_frame;
//     vertex.uv = shading_info.uv;
//     vertex.mean_curvature = shading_info.mean_curvature;
//     vertex.ray_radius = transfer(ray_diff, distance(ray.org, vertex.position));
//     vertex.uv_screen_size = vertex.ray_radius / shading_info.inv_uv_size;

//     if (dot(vertex.geometric_normal, vertex.shading_frame.n) < 0) {
//         vertex.geometric_normal = -vertex.geometric_normal;
//     }

//     return vertex;
// }

// __host__ __device__ bool occluded(const cuda_bvh &bvh, const Shape *shapes, const Ray &ray) {
//     Real t_min = INFINITY;
//     int hit_shape_idx = -1;
//     int hit_primitive = -1;
//     Real u = 0, v = 0;
//     return bvh.intersect(ray, shapes, t_min, hit_shape_idx, hit_primitive, u, v);
// }

// __host__ __device__ Spectrum emission(const PathVertex &v,
//                   const Vector3 &view_dir,
//                   const cudaScene &scene) {
//     int light_id = get_area_light_id(scene.shapes[v.shape_id]);
//     assert(light_id >= 0);
//     const Light &light = scene.lights[light_id];
//     return emission(light,
//                     view_dir,
//                     v.uv_screen_size,
//                     PointAndNormal{v.position, v.geometric_normal},
//                     scene);
// }
