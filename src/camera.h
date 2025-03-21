#pragma once

#include "lajolla.h"
#include "transform.h"
#include "filter.h"
#include "matrix.h"
#include "vector.h"
#include "ray.h"

/// Currently we only support a pinhole perspective camera
struct Camera {
    Camera() = default;
    __host__ Camera(const Matrix4x4 &cam_to_world,
           Real fov, // in degree
           int width, int height,
           const Filter &filter);

    Matrix4x4 sample_to_cam, cam_to_sample;
    Matrix4x4 cam_to_world, world_to_cam;
    int width, height;
    Filter filter;
};

/// Given screen position in [0, 1] x [0, 1],
/// generate a camera ray.
__device__ inline Ray sample_primary(const Camera &camera,
                                     const Vector2 &screen_pos) {
     // screen_pos' domain is [0, 1]^2
     Vector2 pixel_pos{screen_pos.x * camera.width, screen_pos.y * camera.height};
 
     // Importance sample from the pixel filter (see filter.h for more explanation).
     // We first extract the subpixel offset.
     Real dx = pixel_pos.x - floor(pixel_pos.x);
     Real dy = pixel_pos.y - floor(pixel_pos.y);
     // dx/dy are uniform variables in [0, 1],
     // so let's remap them using importance sampling.
     Vector2 offset = sample(camera.filter, Vector2{dx, dy});
     // Filters are placed at pixel centers.
     Vector2 remapped_pos{
       (floor(pixel_pos.x) + Real(0.5) + offset.x) / camera.width,
       (floor(pixel_pos.y) + Real(0.5) + offset.y) / camera.height};
 
     Vector3 pt = xform_point(camera.sample_to_cam,
         Vector3(remapped_pos[0], remapped_pos[1], Real(0)));
     Vector3 dir = normalize(pt);
     return Ray{xform_point(camera.cam_to_world, Vector3{0, 0, 0}),
                // the last normalize might not be necessary
                normalize(xform_vector(camera.cam_to_world, dir)),
                Real(0), infinity<Real>()};
 }