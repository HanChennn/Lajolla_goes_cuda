#include "camera.h"
#include "lajolla.h"
#include "transform.h"

#include <cmath>

__host__ __device__ Camera::Camera(const Matrix4x4 &cam_to_world,
               Real fov,
               int width, int height,
               const Filter &filter,
               int medium_id)
    : cam_to_world(cam_to_world),
      world_to_cam(inverse(cam_to_world)),
      width(width), height(height),
      filter(filter), medium_id(medium_id) {
    Real aspect = (Real)width / (Real)height;
    cam_to_sample = scale(Vector3(-Real(0.5), -Real(0.5) * aspect, Real(1.0))) *
                    translate(Vector3(-Real(1.0), -Real(1.0) / aspect, Real(0.0))) *
                    perspective(fov);
    sample_to_cam = inverse(cam_to_sample);
}

