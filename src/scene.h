#pragma once

#include "lajolla.h"
#include "camera.h"
#include "light.h"
#include "material.h"
#include "shape.h"

namespace parser{
    struct Scene;
}

enum class Integrator {
    Depth,
    ShadingNormal,
    MeanCurvature,
    RayDifferential, // visualize radius & spread
    MipmapLevel,
    Path,
    VolPath
};

struct RenderOptions {
    Integrator integrator = Integrator::Path;
    int samples_per_pixel = 4;
    int max_depth = -1;
    int rr_depth = 5;
    int vol_path_version = 0;
    int max_null_collisions = 1000;
};

/// Bounding sphere
struct BSphere {
    Real radius;
    Vector3 center;
};

/// A "Scene" contains the camera, materials, geometry (shapes), lights,
/// and also the rendering options such as number of samples per pixel or
/// the parameters of our renderer.
struct Scene {
    Scene() {}
    Scene(const parser::Scene& parsed_scene);
    ~Scene();
    Scene(const Scene& t) = delete;
    Scene& operator=(const Scene& t) = delete;

    Camera camera;
    CUArray<Material> materials;
    CUArray<Shape> shapes;
    CUArray<Light> lights;
    int envmap_light_id;
    TexturePool texture_pool;
    BSphere bounds;
    RenderOptions options;
    TableDist1D light_dist;
};

/// Sample a light source from the scene given a random number u \in [0, 1]
__device__ inline int sample_light(const Scene &scene, Real u) {
    return sample(scene.light_dist, u);
}

/// The probability mass function of the sampling procedure above.
__device__ inline Real light_pmf(const Scene &scene, int light_id) {
    return pmf(scene.light_dist, light_id);
}

__device__ inline bool has_envmap(const Scene &scene) {
    return scene.envmap_light_id != -1;
}

__device__ inline const Light &get_envmap(const Scene &scene) {
    assert(scene.envmap_light_id != -1);
    return scene.lights[scene.envmap_light_id];
}

__device__ inline Real get_shadow_epsilon(const Scene &scene) {
    return min(scene.bounds.radius * Real(1e-5), Real(0.01));
}

__device__ inline Real get_intersection_epsilon(const Scene &scene) {
    return min(scene.bounds.radius * Real(1e-5), Real(0.01));
}
