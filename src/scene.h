#pragma once

#include "lajolla.h"
#include "camera.h"
#include "light.h"
#include "material.h"
#include "medium.h"
#include "shape.h"
#include "volume.h"
#include "bvh.h"
#include "AABB.h"
#include "parse_scene.h"

#include <memory>
#include <vector>

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
    Scene(const Camera &camera,
          const std::vector<Material> &materials,
          const std::vector<ParsedShape> &shapes,
          const std::vector<ParsedLight> &lights,
          const std::vector<Medium> &media,
          int envmap_light_id, /* -1 if the scene has no envmap */
          const ParsedTexturePool &texture_pool,
          const RenderOptions &options,
          const std::string &output_filename);
    ~Scene();
    Scene(const Scene& t) = delete;
    Scene& operator=(const Scene& t) = delete;

    // We decide to maintain a copy of the scene here.
    // This allows us to manage the memory of the scene ourselves and decouple
    // from the scene parser, but it's obviously less efficient.
    Camera camera;
    // For now we use stl vectors to store scene content.
    // This wouldn't work if we want to extend this to run on GPUs.
    // If we want to port this to GPUs later, we need to maintain a thrust vector or something similar.
    const std::vector<Material> materials;
    const std::vector<ParsedShape> shapes;
    const std::vector<ParsedLight> lights;
    const std::vector<Medium> media;
    int envmap_light_id;
    const ParsedTexturePool texture_pool;

    // Bounding sphere of the scene.
    BSphere bounds;
    
    RenderOptions options;
    std::string output_filename;

    // For sampling lights
    ParsedTableDist1D light_dist;
};

struct cudaScene{
    int len_materials, len_shapes, len_lights, len_media;
    Material *materials;
    Shape *shapes;
    Light *lights;
    Medium *media;
    int envmap_light_id;
    TexturePool texture_pool;
    BSphere bounds;
    Camera camera;

    RenderOptions options;
    TableDist1D light_dist;
    cuda_bvh bvh;
};

/// Sample a light source from the scene given a random number u \in [0, 1]
int sample_light(const Scene &scene, Real u);
// __device__ int sample_light(const cudaScene &scene, Real u);
__host__ __device__ inline int sample_light(const cudaScene &scene, Real u) {
    return sample_1d(scene.light_dist, u);
}

/// The probability mass function of the sampling procedure above.
Real light_pmf(const Scene &scene, int light_id);
// __device__ Real light_pmf(const cudaScene &scene, int light_id);
__host__ __device__ inline Real light_pmf(const cudaScene &scene, int light_id) {
    return pmf_1d(scene.light_dist, light_id);
}

inline bool has_envmap(const Scene &scene) {
    return scene.envmap_light_id != -1;
}

inline const ParsedLight &get_envmap(const Scene &scene) {
    assert(scene.envmap_light_id != -1);
    return scene.lights[scene.envmap_light_id];
}

inline Real get_shadow_epsilon(const Scene &scene) {
    return min(scene.bounds.radius * Real(1e-5), Real(0.01));
}

inline Real get_intersection_epsilon(const Scene &scene) {
    return min(scene.bounds.radius * Real(1e-5), Real(0.01));
}

__device__ inline bool has_envmap(const cudaScene &scene) {
    return scene.envmap_light_id != -1;
}

__device__ inline const Light &get_envmap(const cudaScene &scene) {
    assert(scene.envmap_light_id != -1);
    return scene.lights[scene.envmap_light_id];
}

__device__ inline Real get_shadow_epsilon(const cudaScene &scene) {
    return min(scene.bounds.radius * Real(1e-5), Real(0.01));
}

__device__ inline Real get_intersection_epsilon(const cudaScene &scene) {
    return min(scene.bounds.radius * Real(1e-5), Real(0.01));
}