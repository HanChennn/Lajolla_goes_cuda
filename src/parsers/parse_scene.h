#pragma once

#include "lajolla.h"
#include "scene.h"
#include "parsed_light.h"
#include "parsed_shape.h"
#include "parsed_texture.h"
#include "parsed_table_dist.h"
#include <string>
#include <memory>

namespace parser 
{

struct Scene {
    Scene() {}
    Scene(const Camera &camera,
            const std::vector<Material> &materials,
            const std::vector<Shape> &shapes,
            const std::vector<Light> &lights,
            int envmap_light_id, /* -1 if the scene has no envmap */
            const TexturePool &texture_pool,
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
    const std::vector<Shape> shapes;
    const std::vector<Light> lights;
    int envmap_light_id;
    const TexturePool texture_pool;

    // Bounding sphere of the scene.
    BSphere bounds;
    
    RenderOptions options;
    std::string output_filename;

    // For sampling lights
    TableDist1D light_dist;
};

/// Parse Mitsuba's XML scene format.
std::unique_ptr<Scene> parse_scene(const fs::path &filename);

}