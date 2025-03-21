#include "scene.h"
#include "table_dist.h"
#include "light.h"
#include "parse_scene.h"

Scene::Scene(const Camera &camera,
             const std::vector<Material> &materials,
             const std::vector<ParsedShape> &shapes,
             const std::vector<ParsedLight> &lights,
             const std::vector<Medium> &media,
             int envmap_light_id,
             const ParsedTexturePool &texture_pool,
             const RenderOptions &options,
             const std::string &output_filename) : 
        camera(camera), materials(materials),
        shapes(shapes), lights(lights), media(media),
        envmap_light_id(envmap_light_id),
        texture_pool(texture_pool), options(options),
        output_filename(output_filename) {

    Vector3 lb{0, 0, 0};
    Vector3 ub{0, 0, 0};
    std::vector<ParsedShape> &mod_shapes = const_cast<std::vector<ParsedShape>&>(this->shapes);
    for (ParsedShape &shape : mod_shapes) {
        if(auto *m = std::get_if<Sphere>(&shape))
        {
            lb = min(lb, m->position - m->radius);
            ub = max(ub, m->position + m->radius);
        }
        else if(auto *m = std::get_if<ParsedTriangleMesh>(&shape))
        {
            for (auto& pos : m->positions)
            {
                lb = min(lb, pos);
                ub = max(ub, pos);
            }
        }
    }
    bounds = BSphere{distance(ub, lb) / 2, (lb + ub) / Real(2)};

    // build shape & light sampling distributions if necessary
    // TODO: const_cast is a bit ugly...
    std::vector<ParsedShape> &mod_shapes = const_cast<std::vector<ParsedShape>&>(this->shapes);
    for (ParsedShape &shape : mod_shapes) {
        init_sampling_dist(shape);
    }
    std::vector<ParsedLight> &mod_lights = const_cast<std::vector<ParsedLight>&>(this->lights);
    for (ParsedLight &light : mod_lights) {
        init_sampling_dist(light, *this);
    }

    // build a sampling distributino for all the lights
    std::vector<Real> power(this->lights.size());
    for (int i = 0; i < (int)this->lights.size(); i++) {
        power[i] = light_power(this->lights[i], *this);
    }
    light_dist = make_table_dist_1d(power);
}

Scene::~Scene() {
}

int sample_light(const Scene &scene, Real u) {
    return sample_1d_parsed(scene.light_dist, u);
}

Real light_pmf(const Scene &scene, int light_id) {
    return pmf_1d_parsed(scene.light_dist, light_id);
}
