#include "scene.h"
#include "table_dist.h"
#include "parsers/parse_scene.h"
#include "parsers/parsed_shape.h"
#include "parsers/parsed_light.h"
#include "parsers/parsed_texture.h"
#include "parsers/parsed_table_dist.h"

Scene::Scene(const parser::Scene& parsed_scene) : 
        camera(parsed_scene.camera), 
        envmap_light_id(parsed_scene.envmap_light_id),
        bounds(parsed_scene.bounds), options(parsed_scene.options) {

    materials.init(parsed_scene.materials);
    
    std::vector<Shape> shapes_vector;
    for(auto& shape : parsed_scene.shapes){
        if(auto *s = std::get_if<Sphere>(&shape))
        {
            shapes_vector.push_back(*s);
        }
        else if(auto *s = std::get_if<parser::TriangleMesh>(&shape))
        {
            TriangleMesh mesh;
            mesh.area_light_id = s->area_light_id;
            mesh.material_id = s->material_id;
            mesh.shape_id = s->shape_id;
            mesh.total_area = s->total_area;
            mesh.positions.init(s->positions);
            mesh.indices.init(s->indices);
            mesh.normals.init(s->normals);
            mesh.uvs.init(s->uvs);
            mesh.triangle_sampler.cdf.init(s->triangle_sampler.cdf);
            mesh.triangle_sampler.pmf.init(s->triangle_sampler.pmf);

            shapes_vector.push_back(std::move(mesh));
        }
    }
    shapes.init(shapes_vector);

    std::vector<Light> lights_vector;
    for(auto& light : parsed_scene.lights){
        if(auto *l = std::get_if<DiffuseAreaLight>(&light))
        {
            lights_vector.push_back(*l);
        }
        else if(auto *l = std::get_if<parser::Envmap>(&light))
        {
            Envmap envmap;
            envmap.values = l->values;
            envmap.to_local = l->to_local;
            envmap.to_world = l->to_world;
            envmap.scale = l->scale;
            envmap.sampling_dist.height = l->sampling_dist.height;
            envmap.sampling_dist.width = l->sampling_dist.width;
            envmap.sampling_dist.total_values = l->sampling_dist.total_values;
            envmap.sampling_dist.cdf_marginals.init(l->sampling_dist.cdf_marginals);
            envmap.sampling_dist.cdf_rows.init(l->sampling_dist.cdf_rows);
            envmap.sampling_dist.pdf_marginals.init(l->sampling_dist.pdf_marginals);
            envmap.sampling_dist.pdf_rows.init(l->sampling_dist.pdf_rows);

            lights_vector.push_back(std::move(envmap));
        }
    }
    lights.init(lights_vector);

    std::vector<Mipmap1> mipmap1_vector;
    for(auto& m : parsed_scene.texture_pool.image1s){
        Mipmap1 mipmap;
        std::vector<CUImage1> img1_vector;
        for(auto& i : m.images){
            CUImage1 img;
            img.width = i.width;
            img.height = i.height;
            img.data.init(i.data);
            img1_vector.push_back(std::move(img));
        }
        mipmap.images.init(img1_vector);
        mipmap1_vector.push_back(mipmap);
    }
    texture_pool.image1s.init(mipmap1_vector);

    std::vector<Mipmap3> mipmap3_vector;
    for(auto& m : parsed_scene.texture_pool.image3s){
        Mipmap3 mipmap;
        std::vector<CUImage3> img3_vector;
        for(auto& i : m.images){
            CUImage3 img;
            img.width = i.width;
            img.height = i.height;
            img.data.init(i.data);
            img3_vector.push_back(std::move(img));
        }
        mipmap.images.init(img3_vector);
        mipmap3_vector.push_back(mipmap);
    }
    texture_pool.image3s.init(mipmap3_vector);

    light_dist.cdf.init(parsed_scene.light_dist.cdf);
    light_dist.pmf.init(parsed_scene.light_dist.pmf);
}

Scene::~Scene() {
}
