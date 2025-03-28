#pragma once

#include "lajolla.h"
#include "texture.h"
#include "parse_scene.h"
#include "parsed_shape.h"
#include "parsed_light.h"
#include "parsed_table_dist.h"

namespace parser
{
struct init_sampling_dist_on_light_op {
    void operator()(DiffuseAreaLight &light) const;
    void operator()(Envmap &light) const;

    const Scene &scene;
};

void init_sampling_dist_on_light_op::operator()(DiffuseAreaLight &light) const {
}

void init_sampling_dist_on_light_op::operator()(Envmap &light) const {
    if (auto *t = std::get_if<ImageTexture<Spectrum>>(&light.values)) {
        // Only need to initialize sampling distribution
        // if the envmap is an image.
        const Mipmap3 &mipmap = get_img(*t, scene.texture_pool);
        int w = get_width(mipmap), h = get_height(mipmap);
        std::vector<Real> f(w * h);
        int i = 0;
        for (int y = 0; y < h; y++) {
            // We shift the grids by 0.5 pixels because we are approximating
            // a piecewise bilinear distribution with a piecewise constant
            // distribution. This shifting is necessary to make the sampling
            // unbiased, as we can interpolate at a position of a black pixel
            // and get a non-zero contribution.
            Real v = (y + Real(0.5)) / Real(h);
            Real sin_elevation = sin(c_PI * v);
            for (int x = 0; x < w; x++) {
                Real u = (x + Real(0.5)) / Real(w);
                f[i++] = luminance(lookup(mipmap, u, v, 0)) * sin_elevation;
            }
        }
        light.sampling_dist = make_table_dist_2d(f, w, h);
    }
}

void init_sampling_dist(Light &light, const Scene &scene) {
    return std::visit(init_sampling_dist_on_light_op{scene}, light);
}

struct light_power_op {
    Real operator()(const DiffuseAreaLight &light) const;
    Real operator()(const Envmap &light) const;

    const Scene &scene;
};

Real light_power_op::operator()(const DiffuseAreaLight &light) const {
    return luminance(light.intensity) * surface_area(scene.shapes[light.shape_id]) * c_PI;
}

Real light_power_op::operator()(const Envmap &light) const {
    return c_PI * scene.bounds.radius * scene.bounds.radius *
           light.sampling_dist.total_values /
           (light.sampling_dist.width * light.sampling_dist.height);
}

Real light_power(const Light &light, const Scene &scene) {
    return std::visit(light_power_op{scene}, light);
}

}