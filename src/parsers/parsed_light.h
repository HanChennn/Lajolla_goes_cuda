#pragma once

#include "lajolla.h"
#include "texture.h"
#include "matrix.h"
#include "light.h"
#include "parsed_shape.h"
#include "parsed_table_dist.h"

namespace parser
{
struct Scene;

struct Envmap {
    Texture<Spectrum> values;
    Matrix4x4 to_world, to_local;
    Real scale;

    // For sampling a point on the envmap
    TableDist2D sampling_dist;
};

using Light = std::variant<DiffuseAreaLight, Envmap>;

inline void set_area_light_id(Shape &shape, int area_light_id) {
    std::visit([&](auto &s) { s.area_light_id = area_light_id; }, shape);
}
inline int get_area_light_id(const Shape &shape) {
    return std::visit([&](const auto &s) { return s.area_light_id; }, shape);
}
inline bool is_light(const Shape &shape) {
    return get_area_light_id(shape) >= 0;
}

void init_sampling_dist(Light &light, const Scene &scene);

Real light_power(const Light &light, const Scene &scene);
}