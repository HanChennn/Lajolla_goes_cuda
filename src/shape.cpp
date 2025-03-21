#include "shape.h"
#include "intersection.h"
#include "point_and_normal.h"
#include "ray.h"

struct Parsed_sample_point_on_shape_op {
    PointAndNormal operator()(const Sphere &sphere) const;
    PointAndNormal operator()(const ParsedTriangleMesh &mesh) const;

    const Vector3 &ref_point;
    const Vector2 &uv; // for selecting a point on a 2D surface
    const Real &w; // for selecting triangles
};

//////////////////////////////////////////////////////////////////////////////////////////
struct Parsed_surface_area_op {
    Real operator()(const Sphere &sphere) const;
    Real operator()(const ParsedTriangleMesh &mesh) const;
};
//////////////////////////////////////////////////////////////////////////////////////////
struct Parsed_pdf_point_on_shape_op {
    Real operator()(const Sphere &sphere) const;
    Real operator()(const ParsedTriangleMesh &mesh) const;

    const PointAndNormal &point_on_shape;
    const Vector3 &ref_point;
};
//////////////////////////////////////////////////////////////////////////////////////////
struct init_sampling_dist_op {
    void operator()(Sphere &sphere) const;
    void operator()(ParsedTriangleMesh &mesh) const;
};
//////////////////////////////////////////////////////////////////////////////////////////
struct Parsed_compute_shading_info_op {
    ShadingInfo operator()(const Sphere &sphere) const;
    ShadingInfo operator()(const ParsedTriangleMesh &mesh) const;

    const PathVertex &vertex;
};

#include "shapes/sphere.inl"
#include "shapes/triangle_mesh.inl"

PointAndNormal sample_point_on_shape(const ParsedShape &shape,
    const Vector3 &ref_point,
    const Vector2 &uv,
    Real w) {
    return std::visit(Parsed_sample_point_on_shape_op{ref_point, uv, w}, shape);
}

Real pdf_point_on_shape(const ParsedShape &shape,
                        const PointAndNormal &point_on_shape,
                        const Vector3 &ref_point) {
    return std::visit(Parsed_pdf_point_on_shape_op{point_on_shape, ref_point}, shape);
}

Real surface_area(const ParsedShape &shape) {
    return std::visit(Parsed_surface_area_op{}, shape);
}

void init_sampling_dist(ParsedShape &shape) {
    return std::visit(init_sampling_dist_op{}, shape);
}

ShadingInfo compute_shading_info(const ParsedShape &shape, const PathVertex &vertex) {
    return std::visit(Parsed_compute_shading_info_op{vertex}, shape);
}


void set_material_id(ParsedShape &shape, int material_id) {
    std::visit([&](auto &s) { s.material_id = material_id; }, shape);
}
void set_area_light_id(ParsedShape &shape, int area_light_id) {
    std::visit([&](auto &s) { s.area_light_id = area_light_id; }, shape);
}
void set_interior_medium_id(ParsedShape &shape, int interior_medium_id) {
    std::visit([&](auto &s) { s.interior_medium_id = interior_medium_id; }, shape);
}
void set_exterior_medium_id(ParsedShape &shape, int exterior_medium_id) {
    std::visit([&](auto &s) { s.exterior_medium_id = exterior_medium_id; }, shape);
}


int get_material_id(const ParsedShape &shape) {
    return std::visit([&](const auto &s) { return s.material_id; }, shape);
}
int get_area_light_id(const ParsedShape &shape) {
    return std::visit([&](const auto &s) { return s.area_light_id; }, shape);
}
int get_interior_medium_id(const ParsedShape &shape) {
    return std::visit([&](const auto &s) { return s.interior_medium_id; }, shape);
}
int get_exterior_medium_id(const ParsedShape &shape) {
    return std::visit([&](const auto &s) { return s.exterior_medium_id; }, shape);
}
bool is_light(const ParsedShape &shape) {
    return get_area_light_id(shape) >= 0;
}