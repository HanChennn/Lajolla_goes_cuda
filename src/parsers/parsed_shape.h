#pragma once

#include "lajolla.h"
#include "shape.h"
#include "parsed_table_dist.h"

namespace parser
{

struct TriangleMesh : public ShapeBase {
    std::vector<Vector3> positions;
    std::vector<Vector3i> indices;
    std::vector<Vector3> normals;
    std::vector<Vector2> uvs;
    /// Below are used only when the mesh is associated with an area light
    Real total_area;
    /// For sampling a triangle based on its area
    TableDist1D triangle_sampler;
};

using Shape = std::variant<Sphere, TriangleMesh>;

struct surface_area_op {
    inline Real operator()(const Sphere &sphere) const;
    inline Real operator()(const TriangleMesh &mesh) const;
};

inline Real surface_area_op::operator()(const Sphere &sphere) const {
    return 4 * c_PI * sphere.radius * sphere.radius;
}

inline Real surface_area_op::operator()(const TriangleMesh &mesh) const {
    return mesh.total_area;
}

inline Real surface_area(const Shape &shape) {
    return std::visit(surface_area_op{}, shape);
}

struct init_sampling_dist_on_shape_op {
    inline void operator()(Sphere &sphere) const;
    inline void operator()(TriangleMesh &mesh) const;
};

inline void init_sampling_dist_on_shape_op::operator()(TriangleMesh &mesh) const {
    std::vector<Real> tri_areas(mesh.indices.size(), Real(0));
    Real total_area = 0;
    for (int tri_id = 0; tri_id < (int)mesh.indices.size(); tri_id++) {
        Vector3i index = mesh.indices[tri_id];
        Vector3 v0 = mesh.positions[index[0]];
        Vector3 v1 = mesh.positions[index[1]];
        Vector3 v2 = mesh.positions[index[2]];
        Vector3 e1 = v1 - v0;
        Vector3 e2 = v2 - v0;
        tri_areas[tri_id] = length(cross(e1, e2)) / 2;
        total_area += tri_areas[tri_id];
    }
    mesh.triangle_sampler = make_table_dist_1d(tri_areas);
    mesh.total_area = total_area;
}

inline void init_sampling_dist_on_shape_op::operator()(Sphere &sphere) const {
}

inline void init_sampling_dist(Shape &shape) {
    return std::visit(init_sampling_dist_on_shape_op{}, shape);
}

inline void set_material_id(Shape &shape, int material_id) {
    std::visit([&](auto &s) { s.material_id = material_id; }, shape);
}

}