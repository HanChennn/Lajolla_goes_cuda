#pragma once

#include "lajolla.h"
#include "frame.h"
#include "point_and_normal.h"
#include "ray.h"
#include "table_dist.h"
#include "vector.h"
#include <variant>
#include <optional>

struct ShadingInfo {
    Vector2 uv; // UV coordinates for texture mapping
    Frame shading_frame; // the coordinate basis for shading
    Real mean_curvature; // 0.5 * (dN/du + dN/dv)
    // Stores min(length(dp/du), length(dp/dv)), for ray differentials.
    Real inv_uv_size;
};

/// A Shape is a geometric entity that describes a surface. E.g., a sphere, a triangle mesh, a NURBS, etc.
/// For each shape, we also store an integer "material ID" that points to a material, and an integer
/// "area light ID" that points to a light source if the shape is an area light. area_lightID is set to -1
/// if the shape is not an area light.
struct ShapeBase {
    int material_id = -1;
    int area_light_id = -1;
    int shape_id = -1;
};

struct Sphere : public ShapeBase {
    Vector3 position;
    Real radius;
};

struct TriangleMesh : public ShapeBase {
    CUArray<Vector3> positions;
    CUArray<Vector3i> indices;
    CUArray<Vector3> normals;
    CUArray<Vector2> uvs;
    /// Below are used only when the mesh is associated with an area light
    Real total_area;
    /// For sampling a triangle based on its area
    TableDist1D triangle_sampler;
};

// To add more shapes, first create a struct for the shape, add it to the variant below,
// then implement all the relevant functions below.
using Shape = std::variant<Sphere, TriangleMesh>;

/// Sample a point on the surface given a reference point.
/// uv & w are uniform random numbers.
struct sample_point_on_shape_op {
    __device__ inline PointAndNormal operator()(const Sphere &sphere) const;
    __device__ inline PointAndNormal operator()(const TriangleMesh &mesh) const;

    const Vector3 &ref_point;
    const Vector2 &uv; // for selecting a point on a 2D surface
    const Real &w; // for selecting triangles
};

struct surface_area_op {
    __device__ inline Real operator()(const Sphere &sphere) const;
    __device__ inline Real operator()(const TriangleMesh &mesh) const;
};

/// Probability density of the operation above
struct pdf_point_on_shape_op {
    __device__ inline Real operator()(const Sphere &sphere) const;
    __device__ inline Real operator()(const TriangleMesh &mesh) const;

    const PointAndNormal &point_on_shape;
    const Vector3 &ref_point;
};

struct compute_shading_info_op {
    __device__ inline ShadingInfo operator()(const Sphere &sphere) const;
    __device__ inline ShadingInfo operator()(const TriangleMesh &mesh) const;

    const PathVertex &vertex;
};

struct intersect_op {
    __device__ inline std::optional<PathVertex> operator()(const Sphere &sphere) const;
    __device__ inline std::optional<PathVertex> operator()(const TriangleMesh &mesh) const;

    const Ray &ray;
    const RayDifferential &ray_diff;
};

struct occluded_op {
    __device__ inline bool operator()(const Sphere &sphere) const;
    __device__ inline bool operator()(const TriangleMesh &mesh) const;

    const Ray &ray;
};

#include "shapes/sphere.inl"
#include "shapes/triangle_mesh.inl"

__device__ inline PointAndNormal sample_point_on_shape(const Shape &shape,
                                     const Vector3 &ref_point,
                                     const Vector2 &uv,
                                     Real w) {
    return std::visit(sample_point_on_shape_op{ref_point, uv, w}, shape);
}

__device__ inline Real pdf_point_on_shape(const Shape &shape,
                        const PointAndNormal &point_on_shape,
                        const Vector3 &ref_point) {
    return std::visit(pdf_point_on_shape_op{point_on_shape, ref_point}, shape);
}

__device__ inline Real surface_area(const Shape &shape) {
    return std::visit(surface_area_op{}, shape);
}

__device__ inline ShadingInfo compute_shading_info(const Shape &shape, const PathVertex &vertex) {
    return std::visit(compute_shading_info_op{vertex}, shape);
}

__device__ inline int get_material_id(const Shape &shape) {
    return std::visit([&](const auto &s) { return s.material_id; }, shape);
}

__device__ inline int get_area_light_id(const Shape &shape) {
    return std::visit([&](const auto &s) { return s.area_light_id; }, shape);
}

__device__ inline bool is_light(const Shape &shape) {
    return get_area_light_id(shape) >= 0;
}
