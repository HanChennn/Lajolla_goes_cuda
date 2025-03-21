#pragma once

#include "lajolla.h"
#include "frame.h"
#include "table_dist.h"
#include "vector.h"
#include "point_and_normal.h"
#include "intersection.h"
#include <variant>
#include <vector>

struct PointAndNormal;
struct PathVertex;

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

    int interior_medium_id = -1;
    int exterior_medium_id = -1;
};

struct Sphere : public ShapeBase {
    Vector3 position;
    Real radius;
};

struct ParsedTriangleMesh : public ShapeBase {
    /// TODO: make these portable to GPUs
    std::vector<Vector3> positions;
    std::vector<Vector3i> indices;
    std::vector<Vector3> normals;
    std::vector<Vector2> uvs;
    /// Below are used only when the mesh is associated with an area light
    Real total_area;
    /// For sampling a triangle based on its area
    ParsedTableDist1D triangle_sampler;
};

struct TriangleMesh : public ShapeBase {
    /// TODO: make these portable to GPUs
    Vector3 *positions;
    Vector3i *indices;
    Vector3 *normals;
    Vector2 *uvs;
    int len_positions, len_indices, len_normals, len_uvs;
    /// Below are used only when the mesh is associated with an area light
    Real total_area;
    /// For sampling a triangle based on its area
    TableDist1D triangle_sampler;
    
    __host__ __device__ TriangleMesh()
        : positions(nullptr), indices(nullptr), normals(nullptr), uvs(nullptr),
          total_area(0.0f) {}

    __host__ __device__ TriangleMesh(Vector3* pos, Vector3i* idx, Vector3* norm, Vector2* tex, Real area, TableDist1D sampler)
        : positions(pos), indices(idx), normals(norm), uvs(tex),
          total_area(area), triangle_sampler(sampler) {}
};

// To add more shapes, first create a struct for the shape, add it to the variant below,
// then implement all the relevant functions below.
using ParsedShape = std::variant<Sphere, ParsedTriangleMesh>;
using Shape = std::variant<Sphere, TriangleMesh>;

/// Sample a point on the surface given a reference point.
/// uv & w are uniform random numbers.
__host__ PointAndNormal sample_point_on_shape(const ParsedShape &shape,
                                     const Vector3 &ref_point,
                                     const Vector2 &uv,
                                     Real w);

/// Probability density of the operation above

__host__ Real pdf_point_on_shape(const ParsedShape &shape,
                        const PointAndNormal &point_on_shape,
                        const Vector3 &ref_point);

/// Useful for sampling.
__host__ Real surface_area(const ParsedShape &shape);

/// Some shapes require storing sampling data structures inside. This function initialize them.
__host__ void init_sampling_dist(ParsedShape &shape);

ShadingInfo compute_shading_info(const ParsedShape &shape, const PathVertex &vertex);

void set_material_id(ParsedShape &shape, int material_id);
void set_area_light_id(ParsedShape &shape, int area_light_id);
void set_interior_medium_id(ParsedShape &shape, int interior_medium_id);
void set_exterior_medium_id(ParsedShape &shape, int exterior_medium_id);
// __host__ __device__ int get_material_id(const Shape &shape);
// __host__ __device__ int get_area_light_id(const Shape &shape);
// __host__ __device__ int get_interior_medium_id(const Shape &shape);
// __host__ __device__ int get_exterior_medium_id(const Shape &shape);
// __host__ __device__ bool is_light(const Shape &shape);
int get_material_id(const ParsedShape &shape);
int get_area_light_id(const ParsedShape &shape);
int get_interior_medium_id(const ParsedShape &shape);
int get_exterior_medium_id(const ParsedShape &shape);
bool is_light(const ParsedShape &shape);

__host__ __device__ inline int get_material_id(const Shape &shape) {
    return std::visit([&](const auto &s) { return s.material_id; }, shape);
}
__host__ __device__ inline int get_area_light_id(const Shape &shape) {
    return std::visit([&](const auto &s) { return s.area_light_id; }, shape);
}
__host__ __device__ inline int get_interior_medium_id(const Shape &shape) {
    return std::visit([&](const auto &s) { return s.interior_medium_id; }, shape);
}
__host__ __device__ inline int get_exterior_medium_id(const Shape &shape) {
    return std::visit([&](const auto &s) { return s.exterior_medium_id; }, shape);
}
__host__ __device__ inline bool is_light(const Shape &shape) {
    return get_area_light_id(shape) >= 0;
}

struct sample_point_on_shape_op {
    __host__ __device__ inline PointAndNormal operator()(const Sphere &sphere) const;
    __host__ __device__ inline PointAndNormal operator()(const TriangleMesh &mesh) const;

    const Vector3 &ref_point;
    const Vector2 &uv; // for selecting a point on a 2D surface
    const Real &w; // for selecting triangles
};
__host__ __device__ inline PointAndNormal sample_point_on_shape(const Shape &shape,
                                     const Vector3 &ref_point,
                                     const Vector2 &uv,
                                     Real w) {
    // return std::visit(sample_point_on_shape_op{ref_point, uv, w}, shape);
    sample_point_on_shape_op op{ref_point, uv, w};
    if (auto *s = std::get_if<Sphere>(&shape)) return op(*s);
    else if (auto *s = std::get_if<TriangleMesh>(&shape)) return op(*s);
    else return PointAndNormal();  // 处理未知情况，返回默认值
}

struct pdf_point_on_shape_op {
    __host__ __device__ inline Real operator()(const Sphere &sphere) const;
    __host__ __device__ inline Real operator()(const TriangleMesh &mesh) const;

    const PointAndNormal &point_on_shape;
    const Vector3 &ref_point;
};
__host__ __device__ inline Real pdf_point_on_shape(const Shape &shape,
                        const PointAndNormal &point_on_shape,
                        const Vector3 &ref_point) {
    // return std::visit(pdf_point_on_shape_op{point_on_shape, ref_point}, shape);
    pdf_point_on_shape_op op{point_on_shape, ref_point};
    if (auto *s = std::get_if<Sphere>(&shape)) return op(*s);
    else if (auto *s = std::get_if<TriangleMesh>(&shape)) return op(*s);
    else return Real(0.0);  // 处理未知情况，返回默认 PDF
}


struct surface_area_op {
    __host__ __device__ inline Real operator()(const Sphere &sphere) const;
    __host__ __device__ inline Real operator()(const TriangleMesh &mesh) const;
};
__host__ __device__ inline Real surface_area(const Shape &shape) {
    // return std::visit(surface_area_op{}, shape);
    surface_area_op op{};
    if (auto *s = std::get_if<Sphere>(&shape)) return op(*s);
    else if (auto *s = std::get_if<TriangleMesh>(&shape)) return op(*s);
    else return Real(0.0);  // 处理未知情况，返回默认面积
}


struct compute_shading_info_op {
    __host__ __device__ inline ShadingInfo operator()(const Sphere &sphere) const;
    __host__ __device__ inline ShadingInfo operator()(const TriangleMesh &mesh) const;

    // const PathVertex &vertex;
    const Vector3 geometric_normal;
    const Vector2 st;
    const int primitive_id;
};
__host__ __device__ inline ShadingInfo compute_shading_info(const Shape &shape, const Vector3 geometric_normal, const Vector2 st, const int primitive_id) {
    // return std::visit(compute_shading_info_op{geometric_normal, st, primitive_id}, shape);
    compute_shading_info_op op{geometric_normal, st, primitive_id};
    if (auto *s = std::get_if<Sphere>(&shape))  return op(*s);
    else if (auto *s = std::get_if<TriangleMesh>(&shape))return op(*s);
    else return ShadingInfo{};  // 处理未知情况，返回默认 ShadingInfo
}

__host__ __device__ inline PointAndNormal sample_point_on_shape_op::operator()(const Sphere &sphere) const {
    // https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#x2-SamplingSpheres
    const Vector3 &center = sphere.position;
    const Real &r = sphere.radius;

    if (distance_squared(ref_point, center) < r * r) {
        // If the reference point is inside the sphere, just sample the whole sphere uniformly
        Real z = 1 - 2 * uv.x;
        Real r_ = sqrt(fmax(Real(0), 1 - z * z));
        Real phi = 2 * c_PI * uv.y;
        Vector3 offset(r_ * cos(phi), r_ * sin(phi), z);
        Vector3 position = center + r * offset;
        Vector3 normal = offset;
        return PointAndNormal{position, normal};
    }

    // Otherwise sample a ray inside a cone towards the sphere center.

    // Build a coordinate system with n pointing towards the sphere
    Vector3 dir_to_center = normalize(center - ref_point);
    Frame frame(dir_to_center);

    // These are not exactly "elevation" and "azimuth": elevation here
    // stands for the extended angle of the cone, and azimuth here stands
    // for the polar coordinate angle on the substended disk.
    // I just don't like the theta/phi naming convention...
    Real sin_elevation_max_sq = r * r / distance_squared(ref_point, center);
    Real cos_elevation_max = sqrt(max(Real(0), 1 - sin_elevation_max_sq));
    // Uniformly interpolate between 1 (angle 0) and max
    Real cos_elevation = (1 - uv[0]) + uv[0] * cos_elevation_max;
    Real sin_elevation = sqrt(max(Real(0), 1 - cos_elevation * cos_elevation));
    Real azimuth = uv[1] * 2 * c_PI;

    // Now we have a ray direction and a sphere, we can just ray trace and find
    // the intersection point. Pbrt uses an more clever and numerically robust
    // approach which I will just shamelessly copy here.
    Real dc = distance(ref_point, center);
    Real ds = dc * cos_elevation -
        sqrt(max(Real(0), r * r - dc * dc * sin_elevation * sin_elevation));
    Real cos_alpha = (dc * dc + r * r - ds * ds) / (2 * dc * r);
    Real sin_alpha = sqrt(max(Real(0), 1 - cos_alpha * cos_alpha));
    // Add negative sign since normals point outwards.
    Vector3 n_on_sphere = -to_world(frame,
        Vector3{sin_alpha * cos(azimuth),
                sin_alpha * sin(azimuth),
                cos_alpha});
    Vector3 p_on_sphere = r * n_on_sphere + center;
    return PointAndNormal{p_on_sphere, n_on_sphere};
}
__host__ __device__ inline Real surface_area_op::operator()(const Sphere &sphere) const {
    return 4 * c_PI * sphere.radius * sphere.radius;
}
__host__ __device__ inline Real pdf_point_on_shape_op::operator()(const Sphere &sphere) const {
    // https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#x2-SamplingSpheres
    const Vector3 &center = sphere.position;
    const Real &r = sphere.radius;

    if (distance_squared(ref_point, center) < r * r) {
        // If the reference point is inside the sphere, just sample the whole sphere uniformly
        return 1 / surface_area_op{}(sphere);
    }
    
    Real sin_elevation_max_sq = r * r / distance_squared(ref_point, center);
    Real cos_elevation_max = sqrt(max(Real(0), 1 - sin_elevation_max_sq));
    // Uniform sampling PDF of a cone.
    Real pdf_solid_angle = 1 / (2 * c_PI * (1 - cos_elevation_max));
    // Convert it back to area measure
    Vector3 p_on_sphere = point_on_shape.position;
    Vector3 n_on_sphere = point_on_shape.normal;
    Vector3 dir = normalize(p_on_sphere - ref_point);
    return pdf_solid_angle * fabs(dot(n_on_sphere, dir)) /
        distance_squared(ref_point, p_on_sphere);
}
__host__ __device__ inline ShadingInfo compute_shading_info_op::operator()(const Sphere &sphere) const {
    // To compute the shading frame, we use the geometry normal as normal,
    // and dpdu as one of the tangent vector. 
    // We use the azimuthal angle as u, and the elevation as v, 
    // thus the point p on sphere and u, v has the following relationship:
    // p = center + {r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v)}
    // thus dpdu = {-r * sin(u) * sin(v), r * cos(u) * sin(v), 0}
    //      dpdv = { r * cos(u) * cos(v), r * sin(u) * cos(v), - r * sin(v)}
    Vector3 dpdu{-sphere.radius * sin(st[0]) * sin(st[1]),
                  sphere.radius * cos(st[0]) * sin(st[1]),
                 Real(0)};
    Vector3 dpdv{ sphere.radius * cos(st[0]) * cos(st[1]),
                  sphere.radius * sin(st[0]) * cos(st[1]),
                 -sphere.radius * sin(st[1])};
    // dpdu may not be orthogonal to shading normal:
    // subtract the projection of shading_normal onto dpdu to make them orthogonal
    Vector3 tangent = normalize(
        dpdu - geometric_normal * dot(geometric_normal, dpdu));
    Frame shading_frame(tangent,
                        normalize(cross(geometric_normal, tangent)),
                        geometric_normal);
    return ShadingInfo{st,
                       shading_frame,
                       1 / sphere.radius, /* mean curvature */
                       (length(dpdu) + length(dpdv)) / 2};
}
__host__ __device__ inline PointAndNormal sample_point_on_shape_op::operator()(const TriangleMesh &mesh) const {
    int tri_id = sample_1d(mesh.triangle_sampler, w);
    assert(tri_id >= 0 && tri_id < (int)mesh.len_indices);
    Vector3i index = mesh.indices[tri_id];
    Vector3 v0 = mesh.positions[index[0]];
    Vector3 v1 = mesh.positions[index[1]];
    Vector3 v2 = mesh.positions[index[2]];
    Vector3 e1 = v1 - v0;
    Vector3 e2 = v2 - v0;
    // https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#SamplingaTriangle
    Real a = sqrt(std::clamp(uv[0], Real(0), Real(1)));
    Real b1 = 1 - a;
    Real b2 = a * uv[1];
    Vector3 geometric_normal = normalize(cross(e1, e2));
    // Flip the geometric normal to the same side as the shading normal
    if (mesh.len_normals > 0) {
        Vector3 n0 = mesh.normals[index[0]];
        Vector3 n1 = mesh.normals[index[1]];
        Vector3 n2 = mesh.normals[index[2]];
        Vector3 shading_normal = normalize(
            (1 - b1 - b2) * n0 + b1 * n1 + b2 * n2);
        if (dot(geometric_normal, shading_normal) < 0) {
            geometric_normal = -geometric_normal;
        }
    }
    return PointAndNormal{v0 + (e1 * b1) + (e2 * b2), geometric_normal};
}
__host__ __device__ inline Real surface_area_op::operator()(const TriangleMesh &mesh) const {
    return mesh.total_area;
}
__host__ __device__ inline Real pdf_point_on_shape_op::operator()(const TriangleMesh &mesh) const {
    return 1 / surface_area_op{}(mesh);
}
__host__ __device__ inline ShadingInfo compute_shading_info_op::operator()(const TriangleMesh &mesh) const {
    // Get UVs of the three vertices
    assert(primitive_id >= 0);
    Vector3i index = mesh.indices[primitive_id];
    Vector2 uvs[3];
    if (mesh.len_uvs > 0) {
        uvs[0] = mesh.uvs[index[0]];
        uvs[1] = mesh.uvs[index[1]];
        uvs[2] = mesh.uvs[index[2]];
    } else {
        // Use barycentric coordinates
        uvs[0] = Vector2{0, 0};
        uvs[1] = Vector2{1, 0};
        uvs[2] = Vector2{1, 1};
    }
    // Barycentric coordinates are stored in vertex.st
    Vector2 uv = (1 - st[0] - st[1]) * uvs[0] +
                 st[0] * uvs[1] +
                 st[1] * uvs[2];
    Vector3 p0 = mesh.positions[index[0]],
            p1 = mesh.positions[index[1]],
            p2 = mesh.positions[index[2]];
    // We want to derive dp/du & dp/dv. We have the following
    // relation:
    // p  = (1 - s - t) * p0   + s * p1   + t * p2
    // uv = (1 - s - t) * uvs0 + s * uvs1 + t * uvs2
    // dp/duv = dp/dst * dst/duv = dp/dst * (duv/dst)^{-1}
    // where dp/dst is a 3x2 matrix, duv/dst and dst/duv is a 2x2 matrix,
    // and dp/duv is a 3x2 matrix.

    // Let's build duv/dst first. To be clearer, it is
    // [du/ds, du/dt]
    // [dv/ds, dv/dt]
    Vector2 duvds = uvs[2] - uvs[0];
    Vector2 duvdt = uvs[2] - uvs[1];
    // The inverse of this matrix is
    // (1/det) [ dv/dt, -du/dt]
    //         [-dv/ds,  du/ds]
    // where det = duds * dvdt - dudt * dvds
    Real det = duvds[0] * duvdt[1] - duvdt[0] * duvds[1];
    Real dsdu =  duvdt[1] / det;
    Real dtdu = -duvds[1] / det;
    Real dsdv =  duvdt[0] / det;
    Real dtdv = -duvds[0] / det;
    Vector3 dpdu, dpdv;
    if (fabs(det) > 1e-8f) {
        // Now we just need to do the matrix multiplication
        Vector3 dpds = p2 - p0;
        Vector3 dpdt = p2 - p1;
        dpdu = dpds * dsdu + dpdt * dtdu;
        dpdv = dpds * dsdv + dpdt * dtdv;
    } else {
        // degenerate uvs. Use an arbitrary coordinate system
        // std::tie(dpdu, dpdv) =
        coordinate_system(geometric_normal, dpdu, dpdv);
    }

    // Now let's get the shading normal & mean_curvature.
    // By default it is the geometry normal and we have zero curvature.
    Vector3 shading_normal = geometric_normal;
    Real mean_curvature = 0;
    Vector3 tangent, bitangent;
    // However if we have vertex normals, that overrides the geometry normal.
    if (mesh.len_normals > 0) {
        Vector3 n0 = mesh.normals[index[0]],
                n1 = mesh.normals[index[1]],
                n2 = mesh.normals[index[2]];
        shading_normal = normalize(
            (1 - st[0] - st[1]) * n0 + 
                                st[0] * n1 +
                                st[1] * n2);
        // dpdu may not be orthogonal to shading normal:
        // subtract the projection of shading_normal onto dpdu to make them orthogonal
        tangent = normalize(dpdu - shading_normal * dot(shading_normal, dpdu));

        // We want to compute dn/du & dn/dv for mean curvature.
        // This is computed in a similar way to dpdu.
        // dn/duv = dn/dst * dst/duv = dn/dst * (duv/dst)^{-1}
        Vector3 dnds = n2 - n0;
        Vector3 dndt = n2 - n1;
        Vector3 dndu = dnds * dsdu + dndt * dtdu;
        Vector3 dndv = dnds * dsdv + dndt * dtdv;
        bitangent = normalize(cross(shading_normal, tangent));
        mean_curvature = (dot(dndu, tangent) + 
                          dot(dndv, bitangent)) / Real(2);
    } else {
        tangent = normalize(dpdu - shading_normal * dot(shading_normal, dpdu));
        bitangent = normalize(cross(shading_normal, tangent));
    }

    Frame shading_frame(tangent, bitangent, shading_normal);
    return ShadingInfo{uv, shading_frame, mean_curvature,
                       max(length(dpdu), length(dpdv)) /* inv_uv_size */};
}
