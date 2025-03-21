#include "bvh.h"
#include "AABB.h"
#include <algorithm>
#include <variant>

__host__ __device__ AABB get_bounds(const ParsedShape &shape) {
    return std::visit([](const auto &s) -> AABB {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>, Sphere>) {
            return AABB(s.position - Vector3(s.radius, s.radius, s.radius), s.position + Vector3(s.radius, s.radius, s.radius));
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>, ParsedTriangleMesh>) {
            AABB bounds;
            for (const auto &pos : s.positions) {
                bounds.expand(pos);
            }
            return bounds;
        }
    }, shape);
}
__host__ __device__ AABB get_bounds(const Shape &shape) {
    return std::visit([](const auto &s) -> AABB {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>, Sphere>) {
            return AABB(s.position - Vector3(s.radius, s.radius, s.radius), s.position + Vector3(s.radius, s.radius, s.radius));
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>, TriangleMesh>) {
            AABB bounds;
            for (int i=0;i<s.len_positions;i++) {
                bounds.expand(s.positions[i]);
            }
            return bounds;
        }
    }, shape);
}

__host__ __device__ bool ray_triangle_intersect(const Ray &ray, const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, Real &t, Real &u, Real &v) {
    const Real EPSILON = 1e-6f;
    Vector3 edge1 = v1 - v0;
    Vector3 edge2 = v2 - v0;
    Vector3 h = cross(ray.dir, edge2);
    Real a = dot(edge1, h);
    if (fabs(a) < EPSILON) return false;

    Real f = 1.0f / a;
    Vector3 s = ray.org - v0;
    u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    Vector3 q = cross(s, edge1);
    v = f * dot(ray.dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    t = f * dot(edge2, q);
    return t > EPSILON;
}



__host__ __device__ void BVH::build(const std::vector<ParsedShape> &shapes) {
    shape_indices.resize(shapes.size());
    for (size_t i = 0; i < shapes.size(); i++) {
        shape_indices[i] = i;
    }
    node_count = 0;
    build_recursive(0, shapes.size(), shapes);
}

__host__ __device__ int BVH::build_recursive(int start, int end, const std::vector<ParsedShape> &shapes) {
    if (node_count >= MAX_BVH_NODES) return -1;

    int node_idx = node_count++;
    BVHNode &node = nodes.emplace_back();

    node.shape_start = start;
    node.shape_count = end - start;
    node.left_child = -1;
    node.right_child = -1;

    AABB bounds;
    for (int i = start; i < end; i++) {
        bounds.expand(get_bounds(shapes[shape_indices[i]]));
    }
    node.bounds = bounds;

    if (node.shape_count <= 2) return node_idx;

    Vector3 size = node.bounds.maxx - node.bounds.minn;
    int axis = (size.x >= size.y && size.x >= size.z) ? 0 : (size.y >= size.z) ? 1 : 2;

    std::sort(shape_indices.begin() + start, shape_indices.begin() + end, [&](int a, int b) {
        return get_bounds(shapes[a]).center()[axis] < get_bounds(shapes[b]).center()[axis];
    });

    int mid = (start + end) / 2;
    node.left_child = build_recursive(start, mid, shapes);
    node.right_child = build_recursive(mid, end, shapes);
    node.shape_count = 0;

    return node_idx;
}

// __host__ __device__ bool cuda_bvh::intersect(const Ray &ray, const Shape *shapes, Real &t_min, int &hit_shape_idx, int &hit_primitive, Real &u, Real &v) const {
//     if (node_count == 0) return false;

//     bool hit = false;
//     int stack[100] = {0}, idx = 0;

//     while (idx >= 0) {
//         int node_idx = stack[idx];
//         idx--;
//         const BVHNode &node = nodes[node_idx];

//         if (!node.bounds.intersect(ray, t_min)) continue;

//         if (node.is_leaf()) {
//             for (int i = node.shape_start; i < node.shape_start + node.shape_count; i++) {
//                 int shape_idx = shape_indices[i];
//                 Real t;
//                 if (intersect_shape(shapes[shape_idx], ray, t, u, v) && t < t_min) {
//                     t_min = t;
//                     hit_shape_idx = shape_idx;
//                     hit_primitive = i;
//                     hit = true;
//                 }
//             }
//         } else {
//             if (node.right_child != -1) stack[++idx] = node.right_child;
//             if (node.left_child != -1) stack[++idx] = node.left_child;
//         }
//     }
//     return hit;
// }


