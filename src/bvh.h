#ifndef BVH_H
#define BVH_H

#include <vector>
#include <variant>
#include "lajolla.h"
#include "AABB.h"
#include "ray.h"
#include "shape.h"
#include "vector.h"

constexpr int MAX_BVH_NODES = 100;


struct BVHNode {
    AABB bounds;
    int left_child = -1;
    int right_child = -1;
    int shape_start = 0;
    int shape_count = 0;

    __host__ __device__ bool is_leaf() const { return shape_count > 0; } 
};

class BVH {
public:
    std::vector<BVHNode> nodes;
    std::vector<int> shape_indices;
    int node_count = 0;

    __host__ __device__ void build(const std::vector<ParsedShape> &shapes);
private:
    __host__ __device__ int build_recursive(int start, int end, const std::vector<ParsedShape> &shapes);
};

class cuda_bvh{
public:
    BVHNode *nodes;
    int *shape_indices;
    int node_count = 0, len_shape_indices;

    __host__ __device__ bool intersect(const Ray &ray, const Shape *shapes, Real &t_min, int &hit_shape_idx, int &hit_primitive, Real &u, Real &v) const;
};

__host__ __device__ AABB get_bounds(const ParsedShape &shape);
__host__ __device__ AABB get_bounds(const Shape &shape);
// __host__ __device__ bool intersect_shape(const Shape &shape, const Ray &ray, Real &t, Real &u, Real &v);
__host__ __device__ inline bool intersect_shape(const Shape &shape, const Ray &ray, Real &t, Real &u, Real &v) {
    return std::visit([&](const auto &s) -> bool {
        if constexpr (std::is_same_v<std::decay_t<decltype(s)>, Sphere>) {
            Vector3 oc = ray.org - s.position;
            Real a = dot(ray.dir, ray.dir);
            Real b = 2.0f * dot(oc, ray.dir);
            Real c = dot(oc, oc) - s.radius * s.radius;
            Real discriminant = b * b - 4 * a * c;
            if (discriminant < 0) return false;
            t = (-b - sqrt(discriminant)) / (2.0f * a);
            u = v = 0; // Not used for spheres
            return true;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>, ParsedTriangleMesh>) {
            bool hit = false;
            for (size_t i = 0; i < s.indices.size(); ++i) {
                const auto &v0 = s.positions[s.indices[i][0]];
                const auto &v1 = s.positions[s.indices[i][1]];
                const auto &v2 = s.positions[s.indices[i][2]];
                Real temp_t, temp_u, temp_v;
                if (ray_triangle_intersect(ray, v0, v1, v2, temp_t, temp_u, temp_v) && temp_t < t) {
                    t = temp_t;
                    u = temp_u;
                    v = temp_v;
                    hit = true;
                }
            }
            return hit;
        }
        return false;
    }, shape);
}

__host__ __device__ bool ray_triangle_intersect(const Ray &ray, const Vector3 &v0, const Vector3 &v1, const Vector3 &v2, Real &t, Real &u, Real &v);

__host__ __device__ inline bool cuda_bvh::intersect(const Ray &ray, const Shape *shapes, Real &t_min, int &hit_shape_idx, int &hit_primitive, Real &u, Real &v) const {
    if (node_count == 0) return false;

    bool hit = false;
    int stack[100] = {0}, idx = 0;

    while (idx >= 0) {
        int node_idx = stack[idx];
        idx--;
        const BVHNode &node = nodes[node_idx];

        if (!node.bounds.intersect(ray, t_min)) continue;

        if (node.is_leaf()) {
            for (int i = node.shape_start; i < node.shape_start + node.shape_count; i++) {
                int shape_idx = shape_indices[i];
                Real t;
                if (intersect_shape(shapes[shape_idx], ray, t, u, v) && t < t_min) {
                    t_min = t;
                    hit_shape_idx = shape_idx;
                    hit_primitive = i;
                    hit = true;
                }
            }
        } else {
            if (node.right_child != -1) stack[++idx] = node.right_child;
            if (node.left_child != -1) stack[++idx] = node.left_child;
        }
    }
    return hit;
}


#endif // BVH_H