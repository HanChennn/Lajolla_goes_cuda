#ifndef AABB_H
#define AABB_H

#include "lajolla.h"
#include "vector.h"
#include "ray.h"

class AABB {
public:
    Vector3 minn;
    Vector3 maxx;

    __host__ __device__ AABB() 
        : minn(Vector3(INFINITY, INFINITY, INFINITY)), 
          maxx(Vector3(-INFINITY, -INFINITY, -INFINITY)) {}

    __host__ __device__ AABB(const Vector3 &minn, const Vector3 &maxx) : minn(minn), maxx(maxx) {}

    // 扩展 AABB 以包含另一个 AABB
    __host__ __device__ void expand(const AABB &other) {
        minn = min(minn, other.minn);
        maxx = max(maxx, other.maxx);
    }

    // 扩展 AABB 以包含一个点
    __host__ __device__ void expand(const Vector3 &point) {
        minn = min(minn, point);
        maxx = max(maxx, point);
    }

    // 计算 AABB 的中心点
    __host__ __device__ Vector3 center() const {
        return (0.5 * minn + 0.5 * maxx);
    }

    // AABB 和光线求交
    __host__ __device__ bool intersect(const Ray &ray, Real &t_min) const {
        Real t0 = 0, t1 = INFINITY;
        
        for (int i = 0; i < 3; i++) {
            Real invD = 1.0f / ray.dir[i];
            Real tNear = (minn[i] - ray.org[i]) * invD;
            Real tFar  = (maxx[i] - ray.org[i]) * invD;
            if (invD < 0) {
                Real _ = tNear;
                tNear = tFar;
                tFar = _;
                // std::swap(tNear, tFar);
            }
            
            t0 = tNear>t0?tNear:t0;
            t1 = tFar<t1?tFar:t1;
            if (t0 > t1) return false;
        }
        
        t_min = t0;
        return true;
    }
};
#endif // AABB_H
