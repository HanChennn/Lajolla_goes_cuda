#pragma once

#include "lajolla.h"
#include "image.h"

constexpr int c_max_mipmap_levels = 8;

template <typename T>
struct Mipmap {
    CUArray<CUImage<T>> images;
};

template <typename T>
__device__ inline int get_width(const Mipmap<T> &mipmap) {
    assert(mipmap.images.size() > 0);
    return mipmap.images[0].width;
}

template <typename T>
__device__ inline int get_height(const Mipmap<T> &mipmap) {
    assert(mipmap.images.size() > 0);
    return mipmap.images[0].height;
}

/// Bilinear lookup of a mipmap at location (uv) with an integer level
template <typename T>
__device__ inline T lookup(const Mipmap<T> &mipmap, Real u, Real v, int level) {
    assert(level >= 0 && level < (int)mipmap.images.size());
    // Bilinear interpolation
    // (-0.5 to match Mitsuba's coordinates)
    u = u * mipmap.images[level].width - Real(0.5);
    v = v * mipmap.images[level].height - Real(0.5);
    int ufi = modulo(int(u), mipmap.images[level].width);
    int vfi = modulo(int(v), mipmap.images[level].height);
    int uci = modulo(ufi + 1, mipmap.images[level].width);
    int vci = modulo(vfi + 1, mipmap.images[level].height);
    Real u_off = u - ufi;
    Real v_off = v - vfi;
    T val_ff = mipmap.images[level](ufi, vfi);
    T val_fc = mipmap.images[level](ufi, vci);
    T val_cf = mipmap.images[level](uci, vfi);
    T val_cc = mipmap.images[level](uci, vci);
    return val_ff * (1 - u_off) * (1 - v_off) +
           val_fc * (1 - u_off) *      v_off +
           val_cf *      u_off  * (1 - v_off) +
           val_cc *      u_off  *      v_off;
}

/// Trilinear look of of a mipmap at (u, v, level)
template <typename T>
__device__ inline T lookup(const Mipmap<T> &mipmap, Real u, Real v, Real level) {
    if (level <= 0) {
        return lookup(mipmap, u, v, 0);
    } else if (level < Real(mipmap.images.size() - 1)) {
        int flevel = std::clamp((int)floor(level), 0, (int)mipmap.images.size() - 1);
        int clevel = std::clamp(flevel + 1, 0, (int)mipmap.images.size() - 1);
        Real level_off = level - flevel;
        return lookup(mipmap, u, v, flevel) * (1 - level_off) +
               lookup(mipmap, u, v, clevel) *      level_off;
    } else {
        return lookup(mipmap, u, v, int(mipmap.images.size() - 1));
    }
}

using Mipmap1 = Mipmap<Real>;
using Mipmap3 = Mipmap<Vector3>;
