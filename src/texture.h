#pragma once

#include "lajolla.h"
#include "mipmap.h"
#include "spectrum.h"
#include <variant>

/// Can be replaced by a more advanced texture caching system,
/// where we only load images from files when necessary.
/// See OpenImageIO for example https://github.com/OpenImageIO/oiio
struct TexturePool {
    CUArray<Mipmap1> image1s;
    CUArray<Mipmap3> image3s;
};

__device__ inline const Mipmap1 &get_img1(const TexturePool &pool, int texture_id) {
    assert(texture_id >= 0 && texture_id < (int)pool.image1s.size());
    return pool.image1s[texture_id];
}

__device__ inline const Mipmap3 &get_img3(const TexturePool &pool, int texture_id) {
    assert(texture_id >= 0 && texture_id < (int)pool.image3s.size());
    return pool.image3s[texture_id];
}

template <typename T>
struct ConstantTexture {
    T value;
};

template <typename T>
struct ImageTexture {
    int texture_id;
    Real uscale, vscale;
    Real uoffset, voffset;
};

template <typename T>
struct CheckerboardTexture {
    T color0, color1;
    Real uscale, vscale;
    Real uoffset, voffset;
};

template <typename T>
__device__ inline const Mipmap<T> &get_img(const ImageTexture<T> &t, const TexturePool &pool) {
    return Mipmap<T>{};
}
template <>
__device__ inline const Mipmap<Real> &get_img(const ImageTexture<Real> &t, const TexturePool &pool) {
    return get_img1(pool, t.texture_id);
}
template <>
__device__ inline const Mipmap<Vector3> &get_img(const ImageTexture<Vector3> &t, const TexturePool &pool) {
    return get_img3(pool, t.texture_id);
}

template <typename T>
using Texture = std::variant<ConstantTexture<T>, ImageTexture<T>, CheckerboardTexture<T>>;
using Texture1 = Texture<Real>;
using TextureSpectrum = Texture<Spectrum>;

__device__ inline ConstantTexture<Spectrum> make_constant_spectrum_texture(const Spectrum &spec) {
    return ConstantTexture<Spectrum>{spec};
}

template <typename T>
struct eval_texture_op {
    __device__ inline T operator()(const ConstantTexture<T> &t) const;
    __device__ inline T operator()(const ImageTexture<T> &t) const;
    __device__ inline T operator()(const CheckerboardTexture<T> &t) const;

    const Vector2 &uv;
    const Real &footprint;
    const TexturePool &pool;
};
template <typename T>
__device__ inline T eval_texture_op<T>::operator()(const ConstantTexture<T> &t) const {
    return t.value;
}
template <typename T>
__device__ inline T eval_texture_op<T>::operator()(const ImageTexture<T> &t) const {
    const Mipmap<T> &img = get_img(t, pool);
    Vector2 local_uv{modulo(uv[0] * t.uscale + t.uoffset, Real(1)),
                     modulo(uv[1] * t.vscale + t.voffset, Real(1))};
    Real scaled_footprint = max(get_width(img), get_height(img)) * max(t.uscale, t.vscale) * footprint;
    Real level = log2(max(scaled_footprint, Real(1e-8f)));
    return lookup(img, local_uv[0], local_uv[1], level);
}
template <typename T>
__device__ inline T eval_texture_op<T>::operator()(const CheckerboardTexture<T> &t) const {
    Vector2 local_uv{modulo(uv[0] * t.uscale + t.uoffset, Real(1)),
                     modulo(uv[1] * t.vscale + t.voffset, Real(1))};
    int x = 2 * modulo((int)(local_uv.x * 2), 2) - 1,
        y = 2 * modulo((int)(local_uv.y * 2), 2) - 1;

    if (x * y == 1) {
        return t.color0;
    } else {
        return t.color1;
    }
}

/// Evaluate the texture at location uv.
/// Footprint should be approximatedly min(du/dx, du/dy, dv/dx, dv/dy) for texture filtering.
template <typename T>
__device__ inline T eval(const Texture<T> &texture, const Vector2 &uv, Real footprint, const TexturePool &pool) {
    return std::visit(eval_texture_op<T>{uv, footprint, pool}, texture);
}