#pragma once

#include "lajolla.h"
#include "image.h"
#include "intersection.h"
#include "mipmap.h"
#include <map>
#include <variant>

/// Can be replaced by a more advanced texture caching system,
/// where we only load images from files when necessary.
/// See OpenImageIO for example https://github.com/OpenImageIO/oiio
struct ParsedTexturePool {
    std::map<std::string, int> image1s_map;
    std::map<std::string, int> image3s_map;

    std::vector<ParsedMipmap1> image1s;
    std::vector<ParsedMipmap3> image3s;
};
struct TexturePool {
    Mipmap1 *image1s;
    Mipmap3 *image3s;
    int len_images1s, len_images3s;
};

__host__ __device__ inline bool texture_id_exists(const ParsedTexturePool &pool, const std::string &texture_name) {
    return pool.image1s_map.find(texture_name) != pool.image1s_map.end() ||
           pool.image3s_map.find(texture_name) != pool.image3s_map.end();
}

inline int insert_image1(ParsedTexturePool &pool, const std::string &texture_name, const fs::path &filename) {
    if (pool.image1s_map.find(texture_name) != pool.image1s_map.end()) {
        // We don't check if img is the same as the one in the cache!
        return pool.image1s_map[texture_name];
    }
    int id = (int)pool.image1s.size();
    pool.image1s_map[texture_name] = id;
    pool.image1s.push_back(make_mipmap(imread1(filename)));
    return id;
}

inline int insert_image1(ParsedTexturePool &pool, const std::string &texture_name, const ParsedImage1 &img) {
    if (pool.image1s_map.find(texture_name) != pool.image1s_map.end()) {
        // We don't check if img is the same as the one in the cache!
        return pool.image1s_map[texture_name];
    }
    int id = (int)pool.image1s.size();
    pool.image1s_map[texture_name] = id;
    pool.image1s.push_back(make_mipmap(img));
    return id;
}

inline int insert_image3(ParsedTexturePool &pool, const std::string &texture_name, const fs::path &filename) {
    if (pool.image3s_map.find(texture_name) != pool.image3s_map.end()) {
        // We don't check if img is the same as the one in the cache!
        return pool.image3s_map[texture_name];
    }
    int id = (int)pool.image3s.size();
    pool.image3s_map[texture_name] = id;
    pool.image3s.push_back(make_mipmap(imread3(filename)));
    return id;
}

inline int insert_image3(ParsedTexturePool &pool, const std::string &texture_name, const ParsedImage3 &img) {
    if (pool.image3s_map.find(texture_name) != pool.image3s_map.end()) {
        // We don't check if img is the same as the one in the cache!
        return pool.image3s_map[texture_name];
    }
    int id = (int)pool.image3s.size();
    pool.image3s_map[texture_name] = id;
    pool.image3s.push_back(make_mipmap(img));
    return id;
}

__host__ __device__ inline const Mipmap1 &get_img1(const TexturePool &pool, int texture_id) {
    assert(texture_id >= 0 && texture_id < (int)pool.len_images1s);
    return pool.image1s[texture_id];
}

__host__ __device__ inline const Mipmap3 &get_img3(const TexturePool &pool, int texture_id) {
    assert(texture_id >= 0 && texture_id < (int)pool.len_images3s);
    return pool.image3s[texture_id];
}

inline const ParsedMipmap1 &get_img1(const ParsedTexturePool &pool, int texture_id) {
    assert(texture_id >= 0 && texture_id < (int)pool.image1s.size());
    return pool.image1s[texture_id];
}

inline const ParsedMipmap3 &get_img3(const ParsedTexturePool &pool, int texture_id) {
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
inline const ParsedMipmap<T> &get_img(const ImageTexture<T> &t, const ParsedTexturePool &pool) {
    return ParsedMipmap<T>{};
}
template <>
inline const ParsedMipmap<Real> &get_img(const ImageTexture<Real> &t, const ParsedTexturePool &pool) {
    return get_img1(pool, t.texture_id);
}
template <>
inline const ParsedMipmap<Vector3> &get_img(const ImageTexture<Vector3> &t, const ParsedTexturePool &pool) {
    return get_img3(pool, t.texture_id);
}

template <typename T>
__host__ __device__ inline const Mipmap<T> &get_img(const ImageTexture<T> &t, const TexturePool &pool) {
    return Mipmap<T>{};
}
template <>
__host__ __device__ inline const Mipmap<Real> &get_img(const ImageTexture<Real> &t, const TexturePool &pool) {
    return get_img1(pool, t.texture_id);
}
template <>
__host__ __device__ inline const Mipmap<Vector3> &get_img(const ImageTexture<Vector3> &t, const TexturePool &pool) {
    return get_img3(pool, t.texture_id);
}

template <typename T>
using Texture = std::variant<ConstantTexture<T>, ImageTexture<T>, CheckerboardTexture<T>>;
using Texture1 = Texture<Real>;
using TextureSpectrum = Texture<Spectrum>;

template <typename T>
struct Parsed_eval_texture_op {
    T operator()(const ConstantTexture<T> &t) const;
    T operator()(const ImageTexture<T> &t) const;
    T operator()(const CheckerboardTexture<T> &t) const;

    const Vector2 &uv;
    const Real &footprint;
    const ParsedTexturePool &pool;
};
template <typename T>
T Parsed_eval_texture_op<T>::operator()(const ConstantTexture<T> &t) const {
    return t.value;
}
template <typename T>
T Parsed_eval_texture_op<T>::operator()(const ImageTexture<T> &t) const {
    const ParsedMipmap<T> &img = get_img(t, pool);
    Vector2 local_uv{modulo(uv[0] * t.uscale + t.uoffset, Real(1)),
                     modulo(uv[1] * t.vscale + t.voffset, Real(1))};
    Real scaled_footprint = max(get_width(img), get_height(img)) * max(t.uscale, t.vscale) * footprint;
    Real level = log2(max(scaled_footprint, Real(1e-8f)));
    return lookup(img, local_uv[0], local_uv[1], level);
}
template <typename T>
T Parsed_eval_texture_op<T>::operator()(const CheckerboardTexture<T> &t) const {
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

template <typename T>
struct eval_texture_op {
    __host__ __device__ inline T operator()(const ConstantTexture<T> &t) const;
    __host__ __device__ inline T operator()(const ImageTexture<T> &t) const;
    __host__ __device__ inline T operator()(const CheckerboardTexture<T> &t) const;

    const Vector2 &uv;
    const Real &footprint;
    const TexturePool &pool;
};
template <typename T>
__host__ __device__ inline T eval_texture_op<T>::operator()(const ConstantTexture<T> &t) const {
    return t.value;
}
template <typename T>
__host__ __device__ inline T eval_texture_op<T>::operator()(const ImageTexture<T> &t) const {
    const Mipmap<T> &img = get_img(t, pool);
    Vector2 local_uv{modulo(uv[0] * t.uscale + t.uoffset, Real(1)),
                     modulo(uv[1] * t.vscale + t.voffset, Real(1))};
    Real scaled_footprint = max(get_width(img), get_height(img)) * max(t.uscale, t.vscale) * footprint;
    Real level = log2(max(scaled_footprint, Real(1e-8f)));
    return lookup(img, local_uv[0], local_uv[1], level);
}
template <typename T>
__host__ __device__ inline T eval_texture_op<T>::operator()(const CheckerboardTexture<T> &t) const {
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
T eval(const Texture<T> &texture, const Vector2 &uv, Real footprint, const ParsedTexturePool &pool) {
    return std::visit(Parsed_eval_texture_op<T>{uv, footprint, pool}, texture);
}
template <typename T>
__host__ __device__ inline T eval(const Texture<T> &texture, const Vector2 &uv, Real footprint, const TexturePool &pool) {
    // return std::visit(eval_texture_op<T>{uv, footprint, pool}, texture);
    eval_texture_op<T> op{uv, footprint, pool};
    if (auto *t = std::get_if<ConstantTexture<T>>(&texture)) return op(*t);
    else if (auto *t = std::get_if<ImageTexture<T>>(&texture)) return op(*t);
    else if (auto *t = std::get_if<CheckerboardTexture<T>>(&texture)) return op(*t);
    else return T{};  // 默认值处理未知类型
}

__host__ __device__ inline ConstantTexture<Spectrum> make_constant_spectrum_texture(const Spectrum &spec) {
    return ConstantTexture<Spectrum>{spec};
}

__host__ __device__ inline ConstantTexture<Real> make_constant_float_texture(Real f) {
    return ConstantTexture<Real>{f};
}

inline ImageTexture<Spectrum> make_image_spectrum_texture(
        const std::string &texture_name,
        const fs::path &filename,
        ParsedTexturePool &pool,
        Real uscale = 1,
        Real vscale = 1,
        Real uoffset = 0,
        Real voffset = 0) {
    return ImageTexture<Spectrum>{insert_image3(pool, texture_name, filename),
        uscale, vscale, uoffset, voffset};
}

inline ImageTexture<Spectrum> make_image_spectrum_texture(
        const std::string &texture_name,
        const ParsedImage3 &img,
        ParsedTexturePool &pool,
        Real uscale = 1,
        Real vscale = 1,
        Real uoffset = 0,
        Real voffset = 0) {
    return ImageTexture<Spectrum>{insert_image3(pool, texture_name, img),
        uscale, vscale, uoffset, voffset};
}

inline ImageTexture<Real> make_image_float_texture(
        const std::string &texture_name,
        const fs::path &filename,
        ParsedTexturePool &pool,
        Real uscale = 1,
        Real vscale = 1,
        Real uoffset = 0,
        Real voffset = 0) {
    return ImageTexture<Real>{insert_image1(pool, texture_name, filename),
        uscale, vscale, uoffset, voffset};
}

inline ImageTexture<Real> make_image_float_texture(
        const std::string &texture_name,
        const ParsedImage1 &img,
        ParsedTexturePool &pool,
        Real uscale = 1,
        Real vscale = 1,
        Real uoffset = 0,
        Real voffset = 0) {
    return ImageTexture<Real>{insert_image1(pool, texture_name, img),
        uscale, vscale, uoffset, voffset};
}

inline CheckerboardTexture<Spectrum> make_checkerboard_spectrum_texture(
        const Spectrum &color0, const Spectrum &color1,
        Real uscale = 1, Real vscale = 1,
        Real uoffset = 0, Real voffset = 0) {
    return CheckerboardTexture<Spectrum>{
        color0, color1, uscale, vscale, uoffset, voffset};
}

inline CheckerboardTexture<Real> make_checkerboard_float_texture(
        Real color0, Real color1,
        Real uscale = 1, Real vscale = 1,
        Real uoffset = 0, Real voffset = 0) {
    return CheckerboardTexture<Real>{
        color0, color1, uscale, vscale, uoffset, voffset};
}
