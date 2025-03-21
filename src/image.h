#pragma once

#include "vector.h"

#include <string>
#include <cstring>
#include <vector>

/// A N-channel image stored in a contiguous vector
/// The storage format is HWC -- outer dimension is height
/// then width, then channels.
template<typename T>
struct ParsedImage {
    ParsedImage() {}
    ParsedImage(int w, int h) : width(w), height(h) {
        data.resize(w * h);
        memset(data.data(), 0, sizeof(T) * data.size());
    }

    T &operator()(int x) {
        return data[x];
    }

    const T &operator()(int x) const {
        return data[x];
    }

    T &operator()(int x, int y) {
        return data[y * width + x];
    }

    const T &operator()(int x, int y) const {
        return data[y * width + x];
    }

    int width;
    int height;
    std::vector<T> data;
};
template<typename T>
struct Image {
    __host__ __device__ Image() {}
    __host__ __device__ Image(int w, int h) : width(w), height(h) {
        // data.resize(w * h);
        // memset(data.data(), 0, sizeof(T) * data.size());
    }

    __host__ __device__ T &operator()(int x) {
        return data[x];
    }

    __host__ __device__ const T &operator()(int x) const {
        return data[x];
    }

    __host__ __device__ T &operator()(int x, int y) {
        return data[y * width + x];
    }

    __host__ __device__ const T &operator()(int x, int y) const {
        return data[y * width + x];
    }

    int width;
    int height;
    T *data;
};

using Image1 = Image<Real>;
using Image3 = Image<Vector3>;
using ParsedImage1 = ParsedImage<Real>;
using ParsedImage3 = ParsedImage<Vector3>;

/// Read from an 1 channel image. If the image is not actually
/// single channel, the first channel is used.
/// Supported formats: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
ParsedImage1 imread1(const fs::path &filename);
/// Read from a 3 channels image. 
/// If the image only has 1 channel, we set all 3 channels to the same color.
/// If the image has more than 3 channels, we truncate it to 3.
/// Undefined behavior if the image has 2 channels (does that even happen?)
/// Supported formats: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
ParsedImage3 imread3(const fs::path &filename);

/// Save an image to a file.
/// Supported formats: PFM & exr
void imwrite(const fs::path &filename, const ParsedImage3 &image);

inline ParsedImage3 to_image3(const ParsedImage1 &img) {
    ParsedImage3 out(img.width, img.height);
    std::transform(img.data.cbegin(), img.data.cend(), out.data.begin(),
        [] (Real v) {return Vector3(v, v, v);});
    return out;
}

inline ParsedImage1 to_image1(const ParsedImage3 &img) {
    ParsedImage1 out(img.width, img.height);
    std::transform(img.data.cbegin(), img.data.cend(), out.data.begin(),
        [] (const Vector3 &v) {return average(v);});
    return out;
}
