#pragma once

#include "lajolla.h"
#include "vector.h"

/// TableDist1D stores a tabular discrete distribution
/// that we can sample from using the functions below.
/// Useful for light source sampling.
struct TableDist1D {
    CUArray<Real> pmf;
    CUArray<Real> cdf;
};

template <typename T>
__device__ inline const T* upper_bound(const T* first, const T* last, const T& value) {
    while (first < last) {
        const T* mid = first + (last - first) / 2;
        if (*mid <= value) {
            first = mid + 1;
        } else {
            last = mid;
        }
    }
    return first;
}

/// Sample an entry from the discrete table given a random number in [0, 1]
__device__ inline int sample(const TableDist1D &table, Real rnd_param) {
    int size = table.pmf.size();
    assert(size > 0);
    const Real *ptr = upper_bound(table.cdf.data(), table.cdf.data() + size + 1, rnd_param);
    int offset = std::clamp(int(ptr - table.cdf.data() - 1), 0, size - 1);
    return offset;
}

/// The probability mass function of the sampling procedure above.
__device__ inline Real pmf(const TableDist1D &table, int id) {
    assert(id >= 0 && id < (int)table.pmf.size());
    return table.pmf[id];
}

/// TableDist2D stores a 2D piecewise constant distribution
/// that we can sample from using the functions below.
/// Useful for envmap sampling.
struct TableDist2D {
    // cdf_rows & pdf_rows store a 1D piecewise constant distribution
    // for each row.
    CUArray<Real> cdf_rows, pdf_rows;
    // cdf_maringlas & pdf_marginals store a single 1D piecewise
    // constant distribution for sampling a row
    CUArray<Real> cdf_marginals, pdf_marginals;
    Real total_values;
    int width, height;
};

/// Given two random number in [0, 1]^2, sample a point in the 2D domain [0, 1]^2
/// with distribution proportional to f above.
__device__ inline Vector2 sample(const TableDist2D &table, const Vector2 &rnd_param) {
    int w = table.width, h = table.height;
    // We first sample a row from the marginal distribution
    const Real *y_ptr = upper_bound(
        table.cdf_marginals.data(),
        table.cdf_marginals.data() + h + 1,
        rnd_param[1]);
    int y_offset = std::clamp(int(y_ptr - table.cdf_marginals.data() - 1), 0, h - 1);
    // Uniformly remap rnd_param[1] to find the continuous offset 
    Real dy = rnd_param[1] - table.cdf_marginals[y_offset];
    if ((table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]) > 0) {
        dy /= (table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]);
    }
    // Sample a column at the row y_offset
    const Real *cdf = &table.cdf_rows[y_offset * (w + 1)];
    const Real *x_ptr = upper_bound(cdf, cdf + w + 1, rnd_param[0]);
    int x_offset = std::clamp(int(x_ptr - cdf - 1), 0, w - 1);
    // Uniformly remap rnd_param[0]
    Real dx = rnd_param[0] - cdf[x_offset];
    if (cdf[x_offset + 1] - cdf[x_offset] > 0) {
        dx /= (cdf[x_offset + 1] - cdf[x_offset]);
    }
    return Vector2{(x_offset + dx) / w, (y_offset + dy) / h};
}

/// Probability density of the sampling procedure above.
__device__ inline Real pdf(const TableDist2D &table, const Vector2 &xy) {
    // Convert xy to integer rows & columns
    int w = table.width, h = table.height;
    int x = std::clamp(xy.x * w, Real(0), Real(w - 1));
    int y = std::clamp(xy.y * h, Real(0), Real(h - 1));
    // What's the PDF for sampling row y?
    Real pdf_y = table.pdf_marginals[y];
    // What's the PDF for sampling row x?
    Real pdf_x = table.pdf_rows[y * w + x];
    return pdf_y * pdf_x * w * h;
}
