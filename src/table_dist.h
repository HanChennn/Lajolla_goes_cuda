#pragma once

#include "lajolla.h"
#include "vector.h"
#include <vector>

/// TableDist1D stores a tabular discrete distribution
/// that we can sample from using the functions below.
/// Useful for light source sampling.

// Custom implementation of upper_bound
template <typename T>
__host__ __device__ const T* upper_bound(const T* first, const T* last, const T& value) {
    while (first < last) {
        const T* mid = first + (last - first) / 2;
        if (*mid > value) {
            last = mid;
        } else {
            first = mid + 1;
        }
    }
    return first;
}

// Custom implementation of clamp
template <typename T>
__host__ __device__ T clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

struct ParsedTableDist1D {
    std::vector<Real> pmf;
    std::vector<Real> cdf;
};
struct TableDist1D {
    Real *pmf;
    Real *cdf;
    int len_pmf, len_cdf;
};

/// Construct the tabular discrete distribution given a vector of positive numbers.
ParsedTableDist1D make_table_dist_1d(const std::vector<Real> &f);

/// Sample an entry from the discrete table given a random number in [0, 1]
int sample_1d_parsed(const ParsedTableDist1D &table, Real rnd_param);
// __host__ __device__  int sample_1d(const TableDist1D &table, Real rnd_param);
__host__ __device__ inline int sample_1d(const TableDist1D &table, Real rnd_param) {
    int size = table.len_pmf;
    assert(size > 0);
    const Real *ptr = upper_bound(table.cdf, table.cdf + size + 1, rnd_param);
    int offset = clamp(int(ptr - table.cdf - 1), 0, size - 1);
    return offset;
}

/// The probability mass function of the sampling procedure above.
Real pmf_1d_parsed(const ParsedTableDist1D &table, int id);
// __host__ __device__ Real pmf_1d(const TableDist1D &table, int id);
__host__ __device__ inline Real pmf_1d(const TableDist1D &table, int id) {
    assert(id >= 0 && id < (int)table.len_pmf);
    return table.pmf[id];
}

/// TableDist2D stores a 2D piecewise constant distribution
/// that we can sample from using the functions below.
/// Useful for envmap sampling.
struct ParsedTableDist2D {
    // cdf_rows & pdf_rows store a 1D piecewise constant distribution
    // for each row.
    std::vector<Real> cdf_rows, pdf_rows;
    // cdf_maringlas & pdf_marginals store a single 1D piecewise
    // constant distribution for sampling a row
    std::vector<Real> cdf_marginals, pdf_marginals;
    Real total_values;
    int width, height;
};

struct TableDist2D {
    // cdf_rows & pdf_rows store a 1D piecewise constant distribution
    // for each row.
    Real* cdf_rows;
    Real* pdf_rows;
    // cdf_maringlas & pdf_marginals store a single 1D piecewise
    // constant distribution for sampling a row
    Real* cdf_marginals;
    Real* pdf_marginals;
    int len_cdf_rows, len_pdf_rows, len_cdf_marginals, len_pdf_marginals;
    Real total_values;
    int width, height;
};

/// Construct the 2D piecewise constant distribution given a vector of positive numbers
/// and width & height.
ParsedTableDist2D make_table_dist_2d(const std::vector<Real> &f, int width, int height);

/// Given two random number in [0, 1]^2, sample a point in the 2D domain [0, 1]^2
/// with distribution proportional to f above.
Vector2 sample_2d_parsed(const ParsedTableDist2D &table, const Vector2 &rnd_param);
// __host__ __device__ Vector2 sample_2d(const TableDist2D &table, const Vector2 &rnd_param);
__host__ __device__ inline Vector2 sample_2d(const TableDist2D &table, const Vector2 &rnd_param) {
    int w = table.width, h = table.height;
    // We first sample a row from the marginal distribution
    const Real *y_ptr = upper_bound(
        table.cdf_marginals,
        table.cdf_marginals + h + 1,
        rnd_param[1]);
    int y_offset = clamp(int(y_ptr - table.cdf_marginals - 1), 0, h - 1);
    // Uniformly remap rnd_param[1] to find the continuous offset 
    Real dy = rnd_param[1] - table.cdf_marginals[y_offset];
    if ((table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]) > 0) {
        dy /= (table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]);
    }
    // Sample a column at the row y_offset
    const Real *cdf = &table.cdf_rows[y_offset * (w + 1)];
    const Real *x_ptr = upper_bound(cdf, cdf + w + 1, rnd_param[0]);
    int x_offset = clamp(int(x_ptr - cdf - 1), 0, w - 1);
    // Uniformly remap rnd_param[0]
    Real dx = rnd_param[0] - cdf[x_offset];
    if (cdf[x_offset + 1] - cdf[x_offset] > 0) {
        dx /= (cdf[x_offset + 1] - cdf[x_offset]);
    }
    return Vector2{(x_offset + dx) / w, (y_offset + dy) / h};
}

/// Probability density of the sampling procedure above.
Real pdf_2d_parsed(const ParsedTableDist2D &table, const Vector2 &xy);
// __host__ __device__ Real pdf_2d(const TableDist2D &table, const Vector2 &xy);
__host__ __device__ inline Real pdf_2d(const TableDist2D &table, const Vector2 &xy) {
    // Convert xy to integer rows & columns
    int w = table.width, h = table.height;
    int x = clamp(xy.x * w, Real(0), Real(w - 1));
    int y = clamp(xy.y * h, Real(0), Real(h - 1));
    // What's the PDF for sampling row y?
    Real pdf_y = table.pdf_marginals[y];
    // What's the PDF for sampling row x?
    Real pdf_x = table.pdf_rows[y * w + x];
    return pdf_y * pdf_x * w * h;
}
