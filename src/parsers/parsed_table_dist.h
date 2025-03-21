#pragma once

#include "lajolla.h"

namespace parser
{

struct TableDist1D {
    std::vector<Real> pmf;
    std::vector<Real> cdf;
};

struct TableDist2D {
    std::vector<Real> cdf_rows, pdf_rows;
    std::vector<Real> cdf_marginals, pdf_marginals;
    Real total_values;
    int width, height;
};

/// Construct the tabular discrete distribution given a vector of positive numbers.
inline TableDist1D make_table_dist_1d(const std::vector<Real> &f) {
    std::vector<Real> pmf = f;
    std::vector<Real> cdf(f.size() + 1);
    cdf[0] = 0;
    for (int i = 0; i < (int)f.size(); i++) {
        assert(pmf[i] >= 0);
        cdf[i + 1] = cdf[i] + pmf[i];
    }
    Real total = cdf.back();
    if (total > 0) {
        for (int i = 0; i < (int)pmf.size(); i++) {
            pmf[i] /= total;
            cdf[i] /= total;
        }
    } else {
        for (int i = 0; i < (int)pmf.size(); i++) {
            pmf[i] = Real(1) / Real(pmf.size());
            cdf[i] = Real(i) / Real(pmf.size());
        }
        cdf.back() = 1;
    }
    return TableDist1D{pmf, cdf};
}

/// Construct the 2D piecewise constant distribution given a vector of positive numbers
/// and width & height.
inline TableDist2D make_table_dist_2d(const std::vector<Real> &f, int width, int height) {
    // Construct a 1D distribution for each row
    std::vector<Real> cdf_rows(height * (width + 1));
    std::vector<Real> pdf_rows(height * width);
    for (int y = 0; y < height; y++) {
        cdf_rows[y * (width + 1)] = 0;
        for (int x = 0; x < width; x++) {
            cdf_rows[y * (width + 1) + (x + 1)] =
                cdf_rows[y * (width + 1) + x] + f[y * width + x];
        }
        Real integral = cdf_rows[y * (width + 1) + width];
        if (integral > 0) {
            // Normalize
            for (int x = 0; x < width; x++) {
                cdf_rows[y * (width + 1) + x] /= integral;
            }
            // Note that after normalization, the last entry of each row for
            // cdf_rows is still the "integral".
            // We need this later for constructing the marginal distribution.

            // Setup the pmf/pdf
            for (int x = 0; x < width; x++) {
                pdf_rows[y * width + x] = f[y * width + x] / integral;
            }
        } else {
            // We shouldn't sample this row, but just in case we
            // set up a uniform distribution.
            for (int x = 0; x < width; x++) {
                pdf_rows[y * width + x] = Real(1) / Real(width);
                cdf_rows[y * (width + 1) + x] = Real(x) / Real(width);
            }
            cdf_rows[y * (width + 1) + width] = 1;
        }
    }
    // Now construct the marginal CDF for each column.
    std::vector<Real> cdf_marginals(height + 1);
    std::vector<Real> pdf_marginals(height);
    cdf_marginals[0] = 0;
    for (int y = 0; y < height; y++) {
        Real weight = cdf_rows[y * (width + 1) + width];
        cdf_marginals[y + 1] = cdf_marginals[y] + weight;
    }
    Real total_values = cdf_marginals.back();
    if (total_values > 0) {
        // Normalize
        for (int y = 0; y < height; y++) {
            cdf_marginals[y] /= total_values;
        }
        cdf_marginals[height] = 1;
        // Setup pdf cols
        for (int y = 0; y < height; y++) {
            Real weight = cdf_rows[y * (width + 1) + width];
            pdf_marginals[y] = weight / total_values;
        }
    } else {
        // The whole thing is black...why are we even here?
        // Still set up a uniform distribution.
        for (int y = 0; y < height; y++) {
            pdf_marginals[y] = Real(1) / Real(height);
            cdf_marginals[y] = Real(y) / Real(height);
        }
        cdf_marginals[height] = 1;
    }
    // We finally normalize the last entry of each cdf row to 1
    for (int y = 0; y < height; y++) {
        cdf_rows[y * (width + 1) + width] = 1;
    }

    return TableDist2D{
        cdf_rows, pdf_rows,
        cdf_marginals, pdf_marginals,
        total_values,
        width, height
    };
}

}