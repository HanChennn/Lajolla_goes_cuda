// struct DisneySheen {
//     Texture<Spectrum> base_color;
//     Texture<Real> sheen_tint;
// };

// const Vector3 &dir_in;
// const Vector3 &dir_out;
// const PathVertex &vertex;
// const TexturePool &texture_pool;
// const TransportDirection &dir;

#include "../microfacet.h"

__host__ __device__ Spectrum eval_op::operator()(const DisneySheen &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // Homework 1: implement this!
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    
    Vector3 half_v = normalize(dir_in+dir_out);
    Real hw_in = dot(half_v, dir_in), hw_out = dot(half_v, dir_out);
    Real nw_in = dot(frame.n, dir_in), nw_out = dot(frame.n, dir_out);

    Spectrum C_tint= luminance(base_color)>0? base_color/luminance(base_color):Vector3(1,1,1);
    Spectrum C_sheen = (1-sheen_tint) + sheen_tint * C_tint;
    Spectrum f_sheen = C_sheen * (1-fabs(hw_out))*(1-fabs(hw_out))*(1-fabs(hw_out))*(1-fabs(hw_out))*(1-fabs(hw_out))* fabs(nw_out);
    return f_sheen;
}

__host__ __device__ Real pdf_sample_bsdf_op::operator()(const DisneySheen &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // Homework 1: implement this!
    return fmax(dot(frame.n, dir_out), Real(0)) / c_PI;
}

__host__ __device__ std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneySheen &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }

    // Homework 1: implement this!
    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, Real(0)/* roughness */};
}

__host__ __device__ TextureSpectrum get_texture_op::operator()(const DisneySheen &bsdf) const {
    return bsdf.base_color;
}
