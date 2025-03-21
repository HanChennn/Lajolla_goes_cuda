// const Vector3 &dir_in;
// const Vector3 &dir_out;
// const PathVertex &vertex;
// const TexturePool &texture_pool;
// const TransportDirection &dir;

// struct DisneyClearcoat {
//     Texture<Real> clearcoat_gloss;
// };


#include "../microfacet.h"

__host__ __device__ Spectrum eval_op::operator()(const DisneyClearcoat &bsdf) const {
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

    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_v = normalize(dir_in+dir_out);
    Vector3 half_v_local = to_local(frame, half_v);
    Real nw_in = fabs(dot(frame.n, dir_in)), nw_out = fabs(dot(frame.n, dir_out));
    Real hw_out = fabs(dot(half_v, dir_out)), one_hw_out = 1.0 - hw_out;

    Real eta = 1.5;

    Real R0_eta = (eta-1)*(eta-1)/(eta+1)/(eta+1);
    Real F_c = R0_eta + (1-R0_eta)*one_hw_out*one_hw_out*one_hw_out*one_hw_out*one_hw_out;
    
    Real alpha_g = (1-clearcoat_gloss)*0.1 + clearcoat_gloss*0.001;
    Real D_c = (alpha_g*alpha_g - 1.0) / ( c_PI*log(alpha_g*alpha_g)*( 1+( alpha_g*alpha_g-1 )*half_v_local.z*half_v_local.z ) );

    Vector3 dir_in_local = to_local(frame, dir_in), dir_out_local = to_local(frame, dir_out);
    auto lambda = [](Vector3 w) -> double {
        return ( sqrt(1+((w.x*0.25)*(w.x*0.25)+(w.y*0.25)*(w.y*0.25))/(w.z*w.z)) - 1) / 2.0;
    };
    Real lambda_w_in = lambda(dir_in_local), lambda_w_out = lambda(dir_out_local);
    Real Gw_in = 1.0 / (1.0+lambda_w_in), Gw_out = 1.0 / (1.0+lambda_w_out);
    Real G_c = Gw_in*Gw_out;

    Spectrum f_clearcoat = D_c * G_c / 4.0 / nw_in * F_c * Vector3(1.0,1.0,1.0);
    return f_clearcoat;
}

__host__ __device__ Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
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

    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_v = normalize(dir_in+dir_out);
    Vector3 half_v_local = to_local(frame, half_v);
    Real hw_out = fabs(dot(half_v, dir_out));
    
    Real alpha_g = (1-clearcoat_gloss)*0.1 + clearcoat_gloss*0.001;
    Real D_c = (alpha_g*alpha_g - 1.0) / ( c_PI*log(alpha_g*alpha_g)*( 1+( alpha_g*alpha_g-1 )*half_v_local.z*half_v_local.z ) );

    return D_c*fabs(dot(frame.n, half_v))/4.0/hw_out;
}

__host__ __device__ std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
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
    Vector3 local_dir_in = to_local(frame, dir_in);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real alpha_g = (1-clearcoat_gloss)*0.1 + clearcoat_gloss*0.001;
 
    
    Real cos_h_elevation = sqrt((1-pow(alpha_g*alpha_g, 1-rnd_param_uv.x))/(1-alpha_g*alpha_g));
    Real h_azimuth = 2.0*c_PI*rnd_param_uv.y;

    Vector3 local_micro_normal = {  sqrt(1-cos_h_elevation*cos_h_elevation)*cos(h_azimuth),
                                    sqrt(1-cos_h_elevation*cos_h_elevation)*sin(h_azimuth),
                                    cos_h_elevation};
    
    
    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(0) /* eta */, Real(0) /* roughness */
    };
}

__host__ __device__ TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
