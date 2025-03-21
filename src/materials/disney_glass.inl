// const Vector3 &dir_in;
// const Vector3 &dir_out;
// const PathVertex &vertex;
// const TexturePool &texture_pool;
// const TransportDirection &dir;
// struct DisneyGlass {
//     Texture<Spectrum> base_color;
//     Texture<Real> roughness;
//     Texture<Real> anisotropic;

//     Real eta; // internal IOR / externalIOR
// };

#include "../microfacet.h"

__host__ __device__ Spectrum eval_op::operator()(const DisneyGlass &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    
    // Real eta = bsdf.eta;

    Vector3 half_v = normalize(dir_in+dir_out);
    if (!reflect) {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_v = normalize(dir_in + dir_out * eta);
    }
    // Flip half-vector if it's below surface
    if (dot(half_v, frame.n) < 0) {
        half_v = -half_v;
    }

    // Real hw_in = dot(half_v,dir_in), hw_out = dot(half_v,dir_out);
    Real hw_in = fabs(dot(half_v,dir_in)), hw_out = fabs(dot(half_v,dir_out));
    Real nw_in = fabs(dot(frame.n, dir_in)), nw_out = fabs(dot(frame.n, dir_out));
    
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Real eta = bsdf.eta;
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);


    //////// FFFFFFFFFFFFFFFFF
    // Real R_0 = ((eta-1)*(eta-1))/((eta+1)*(eta+1));
    // Real one_minus_cos = 1.0 - dot(dir_in, half_v);
    // Real F_g = R_0 + (1-R_0)*one_minus_cos*one_minus_cos*one_minus_cos*one_minus_cos*one_minus_cos;

    Real R_s = ( hw_in - eta*hw_out ) / ( hw_in + eta*hw_out );
    Real R_p = ( eta*hw_in - hw_out ) / ( eta*hw_in + hw_out );
    Real F_g = 0.5 * (R_s*R_s + R_p*R_p);
    assert(F_g>=0 && F_g<=1);

    //////// DDDDDDDDDDDDDDDDD
    Real aspect = sqrt(1-0.9*anisotropic);
    Real alpha_x = max(0.0001, roughness*roughness/aspect);
    Real alpha_y = max(0.0001, roughness*roughness*aspect);
    Vector3 h_in_local = to_local(frame, half_v);
    Real _ = h_in_local[0]*h_in_local[0]/alpha_x/alpha_x + h_in_local[1]*h_in_local[1]/alpha_y/alpha_y + h_in_local[2]*h_in_local[2];
    Real D_g = 1.0/(c_PI*alpha_x*alpha_y*_*_);

    //////// GGGGGGGGGGGGGGGGG
    Vector3 dir_in_local = to_local(frame, dir_in), dir_out_local = to_local(frame, dir_out);
    auto lambda = [](Vector3 w, double alpha_x, double alpha_y) -> double {
        return ( sqrt(1+((w.x*alpha_x)*(w.x*alpha_x)+(w.y*alpha_y)*(w.y*alpha_y))/(w.z*w.z)) - 1 ) / 2.0;
    };
    Real lambda_w_in = lambda(dir_in_local, alpha_x, alpha_y), lambda_w_out = lambda(dir_out_local, alpha_x, alpha_y);
    Real Gw_in = 1.0 / (1.0+lambda_w_in), Gw_out = 1.0 / (1.0+lambda_w_out);
    Real G_g = Gw_in*Gw_out;

    Spectrum f_glass = Vector3(0.0, 0.0, 0.0);
    if (dot(vertex.geometric_normal, dir_in)*dot(vertex.geometric_normal, dir_out)>0){
        f_glass = (F_g*D_g*G_g/4.0/nw_in) * base_color;
    } else {
        hw_in = dot(half_v, dir_in);
        hw_out = dot(half_v, dir_out);
        f_glass = ((1-F_g)*D_g*G_g*fabs(hw_out*hw_in)/nw_in/(hw_in+eta*hw_out)/(hw_in+eta*hw_out)) * sqrt(base_color);
    }
    return f_glass;
}

__host__ __device__ Real pdf_sample_bsdf_op::operator()(const DisneyGlass &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    // If we are going into the surface, then we use normal eta
    // (internal/external), otherwise we use external/internal.
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    assert(eta > 0);

    Vector3 half_vector;
    if (reflect) {
        half_vector = normalize(dir_in + dir_out);
    } else {
        // "Generalized half-vector" from Walter et al.
        // See "Microfacet Models for Refraction through Rough Surfaces"
        half_vector = normalize(dir_in + dir_out * eta);
    }

    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    Real hw_in = fabs(dot(half_vector,dir_in)), hw_out = fabs(dot(half_vector,dir_out));
    Real nw_in = fabs(dot(frame.n, dir_in)), nw_out = fabs(dot(frame.n, dir_out));
    
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    // Real R_0 = ((eta-1)*(eta-1))/((eta+1)*(eta+1));
    // Real one_minus_cos = 1.0 - dot(dir_in, half_vector);
    // Real F = R_0 + (1-R_0)*one_minus_cos*one_minus_cos*one_minus_cos*one_minus_cos*one_minus_cos;
    Real R_s = ( hw_in - eta*hw_out ) / ( hw_in + eta*hw_out );
    Real R_p = ( eta*hw_in - hw_out ) / ( eta*hw_in + hw_out );
    Real F = 0.5 * (R_s*R_s + R_p*R_p);
    
    assert(F>=0 && F<=1);

    Real aspect = sqrt(1-0.9*anisotropic);
    Real alpha_x = max(0.0001, roughness*roughness/aspect);
    Real alpha_y = max(0.0001, roughness*roughness*aspect);
    
    Vector3 h_in_local = to_local(frame, half_vector);
    Real _ = h_in_local[0]*h_in_local[0]/alpha_x/alpha_x + h_in_local[1]*h_in_local[1]/alpha_y/alpha_y + h_in_local[2]*h_in_local[2];
    Real D = 1.0/(c_PI*alpha_x*alpha_y*_*_);

    Vector3 dir_in_local = to_local(frame, dir_in);
    auto lambda = [](Vector3 w, double alpha_x, double alpha_y) -> double {
        return ( sqrt(1+((w.x*alpha_x)*(w.x*alpha_x)+(w.y*alpha_y)*(w.y*alpha_y))/(w.z*w.z)) - 1) / 2.0;
    };
    Real lambda_w_in = lambda(dir_in_local, alpha_x, alpha_y);
    Real G_in = 1.0 / (1.0+lambda_w_in);


    // We sample the visible normals, also we use F to determine
    // whether to sample reflection or refraction
    // so PDF ~ F * D * G_in for reflection, PDF ~ (1 - F) * D * G_in for refraction.
    Real h_dot_in = dot(half_vector, dir_in);
    // Real F = fresnel_dielectric(h_dot_in, eta);
    // Real D = GTR2(dot(half_vector, frame.n), roughness);
    // Real G_in = smith_masking_gtr2(to_local(frame, dir_in), roughness);


    if (reflect) {
        return (F * D * G_in) / (4 * fabs(dot(frame.n, dir_in)));
    } else {
        Real h_dot_out = dot(half_vector, dir_out);
        Real sqrt_denom = h_dot_in + eta * h_dot_out;
        Real dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        return (1 - F) * D * G_in * fabs(dh_dout * h_dot_in / dot(frame.n, dir_in));
    }
    return 0;
}

__host__ __device__ std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyGlass &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    

    Real aspect = sqrt(1-0.9*anisotropic);
    Real alpha_x = max(0.0001, roughness*roughness/aspect);
    Real alpha_y = max(0.0001, roughness*roughness*aspect);

    // Sample a micro normal and transform it to world space -- this is our half-vector.
    Vector3 local_dir_in = to_local(frame, dir_in);
    Vector3 local_micro_normal =
        sample_visible_normals_anisotropic(local_dir_in, alpha_x, alpha_y, rnd_param_uv);

    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Flip half-vector if it's below surface
    if (dot(half_vector, frame.n) < 0) {
        half_vector = -half_vector;
    }

    // Now we need to decide whether to reflect or refract.
    // We do this using the Fresnel term.
    Real h_dot_in = dot(half_vector, dir_in);
    Real F = fresnel_dielectric(h_dot_in, eta);
    // Real R_0 = ((eta-1)*(eta-1))/((eta+1)*(eta+1));
    // Real one_minus_cos = 1.0 - h_dot_in;
    // Real F = R_0 + (1-R_0)*one_minus_cos*one_minus_cos*one_minus_cos*one_minus_cos*one_minus_cos;

    if (rnd_param_w <= F) {
        // Reflection
        Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
        // set eta to 0 since we are not transmitting
        return BSDFSampleRecord{reflected, Real(0), roughness};
    } else {
        // Refraction
        // https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        // (note that our eta is eta2 / eta1, and l = -dir_in)
        Real h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
        if (h_dot_out_sq <= 0) {
            // Total internal reflection
            // This shouldn't really happen, as F will be 1 in this case.
            return {};
        }
        // flip half_vector if needed
        if (h_dot_in < 0) {
            half_vector = -half_vector;
        }
        Real h_dot_out= sqrt(h_dot_out_sq);
        Vector3 refracted = -dir_in / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
        return BSDFSampleRecord{refracted, eta, roughness};
    }
}

__host__ __device__ TextureSpectrum get_texture_op::operator()(const DisneyGlass &bsdf) const {
    return bsdf.base_color;
}
