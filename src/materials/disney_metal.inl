#include "../microfacet.h"

// eval_op::
// const Vector3 &dir_in;
// const Vector3 &dir_out;
// const PathVertex &vertex;
// const TexturePool &texture_pool;
// const TransportDirection &dir;

// struct DisneyMetal {
//     Texture<Spectrum> base_color;
//     Texture<Real> roughness;
//     Texture<Real> anisotropic;
// };

__host__ __device__ Spectrum eval_op::operator()(const DisneyMetal &bsdf) const {
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
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_v = normalize(dir_in+dir_out);
    Real hw_out = fabs(dot(half_v, dir_out)), one_hw_out = 1 - hw_out;
    Real nw_in = fabs(dot(frame.n, dir_in)), nw_out = fabs(dot(frame.n, dir_out));
    Spectrum F_m = base_color + one_hw_out*one_hw_out*one_hw_out*one_hw_out*one_hw_out * (Vector3(1,1,1)-base_color);

    Real aspect = sqrt(1-0.9*anisotropic);
    Real alpha_x = max(0.0001, roughness*roughness/aspect);
    Real alpha_y = max(0.0001, roughness*roughness*aspect);
    // Real h_x = dot(half_v, frame.x), h_y = dot(half_v, frame.y), h_z = dot(half_v, frame.n);
    // Real _ = (h_x*h_x/alpha_x/alpha_x+h_y*h_y/alpha_y/alpha_y+h_z*h_z);
    Vector3 h_in_local = to_local(frame, half_v);
    Real _ = h_in_local[0]*h_in_local[0]/alpha_x/alpha_x + h_in_local[1]*h_in_local[1]/alpha_y/alpha_y + h_in_local[2]*h_in_local[2];
    Real D_m = 1.0/(c_PI*alpha_x*alpha_y*_*_);

    Vector3 dir_in_local = to_local(frame, dir_in), dir_out_local = to_local(frame, dir_out);
    // Vector3 dir_in_local = dir_in, dir_out_local = dir_out;
    Real lambda_w_in = (sqrt(1 + ((dir_in_local.x*alpha_x)*(dir_in_local.x*alpha_x)+(dir_in_local.y*alpha_y)*(dir_in_local.y*alpha_y))
                        /dir_in_local.z/dir_in_local.z)-1) / 2.0;
    Real lambda_w_out = (sqrt(1 + ((dir_out_local.x*alpha_x)*(dir_out_local.x*alpha_x)+(dir_out_local.y*alpha_y)*(dir_out_local.y*alpha_y))
                        /dir_out_local.z/dir_out_local.z)-1) / 2.0;
    Real G_w_in = 1.0 / (1.0+lambda_w_in), G_w_out = 1.0 / (1.0+lambda_w_out);
    Real G_m = G_w_in * G_w_out;

    Spectrum f_metal = (D_m * G_m /4.0 /nw_in) * F_m;
    
    return f_metal;
}

__host__ __device__ Real pdf_sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
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
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);

    Vector3 half_v = normalize(dir_in+dir_out);
    Real hw_in = fabs(dot(half_v, dir_in));
    Real nw_in = fabs(dot(frame.n, dir_in));

    Real aspect = sqrt(1-0.9*anisotropic);
    Real alpha_x = max(0.0001, roughness*roughness/aspect);
    Real alpha_y = max(0.0001, roughness*roughness*aspect);
    // Real h_x = dot(half_v, frame.x), h_y = dot(half_v, frame.y), h_z = dot(half_v, frame.n);
    // Real _ = (h_x*h_x/alpha_x/alpha_x+h_y*h_y/alpha_y/alpha_y+h_z*h_z);
    Vector3 h_in_local = to_local(frame, half_v);
    Real _ = h_in_local[0]*h_in_local[0]/alpha_x/alpha_x + h_in_local[1]*h_in_local[1]/alpha_y/alpha_y + h_in_local[2]*h_in_local[2];
    Real D_m = 1.0/(c_PI*alpha_x*alpha_y*_*_);

    Vector3 dir_in_local = to_local(frame, dir_in);
    Real lambda_w_in = (sqrt(1 + ((dir_in_local.x*alpha_x)*(dir_in_local.x*alpha_x)+(dir_in_local.y*alpha_y)*(dir_in_local.y*alpha_y))
                        /dir_in_local.z/dir_in_local.z)-1) / 2.0;
    Real G_w_in = 1.0 / (1.0+lambda_w_in);
    return D_m*G_w_in*max(0.0,hw_in)/nw_in/4/hw_in;
}

__host__ __device__ std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
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

    // Convert the incoming direction to local coordinates
    Vector3 local_dir_in = to_local(frame, dir_in);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real aspect = sqrt(1-0.9*anisotropic);

    // Clamp roughness to avoid numerical issues.
    roughness = std::clamp(roughness, Real(0.01), Real(1));
    Real alpha_x = max(0.0001, roughness*roughness/aspect);
    Real alpha_y = max(0.0001, roughness*roughness*aspect);

    Vector3 local_micro_normal =
        sample_visible_normals_anisotropic(local_dir_in, alpha_x, alpha_y, rnd_param_uv);
    
    // Transform the micro normal to world space
    Vector3 half_vector = to_world(frame, local_micro_normal);
    // Reflect over the world space normal
    Vector3 reflected = normalize(-dir_in + 2 * dot(dir_in, half_vector) * half_vector);
    return BSDFSampleRecord{
        reflected,
        Real(0) /* eta */, roughness /* roughness */
    };
}

__host__ __device__ TextureSpectrum get_texture_op::operator()(const DisneyMetal &bsdf) const {
    return bsdf.base_color;
}
