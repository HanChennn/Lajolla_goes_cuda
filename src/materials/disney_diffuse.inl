// eval_op::
// const Vector3 &dir_in;
// const Vector3 &dir_out;
// const PathVertex &vertex;
// const TexturePool &texture_pool;
// const TransportDirection &dir;


// struct DisneyDiffuse {
//     Texture<Spectrum> base_color;
//     Texture<Real> roughness;
//     Texture<Real> subsurface;
// };


__host__ __device__ Spectrum eval_op::operator()(const DisneyDiffuse &bsdf) const {
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
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);


    Vector3 half_v = normalize(dir_in+dir_out);
    Real hw_out = fabs(dot(dir_out, half_v));
    // Real nw_in = fabs(dot(vertex.geometric_normal, dir_in)), nw_out = fabs(dot(vertex.geometric_normal, dir_out));
    Real nw_in = fabs(dot(frame.n, dir_in)), nw_out = fabs(dot(frame.n, dir_out));
    Real F_D90 = (Real)0.5 + 2 * roughness * hw_out * hw_out;
    Real F_Dw_in = (1 + (F_D90-1)*(1-nw_in*nw_in*nw_in*nw_in*nw_in));
    Real F_Dw_out = (1 + (F_D90-1)*(1-nw_out*nw_out*nw_out*nw_out*nw_out));
    Spectrum f_baseDiffuse = (Real)(1.0/c_PI * F_Dw_in * F_Dw_out * nw_out) * base_color;

    Real F_SS90 = roughness * hw_out*hw_out;
    Real F_SSw_in = (1 + (F_SS90-1)*(1-nw_in*nw_in*nw_in*nw_in*nw_in));
    Real F_SSw_out = (1 + (F_SS90-1)*(1-nw_out*nw_out*nw_out*nw_out*nw_out));
    Spectrum f_subsurface = (1.25/c_PI*(F_SSw_in*F_SSw_out*(1.0/(nw_in+nw_out)-0.5)+0.5)*nw_out) * base_color;

    Spectrum f_diffuse = (1-subsurface)*f_baseDiffuse +subsurface * f_subsurface;

    return f_diffuse;
}

__host__ __device__ Real pdf_sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
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

__host__ __device__ std::optional<BSDFSampleRecord> sample_bsdf_op::operator()(const DisneyDiffuse &bsdf) const {
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
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    return BSDFSampleRecord{
        to_world(frame, sample_cos_hemisphere(rnd_param_uv)),
        Real(0) /* eta */, roughness /* roughness */};
}

__host__ __device__ TextureSpectrum get_texture_op::operator()(const DisneyDiffuse &bsdf) const {
    return bsdf.base_color;
}
