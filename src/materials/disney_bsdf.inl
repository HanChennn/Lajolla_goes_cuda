// Texture<Spectrum> base_color;
// Texture<Real> specular_transmission;
// Texture<Real> metallic;
// Texture<Real> subsurface;
// Texture<Real> specular;
// Texture<Real> roughness;
// Texture<Real> specular_tint;
// Texture<Real> anisotropic;
// Texture<Real> sheen;
// Texture<Real> sheen_tint;
// Texture<Real> clearcoat;
// Texture<Real> clearcoat_gloss;

// const Vector3 &dir_in;
// const Vector3 &dir_out;
// const PathVertex &vertex;
// const TexturePool &texture_pool;
// const TransportDirection &dir;

#include "../microfacet.h"

__host__ __device__ Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    // (void)reflect; // silence unuse warning, remove this when implementing hw
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_tint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen_tint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real eta = dot(vertex.geometric_normal, dir_in) > 0 ? bsdf.eta : 1 / bsdf.eta;
    // Real eta = bsdf.eta;

    Vector3 half_vector = normalize(dir_in+dir_out);
    Real hw_in = dot(half_vector, dir_in), hw_out = dot(half_vector, dir_out);
    Real nw_in = dot(frame.n, dir_in), nw_out = dot(frame.n, dir_out);
    Real one_minus_hw_out = 1- hw_out;
    Real aspect = sqrt(1-0.9*anisotropic);
    Real alpha_x = max(0.0001, roughness*roughness/aspect);
    Real alpha_y = max(0.0001, roughness*roughness*aspect);

    // Metal:
    Spectrum f_metal = make_zero_spectrum();
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {}
    else {
        Frame frame = vertex.shading_frame;
        if (dot(frame.n, dir_in) < 0) {
            frame = -frame;
        }
        Spectrum C_tint= luminance(base_color)>0? base_color/luminance(base_color):Vector3(1,1,1);
        Spectrum Ks = (1-specular_tint) + specular_tint*C_tint;
        Real R_0_eta = ((eta-1)*(eta-1)) / ((eta+1)*(eta+1));
        Spectrum C_0 = specular*R_0_eta*(1-metallic)*Ks + metallic*base_color;
        Spectrum F_m = C_0 + (Vector3(1,1,1)-C_0) * one_minus_hw_out * one_minus_hw_out * one_minus_hw_out * one_minus_hw_out * one_minus_hw_out;

        Vector3 h_in_local = to_local(frame, half_vector);
        Real _ = h_in_local[0]*h_in_local[0]/alpha_x/alpha_x + h_in_local[1]*h_in_local[1]/alpha_y/alpha_y + h_in_local[2]*h_in_local[2];
        Real D_m = 1.0/(c_PI*alpha_x*alpha_y*_*_);

        Vector3 dir_in_local = to_local(frame, dir_in), dir_out_local = to_local(frame, dir_out);
        auto lambda = [](Vector3 w, double alpha_x, double alpha_y) -> double {
            return ( sqrt(1+((w.x*alpha_x)*(w.x*alpha_x)+(w.y*alpha_y)*(w.y*alpha_y))/(w.z*w.z)) - 1 ) / 2.0;
        };
        Real lambda_w_in = lambda(dir_in_local, alpha_x, alpha_y), lambda_w_out = lambda(dir_out_local, alpha_x, alpha_y);
        Real Gw_in = 1.0 / (1.0+lambda_w_in), Gw_out = 1.0 / (1.0+lambda_w_out);
        Real G_m = Gw_in * Gw_out;

        f_metal = (D_m * G_m /4.0 /fabs(nw_in)) * F_m;
    }

    // Diffuse:
    DisneyDiffuse disneyDiffuseInstance{
        std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.roughness),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.subsurface)
    };
    Material material = disneyDiffuseInstance;
    Spectrum f_diffuse = eval(material, dir_in, dir_out, vertex, texture_pool);

    // Clearcoat:
    DisneyClearcoat DisneyClearcoatInstance{
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.clearcoat_gloss)
    };
    material = DisneyClearcoatInstance;
    Spectrum f_clearcoat = eval(material, dir_in, dir_out, vertex, texture_pool);

    // Glass:
    DisneyGlass DisneyGlassInstance{
        std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.roughness),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.anisotropic),
        bsdf.eta
    };
    material = DisneyGlassInstance;
    Spectrum f_glass = eval(material, dir_in, dir_out, vertex, texture_pool);

    // if (dot(dir_in, frame.n)<=0) return (1.0-metallic)*specular_transmission*f_glass;

    // Sheen:
    DisneySheen DisneySheenInstance{
        std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.sheen_tint)
    };
    material = DisneySheenInstance;
    Spectrum f_sheen = eval(material, dir_in, dir_out, vertex, texture_pool);
    
    if (dot(dir_in, frame.n)<=0){
        f_diffuse = f_clearcoat = f_metal = f_sheen = Vector3(0,0,0);
    }

    Spectrum f_disney = (1.0-specular_transmission)*(1.0-metallic)*f_diffuse +
                        (1.0-metallic)*sheen*f_sheen +
                        (1.0-specular_transmission*(1.0-metallic))*f_metal +
                        0.25*clearcoat*f_clearcoat +
                        (1.0-metallic)*specular_transmission*f_glass;

    return f_disney;
}

__host__ __device__ Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    bool reflect = dot(vertex.geometric_normal, dir_in) *
                   dot(vertex.geometric_normal, dir_out) > 0;
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Vector4 weight = Vector4(
        (1.0-specular_transmission)*(1.0-metallic),        // diffuse
        (1.0-specular_transmission*(1.0-metallic)),        // metal
        (1.0-metallic)*specular_transmission,              // glass
        0.25*clearcoat                                     // clearcoat
    );

    if (dot(dir_in, frame.n)<=0) weight = Vector4(0.0,0.0,1.0,0.0);
    else weight = weight/(weight[0]+weight[1]+weight[2]+weight[3]);

    // printf("%0.3f %0.3f %0.3f %0.3f\n", weight[0], weight[1], weight[2], weight[3]);

    Material material;
    Real pdf = 0;
    
    // weight = Vector4(0.0,0.0,1.0,0.0);
    // Diffuse:
    DisneyDiffuse disneyDiffuseInstance{
        std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.roughness),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.subsurface)
    };
    material = disneyDiffuseInstance;
    pdf += weight[0] * std::visit(pdf_sample_bsdf_op{dir_in, dir_out, vertex, texture_pool, dir}, material);

    // Metal:
    DisneyMetal disneyMetalInstance{
        std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.roughness),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.anisotropic)
    };
    material = disneyMetalInstance;
    pdf += weight[1] * std::visit(pdf_sample_bsdf_op{dir_in, dir_out, vertex, texture_pool, dir}, material);

    // Glass:
    DisneyGlass DisneyGlassInstance{
        std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.roughness),
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.anisotropic),
        bsdf.eta
    };
    material = DisneyGlassInstance;
    pdf += weight[2] * std::visit(pdf_sample_bsdf_op{dir_in, dir_out, vertex, texture_pool, dir}, material);

    // Clearcoat:
    DisneyClearcoat DisneyClearcoatInstance{
        std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.clearcoat_gloss)
    };
    material = DisneyClearcoatInstance;
    pdf += weight[3] * std::visit(pdf_sample_bsdf_op{dir_in, dir_out, vertex, texture_pool, dir}, material);

    return pdf;
}

__host__ __device__ std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }
    // Homework 1: implement this!
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Vector4 weight = Vector4(
        (1.0-specular_transmission)*(1.0-metallic),
        (1.0-specular_transmission*(1.0-metallic)),
        (1.0-metallic)*specular_transmission,
        0.25*clearcoat
    );

    if (dot(dir_in, frame.n)<=0) weight = Vector4(0.0,0.0,1.0,0.0);
    else weight = weight/(weight[0]+weight[1]+weight[2]+weight[3]);

    // weight = Vector4(0.0,0.0,1.0,0.0);
    Material material;
    if(rnd_param_w<=weight[0] && weight[0]!=0){
        // Diffuse:
        DisneyDiffuse disneyDiffuseInstance{
            std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
            std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.roughness),
            std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.subsurface)
        };
        material = disneyDiffuseInstance;
        return std::visit(sample_bsdf_op{dir_in, vertex, texture_pool, rnd_param_uv, rnd_param_w/weight[0], dir}, material);
    } else if (rnd_param_w>weight[0] && rnd_param_w<=weight[0]+weight[1] && weight[1]!=0) {
        // Metal:
        DisneyMetal disneyMetalInstance{
            std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
            std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.roughness),
            std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.anisotropic)
        };
        material = disneyMetalInstance;
        return std::visit(sample_bsdf_op{dir_in, vertex, texture_pool, rnd_param_uv, (rnd_param_w-weight[0])/weight[1], dir}, material);
    } else if (rnd_param_w>weight[0]+weight[1] && rnd_param_w<=weight[0]+weight[1]+weight[2] && weight[2]!=0) {
        // Glass:
        DisneyGlass DisneyGlassInstance{
            std::visit([](auto&& texture) -> Texture<Spectrum> { return texture; }, bsdf.base_color),
            std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.roughness),
            std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.anisotropic),
            bsdf.eta
        };
        material = DisneyGlassInstance;
        return std::visit(sample_bsdf_op{dir_in, vertex, texture_pool, rnd_param_uv, (rnd_param_w-weight[0]-weight[1])/weight[2], dir}, material);
    } else if (rnd_param_w>weight[0]+weight[1]+weight[2] && weight[3]!=0) {
        // Clearcoat:
        DisneyClearcoat DisneyClearcoatInstance{
            std::visit([](auto&& texture) -> Texture<Real> { return texture; }, bsdf.clearcoat_gloss)
        };
        material = DisneyClearcoatInstance;
        return std::visit(sample_bsdf_op{dir_in, vertex, texture_pool, rnd_param_uv, (rnd_param_w-weight[0]-weight[1]-weight[2])/weight[3], dir}, material);
    }
    return BSDFSampleRecord { Vector3(0,0,0),0,0};
}

__host__ __device__ TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}
