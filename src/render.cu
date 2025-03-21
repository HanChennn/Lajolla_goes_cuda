#include "render.h"
#include "intersection.h"
#include "material.h"
#include "path_tracing.h"
#include "pcg.h"
#include "scene.h"
#include <driver_types.h>

#include "cuda_log.h"

__global__ void path_render(Scene* scene_ptr, Vector3 *img)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = bx * TILE_WIDTH + tx;
    int y = by * TILE_WIDTH + ty;

    const Scene& scene = *scene_ptr;
    int w = scene.camera.width, h = scene.camera.height;

    if (x >= w || y >= h)
        return;

    // DEBUG
    // if (x == 256 && y == 256)
    //     printScene(scene);

    // random number initialization
    pcg32_state rng = init_pcg32(y * w + x);

    Spectrum radiance = make_zero_spectrum();
    int spp = scene.options.samples_per_pixel;
    
    for (int s = 0; s < spp; s++) {
        radiance += path_tracing(scene, x, y, rng);
    }
    Spectrum color = radiance / Real(spp);

    // Set output pixel
    int img_pos = w * y + x;
    img[img_pos] = color;
}

__global__ void aux_render(Scene* scene_ptr, Vector3 *img)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = bx * TILE_WIDTH + tx;
    int y = by * TILE_WIDTH + ty;

    const Scene& scene = *scene_ptr;
    int w = scene.camera.width, h = scene.camera.height;

    if (x >= w || y >= h)
        return;

    Vector3 color{0, 0, 0};
    Ray ray = sample_primary(scene.camera, Vector2((x + Real(0.5)) / w, (y + Real(0.5)) / h));
    RayDifferential ray_diff = init_ray_differential(w, h);
    if (std::optional<PathVertex> vertex = intersect(scene.shapes, ray, ray_diff)) {
        Real dist = distance(vertex->position, ray.org);
        if (scene.options.integrator == Integrator::Depth) {
            color = Vector3{dist, dist, dist};
        } else if (scene.options.integrator == Integrator::ShadingNormal) {
            // color = (vertex->shading_frame.n + Vector3{1, 1, 1}) / Real(2);
            color = vertex->shading_frame.n;
        } else if (scene.options.integrator == Integrator::MeanCurvature) {
            Real kappa = vertex->mean_curvature;
            color = Vector3{kappa, kappa, kappa};
        } else if (scene.options.integrator == Integrator::RayDifferential) {
            color = Vector3{ray_diff.radius, ray_diff.spread, Real(0)};
        } else if (scene.options.integrator == Integrator::MipmapLevel) {
            const Material &mat = scene.materials[vertex->material_id];
            const TextureSpectrum &texture = get_texture(mat);
            auto *t = std::get_if<ImageTexture<Spectrum>>(&texture);
            if (t != nullptr) {
                const Mipmap3 &mipmap = get_img3(scene.texture_pool, t->texture_id);
                Vector2 uv{modulo(vertex->uv[0] * t->uscale, Real(1)),
                            modulo(vertex->uv[1] * t->vscale, Real(1))};
                // ray_diff.radius stores approximatedly dpdx,
                // but we want dudx -- we get it through
                // dpdx / dpdu
                Real footprint = vertex->uv_screen_size;
                Real scaled_footprint = max(get_width(mipmap), get_height(mipmap)) *
                                        max(t->uscale, t->vscale) * footprint;
                Real level = log2(max(scaled_footprint, Real(1e-8f)));
                color = Vector3{level, level, level};
            }
        } else {
            // G-buffers used for debugging
            // color = vertex->position;
            // color = vertex->geometric_normal;
            // color = Vector3{vertex->shape_id, vertex->primitive_id, vertex->material_id};
            // color = Vector3{vertex->st.x, vertex->st.y, 0.0};
            // color = vertex->shading_frame.n;
            // color = vertex->shading_frame.x;
            // color = vertex->shading_frame.y;
            // color = Vector3{vertex->uv.x, vertex->uv.y, 0.0};
            // color = Vector3{vertex->mean_curvature, vertex->ray_radius, vertex->uv_screen_size};
        }
    } else {
        color = Vector3{0, 0, 0};
    }

    // Set output pixel
    int img_pos = w * y + x;
    img[img_pos] = color;
}

Image3 render(const Scene &scene) {
    int w = scene.camera.width, h = scene.camera.height;
    Image3 img(w, h);

    Vector3* deviceImg;
    cudaMalloc((void **)&deviceImg, img.height * img.width * sizeof(Vector3));
    Scene* device_scene;
    cudaMalloc((void **)&device_scene, sizeof(Scene));
    cudaMemcpy(device_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice);

    // Kernel Init
    dim3 DimGrid(ceil(((float)img.width) / TILE_WIDTH), ceil(((float)img.height) / TILE_WIDTH), 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    if (scene.options.integrator == Integrator::Depth ||
            scene.options.integrator == Integrator::ShadingNormal ||
            scene.options.integrator == Integrator::MeanCurvature ||
            scene.options.integrator == Integrator::RayDifferential ||
            scene.options.integrator == Integrator::MipmapLevel) {
        cudaFuncSetCacheConfig(aux_render, cudaFuncCachePreferL1);
        aux_render<<<DimGrid, DimBlock>>>(device_scene, deviceImg);
    } else if (scene.options.integrator == Integrator::Path) {
        cudaFuncSetCacheConfig(path_render, cudaFuncCachePreferL1);
        path_render<<<DimGrid, DimBlock>>>(device_scene, deviceImg);
    } else {
        assert(false);
        return Image3();
    }

    cudaDeviceSynchronize();

    // Checking Errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    // Copy the device image back to the host
    cudaMemcpy(img.data.data(), deviceImg, img.height * img.width * sizeof(Vector3), cudaMemcpyDeviceToHost);

    // Free device memory
    // DeviceMemTracker::print();
    DeviceMemTracker::free();
    cudaFree(deviceImg);

    return img;
}