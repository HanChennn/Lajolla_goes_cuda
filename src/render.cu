#include<stdio.h>

#include "intersection.h"
#include "material.h"
#include "path_tracing.h"
// #include "vol_path_tracing.h"
#include "pcg.h"
#include "progress_reporter.h"
#include "render_cu.h"

struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};


__global__  void path_render_kernel(int tile_size, int w, int h, cudaScene *scene_cu, Image3 *img){
    // int x[2] = {int(blockIdx.x * tile_size), std::min<int>(int(blockIdx.x * tile_size + tile_size), w)};
    // int y[2] = {int(blockIdx.y * tile_size), std::min<int>(int(blockIdx.y * tile_size + tile_size), h)};
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("YES YOU RUN THE KERNEL! %d %d\n",idx,idy);
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    pcg32_state rng = init_pcg32(blockIdx.y * num_tiles_x + blockIdx.x);
    __syncthreads();

    // printf("%d %d %0.2f\n",idx, idy, scene_cu->texture_pool.image3s[0].images[0].data[5000].y);
    // Shape* shape_ptr = &(scene_cu->shapes[2]);/* 指向一个 Shape */;
    // if (auto* mesh = std::get_if<TriangleMesh>(shape_ptr)) {
    //     printf("indice x y z ID: %d %d %d\n",mesh->indices[1].x,mesh->indices[1].y,mesh->indices[1].z);
    // } 
    if (idx>=w || idy >=h) return;

    Spectrum radiance = make_zero_spectrum();
    int spp = scene_cu->options.samples_per_pixel;
    for (int s = 0; s < spp; s++) {
        Spectrum update = make_zero_spectrum();
        update =  path_tracing(scene_cu, idx, idy, rng);
        if(update.x!=0 || update.y!=0 || update.z!=0){
            printf("%d %d %0.2f %0.2f %0.2f\n",idx,idy,update.x,update.y,update.z);
        }
        radiance = radiance + update;
    }
    img->data[idy*w+idx] = radiance / Real(spp);
    return;
}

ParsedImage3 path_render_launch(const Scene &scene) {
    // BVH
    BVH Parsed_bvh;
    Parsed_bvh.build(scene.shapes);

    cuda_bvh temp_bvh;
    temp_bvh.len_shape_indices = Parsed_bvh.shape_indices.size();
    temp_bvh.node_count = Parsed_bvh.node_count;

    BVHNode *raw_nodes;
    cudaMalloc((void**)&raw_nodes, Parsed_bvh.node_count*sizeof(BVHNode));
    std::unique_ptr<BVHNode, CudaDeleter> nodes(raw_nodes, CudaDeleter());
    cudaMemcpy(nodes.get(), Parsed_bvh.nodes.data(), Parsed_bvh.node_count * sizeof(BVHNode), cudaMemcpyHostToDevice);

    int* raw_int;
    cudaMalloc((void**)&raw_int, Parsed_bvh.shape_indices.size()*sizeof(int));
    std::unique_ptr<int, CudaDeleter> shape_indicies(raw_int, CudaDeleter());
    cudaMemcpy(shape_indicies.get(), Parsed_bvh.shape_indices.data(), Parsed_bvh.shape_indices.size()*sizeof(int), cudaMemcpyHostToDevice);

    // 使用 unique_ptr 来管理 CUDA 内存
    cudaScene temp;
    std::vector<std::unique_ptr<Material, CudaDeleter>> gpu_materials_store; // 存储 GPU 指针，防止悬空指针
    std::vector<std::unique_ptr<Medium, CudaDeleter>> gpu_media_store;

    // 申请 GPU 内存
    cudaScene* raw_scene_cu = nullptr;
    cudaMalloc((void**)&raw_scene_cu, sizeof(cudaScene));
    std::unique_ptr<cudaScene, CudaDeleter> scene_cu(raw_scene_cu, CudaDeleter());

    Material* raw_materials = nullptr;
    cudaMalloc((void**)&raw_materials, scene.materials.size() * sizeof(Material));
    std::unique_ptr<Material, CudaDeleter> materials(raw_materials, CudaDeleter());

    Medium* raw_media = nullptr;
    cudaMalloc((void**)&raw_media, scene.media.size() * sizeof(Medium));
    std::unique_ptr<Medium, CudaDeleter> media(raw_media, CudaDeleter());

    // **确保数据不为空后再拷贝**
    cudaMemcpy(materials.get(), scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    temp.len_materials = scene.materials.size();
    temp.len_lights = scene.lights.size();
    temp.len_media = scene.media.size();
    temp.len_shapes = scene.shapes.size();

    gpu_materials_store.push_back(std::move(materials));
    gpu_media_store.push_back(std::move(media));

    temp.materials = gpu_materials_store[0].get();
    temp.media = gpu_media_store[0].get(); // ignore now
    temp.bounds = scene.bounds;
    temp.options = scene.options;
    temp.camera = scene.camera;
    temp.envmap_light_id = scene.envmap_light_id;


    // 处理light
    std::vector<Light> temp_light(scene.lights.size());
    std::vector<std::unique_ptr<Real, CudaDeleter>> gpu_cdf_rows_store;
    std::vector<std::unique_ptr<Real, CudaDeleter>> gpu_pdf_rows_store;
    std::vector<std::unique_ptr<Real, CudaDeleter>> gpu_cdf_marginals_store;
    std::vector<std::unique_ptr<Real, CudaDeleter>> gpu_pdf_marginals_store;

    for (int i = 0; i < scene.lights.size(); i++) {
        if (auto light = std::get_if<DiffuseAreaLight>(&scene.lights[i])) {
            // temp_light[i] = scene.lights[i];
            temp_light[i] = std::get<DiffuseAreaLight>(scene.lights[i]);
        } else if (auto mesh = std::get_if<ParsedEnvmap>(&scene.lights[i])) {
            TableDist2D temp_table;
            Real* raw_cdf_rows = nullptr;
            Real* raw_pdf_rows = nullptr;
            Real* raw_cdf_marginals = nullptr;
            Real* raw_pdf_marginals = nullptr;

            cudaMalloc((void**)&raw_cdf_rows, mesh->sampling_dist.cdf_rows.size() * sizeof(Real));
            cudaMalloc((void**)&raw_pdf_rows, mesh->sampling_dist.pdf_rows.size() * sizeof(Real));
            cudaMalloc((void**)&raw_cdf_marginals, mesh->sampling_dist.cdf_marginals.size() * sizeof(Real));
            cudaMalloc((void**)&raw_pdf_marginals, mesh->sampling_dist.pdf_marginals.size() * sizeof(Real));
            

            std::unique_ptr<Real, CudaDeleter> cdf_rows(raw_cdf_rows, CudaDeleter());
            std::unique_ptr<Real, CudaDeleter> pdf_rows(raw_pdf_rows, CudaDeleter());
            std::unique_ptr<Real, CudaDeleter> cdf_marginals(raw_cdf_marginals, CudaDeleter());
            std::unique_ptr<Real, CudaDeleter> pdf_marginals(raw_pdf_marginals, CudaDeleter());

            cudaMemcpy(cdf_rows.get(), mesh->sampling_dist.cdf_rows.data(), 
                        mesh->sampling_dist.cdf_rows.size() * sizeof(Real), cudaMemcpyHostToDevice);
            cudaMemcpy(pdf_rows.get(), mesh->sampling_dist.pdf_rows.data(), 
                        mesh->sampling_dist.pdf_rows.size() * sizeof(Real), cudaMemcpyHostToDevice);
            cudaMemcpy(cdf_marginals.get(), mesh->sampling_dist.cdf_marginals.data(), 
                        mesh->sampling_dist.cdf_marginals.size() * sizeof(Real), cudaMemcpyHostToDevice);
            cudaMemcpy(pdf_marginals.get(), mesh->sampling_dist.pdf_marginals.data(), 
                        mesh->sampling_dist.pdf_marginals.size() * sizeof(Real), cudaMemcpyHostToDevice);

            gpu_cdf_rows_store.push_back(std::move(cdf_rows));
            gpu_pdf_rows_store.push_back(std::move(pdf_rows));
            gpu_cdf_marginals_store.push_back(std::move(cdf_marginals));
            gpu_pdf_marginals_store.push_back(std::move(pdf_marginals));

            temp_table.cdf_rows = gpu_cdf_rows_store.back().get();
            temp_table.pdf_rows = gpu_pdf_rows_store.back().get();
            temp_table.cdf_marginals = gpu_cdf_marginals_store.back().get();
            temp_table.pdf_marginals = gpu_pdf_marginals_store.back().get();

            Envmap temp_envmap;
            temp_envmap.sampling_dist = temp_table;
            temp_envmap.scale = mesh->scale;
            temp_envmap.to_local = mesh->to_local;
            temp_envmap.to_world = mesh->to_world;
            temp_envmap.values = mesh->values;

            temp_light[i] = std::move(temp_envmap);
        }
    }
    std::vector<std::unique_ptr<Light, CudaDeleter>> gpu_lights_store;

    Light* raw_lights = nullptr;
    cudaMalloc((void**)&raw_lights, scene.lights.size() * sizeof(Light));
    std::unique_ptr<Light, CudaDeleter> lights(raw_lights, CudaDeleter());
    cudaMemcpy(lights.get(), temp_light.data(), scene.lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    // ✅ 确保 `lights` 的生命周期正确
    gpu_lights_store.push_back(std::move(lights));
    temp.lights = gpu_lights_store.back().get();


    // 处理 light_dist
    TableDist1D light_dist;
    std::vector<std::unique_ptr<Real, CudaDeleter>> gpu_light_dist_store; // 存储 GPU 指针，防止悬空指针

    Real* raw_light_pmf = nullptr;
    Real* raw_light_cdf = nullptr;
    cudaMalloc((void**)&raw_light_pmf, scene.light_dist.pmf.size() * sizeof(Real));
    cudaMalloc((void**)&raw_light_cdf, scene.light_dist.cdf.size() * sizeof(Real));
    std::unique_ptr<Real, CudaDeleter> light_pmf(raw_light_pmf, CudaDeleter());
    std::unique_ptr<Real, CudaDeleter> light_cdf(raw_light_cdf, CudaDeleter());
    cudaMemcpy(light_pmf.get(), scene.light_dist.pmf.data(), scene.light_dist.pmf.size() * sizeof(Real), cudaMemcpyHostToDevice);
    cudaMemcpy(light_cdf.get(), scene.light_dist.cdf.data(), scene.light_dist.cdf.size() * sizeof(Real), cudaMemcpyHostToDevice);

    // **确保 GPU 指针不会被过早释放**
    gpu_light_dist_store.push_back(std::move(light_pmf));
    gpu_light_dist_store.push_back(std::move(light_cdf));

    light_dist.pmf = gpu_light_dist_store[0].get();
    light_dist.cdf = gpu_light_dist_store[1].get();
    light_dist.len_pmf = scene.light_dist.pmf.size();
    light_dist.len_cdf = scene.light_dist.cdf.size();

    // 确保 temp.light_dist 仍然持有 GPU 内存
    temp.light_dist = light_dist;


    // 处理 Shape
    std::vector<Shape> temp_shape(scene.shapes.size());
    std::vector<std::unique_ptr<Shape, CudaDeleter>> gpu_shape_store; // 存储 GPU 形状指针，防止提前释放
    std::vector<std::unique_ptr<Vector3, CudaDeleter>> gpu_positions_store;
    std::vector<std::unique_ptr<Vector3i, CudaDeleter>> gpu_indices_store;
    std::vector<std::unique_ptr<Vector3, CudaDeleter>> gpu_normals_store;
    std::vector<std::unique_ptr<Vector2, CudaDeleter>> gpu_uvs_store;
    std::vector<std::unique_ptr<Real, CudaDeleter>> gpu_pmf_store;
    std::vector<std::unique_ptr<Real, CudaDeleter>> gpu_cdf_store;

    for (int i = 0; i < scene.shapes.size(); i++) {
        if (auto sphere = std::get_if<Sphere>(&scene.shapes[i])) {
            temp_shape[i] = std::get<Sphere>(scene.shapes[i]);
        } else if (auto mesh = std::get_if<ParsedTriangleMesh>(&scene.shapes[i])) {
            // 为 mesh 申请 GPU 内存
            Vector3* raw_positions = nullptr;
            Vector3i* raw_indices = nullptr;
            Vector3* raw_normals = nullptr;
            Vector2* raw_uvs = nullptr;

            cudaMalloc((void**)&raw_positions, mesh->positions.size() * sizeof(Vector3));
            cudaMalloc((void**)&raw_indices, mesh->indices.size() * sizeof(Vector3i));
            cudaMalloc((void**)&raw_normals, mesh->normals.size() * sizeof(Vector3));
            cudaMalloc((void**)&raw_uvs, mesh->uvs.size() * sizeof(Vector2));

            std::unique_ptr<Vector3, CudaDeleter> positions(raw_positions, CudaDeleter());
            std::unique_ptr<Vector3i, CudaDeleter> indices(raw_indices, CudaDeleter());
            std::unique_ptr<Vector3, CudaDeleter> normals(raw_normals, CudaDeleter());
            std::unique_ptr<Vector2, CudaDeleter> uvs(raw_uvs, CudaDeleter());

            cudaMemcpy(positions.get(), mesh->positions.data(), mesh->positions.size() * sizeof(Vector3), cudaMemcpyHostToDevice);
            cudaMemcpy(indices.get(), mesh->indices.data(), mesh->indices.size() * sizeof(Vector3i), cudaMemcpyHostToDevice);
            cudaMemcpy(normals.get(), mesh->normals.data(), mesh->normals.size() * sizeof(Vector3), cudaMemcpyHostToDevice);
            cudaMemcpy(uvs.get(), mesh->uvs.data(), mesh->uvs.size() * sizeof(Vector2), cudaMemcpyHostToDevice);

            gpu_positions_store.push_back(std::move(positions));
            gpu_indices_store.push_back(std::move(indices));
            gpu_normals_store.push_back(std::move(normals));
            gpu_uvs_store.push_back(std::move(uvs));

            // 处理 triangle_sampler
            TableDist1D temp_table;
            Real* raw_tri_pmf = nullptr;
            Real* raw_tri_cdf = nullptr;

            cudaMalloc((void**)&raw_tri_pmf, mesh->triangle_sampler.pmf.size() * sizeof(Real));
            cudaMalloc((void**)&raw_tri_cdf, mesh->triangle_sampler.cdf.size() * sizeof(Real));

            std::unique_ptr<Real, CudaDeleter> tri_pmf(raw_tri_pmf, CudaDeleter());
            std::unique_ptr<Real, CudaDeleter> tri_cdf(raw_tri_cdf, CudaDeleter());

            cudaMemcpy(tri_pmf.get(), mesh->triangle_sampler.pmf.data(), mesh->triangle_sampler.pmf.size() * sizeof(Real), cudaMemcpyHostToDevice);
            cudaMemcpy(tri_cdf.get(), mesh->triangle_sampler.cdf.data(), mesh->triangle_sampler.cdf.size() * sizeof(Real), cudaMemcpyHostToDevice);

            gpu_pmf_store.push_back(std::move(tri_pmf));
            gpu_cdf_store.push_back(std::move(tri_cdf));

            temp_table.pmf = gpu_pmf_store.back().get();
            temp_table.cdf = gpu_cdf_store.back().get();

            TriangleMesh temp_tri(
                gpu_positions_store.back().get(),
                gpu_indices_store.back().get(),
                gpu_normals_store.back().get(),
                gpu_uvs_store.back().get(),
                mesh->total_area,
                temp_table
            );
            temp_tri.material_id = mesh->material_id;
            temp_tri.area_light_id = mesh->area_light_id;
            temp_tri.interior_medium_id = mesh->interior_medium_id;
            temp_tri.exterior_medium_id = mesh->exterior_medium_id;

            temp_tri.len_positions = mesh->positions.size();
            temp_tri.len_indices = mesh->indices.size();
            temp_tri.len_normals = mesh->normals.size();
            temp_tri.len_uvs = mesh->uvs.size();

            temp_shape[i] = std::move(temp_tri);
        }
    }

    // ✅ 分配 GPU 内存，存储 `Shape`
    Shape* raw_shapes = nullptr;
    cudaMalloc((void**)&raw_shapes, scene.shapes.size() * sizeof(Shape));
    std::unique_ptr<Shape, CudaDeleter> shapes(raw_shapes, CudaDeleter());
    cudaMemcpy(shapes.get(), temp_shape.data(), scene.shapes.size() * sizeof(Shape), cudaMemcpyHostToDevice);

    // ✅ 确保 `shapes` 的生命周期正确
    gpu_shape_store.push_back(std::move(shapes));
    temp.shapes = gpu_shape_store.back().get();



    TexturePool texture_pool;
    texture_pool.len_images1s = scene.texture_pool.image1s.size();
    texture_pool.len_images3s = scene.texture_pool.image3s.size();
    
    std::vector<Mipmap1> image1s;
    std::vector<std::unique_ptr<Mipmap1, CudaDeleter>> gpu_image1s_store; // 确保 image1s 生命周期
    std::vector<std::unique_ptr<Image<Real>, CudaDeleter>> gpu_images1_store; // 确保 images 生命周期
    std::vector<std::unique_ptr<Real, CudaDeleter>> gpu_data1_store; // 确保 data 生命周期
    
    for (int i = 0; i < scene.texture_pool.image1s.size(); i++) {
        using T = typename decltype(scene.texture_pool.image1s[i].images[0].data)::value_type;
        std::vector<Image<T>> images;
    
        for (int j = 0; j < scene.texture_pool.image1s[i].images.size(); j++) {
            Image<T> temp_img;
            temp_img.height = scene.texture_pool.image1s[i].images[j].height;
            temp_img.width = scene.texture_pool.image1s[i].images[j].width;
    
            T* raw_gpu_data = nullptr;
            cudaMalloc((void**)&raw_gpu_data, scene.texture_pool.image1s[i].images[j].data.size() * sizeof(T));
            std::unique_ptr<T, CudaDeleter> gpu_data(raw_gpu_data, CudaDeleter());
            cudaMemcpy(gpu_data.get(), scene.texture_pool.image1s[i].images[j].data.data(),
                       scene.texture_pool.image1s[i].images[j].data.size() * sizeof(T), cudaMemcpyHostToDevice);
    
            temp_img.data = gpu_data.get();
            gpu_data1_store.push_back(std::move(gpu_data)); // **确保 data 不被释放**
            images.push_back(temp_img);
        }
    
        Image<T>* raw_gpu_images = nullptr;
        cudaMalloc((void**)&raw_gpu_images, images.size() * sizeof(Image<T>));
        std::unique_ptr<Image<T>, CudaDeleter> gpu_images(raw_gpu_images, CudaDeleter());
        cudaMemcpy(gpu_images.get(), images.data(), images.size() * sizeof(Image<T>), cudaMemcpyHostToDevice);
    
        Mipmap1 temp_mipmap;
        temp_mipmap.len = scene.texture_pool.image1s[i].images.size();
        temp_mipmap.images = gpu_images.get();
        
        gpu_images1_store.push_back(std::move(gpu_images)); // **确保 images 不被释放**
        image1s.push_back(temp_mipmap);
    }
    
    Mipmap1* raw_gpu_image1s = nullptr;
    cudaMalloc((void**)&raw_gpu_image1s, image1s.size() * sizeof(Mipmap1));
    std::unique_ptr<Mipmap1, CudaDeleter> gpu_image1s(raw_gpu_image1s, CudaDeleter());
    cudaMemcpy(gpu_image1s.get(), image1s.data(), image1s.size() * sizeof(Mipmap1), cudaMemcpyHostToDevice);
    
    texture_pool.image1s = gpu_image1s.get();
    gpu_image1s_store.push_back(std::move(gpu_image1s)); // **确保 image1s 不被释放**
    
    
    //  **处理 image3s**
    std::vector<Mipmap3> image3s;
    std::vector<std::unique_ptr<Mipmap3, CudaDeleter>> gpu_image3s_store; // 确保 image3s 生命周期
    std::vector<std::unique_ptr<Image<Spectrum>, CudaDeleter>> gpu_images3_store; // 确保 images 生命周期
    std::vector<std::unique_ptr<Spectrum, CudaDeleter>> gpu_data3_store; // 确保 data 生命周期
    
    for (int i = 0; i < scene.texture_pool.image3s.size(); i++) {
        using T = typename decltype(scene.texture_pool.image3s[i].images[0].data)::value_type;
        std::vector<Image<T>> images;
    
        for (int j = 0; j < scene.texture_pool.image3s[i].images.size(); j++) {
            Image<T> temp_img;
            temp_img.height = scene.texture_pool.image3s[i].images[j].height;
            temp_img.width = scene.texture_pool.image3s[i].images[j].width;
    
            T* raw_gpu_data = nullptr;
            cudaMalloc((void**)&raw_gpu_data, scene.texture_pool.image3s[i].images[j].data.size() * sizeof(T));
            std::unique_ptr<T, CudaDeleter> gpu_data(raw_gpu_data, CudaDeleter());
            cudaMemcpy(gpu_data.get(), scene.texture_pool.image3s[i].images[j].data.data(),
                       scene.texture_pool.image3s[i].images[j].data.size() * sizeof(T), cudaMemcpyHostToDevice);
    
            temp_img.data = gpu_data.get();
            gpu_data3_store.push_back(std::move(gpu_data)); // **确保 data 不被释放**
            images.push_back(temp_img);
        }
    
        Image<T>* raw_gpu_images = nullptr;
        cudaMalloc((void**)&raw_gpu_images, images.size() * sizeof(Image<T>));
        std::unique_ptr<Image<T>, CudaDeleter> gpu_images(raw_gpu_images, CudaDeleter());
        cudaMemcpy(gpu_images.get(), images.data(), images.size() * sizeof(Image<T>), cudaMemcpyHostToDevice);
    
        Mipmap3 temp_mipmap;
        temp_mipmap.len = scene.texture_pool.image3s[i].images.size();
        temp_mipmap.images = gpu_images.get();
        
        gpu_images3_store.push_back(std::move(gpu_images)); // **确保 images 不被释放**
        image3s.push_back(temp_mipmap);
    }
    
    Mipmap3* raw_gpu_image3s = nullptr;
    cudaMalloc((void**)&raw_gpu_image3s, image3s.size() * sizeof(Mipmap3));
    std::unique_ptr<Mipmap3, CudaDeleter> gpu_image3s(raw_gpu_image3s, CudaDeleter());
    cudaMemcpy(gpu_image3s.get(), image3s.data(), image3s.size() * sizeof(Mipmap3), cudaMemcpyHostToDevice);
    
    texture_pool.image3s = gpu_image3s.get();
    gpu_image3s_store.push_back(std::move(gpu_image3s)); // **确保 image3s 不被释放**  
    
    temp.texture_pool = texture_pool;
    
    // 复制到 GPU
    cudaMemcpy(scene_cu.get(), &temp, sizeof(cudaScene), cudaMemcpyHostToDevice);

    // 渲染图像
    int w = scene.camera.width, h = scene.camera.height;
    
    constexpr int tile_size = 16;
    int num_tiles_x = (w + tile_size - 1) / tile_size;
    int num_tiles_y = (h + tile_size - 1) / tile_size;
    dim3 blockDim(tile_size, tile_size);
    dim3 gridDim(num_tiles_x, num_tiles_y);

    Image3* raw_img_cu;
    cudaMalloc((void**)&raw_img_cu, w*h*sizeof(Vector3));
    std::unique_ptr<Image3, CudaDeleter> img_cu(raw_img_cu, CudaDeleter());
    path_render_kernel<<<gridDim, blockDim>>>(tile_size, w, h, scene_cu.get(), img_cu.get());

    ParsedImage3 img;
    img.height = h;
    img.width = w;
    img.data = std::vector<Vector3>(w * h);
    cudaMemcpy(img.data.data(), img_cu.get(), w*h*sizeof(Vector3), cudaMemcpyDeviceToHost);
    // const ParsedShape* shape_ptr = &(scene.shapes[2]);
    // if (auto* mesh = std::get_if<ParsedTriangleMesh>(shape_ptr)) {
    //     printf("Material ID: %d %d %d\n",mesh->indices[1].x, mesh->indices[1].y, mesh->indices[1].z);
    // } 
    // printf("%0.2f\n",scene.texture_pool.image3s[0].images[0].data[5000].y);
    return img;
}