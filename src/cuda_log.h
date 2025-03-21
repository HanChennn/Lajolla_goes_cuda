#include "scene.h"

__device__ inline void printMaterial(const Material& mat, int idx) {
    if (std::get_if<Lambertian>(&mat)) {
        printf("Material[%d]: Lambertian\n", idx);
    } else if (std::get_if<RoughPlastic>(&mat)) {
        printf("Material[%d]: RoughPlastic\n", idx);
    } else if (std::get_if<RoughDielectric>(&mat)) {
        printf("Material[%d]: RoughDielectric\n", idx);
    } else if (std::get_if<DisneyDiffuse>(&mat)) {
        printf("Material[%d]: DisneyDiffuse\n", idx);
    } else if (std::get_if<DisneyMetal>(&mat)) {
        printf("Material[%d]: DisneyMetal\n", idx);
    } else if (std::get_if<DisneyGlass>(&mat)) {
        printf("Material[%d]: DisneyGlass\n", idx);
    } else if (std::get_if<DisneyClearcoat>(&mat)) {
        printf("Material[%d]: DisneyClearcoat\n", idx);
    } else if (std::get_if<DisneySheen>(&mat)) {
        printf("Material[%d]: DisneySheen\n", idx);
    } else if (std::get_if<DisneyBSDF>(&mat)) {
        printf("Material[%d]: DisneyBSDF\n", idx);
    } else {
        printf("Material[%d]: Unknown Type\n", idx);
    }
}

__device__ inline void printTableDist1D(const TableDist1D& dist) {
    printf("  TableDist1D:\n");
    printf("    pmf (size=%d): ", (int)dist.pmf.size());
    for (int i = 0; i < dist.pmf.size(); i++) {
        printf("%f ", (float)dist.pmf[i]);
    }
    printf("\n    cdf (size=%d): ", (int)dist.cdf.size());
    for (int i = 0; i < dist.cdf.size(); i++) {
        printf("%f ", (float)dist.cdf[i]);
    }
    printf("\n");
}

__device__ inline void printTableDist2D(const TableDist2D& dist) {
    printf("  TableDist2D:\n");
    printf("    width=%d, height=%d, total_values=%f\n", dist.width, dist.height, dist.total_values);

    printf("    cdf_rows (size=%d): ", (int)dist.cdf_rows.size());
    for (int i = 0; i < dist.cdf_rows.size(); i++) {
        printf("%f ", (float)dist.cdf_rows[i]);
    }
    printf("\n    pdf_rows (size=%d): ", (int)dist.pdf_rows.size());
    for (int i = 0; i < dist.pdf_rows.size(); i++) {
        printf("%f ", (float)dist.pdf_rows[i]);
    }

    printf("\n    cdf_marginals (size=%d): ", (int)dist.cdf_marginals.size());
    for (int i = 0; i < dist.cdf_marginals.size(); i++) {
        printf("%f ", (float)dist.cdf_marginals[i]);
    }
    printf("\n    pdf_marginals (size=%d): ", (int)dist.pdf_marginals.size());
    for (int i = 0; i < dist.pdf_marginals.size(); i++) {
        printf("%f ", (float)dist.pdf_marginals[i]);
    }
    printf("\n");
}

__device__ inline void printScene(const Scene& scene) {
    printf("========== Scene ==========\n");

    printf("envmap_light_id: %d\n", scene.envmap_light_id);

    printf("Scene Bounds: center=(%f, %f, %f), radius=%f\n",
           scene.bounds.center.x, scene.bounds.center.y, scene.bounds.center.z, scene.bounds.radius);

    printf("\nMaterials count: %d\n", (int)scene.materials.size());
    for (int i = 0; i < scene.materials.size(); i++) {
        printMaterial(scene.materials[i], i);
    }

    printf("\nShapes count: %d\n", (int)scene.shapes.size());
    for (int i = 0; i < scene.shapes.size(); i++) {
        const auto& s = scene.shapes[i];
        if (auto* sp = std::get_if<Sphere>(&s)) {
            printf("Shape[%d]: Sphere pos=(%f, %f, %f), radius=%f, shape_id=%d, material_id=%d, area_light_id=%d\n",
                   i, sp->position.x, sp->position.y, sp->position.z, sp->radius, sp->shape_id, sp->material_id, sp->area_light_id);
        } else if (auto* tm = std::get_if<TriangleMesh>(&s)) {
            printf("Shape[%d]: TriangleMesh shape_id=%d, material_id=%d, area_light_id=%d, total_area=%f\n",
                   i, tm->shape_id, tm->material_id, tm->area_light_id, tm->total_area);

            printf("  positions (count=%d):\n", (int)tm->positions.size());
            for (int p = 0; p < tm->positions.size(); p++) {
                auto v = tm->positions[p];
                printf("    pos[%d] = (%f, %f, %f)\n", p, v.x, v.y, v.z);
            }

            printf("  indices (count=%d):\n", (int)tm->indices.size());
            for (int p = 0; p < tm->indices.size(); p++) {
                auto v = tm->indices[p];
                printf("    idx[%d] = (%d, %d, %d)\n", p, v.x, v.y, v.z);
            }

            printf("  normals (count=%d):\n", (int)tm->normals.size());
            for (int p = 0; p < tm->normals.size(); p++) {
                auto v = tm->normals[p];
                printf("    normal[%d] = (%f, %f, %f)\n", p, v.x, v.y, v.z);
            }

            printf("  uvs (count=%d):\n", (int)tm->uvs.size());
            for (int p = 0; p < tm->uvs.size(); p++) {
                auto v = tm->uvs[p];
                printf("    uv[%d] = (%f, %f)\n", p, v.x, v.y);
            }

            printTableDist1D(tm->triangle_sampler);
        }
    }

    printf("\nLights count: %d\n", (int)scene.lights.size());
    for (int i = 0; i < scene.lights.size(); i++) {
        const auto& l = scene.lights[i];
        if (auto* ll = std::get_if<DiffuseAreaLight>(&l))
            printf("Light[%d]: DiffuseAreaLight shape_id=%d, intensity=(%f, %f, %f)\n", i, ll->shape_id, ll->intensity.x, ll->intensity.y, ll->intensity.z);
        else if (auto* el = std::get_if<Envmap>(&l)) {
            printf("Light[%d]: Envmap scale=%f\n", i, el->scale);
            printTableDist2D(el->sampling_dist);
        }
    }

    printf("\nTexturePool image1s mipmap count: %d\n", (int)scene.texture_pool.image1s.size());
    printf("TexturePool image3s mipmap count: %d\n", (int)scene.texture_pool.image3s.size());

    printf("\nLight distribution table (light_dist):\n");
    printTableDist1D(scene.light_dist);

    printf("Render options: (add fields here if needed)\n");
    printf("==========================\n");
}