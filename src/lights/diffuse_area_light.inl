__host__ __device__ inline PointAndNormal sample_point_on_light_diffuse_area_light(const DiffuseAreaLight &light, const Vector3 &ref_point, const Vector2 &rnd_param_uv, const Real &rnd_param_w, const Shape* shapes) const {
    const Shape &shape = shapes[light.shape_id];
    return sample_point_on_shape(shape, ref_point, rnd_param_uv, rnd_param_w);
}

__host__ __device__ inline Real pdf_point_on_light_diffuse_area_light(const DiffuseAreaLight &light, const PointAndNormal &point_on_light, const Vector3 &ref_point, const Shape* shapes) const {
    return pdf_point_on_shape(
        shapes[light.shape_id], point_on_light, ref_point);
}

__host__ __device__ inline Spectrum emission_diffuse_area_light(const DiffuseAreaLight &light, const Vector3 &view_dir, const PointAndNormal &point_on_light) const {
    if (dot(point_on_light.normal, view_dir) <= 0) {
        return make_zero_spectrum();
    }
    return light.intensity;
}
