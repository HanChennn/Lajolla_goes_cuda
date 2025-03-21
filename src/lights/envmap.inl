__device__ inline PointAndNormal sample_point_on_light_envmap(const Envmap &light, const Vector2 &rnd_param_uv) {
    Vector2 uv = sample(light.sampling_dist, rnd_param_uv);
    // Convert uv to spherical coordinates
    Real azimuth = uv[0] * (2 * c_PI);
    Real elevation = uv[1] * c_PI;
    // Convert spherical coordinates to Cartesian coordinates.
    // (https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates)
    // We use the convention that y is the up axis.
    Vector3 local_dir{sin(azimuth) * sin(elevation),
                      cos(elevation),
                      -cos(azimuth) * sin(elevation)};
    Vector3 world_dir = xform_vector(light.to_world, local_dir);
    return PointAndNormal{Vector3{0, 0, 0}, -world_dir};
}

__device__ inline Real pdf_point_on_light_envmap(const Envmap &light, const PointAndNormal &point_on_light) {
    // We store the direction pointing outwards from light in point_on_light.normal.
    Vector3 world_dir = -point_on_light.normal;
    // Convert the direction to local Catesian coordinates.
    Vector3 local_dir = xform_vector(light.to_local, world_dir);
    // Convert the Cartesian coordinates to the spherical coordinates.
    // We use the convention that y is the up-axis.
    Vector2 uv{atan2(local_dir[0], -local_dir[2]) * c_INVTWOPI,
               acos(std::clamp(local_dir[1], Real(-1), Real(1))) * c_INVPI};
    // atan2 returns -pi to pi, we map [-pi, 0] to [pi, 2pi]
    if (uv[0] < 0) {
        uv[0] += 1;
    }
    Real cos_elevation = local_dir.y;
    Real sin_elevation = sqrt(std::clamp(1 - cos_elevation * cos_elevation, Real(0), Real(1)));
    if (sin_elevation <= 0) {
        // degenerate
        return 0;
    }
    return pdf(light.sampling_dist, uv) / (2 * c_PI * c_PI * sin_elevation);
}

__device__ inline Spectrum emission_envmap(const Envmap &light, const Vector3 &view_dir, Real view_footprint, const TexturePool& texture_pool) {
    // View dir is pointing outwards "from" the light.
    // An environment map stores the light from the opposite direction,
    // so we need to flip view dir.
    // We then transform the direction to the local Cartesian coordinates.
    Vector3 local_dir = xform_vector(light.to_local, -view_dir);
    // Convert the Cartesian coordinates to the spherical coordinates.
    Vector2 uv{atan2(local_dir[0], -local_dir[2]) * c_INVTWOPI,
               acos(std::clamp(local_dir[1], Real(-1), Real(1))) * c_INVPI};
    // atan2 returns -pi to pi, we map [-pi, 0] to [pi, 2pi]
    if (uv[0] < 0) {
        uv[0] += 1;
    }

    // For envmap, view_footprint stores (approximatedly) d view_dir / dx
    // We want to convert it to du/dx -- we do it by computing (d dir / dx) * (d u / d dir)
    // To do this we differentiate through the process above:
    
    // We abbrevite local_dir as w
    Vector3 w = local_dir;
    Real dudwx = -w.z / (w.x * w.x + w.z * w.z);
    Real dudwz = w.x / (w.x * w.x + w.z * w.z);
    Real dvdwy = -1 / sqrt(std::max(1 - w.y * w.y, Real(0)));
    // We only want to know the length of dudw & dvdw
    // The local coordinate transformation is length preserving,
    // so we don't need to differentiate through it.
    Real footprint = min(sqrt(dudwx * dudwx + dudwz * dudwz), dvdwy);

    return eval(light.values, uv, footprint, texture_pool) * light.scale;
}