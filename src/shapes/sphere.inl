/// Numerically stable quadratic equation solver at^2 + bt + c = 0
/// See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
/// returns false when it can't find solutions.
__device__ inline bool solve_quadratic(Real a, Real b, Real c, Real *t0, Real *t1) {
    // Degenerated case
    if (a == 0) {
        if (b == 0) {
            return false;
        }
        *t0 = *t1 = -c / b;
        return true;
    }

    Real discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    }
    Real root_discriminant = sqrt(discriminant);
    if (b >= 0) {
        *t0 = (- b - root_discriminant) / (2 * a);
        *t1 = 2 * c / (- b - root_discriminant);
    } else {
        *t0 = 2 * c / (- b + root_discriminant);
        *t1 = (- b + root_discriminant) / (2 * a);
    }
    return true;
}

__device__ inline PointAndNormal sample_point_on_shape_op::operator()(const Sphere &sphere) const {
    // https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#x2-SamplingSpheres
    const Vector3 &center = sphere.position;
    const Real &r = sphere.radius;

    if (distance_squared(ref_point, center) < r * r) {
        // If the reference point is inside the sphere, just sample the whole sphere uniformly
        Real z = 1 - 2 * uv.x;
        Real r_ = sqrt(fmax(Real(0), 1 - z * z));
        Real phi = 2 * c_PI * uv.y;
        Vector3 offset(r_ * cos(phi), r_ * sin(phi), z);
        Vector3 position = center + r * offset;
        Vector3 normal = offset;
        return PointAndNormal{position, normal};
    }

    // Otherwise sample a ray inside a cone towards the sphere center.

    // Build a coordinate system with n pointing towards the sphere
    Vector3 dir_to_center = normalize(center - ref_point);
    Frame frame(dir_to_center);

    // These are not exactly "elevation" and "azimuth": elevation here
    // stands for the extended angle of the cone, and azimuth here stands
    // for the polar coordinate angle on the substended disk.
    // I just don't like the theta/phi naming convention...
    Real sin_elevation_max_sq = r * r / distance_squared(ref_point, center);
    Real cos_elevation_max = sqrt(max(Real(0), 1 - sin_elevation_max_sq));
    // Uniformly interpolate between 1 (angle 0) and max
    Real cos_elevation = (1 - uv[0]) + uv[0] * cos_elevation_max;
    Real sin_elevation = sqrt(max(Real(0), 1 - cos_elevation * cos_elevation));
    Real azimuth = uv[1] * 2 * c_PI;

    // Now we have a ray direction and a sphere, we can just ray trace and find
    // the intersection point. Pbrt uses an more clever and numerically robust
    // approach which I will just shamelessly copy here.
    Real dc = distance(ref_point, center);
    Real ds = dc * cos_elevation -
        sqrt(max(Real(0), r * r - dc * dc * sin_elevation * sin_elevation));
    Real cos_alpha = (dc * dc + r * r - ds * ds) / (2 * dc * r);
    Real sin_alpha = sqrt(max(Real(0), 1 - cos_alpha * cos_alpha));
    // Add negative sign since normals point outwards.
    Vector3 n_on_sphere = -to_world(frame,
        Vector3{sin_alpha * cos(azimuth),
                sin_alpha * sin(azimuth),
                cos_alpha});
    Vector3 p_on_sphere = r * n_on_sphere + center;
    return PointAndNormal{p_on_sphere, n_on_sphere};
}

__device__ inline Real surface_area_op::operator()(const Sphere &sphere) const {
    return 4 * c_PI * sphere.radius * sphere.radius;
}

__device__ inline Real pdf_point_on_shape_op::operator()(const Sphere &sphere) const {
    // https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources#x2-SamplingSpheres
    const Vector3 &center = sphere.position;
    const Real &r = sphere.radius;

    if (distance_squared(ref_point, center) < r * r) {
        // If the reference point is inside the sphere, just sample the whole sphere uniformly
        return 1 / surface_area_op{}(sphere);
    }
    
    Real sin_elevation_max_sq = r * r / distance_squared(ref_point, center);
    Real cos_elevation_max = sqrt(max(Real(0), 1 - sin_elevation_max_sq));
    // Uniform sampling PDF of a cone.
    Real pdf_solid_angle = 1 / (2 * c_PI * (1 - cos_elevation_max));
    // Convert it back to area measure
    Vector3 p_on_sphere = point_on_shape.position;
    Vector3 n_on_sphere = point_on_shape.normal;
    Vector3 dir = normalize(p_on_sphere - ref_point);
    return pdf_solid_angle * fabs(dot(n_on_sphere, dir)) /
        distance_squared(ref_point, p_on_sphere);
}

__device__ inline ShadingInfo compute_shading_info_op::operator()(const Sphere &sphere) const {
    // To compute the shading frame, we use the geometry normal as normal,
    // and dpdu as one of the tangent vector. 
    // We use the azimuthal angle as u, and the elevation as v, 
    // thus the point p on sphere and u, v has the following relationship:
    // p = center + {r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v)}
    // thus dpdu = {-r * sin(u) * sin(v), r * cos(u) * sin(v), 0}
    //      dpdv = { r * cos(u) * cos(v), r * sin(u) * cos(v), - r * sin(v)}
    Vector3 dpdu{-sphere.radius * sin(vertex.st[0]) * sin(vertex.st[1]),
                  sphere.radius * cos(vertex.st[0]) * sin(vertex.st[1]),
                 Real(0)};
    Vector3 dpdv{ sphere.radius * cos(vertex.st[0]) * cos(vertex.st[1]),
                  sphere.radius * sin(vertex.st[0]) * cos(vertex.st[1]),
                 -sphere.radius * sin(vertex.st[1])};
    // dpdu may not be orthogonal to shading normal:
    // subtract the projection of shading_normal onto dpdu to make them orthogonal
    Vector3 tangent = normalize(
        dpdu - vertex.geometric_normal * dot(vertex.geometric_normal, dpdu));
    Frame shading_frame(tangent,
                        normalize(cross(vertex.geometric_normal, tangent)),
                        vertex.geometric_normal);
    return ShadingInfo{vertex.st,
                       shading_frame,
                       1 / sphere.radius, /* mean curvature */
                       (length(dpdu) + length(dpdv)) / 2};
}

__device__ inline std::optional<PathVertex> intersect_op::operator()(const Sphere &sphere) const {
    // Our sphere is ||p - x||^2 = r^2
    // substitute x = o + d * t, we want to solve for t
    // ||p - (o + d * t)||^2 = r^2
    // (p.x - (o.x + d.x * t))^2 + (p.y - (o.y + d.y * t))^2 + (p.z - (o.z + d.z * t))^2 - r^2 = 0
    // (d.x^2 + d.y^2 + d.z^2) t^2 + 2 * (d.x * (o.x - p.x) + d.y * (o.y - p.y) + d.z * (o.z - p.z)) t + 
    // ((p.x-o.x)^2 + (p.y-o.y)^2 + (p.z-o.z)^2  - r^2) = 0
    // A t^2 + B t + C
    Vector3 v = ray.org - sphere.position;
    Real A = dot(ray.dir, ray.dir);
    Real B = 2 * dot(ray.dir, v);
    Real C = dot(v, v) - sphere.radius * sphere.radius;
    Real t0, t1;
    if (!solve_quadratic(A, B, C, &t0, &t1)) {
        // No intersection
        return {};
    }
    // This can happen due to numerical inaccuracies
    if (t0 > t1) {
        Real tmp = t0;
        t0 = t1; t1 = tmp;
    }

    Real t = -1;
    if (t0 >= ray.tnear && t0 < ray.tfar) {
        t = t0;
    }
    if (t1 >= ray.tnear && t1 < ray.tfar && t < 0) {
        t = t1;
    }

    if (t >= ray.tnear && t < ray.tfar) {
        PathVertex vertex;
        // Record the intersection
        vertex.position = ray.org + t * ray.dir;
        vertex.geometric_normal = vertex.position - sphere.position;
        // We use the spherical coordinates as uv
        Vector3 cartesian = vertex.geometric_normal / sphere.radius;
        // https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
        // We use the convention that y is up axis.
        Real elevation = acos(std::clamp(cartesian.y, Real(-1), Real(1)));
        Real azimuth = atan2(cartesian.z, cartesian.x);
        vertex.st = { azimuth / c_TWOPI, elevation / c_PI };
        vertex.uv = vertex.st;
        vertex.primitive_id = -1;
        vertex.material_id = sphere.material_id;
        vertex.shape_id = sphere.shape_id;


        // To compute the shading frame, we use the geometry normal as normal,
        // and dpdu as one of the tangent vector. 
        // We use the azimuthal angle as u, and the elevation as v, 
        // thus the point p on sphere and u, v has the following relationship:
        // p = center + {r * cos(u) * sin(v), r * sin(u) * sin(v), r * cos(v)}
        // thus dpdu = {-r * sin(u) * sin(v), r * cos(u) * sin(v), 0}
        //      dpdv = { r * cos(u) * cos(v), r * sin(u) * cos(v), - r * sin(v)}
        Vector3 dpdu{-sphere.radius * sin(vertex.st[0]) * sin(vertex.st[1]),
                sphere.radius * cos(vertex.st[0]) * sin(vertex.st[1]),
            Real(0)};
        Vector3 dpdv{ sphere.radius * cos(vertex.st[0]) * cos(vertex.st[1]),
                sphere.radius * sin(vertex.st[0]) * cos(vertex.st[1]),
            -sphere.radius * sin(vertex.st[1])};
        // dpdu may not be orthogonal to shading normal:
        // subtract the projection of shading_normal onto dpdu to make them orthogonal
        Vector3 tangent = normalize(
        dpdu - vertex.geometric_normal * dot(vertex.geometric_normal, dpdu));
        Frame shading_frame(tangent,
                    normalize(cross(vertex.geometric_normal, tangent)),
                    vertex.geometric_normal);

        vertex.shading_frame = shading_frame;
        vertex.mean_curvature = 1 / sphere.radius;
        vertex.ray_radius = transfer(ray_diff, distance(ray.org, vertex.position));
        // vertex.ray_radius stores approximatedly dp/dx, 
        // we get uv_screen_size (du/dx) using (dp/dx)/(dp/du)
        vertex.uv_screen_size = vertex.ray_radius / (length(dpdu) + length(dpdv)) * 2;

        // Flip the geometry normal to the same direction as the shading normal
        if (dot(vertex.geometric_normal, vertex.shading_frame.n) < 0) {
            vertex.geometric_normal = -vertex.geometric_normal;
        }

        return vertex;
    }

    return {};
}

__device__ inline bool occluded_op::operator()(const Sphere &sphere) const {
    // Our sphere is ||p - x||^2 = r^2
    // substitute x = o + d * t, we want to solve for t
    // ||p - (o + d * t)||^2 = r^2
    // (p.x - (o.x + d.x * t))^2 + (p.y - (o.y + d.y * t))^2 + (p.z - (o.z + d.z * t))^2 - r^2 = 0
    // (d.x^2 + d.y^2 + d.z^2) t^2 + 2 * (d.x * (o.x - p.x) + d.y * (o.y - p.y) + d.z * (o.z - p.z)) t + 
    // ((p.x-o.x)^2 + (p.y-o.y)^2 + (p.z-o.z)^2  - r^2) = 0
    // A t^2 + B t + C
    Vector3 v = ray.org - sphere.position;
    Real A = dot(ray.dir, ray.dir);
    Real B = 2 * dot(ray.dir, v);
    Real C = dot(v, v) - sphere.radius * sphere.radius;
    Real t0, t1;
    if (!solve_quadratic(A, B, C, &t0, &t1)) {
        // No intersection
        return false;
    }
    // This can happen due to numerical inaccuracies
    if (t0 > t1) {
        Real tmp = t0;
        t0 = t1; t1 = tmp;
    }

    Real t = -1;
    if (t0 >= ray.tnear && t0 < ray.tfar) {
        t = t0;
    }
    if (t1 >= ray.tnear && t1 < ray.tfar && t < 0) {
        t = t1;
    }

    if (t >= ray.tnear && t < ray.tfar) {
        return true;
    } else {
        return false;
    }
}