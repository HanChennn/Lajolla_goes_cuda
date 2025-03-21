#pragma once

#include "vector.h"
#include "frame.h"

/// An "PathVertex" represents a vertex of a light path.
/// We store the information we need for computing any sort of path contribution & sampling density.
struct PathVertex {
    Vector3 position;
    Vector3 geometric_normal; // always face at the same direction at shading_frame.n
    Frame shading_frame;
    Vector2 st; // A 2D parametrization of the surface. Irrelavant to UV mapping.
                // for triangle this is the barycentric coordinates, which we use
                // for interpolating the uv map.
    Vector2 uv; // The actual UV we use for texture fetching.
    // For texture filtering, stores approximatedly min(abs(du/dx), abs(dv/dx), abs(du/dy), abs(dv/dy))
    Real uv_screen_size;
    Real mean_curvature; // For ray differential propagation.
    Real ray_radius; // For ray differential propagation.
    int shape_id = -1;
    int primitive_id = -1; // For triangle meshes. This indicates which triangle it hits.
    int material_id = -1;
};

/// Convienent class used for storing a point on a surface.
/// Sometimes we will also use it for storing an infinitely far points (environment maps).
/// In that case the normal is the direction of the infinitely far point pointing towards the origin,
/// and position is a point on the scene bounding sphere.
struct PointAndNormal {
	Vector3 position;
	Vector3 normal;
};
