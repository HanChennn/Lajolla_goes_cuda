#pragma once

#include "lajolla.h"
#include "matrix.h"

namespace parser{

struct TriangleMesh;

/// Parse Wavefront obj files. Currently only supports triangles and quads.
/// Throw errors if encountered general polygons.
TriangleMesh parse_obj(const fs::path &filename, const Matrix4x4 &to_world);

}