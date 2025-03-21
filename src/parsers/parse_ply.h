#pragma once

#include "lajolla.h"
#include "matrix.h"

namespace parser{

struct TriangleMesh;

    /// Parse Stanford PLY files.
TriangleMesh parse_ply(const fs::path &filename, const Matrix4x4 &to_world);

}