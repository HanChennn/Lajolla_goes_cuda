#pragma once

#include "lajolla.h"
#include "matrix.h"
#include "shape.h"

/// Parse Stanford PLY files.
ParsedTriangleMesh parse_ply(const fs::path &filename, const Matrix4x4 &to_world);
