#pragma once

#include "lajolla.h"
#include "matrix.h"

namespace parser{

struct TriangleMesh;

/// Load Mitsuba's serialized file format.
TriangleMesh load_serialized(const fs::path &filename,
                             int shape_index,
                             const Matrix4x4 &to_world);

}