#pragma once

#include "lajolla.h"
#include "image.h"
#include <memory>
#include "render_cu.h"

struct Scene;

ParsedImage3 render(const Scene &scene);
