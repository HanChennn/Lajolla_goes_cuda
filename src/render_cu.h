#ifndef RENDER_CU_H
#define RENDER_CU_H

#include "scene.h"  // 确保 Scene 的定义
#include <cuda_runtime.h>
// #include "device_launch_parameters.h"


// 确保 `Scene` 在 `namespace` 里的正确使用
ParsedImage3 path_render_launch(const Scene &scene);

#endif // RENDER_CU_H
