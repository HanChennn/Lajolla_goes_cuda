// #include "filter.h"

// struct sample_op {
//     __device__ inline Vector2 operator()(const Box &filter) const;
//     __device__ inline Vector2 operator()(const Tent &filter) const;
//     __device__ inline Vector2 operator()(const Gaussian &filter) const;

//     const Vector2 &rnd_param;
// };

// // Implementations of the individual filters.
// #include "filters/box.inl"
// #include "filters/tent.inl"
// #include "filters/gaussian.inl"

// __device__ inline Vector2 sample(const Filter &filter, const Vector2 &rnd_param) {
//     return std::visit(sample_op{rnd_param}, filter);
// }

// // __device__ inline Vector2 sample(const Filter &filter, const Vector2 &rnd_param) {
// //     sample_op op{rnd_param};

// //     if (std::holds_alternative<Box>(filter)) {
// //         return op(std::get<Box>(filter));
// //     } else if (std::holds_alternative<Tent>(filter)) {
// //         return op(std::get<Tent>(filter));
// //     } else if (std::holds_alternative<Gaussian>(filter)) {
// //         return op(std::get<Gaussian>(filter));
// //     }
// //     return Vector2{0.0f, 0.0f};
// // }
