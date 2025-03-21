__host__ __device__ Spectrum get_majorant_op::operator()(const HomogeneousMedium &m) {
    return m.sigma_a + m.sigma_s;
}

__host__ __device__ Spectrum get_sigma_s_op::operator()(const HomogeneousMedium &m) {
    return m.sigma_s;
}

__host__ __device__ Spectrum get_sigma_a_op::operator()(const HomogeneousMedium &m) {
    return m.sigma_a;
}
