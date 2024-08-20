#include <cuda_fp16.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "kernel/shift_data_type.cuh"
using namespace qcu::device;
int main() {
    // check half, float, double
    half half_elem;
    float float_elem;
    double double_elem;

    // float to half
    float_elem = 1.232f;
    half_elem = shiftDataType<half, float>(1.232f);
    printf("float_elem = %lf, half_elem = %lf\n", float_elem, __half2float(half_elem));
    assert(fabs(float_elem - __half2float(half_elem)) < 1e-2);

    // half to float
    half_elem = __float2half(1.234f);
    float_elem = shiftDataType<float, half>(half_elem);
    printf("half_elem = %lf, float_elem = %lf\n", __half2float(half_elem), float_elem);
    assert(fabs(__half2float(half_elem) - float_elem) < 1e-2);

    // double to half
    double_elem = 3.2333;
    half_elem = shiftDataType<half, double>(3.2333);
    printf("double_elem = %lf, half_elem = %lf\n", double_elem, double(half_elem));
    assert(fabs(double_elem - double(half_elem)) < 1e-2);
    // half to double
    half_elem = __double2half(2.1356);
    double_elem = shiftDataType<double, half>(half_elem);
    printf("half_elem = %lf, double_elem = %lf\n", double(half_elem), double_elem);
    assert(fabs(double(half_elem) - double_elem) < 1e-2);
}