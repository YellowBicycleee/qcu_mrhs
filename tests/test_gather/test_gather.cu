#include <stdio.h>
#include <vector>
#include "qcu_public.h"
#include "data_format/qcu_data_format_shift.cuh"
#include "check_error/check_cuda.cuh"
using namespace std;
using namespace qcu;

void init (double *a, int len) {
    for (int i = 0; i < len; ++i) {
        a[i] = i;
    }
}
double compare (double *a, double *b, int len) {
    double res = 0;
    for (int i = 0; i < len; ++i) {
        res += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return res;
}

void print12(double *a) {
    for (int i = 0; i < 12; ++i) {
        printf("%lf ", a[i]);
        if (i % 4 == 3) {
            printf("\n");
        }
    }
    printf("\n");
}
int main () {
    int Lx = 4; 
    int Ly = 4;
    int Lz = 4;
    int Lt = 4;
    int nColor = 3;
    int half_Lx = Lx / 2;
    int vol = half_Lx * Ly * Lz * Lt * 4 * nColor;

    int mInput = 1;

    void *d_src;
    void *d_dst;
    double *h_src;
    double *h_dst;
    void** d_lookup_table;
    void** h_lookup_table;

    h_lookup_table = (void **)malloc(sizeof(void *) * mInput);
    cudaMalloc(&d_lookup_table, sizeof(void *) * mInput);
    

    h_src = (double *)malloc(vol * sizeof(double) * 2);
    h_dst = (double *)malloc(vol * sizeof(double) * 2);

    cudaMalloc(&d_src, vol * sizeof(double) * 2);
    cudaMalloc(&d_dst, vol * sizeof(double) * 2);
    printf("d_src = %p, d_dst = %p\n", d_src, d_dst);

    init(h_src, vol * 2);    
    CHECK_CUDA(cudaMemcpy(d_src, h_src, vol * sizeof(double) * 2, cudaMemcpyHostToDevice));

    h_lookup_table[0] = (void*) d_src;
    CHECK_CUDA(cudaMemcpy(d_lookup_table, h_lookup_table, sizeof(void *) * mInput, cudaMemcpyHostToDevice));

    colorSpinorGather(d_dst, QCU_DOUBLE_PRECISION, d_lookup_table, QCU_DOUBLE_PRECISION, Lx, Ly, Lz, Lt, nColor, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
    // colorSpinorScatter(d_dst, QCU_DOUBLE_PRECISION, d_src, QCU_DOUBLE_PRECISION, Lx, Ly, Lz, Lt, nColor, 1);
    CHECK_CUDA(cudaMemcpy(h_dst, d_dst, vol * sizeof(double) * 2, cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(h_src, d_src, vol * sizeof(double) * 2, cudaMemcpyDeviceToHost));
    double res = compare(h_dst, h_src, vol);
    printf("res = %f\n", res);
    
    print12(h_dst);
    print12(h_src);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_lookup_table);
    free(h_src);
    free(h_dst);
    free(h_lookup_table);
}