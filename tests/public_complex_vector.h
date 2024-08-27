#pragma once
#include "complex/qcu_complex.cuh"
#include <vector>
#include <iostream>

using InputFloat  = double;
using OutputFloat = double;

using std::vector;
using std::cout;
using std::endl;

inline bool check_correct (const vector<OutputFloat> & a, const vector<OutputFloat> & b) {
  int size = a.size();
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < size; ++i) {
    if (fabs(a[i] - b[i]) > 1e-12) 
    {
      cout << "Error" << endl;;
      return false;
    }
  }
  std::cout << "Correct" << std::endl;
  cout << "===============" << endl;
  return true;
}

void init_complex_randn_cpu (Complex<double>* a, int N) 
{
  for (int i = 0; i < N; ++i ) {
    a[i] = Complex<double> (rand()%16, rand()%16 + 1);
  }
}

void init_complex_1_0_cpu (Complex<double>* a, int N) 
{
  for (int i = 0; i < N; ++i ) {
    a[i] = Complex<double> (1, 0);
  }
}

void init_complex_1_2_cpu (Complex<double>* a, int N) 
{
  for (int i = 0; i < N; ++i ) {
    a[i] = Complex<double> (1, 2);
  }
}

void init_complex_range_1_N_cpu (Complex<double>* a, int N) 
{
  for (int i = 0; i < N; ++i ) {
    a[i] = Complex<double> (i + 1, 0);
  }
}