#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <taskflow/cudaflow.hpp>
#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <typeinfo>
#include <vector>

// Parameters
using value_t = double;
constexpr size_t N = 257 << 20; // number of elements
constexpr size_t M = 5;         // number of executions

// timing
using time_point_t = std::chrono::high_resolution_clock::time_point;
time_point_t tic();
double toc(const time_point_t &);
// find_if
void find_if();
// transform
void transform();
// reduce
void reduce();
// transform_reduce
void transform_reduce();
// transform_and_reduce
void transform_and_reduce();
// sort
void sort();

// main
int main() {
  std::cout << "Type\t" << typeid(value_t).name() << '\n';
  std::cout << "#Elements\t" << N << '\n';
  std::cout << "#Executions\t" << M << '\n';
  std::cout << std::setprecision(3);
  std::cout << "\nAlgorithm\tThrust\tcudaFlow\n";

  find_if();
  transform();
  reduce();
  transform_reduce();
  transform_and_reduce();
  sort();

  return 0;
}

// timing
time_point_t tic() { return std::chrono::high_resolution_clock::now(); }

double toc(const time_point_t &start) {
  // elapsed time in seconds
  return std::chrono::duration_cast<std::chrono::duration<double>>(tic() -
                                                                   start)
      .count();
}

// find_if
void find_if() {
  std::cout << "find_if\t";
  thrust::host_vector<value_t> h_v(N);
  thrust::sequence(h_v.begin(), h_v.end());
  thrust::device_vector<value_t> d_v;
  time_point_t t;
  double T; // total time
  tf::cudaFlow cf;
  auto pred = [n = N / 2] __device__(const auto &x) { return x > n; };

  // thrust::find_if
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    t = tic();
    thrust::find_if(d_v.begin(), d_v.end(), pred);
    T += toc(t);
  }
  std::cout << (T / M) << '\t';

  // cudaFlow::find_if
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    auto p = thrust::raw_pointer_cast(d_v.data());
    auto ind = tf::cuda_malloc_device<unsigned>(1);
    cf.clear();
    t = tic();
    cf.find_if(p, p + N, ind, pred);
    cf.offload();
    T += toc(t);
  }
  std::cout << (T / M) << '\n';
}

// transform
void transform() {
  std::cout << "transform\t";
  thrust::host_vector<value_t> h_v(N);
  thrust::sequence(h_v.begin(), h_v.end());
  thrust::device_vector<value_t> d_v;
  time_point_t t;
  double T; // total time
  tf::cudaFlow cf;
  auto op = [] __device__(const auto &x) { return -x; };

  // thrust::transform
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    t = tic();
    thrust::transform(d_v.begin(), d_v.end(), d_v.begin(), op);
    T += toc(t);
  }
  std::cout << (T / M) << '\t';

  // cudaFlow::transform
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    auto p = thrust::raw_pointer_cast(d_v.data());
    cf.clear();
    t = tic();
    cf.transform(p, p + N, p, op);
    cf.offload();
    T += toc(t);
  }
  std::cout << (T / M) << '\n';
}

// reduce
void reduce() {
  std::cout << "reduce\t";
  thrust::host_vector<value_t> h_v(N, 1);
  thrust::device_vector<value_t> d_v;
  time_point_t t;
  double T; // total time
  tf::cudaFlow cf;
  auto bop = [] __device__(const auto &a, const auto &x) { return a + x; };

  // thrust::reduce
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    t = tic();
    thrust::reduce(d_v.begin(), d_v.end(), 0, bop);
    T += toc(t);
  }
  std::cout << (T / M) << '\t';

  // cudaFlow::reduce
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    auto p = thrust::raw_pointer_cast(d_v.data());
    auto res = tf::cuda_malloc_shared<int>(1);
    cf.clear();
    t = tic();
    cf.reduce(p, p + N, res, bop);
    cf.offload();
    T += toc(t);
  }
  std::cout << (T / M) << '\n';
}

// transform_reduce
void transform_reduce() {
  std::cout << "transform_reduce\t";
  thrust::host_vector<value_t> h_v(N, 1);
  thrust::device_vector<value_t> d_v;
  time_point_t t;
  double T; // total time
  tf::cudaFlow cf;
  auto op = [] __device__(const auto &x) { return -x; };
  auto bop = [] __device__(const auto &a, const auto &x) { return a + x; };

  // thrust::transform_reduce
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    t = tic();
    thrust::transform_reduce(d_v.begin(), d_v.end(), op, 0, bop);
    T += toc(t);
  }
  std::cout << (T / M) << '\t';

  // cudaFlow::transform_reduce
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    auto p = thrust::raw_pointer_cast(d_v.data());
    auto res = tf::cuda_malloc_shared<int>(1);
    cf.clear();
    t = tic();
    cf.transform_uninitialized_reduce(p, p + N, res, bop, op);
    cf.offload();
    T += toc(t);
  }
  std::cout << (T / M) << '\n';
}

// transform_and_reduce
void transform_and_reduce() {
  std::cout << "transform_and_reduce\t";
  thrust::host_vector<value_t> h_v(N, 1);
  thrust::device_vector<value_t> d_v;
  time_point_t t;
  double T; // total time
  tf::cudaFlow cf;
  auto op = [] __device__(const auto &x) { return -x; };
  auto bop = [] __device__(const auto &a, const auto &x) { return a + x; };

  // thrust::transform_and_reduce
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    t = tic();
    auto tmp = thrust::device_malloc<int>(N);
    thrust::transform(d_v.begin(), d_v.end(), tmp, op);
    thrust::reduce(tmp, tmp + N, 0, bop);
    thrust::device_free(tmp);
    T += toc(t);
  }
  std::cout << (T / M) << '\t';

  // cudaFlow::transform_and_reduce
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    auto p = thrust::raw_pointer_cast(d_v.data());
    auto res = tf::cuda_malloc_shared<int>(1);
    cf.clear();
    t = tic();
    auto tmp = tf::cuda_malloc_device<int>(N);
    cf.transform(p, p + N, tmp, op);
    cf.uninitialized_reduce(tmp, tmp + N, res, bop);
    cf.offload();
    tf::cuda_free(tmp);
    T += toc(t);
  }
  std::cout << (T / M) << '\n';
}

// sort
void sort() {
  std::cout << "sort\t";
  thrust::host_vector<value_t> h_v(N);
  std::generate(h_v.begin(), h_v.end(), std::rand);
  thrust::device_vector<value_t> d_v;
  time_point_t t;
  double T; // total time
  tf::cudaFlow cf;
  auto comp = [] __device__(const auto &a, const auto &b) { return a < b; };

  // thrust::sort
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    t = tic();
    thrust::sort(d_v.begin(), d_v.end(), comp);
    T += toc(t);
  }
  std::cout << (T / M) << '\t';

  // cudaFlow::sort
  T = 0;
  for (size_t i = 0; i < M; ++i) {
    d_v = h_v;
    auto p = thrust::raw_pointer_cast(d_v.data());
    cf.clear();
    t = tic();
    cf.sort(p, p + N, comp);
    cf.offload();
    T += toc(t);
  }
  std::cout << (T / M) << '\n';
}