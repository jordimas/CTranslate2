#include "int8_pack.h"
#include <omp.h>
#include <cstdint>
#include <iostream>
#include <vector>
int main() {
  namespace ex=ctranslate2::experimental;
  using D=ctranslate2::dim_t;
  const D n=127,k=127,m=1;
  std::vector<int8_t> weights(n*k,3),input(m*k,7);
  std::vector<int32_t> co(n,-128*3*k),output(m*n);
  omp_set_num_threads(8);
  auto bytes=ex::pack_int8_weights(weights.data(),true,k,n,1.f,nullptr);
  std::vector<uint64_t> storage((bytes+7)/8);
  auto* p=reinterpret_cast<int8_t*>(storage.data());
  ex::pack_int8_weights(weights.data(),true,k,n,1.f,p);
  omp_set_num_threads(1);
  std::cout<<"BEGIN packed with 8 threads, compute with 1"<<std::endl;
  try {
    ex::packed_int8_gemm(false,true,m,n,k,1.f,input.data(),k,p,k,0.f,output.data(),n,co.data());
  } catch(const std::invalid_argument& e) {
    std::cout<<"REJECTED "<<e.what()<<std::endl;return 0;
  }
  std::cout<<"UNEXPECTED compute returned"<<std::endl;
  return 1;
}
