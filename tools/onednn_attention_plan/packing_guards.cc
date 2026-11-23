// Explicit rejection and historical raw-fallback checks, separate from the
// supported compact numerical matrix. Link with the actual adapter source.
#include "int8_pack.h"
#include <oneapi/dnnl/dnnl.h>
#include <omp.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace ex=ctranslate2::experimental;
using D=ctranslate2::dim_t;
int main() {
  omp_set_num_threads(8);
  const D n=1280,k=1280,m=6;
  std::mt19937 rng(43);
  std::vector<int8_t> weights(n*k);
  for(auto& x:weights)x=int(rng()%255)-127;
  std::vector<uint8_t> input(m*k);for(auto& x:input)x=rng()%256;
  std::vector<int32_t> co(n);
  for(D i=0;i<n;++i)for(D j=0;j<k;++j)co[i]-=128*int(weights[i*k+j]);
  std::vector<int32_t> output(m*n,321),initial=output,expected=output;
  int checks=0;
  auto rejects=[&](auto function) {
    bool threw=false;
    try {function();}catch(const std::exception&){threw=true;}
    if(!threw || output!=initial)throw std::runtime_error("expected rejection with unchanged output");
    ++checks;
  };
  setenv("CT2_DIAG_INT8_PACK_MODE","compact",1);
  rejects([&](){ex::pack_int8_weights(weights.data(),false,k,n,1.f,nullptr);});
  rejects([&](){ex::pack_int8_weights(weights.data(),true,k,n,.5f,nullptr);});
  setenv("CT2_DIAG_INT8_PACK_MODE","invalid",1);
  rejects([&](){ex::pack_int8_weights(weights.data(),true,k,n,1.f,nullptr);});
  setenv("CT2_DIAG_INT8_PACK_MODE","compact",1);
  D bytes=ex::pack_int8_weights(weights.data(),true,k,n,1.f,nullptr);
  std::vector<uint64_t> packed((bytes+7)/8);
  auto* p=reinterpret_cast<int8_t*>(packed.data());
  ex::pack_int8_weights(weights.data(),true,k,n,1.f,p);
  auto invoke=[&](bool ta,bool tb,float alpha,float beta,const int32_t* comp){
    ex::packed_int8_gemm(ta,tb,m,n,k,alpha,reinterpret_cast<int8_t*>(input.data()),k,
                        p,k,beta,output.data(),n,comp);
  };
  rejects([&](){invoke(false,true,1.f,.5f,co.data());});
  rejects([&](){invoke(false,true,.5f,0.f,co.data());});
  rejects([&](){invoke(false,true,1.f,0.f,nullptr);});
  rejects([&](){invoke(true,true,1.f,0.f,co.data());});
  rejects([&](){invoke(false,false,1.f,0.f,co.data());});
  omp_set_num_threads(1);
  rejects([&](){invoke(false,true,1.f,0.f,co.data());});
  omp_set_num_threads(8);
  omp_set_dynamic(1);
  rejects([&](){ex::pack_int8_weights(weights.data(),true,k,n,1.f,nullptr);});
  rejects([&](){invoke(false,true,1.f,0.f,co.data());});
  omp_set_dynamic(0);
  p[0]^=1;
  rejects([&](){invoke(false,true,1.f,0.f,co.data());});
  p[0]^=1;
  if(!ex::pack_int8_weight_name("decoder/layer_0/ffn/linear_0/weight")
     || !ex::pack_int8_weight_name("decoder/projection/weight")
     || ex::pack_int8_weight_name("encoder/layer_0/ffn/linear_0/weight")
     || ex::pack_int8_weight_name("decoder/embeddings/weight")
     || ex::pack_int8_weight_name("decoder/layer_0/attention/linear_1/weight"))
    throw std::runtime_error("wrong compact selection");
  ++checks;
  // Hybrid modes retain raw data and must use it for unsupported accumulation.
  setenv("CT2_DIAG_INT8_PACK_MODE","all",1);
  bytes=ex::pack_int8_weights(weights.data(),true,k,n,1.f,nullptr);
  packed.resize((bytes+7)/8);p=reinterpret_cast<int8_t*>(packed.data());
  ex::pack_int8_weights(weights.data(),true,k,n,1.f,p);
  for(float alpha:{1.f,.5f})for(float beta:{0.f,.5f}) {
    output=initial;expected=initial;
    if(dnnl_gemm_u8s8s32('N','T','R',m,n,k,alpha,input.data(),k,0,weights.data(),k,0,
                          beta,expected.data(),n,co.data())!=dnnl_success)
      throw std::runtime_error("reference failed");
    invoke(false,true,alpha,beta,co.data());
    if(output!=expected)throw std::runtime_error("raw fallback differs");
    ++checks;
  }
  // A raw-retaining historical plan can safely fall back after thread changes.
  omp_set_num_threads(1);output=initial;expected=initial;
  if(dnnl_gemm_u8s8s32('N','T','R',m,n,k,1.f,input.data(),k,0,weights.data(),k,0,
                        0.f,expected.data(),n,co.data())!=dnnl_success)
    throw std::runtime_error("reference failed");
  invoke(false,true,1.f,0.f,co.data());
  if(output!=expected)throw std::runtime_error("thread-change raw fallback differs");
  ++checks;
  std::cout<<"PASS guard_and_fallback_checks="<<checks<<std::endl;
}
