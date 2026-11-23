// Standalone diagnostic of oneDNN 3.1.1's private int8 weight-packing API.
#include <oneapi/dnnl/dnnl.h>
#include "common/c_types_map.hpp"
#include "cpu/gemm/gemm_pack.hpp"
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using D = dnnl_dim_t;
namespace cpu = dnnl::impl::cpu;
void check(dnnl_status_t status) { if (status != dnnl_success) std::abort(); }
int main(int argc, char** argv) {
  omp_set_num_threads(8);
  std::mt19937 rng(123);
  for (int limit : {63, 127}) for (D n : {1280, 5120}) {
    const D k=1280, pack_queries=argc>1?std::atoi(argv[1]):1, ld=k;
    const char ta='T', tb='N', id='A', packed='P', offset='C';
    std::vector<int8_t> weights(n*k);
    for (auto& x:weights) x=int(rng() % (2*limit+1))-limit;
    std::vector<int32_t> co(n);
    for (D i=0;i<n;++i) for(D j=0;j<k;++j) co[i]-=128*int(weights[i*k+j]);
    size_t bytes=0; bool packs=false;
    check(cpu::gemm_s8u8s32_pack_get_size(&id,&ta,&tb,&n,&pack_queries,&k,&ld,&ld,&bytes,&packs));
    std::vector<uint64_t> storage((bytes+7)/8+512);
    check(cpu::gemm_s8u8s32_pack(&id,&ta,&tb,&n,&pack_queries,&k,&ld,&ld,weights.data(),storage.data()));
    for (D queries : {1,5,16,1500}) {
      std::vector<uint8_t> a(queries*k); for(auto& x:a)x=rng()%256;
      std::vector<int32_t> ref(queries*n),actual(queries*n);
      const float zero=0.f;
      auto regular=[&](){check(dnnl_gemm_u8s8s32('N','T','R',queries,n,k,1,a.data(),k,0,
                                weights.data(),k,0,0,ref.data(),n,co.data()));};
      auto optimized=[&](){check(cpu::gemm_s8u8s32_compute(&packed,&tb,&offset,&n,&queries,&k,
                        reinterpret_cast<const int8_t*>(storage.data()),&ld,a.data(),&ld,
                        &zero,actual.data(),&n,co.data()));};
      regular(); optimized();
      size_t changed=0; for(size_t i=0;i<ref.size();++i)changed+=ref[i]!=actual[i];
      const int runs=queries==1500?3:40;
      double times[2]={0,0};
      for(int i=0;i<runs;++i)for(int j=0;j<2;++j){
        int mode=(i+j)%2;auto begin=std::chrono::steady_clock::now();
        if(mode)optimized();else regular();
        times[mode]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
      }
      std::cout<<"limit="<<limit<<" n="<<n<<" k="<<k<<" queries="<<queries
               <<" bytes="<<bytes<<" packs="<<packs<<" changed="<<changed
               <<" regular_ms="<<1000*times[0]/runs<<" packed_ms="<<1000*times[1]/runs
               <<" speedup="<<times[0]/times[1]<<std::endl;
      if(changed) {
        for(D row=0;row<queries;++row) {
          size_t count=0; for(D j=0;j<n;++j) count+=ref[row*n+j]!=actual[row*n+j];
          if(count) std::cerr<<"row="<<row<<" changed="<<count<<" ref0="<<ref[row*n]<<" packed0="<<actual[row*n]<<std::endl;
        }
        return 1;
      }
    }
  }
}
