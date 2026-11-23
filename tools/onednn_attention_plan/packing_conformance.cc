// Compare the actual compact adapter against independent public oneDNN GEMMs.
// Canaries cover private pack writes and padded output rows; input data and beta
// cases exercise saturation, SIMD tails and five-query boundaries.
#include "int8_pack.h"
#include <oneapi/dnnl/dnnl.h>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using D = ctranslate2::dim_t;
namespace ex = ctranslate2::experimental;
constexpr int32_t canary = 0x4badf00d;
struct Buffer {
  size_t size;
  uint8_t* allocation=nullptr;
  explicit Buffer(size_t bytes):size(bytes) {
    if(posix_memalign(reinterpret_cast<void**>(&allocation),64,bytes+128))throw std::bad_alloc();
    std::memset(allocation,0xa5,bytes+128);
  }
  ~Buffer(){std::free(allocation);}
  int8_t* data(){return reinterpret_cast<int8_t*>(allocation+64);}
  bool guards() const {
    for(size_t i=0;i<64;++i)
      if(allocation[i]!=0xa5 || allocation[64+size+i]!=0xa5)return false;
    return true;
  }
};
void check(dnnl_status_t status){if(status!=dnnl_success)throw std::runtime_error("reference GEMM failed");}
int main(int argc,char** argv) {
  const bool cross_threads=argc>1 && std::string(argv[1])=="cross-threads";
  std::mt19937 rng(217);
  size_t cases=0, failures=0, rejected=0;
  std::vector<std::pair<D,D>> shapes={{7,3},{17,33},{31,65},{127,127},{129,129},
    {1024,1024},{3072,1024},{4096,1024},{1024,4096},
    {1280,1280},{3840,1280},{5120,1280},{1280,5120},{51866,1280}};
  if(cross_threads)shapes={{127,127},{1280,1280},{5120,1280}};
  if(argc>2)shapes={{127,127}};
  std::vector<std::pair<int,int>> thread_cases=cross_threads
    ?std::vector<std::pair<int,int>>{{8,1},{8,4},{1,8}}
    :std::vector<std::pair<int,int>>{{1,1},{8,8}};
  for(auto threads:thread_cases) for(auto shape:shapes) {
    const D n=shape.first,k=shape.second;
    for(int pattern=0;pattern<4;++pattern) {
      omp_set_num_threads(threads.first);
      std::vector<int8_t> weights(n*k);
      for(D i=0;i<n*k;++i) {
        if(pattern==0)weights[i]=int(rng()%127)-63;
        if(pattern==1)weights[i]=int(rng()%256)-128;
        if(pattern==2)weights[i]=(i%2)?127:-128;
        if(pattern==3)weights[i]=0;
      }
      auto original=weights;
      std::vector<int32_t> co(n);
      for(D i=0;i<n;++i)for(D j=0;j<k;++j)co[i]-=128*int(weights[i*k+j]);
      D bytes=ex::pack_int8_weights(weights.data(),true,k,n,1.f,nullptr);
      Buffer packed(bytes);
      ex::pack_int8_weights(weights.data(),true,k,n,1.f,packed.data());
      if(weights!=original || !packed.guards())throw std::runtime_error("weight/pack guard changed");
      const std::vector<int8_t> packed_before(packed.data(),packed.data()+bytes);
      omp_set_num_threads(threads.second);
      std::vector<D> query_counts={1,2,3,4,5,6,9,10,11,16,31,64};
      if(cross_threads)query_counts={1,5,6,11};
      if(!cross_threads && n<=4096)query_counts.insert(query_counts.end(),{127,128,129,223,224,225,448,449});
      if(!cross_threads && n==1280 && pattern==0)query_counts.push_back(1500);
      for(D m:query_counts)for(float beta:{0.f,1.f,-1.f,.5f}) {
        const D lda=k+3,ldc=n+7;
        std::vector<uint8_t> input(m*lda+64,0xb7);
        for(D i=0;i<m;++i)for(D j=0;j<k;++j)
          input[i*lda+j]=pattern==2?((i+j)%2?255:0):uint8_t(rng()%256);
        auto input_before=input;
        std::vector<int32_t> ref(m*ldc+64,canary),actual=ref;
        for(D i=0;i<m;++i)for(D j=0;j<n;++j)actual[i*ldc+j]=ref[i*ldc+j]=int(rng()%2001)-1000;
        const bool expect_rejection=cross_threads || beta==.5f;
        if(!expect_rejection)check(dnnl_gemm_u8s8s32('N','T','R',m,n,k,1.f,input.data(),lda,0,
              weights.data(),k,0,beta,ref.data(),ldc,co.data()));
        const auto initial=actual;
        if(argc>2)std::cout<<"BEGIN m="<<m<<" beta="<<beta<<" pattern="<<pattern<<std::endl;
        try {
          ex::packed_int8_gemm(false,true,m,n,k,1.f,reinterpret_cast<const int8_t*>(input.data()),lda,
                              packed.data(),k,beta,actual.data(),ldc,co.data());
          if(expect_rejection)throw std::runtime_error("unsupported context/beta was not rejected");
        } catch(const std::invalid_argument&) {
          if(!expect_rejection || actual!=initial || !packed.guards() || input!=input_before)throw;
          ++cases;++rejected;continue;
        }
        size_t changed=0,padding=0;
        for(D i=0;i<m;++i)for(D j=0;j<n;++j)changed+=ref[i*ldc+j]!=actual[i*ldc+j];
        for(D i=0;i<m;++i)for(D j=n;j<ldc;++j)padding+=actual[i*ldc+j]!=canary;
        for(size_t i=m*ldc;i<actual.size();++i)padding+=actual[i]!=canary;
        bool bad=changed || padding || !packed.guards() || input!=input_before;
        ++cases;failures+=bad;
        if(bad)std::cout<<"FAIL pack_threads="<<threads.first<<" compute_threads="<<threads.second
          <<" n="<<n<<" k="<<k<<" m="<<m<<" pattern="<<pattern<<" beta="<<beta
          <<" changed="<<changed<<" padding="<<padding<<std::endl;
      }
      if(std::memcmp(packed.data(),packed_before.data(),bytes) || weights!=original)
        throw std::runtime_error("compute mutated weights or packed metadata");
      std::cout<<"shape complete pack_threads="<<threads.first<<" compute_threads="<<threads.second
        <<" n="<<n<<" k="<<k<<" pattern="<<pattern<<" cases="<<cases<<" failures="<<failures<<std::endl;
    }
  }
  std::cout<<"TOTAL cases="<<cases<<" numeric="<<cases-rejected<<" expected_rejections="<<rejected
           <<" failures="<<failures<<std::endl;
  return failures?1:0;
}
