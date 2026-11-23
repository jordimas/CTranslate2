#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>
#include "ctranslate2/storage_view.h"
#include "ctranslate2/ops/matmul.h"
#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/ops/fused_attention.h"
using namespace ctranslate2;
int main(int argc, char** argv) {
  const dim_t h=argc>1?std::atoi(argv[1]):20;
  const dim_t t=argc>2?std::atoi(argv[2]):1500;
  const dim_t d=argc>3?std::atoi(argv[3]):64;
  const int runs=argc>4?std::atoi(argv[4]):15;
  omp_set_num_threads(8);
  std::mt19937 rng(123); std::normal_distribution<float> normal;
  auto make=[&](){std::vector<float> data(h*t*d);for(auto& x:data)x=normal(rng);return StorageView({1,h,t,d},data);};
  auto q=make(),k=make(),v=make();
  auto execute=[&](bool fused) {
    StorageView scores,output;
    if(fused) {
      if(!ops::FusedAttention()(q,k,v,nullptr,1.f/std::sqrt(float(d)),false,output)) std::abort();
    } else {
      ops::MatMul(false,true,1.f/std::sqrt(float(d)))(q,k,scores);
      ops::SoftMax()(scores,nullptr,scores);
      ops::MatMul()(scores,v,output);
    }
  };
  execute(false);execute(true);
  std::vector<double> regular,fused;
  for(int i=0;i<runs;++i) {
    for(int j=0;j<2;++j) {
      bool on=(i+j)%2;
      auto begin=std::chrono::steady_clock::now();execute(on);
      double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
      (on?fused:regular).push_back(seconds*1000);
    }
  }
  auto print=[&](const char* name,std::vector<double> x){
    std::sort(x.begin(),x.end());double sum=0;for(auto v:x)sum+=v;
    std::cout<<name<<" mean_ms="<<sum/x.size()<<" median_ms="<<x[x.size()/2]<<" min_ms="<<x.front()<<" samples=";
    for(auto v:x)std::cout<<v<<",";std::cout<<"\n";
  };
  print("regular",regular);print("fused",fused);
}
