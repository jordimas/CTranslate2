#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>
#include "ctranslate2/storage_view.h"
#include "ctranslate2/ops/matmul.h"
#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/ops/quantize.h"
#include "ctranslate2/ops/transpose.h"
#include "ctranslate2/ops/fused_attention.h"
using namespace ctranslate2;
void compare(const char* label, const StorageView& a, const StorageView& b) {
  size_t changed=0; double maxdiff=0; dim_t first=-1,last=-1;
  for(dim_t i=0;i<a.size();++i) {
    double diff=std::abs(double(a.data<float>()[i])-b.data<float>()[i]);
    if(diff!=0) {if(first<0) first=i; last=i;}
    changed += diff!=0; maxdiff=std::max(maxdiff,diff);
  }
  std::cout<<label<<" changed="<<changed<<"/"<<a.size()<<" max="<<maxdiff<<" first="<<first<<" last="<<last<<"\n";
}
int main(int argc,char** argv) {
  omp_set_num_threads(argc>1?std::atoi(argv[1]):8);
  const dim_t h=2,t=argc>2?std::atoi(argv[2]):1500,d=argc>3?std::atoi(argv[3]):64;
  std::mt19937 rng(123); std::normal_distribution<float> dist(0,1);
  auto make=[&](){std::vector<float> data(h*t*d);for(auto& x:data)x=dist(rng);return StorageView({1,h,t,d},data);};
  auto q=make(),k=make(),v=make();
  StorageView scores,probs,expected,actual;
  ops::MatMul(false,true,1.f/std::sqrt(float(d)))(q,k,scores);
  ops::SoftMax()(scores,nullptr,probs);
  ops::MatMul()(probs,v,expected);
  ops::FusedAttention()(q,k,v,nullptr,1.f/std::sqrt(float(d)),false,actual);
  compare("attention",expected,actual);
  StorageView block_scores(scores.shape(),DataType::FLOAT32);
  StorageView block_pv(expected.shape(),DataType::FLOAT32);
  for(dim_t head=0;head<h;++head) {
    StorageView kh({1,1,t,d},std::vector<float>(k.data<float>()+head*t*d,k.data<float>()+(head+1)*t*d));
    StorageView vh({1,1,t,d},std::vector<float>(v.data<float>()+head*t*d,v.data<float>()+(head+1)*t*d));
    for(dim_t start=0;start<t;start+=256) {
      dim_t rows=std::min(dim_t(256),t-start);
      const float* qs=q.data<float>()+(head*t+start)*d;
      const float* ps=probs.data<float>()+(head*t+start)*t;
      StorageView qb({1,1,rows,d},std::vector<float>(qs,qs+rows*d));
      StorageView pb({1,1,rows,t},std::vector<float>(ps,ps+rows*t));
      StorageView sb,ob;
      ops::MatMul(false,true,1.f/std::sqrt(float(d)))(qb,kh,sb);
      ops::MatMul()(pb,vh,ob);
      std::copy(sb.data<float>(),sb.data<float>()+sb.size(),block_scores.data<float>()+(head*t+start)*t);
      std::copy(ob.data<float>(),ob.data<float>()+ob.size(),block_pv.data<float>()+(head*t+start)*d);
    }
  }
  compare("QK full-vs-block",scores,block_scores);
  compare("PV same-probabilities full-vs-block",expected,block_pv);
  StorageView merged_a,merged_b;
  ops::Transpose({0,2,1,3})(expected,merged_a);
  ops::Transpose({0,2,1,3})(actual,merged_b);
  merged_a.reshape({1,t,h*d}); merged_b.reshape({1,t,h*d});
  StorageView qa(DataType::INT8),qb(DataType::INT8),sa,sb;
  ops::Quantize()(merged_a,qa,sa); ops::Quantize()(merged_b,qb,sb);
  size_t changed=0;for(dim_t i=0;i<qa.size();++i)changed+=qa.data<int8_t>()[i]!=qb.data<int8_t>()[i];
  std::cout<<"quantized changed="<<changed<<"/"<<qa.size()<<"\n";
  compare("quantization-scales",sa,sb);
}
