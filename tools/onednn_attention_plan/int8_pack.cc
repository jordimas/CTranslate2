// Experimental oneDNN 3.1.1 decoder packing, compact and diagnostic hybrid modes.
#include "int8_pack.h"
#include <oneapi/dnnl/dnnl.h>
#include <omp.h>
#include "common/c_types_map.hpp"
#include "cpu/gemm/gemm_pack.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ctranslate2 { namespace experimental {
namespace {
namespace cpu = dnnl::impl::cpu;
using D = dnnl_dim_t;
constexpr uint64_t magic = 0x4354325041434b35ULL;
constexpr D pack_queries = 5;
struct Header { uint64_t tag; D n,k,packed_offset,raw_offset,bytes; uint64_t compact; D threads; };
int thread_context() {return omp_in_parallel()?1:omp_get_max_threads();}
std::string mode() {
  const char* value=std::getenv("CT2_DIAG_INT8_PACK_MODE");
  const std::string result=value?value:"compact";
  if(result!="compact" && result!="selective" && result!="all")
    throw std::invalid_argument("invalid experimental INT8 packing mode");
  return result;
}
void check(dnnl_status_t status) {
  if(status != dnnl_success) throw std::runtime_error("private oneDNN INT8 packing failed");
}
}
bool int8_pack_enabled() {
  const char* value=std::getenv("CT2_DIAG_INT8_PACK");
  const auto* v=dnnl_version();
  return value && value[0]!='0' && v->major==3 && v->minor==1 && v->patch==1
    && dnnl_get_effective_cpu_isa()==dnnl_cpu_isa_avx2 && !omp_get_dynamic();
}
bool pack_int8_weight_name(const std::string& name) {
  if(mode()=="all") return true;
  // Whisper: exclude encoder, embeddings and the cross-attention KV projection.
  return name.compare(0,8,"decoder/")==0 && name.find("/embeddings/")==std::string::npos
    && name.find("/attention/linear_1/")==std::string::npos;
}
dim_t pack_int8_weights(const int8_t* src, bool transpose, dim_t k, dim_t n,
                        float alpha, int8_t* dst) {
  if(!transpose || alpha!=1.f) throw std::runtime_error("unsupported experimental weight packing");
  if(omp_get_dynamic())throw std::invalid_argument("compact INT8 packing requires fixed OpenMP teams");
  const D rows=n, depth=k, ld=k;
  const char ta='T',tb='N',id='A';
  size_t size=0;bool packs=false;
  check(cpu::gemm_s8u8s32_pack_get_size(&id,&ta,&tb,&rows,&pack_queries,&depth,&ld,&ld,&size,&packs));
  if(!packs) throw std::runtime_error("private oneDNN packed representation unavailable");
  const D packed_offset=4096;
  const bool compact=mode()=="compact";
  // Numerical verification keeps an oracle copy only in the diagnostic run.
  // Latency/memory measurements never enable that diagnostic option.
  const bool keep_raw=!compact || std::getenv("CT2_DIAG_INT8_PACK_VERIFY");
  const D raw_offset=keep_raw?(packed_offset+size+4095)/4096*4096:0;
  const D bytes=keep_raw?raw_offset+n*k:packed_offset+size;
  if(dst) {
    Header header{magic,n,k,packed_offset,raw_offset,bytes,uint64_t(compact),thread_context()};
    std::memcpy(dst,&header,sizeof(header));
    check(cpu::gemm_s8u8s32_pack(&id,&ta,&tb,&rows,&pack_queries,&depth,&ld,&ld,src,dst+packed_offset));
    if(keep_raw)std::memcpy(dst+raw_offset,src,n*k);
    static const bool logged=[&](){std::fprintf(stderr,"CT2 hybrid INT8 packing active: mode=%s raw_copy=%d\n",mode().c_str(),keep_raw);return true;}();
    (void)logged;
  }
  return bytes;
}
void packed_int8_gemm(bool transpose_a, bool transpose_b, dim_t m, dim_t n, dim_t k,
                      float alpha, const int8_t* a, dim_t lda, const int8_t* b, dim_t ldb,
                      float beta, int32_t* c, dim_t ldc, const int32_t* compensation) {
  Header header;std::memcpy(&header,b,sizeof(header));
  if(header.tag!=magic || header.n!=n || header.k!=k || !transpose_b)
    throw std::runtime_error("invalid hybrid packed weight");
  const auto* raw=b+header.raw_offset;
  const char ta=transpose_a?'T':'N',tb=transpose_b?'T':'N';
  auto regular=[&](int32_t* dst,D stride) {
    if(!header.raw_offset)throw std::runtime_error("compact INT8 representation has no raw fallback");
    if(compensation) check(dnnl_gemm_u8s8s32(ta,tb,'R',m,n,k,alpha,
      reinterpret_cast<const uint8_t*>(a),lda,0,raw,ldb,0,beta,dst,stride,compensation));
    else {const int32_t zero=0;check(dnnl_gemm_s8s8s32(ta,tb,'F',m,n,k,alpha,a,lda,0,
      raw,ldb,0,beta,dst,stride,&zero));}
  };
  // The private packing metadata includes threading partitions. Reusing an
  // eight-thread plan in a one-thread context can silently omit output rows.
  if(omp_get_dynamic() || header.threads!=thread_context()) {
    if(header.raw_offset) {regular(c,ldc);return;}
    throw std::invalid_argument("compact INT8 packed weights require the original fixed thread context");
  }
  // Fractional beta can round differently between the public and packed
  // oneDNN drivers. Only the exactly checked integer accumulation cases are
  // admitted without a raw fallback; Whisper linear layers use beta=0.
  if(beta!=0.f && beta!=1.f && beta!=-1.f) {
    if(header.raw_offset) {regular(c,ldc);return;}
    throw std::invalid_argument("compact INT8 packing supports beta 0, 1 or -1 only");
  }
  // Historical hybrid modes have raw fallback. Compact mode instead executes
  // all query counts through bounded calls into the same packed plan.
  if((m>pack_queries && !header.compact) || transpose_a || !compensation || alpha!=1.f) {
    regular(c,ldc);return;
  }
  const bool verify=std::getenv("CT2_DIAG_INT8_PACK_VERIFY")!=nullptr;
  std::vector<int32_t> expected;
  if(verify) {
    expected.resize(m*n);
    if(beta!=0.f)for(D i=0;i<m;++i)std::memcpy(expected.data()+i*n,c+i*ldc,n*sizeof(int32_t));
    regular(expected.data(),n);
  }
  static const bool logged=[](){std::fprintf(stderr,"CT2 hybrid INT8 packed compute active\n");return true;}();
  (void)logged;
  const D rows=n,depth=k,weight_ld=ldb,query_ld=lda,out_ld=ldc;
  const char packed='P',normal='N',offset='C';
  for(D q=0;q<m;q+=pack_queries) {
    const D queries=std::min<D>(pack_queries,m-q);
    check(cpu::gemm_s8u8s32_compute(&packed,&normal,&offset,&rows,&queries,&depth,
      b+header.packed_offset,&weight_ld,reinterpret_cast<const uint8_t*>(a)+q*lda,&query_ld,
      &beta,c+q*ldc,&out_ld,compensation));
  }
  if(verify)for(D i=0;i<m;++i)if(std::memcmp(c+i*ldc,expected.data()+i*n,n*sizeof(int32_t)))
    throw std::runtime_error("hybrid packed INT8 GEMM differs from reference");
}
}}
