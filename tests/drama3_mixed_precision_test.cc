#include <cstring>
#include "test_utils.h"
#include "ctranslate2/models/drama3.h"
#include "ctranslate2/models/model_reader.h"
#include "ctranslate2/layers/common.h"

namespace {
  void integer(std::string& bytes, uint32_t value, size_t width) {
    for (size_t i=0;i<width;++i) bytes.push_back(char((value>>(8*i))&255));
  }
  void string(std::string& bytes, const std::string& text) {
    integer(bytes,text.size()+1,2);bytes+=text;bytes.push_back('\0');
  }
  template<typename T>
  void tensor(std::string& bytes, const std::string& name, Shape shape,
              uint8_t type, const std::vector<T>& values) {
    string(bytes,name);integer(bytes,shape.size(),1);
    for(auto d:shape)integer(bytes,d,4);
    integer(bytes,type,1);integer(bytes,values.size()*sizeof(T),4);
    for(T v:values) {
      if constexpr(sizeof(T)==4) {uint32_t bits;std::memcpy(&bits,&v,4);integer(bytes,bits,4);}
      else bytes.push_back(char(v));
    }
  }
  models::ModelMemoryReader mixed_fixture(const std::string& policy) {
    std::string bytes;integer(bytes,6,4);string(bytes,"Drama3Spec");integer(bytes,1,4);integer(bytes,8,4);
    tensor<int8_t>(bytes,"backbone/layers_0/mlp/experts_0/gate_proj/weight",{2,2},1,{127,-127,64,-64});
    tensor<float>(bytes,"backbone/layers_0/mlp/experts_0/gate_proj/weight_scale",{2},0,{127,64});
    tensor<float>(bytes,"backbone/layers_0/mlp/router/weight",{2,2},0,{.125f,-.375f,.5f,.75f});
    tensor<float>(bytes,"backbone/layers_0/mlp/shared_expert/up_proj/weight",{2,2},0,{.125f,-.375f,.5f,.75f});
    tensor<float>(bytes,"backbone/embeddings/weight",{2,2},0,{.125f,-.375f,.5f,.75f});
    tensor<float>(bytes,"tower/heads_0/weight",{2,2},0,{.125f,-.375f,.5f,.75f});
    tensor<float>(bytes,"backbone/norm/weight",{2},0,{.25f,-.5f});
    tensor<float>(bytes,"codec/quantizer/upsample/0/1/pwconv1/weight",{2,2},0,{.125f,-.375f,.5f,.75f});
    integer(bytes,0,4);
    models::ModelMemoryReader reader("mixed_precision_fixture");
    reader.register_file("model.bin",std::move(bytes));
    reader.register_file("config.json",nlohmann::json({{"model_type","drama3"},{"ct2_layout_version",1},
        {"source_config",nlohmann::json::object()},{"quantization_policy",policy}}).dump());
    return reader;
  }
}

TEST(Drama3MixedPrecisionTest, PreservesSensitiveFloat32AndResidentInt8ExpertsDuringLoading) {
  if(!mayiuse_int8(Device::CPU))GTEST_SKIP()<<"INT8 backend unavailable";
  auto reader=mixed_fixture("experts_int8_sensitive_float32");
  auto model=models::Model::load(reader,Device::CPU,0,ComputeType::INT8_FLOAT32);
  EXPECT_EQ(model->effective_compute_type(),ComputeType::INT8_FLOAT32);
  const std::string expert="backbone/layers_0/mlp/experts_0/gate_proj/weight";
  expect_storage_eq(model->get_variable(expert),StorageView({2,2},std::vector<int8_t>{127,-127,64,-64}));
  expect_storage_eq(model->get_variable(expert+"_scale"),StorageView({2},std::vector<float>{127,64}));
  for(const auto* name:{"backbone/layers_0/mlp/router/weight","backbone/layers_0/mlp/shared_expert/up_proj/weight",
                       "backbone/embeddings/weight","tower/heads_0/weight",
                       "codec/quantizer/upsample/0/1/pwconv1/weight"}) {
    expect_storage_eq(model->get_variable(name),StorageView({2,2},std::vector<float>{.125f,-.375f,.5f,.75f}));
    EXPECT_EQ(model->get_variable_if_exists(std::string(name)+"_scale"),nullptr);
  }
  layers::Dense router(*model,"backbone/layers_0/mlp/router");
  StorageView input({1,2},std::vector<float>{2,4}),actual;
  router(input,actual);
  expect_storage_eq(actual,StorageView({1,2},std::vector<float>{-1.25f,4.f}));
}

TEST(Drama3MixedPrecisionTest, ExplicitDefaultPolicyStillQuantizesSensitiveMatrices) {
  if(!mayiuse_int8(Device::CPU))GTEST_SKIP()<<"INT8 backend unavailable";
  auto reader=mixed_fixture("all_matrices_int8");
  auto model=models::Model::load(reader,Device::CPU,0,ComputeType::INT8_FLOAT32);
  EXPECT_EQ(model->get_variable("backbone/layers_0/mlp/router/weight").dtype(),DataType::INT8);
  EXPECT_NE(model->get_variable_if_exists("backbone/layers_0/mlp/router/weight_scale"),nullptr);
  EXPECT_EQ(model->get_variable("codec/quantizer/upsample/0/1/pwconv1/weight").dtype(),DataType::FLOAT32);
}

TEST(Drama3MixedPrecisionTest, RejectsUnknownPolicyInsteadOfSilentlyRequantizing) {
  auto reader=mixed_fixture("typo_sensitive_float32");
  EXPECT_THROW(models::Model::load(reader,Device::CPU,0,ComputeType::FLOAT32),std::invalid_argument);
}
