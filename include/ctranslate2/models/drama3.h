#pragma once
#include "model.h"
namespace ctranslate2 { namespace models {
  // Resident CPU checkpoint; inference layers refer to unpacked expert weights.
  class Drama3Model : public Model {
  protected:
    bool is_quantizable(const std::string& name) const override;
    bool is_linear_weight(const std::string& name) const override;
    bool is_packable(const std::string&) const override { return false; }
    void initialize(ModelReader&) override;
    std::unique_ptr<Model> clone() const override;
  };
}}
