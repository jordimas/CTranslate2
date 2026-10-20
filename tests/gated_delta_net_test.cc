#include <cmath>
#include <stdexcept>

#include "test_utils.h"
#include "ops/drama/gated_delta_net.h"

TEST(GatedDeltaNetTest, DecaysStateBeforeComputingResidual) {
  StorageView q({1, 1, 1, 2}, std::vector<float>{2, 1});
  StorageView k({1, 1, 1, 2}, std::vector<float>{1, 2});
  StorageView v({1, 1, 1, 2}, std::vector<float>{11, 6});
  StorageView g({1, 1, 1}, std::log(0.5f));
  StorageView beta({1, 1, 1}, 0.25f);
  StorageView state({1, 1, 2, 2}, std::vector<float>{2, 4, 6, 8});
  StorageView output;
  ops::GatedDeltaNet()(q, k, v, g, beta, state, output);
  expect_storage_eq(state, StorageView({1, 1, 2, 2}, std::vector<float>{2, 1, 5, 2}), 1e-6f);
  expect_storage_eq(
      output, StorageView({1, 1, 1, 2}, std::vector<float>{9 / std::sqrt(2.f), 4 / std::sqrt(2.f)}),
      1e-6f);
}

TEST(GatedDeltaNetTest, ConsecutiveValueHeadsShareKeysAndQueries) {
  StorageView q({1, 1, 2, 1}, std::vector<float>{2, 3});
  StorageView k({1, 1, 2, 1}, std::vector<float>{1, 2});
  StorageView v({1, 1, 4, 1}, std::vector<float>{1, 2, 3, 4});
  StorageView g({1, 1, 4}, 0.f);
  StorageView beta({1, 1, 4}, 0.5f);
  StorageView state({1, 4, 1, 1}, 0.f);
  StorageView output;
  ops::GatedDeltaNet()(q, k, v, g, beta, state, output);
  expect_storage_eq(output, StorageView({1, 1, 4, 1}, std::vector<float>{1, 2, 9, 12}));
  expect_storage_eq(state, StorageView({1, 4, 1, 1}, std::vector<float>{0.5f, 1, 3, 4}));
}

TEST(GatedDeltaNetTest, PrefillMatchesSingleStepsWithNonzeroState) {
  const dim_t batch = 2, steps = 5, key_heads = 2, heads = 4, kd = 3, vd = 2;
  StorageView q({batch, steps, key_heads, kd});
  StorageView k(q.shape());
  StorageView v({batch, steps, heads, vd});
  StorageView g({batch, steps, heads});
  StorageView beta(g.shape());
  StorageView initial({batch, heads, kd, vd});
  for (dim_t i = 0; i < q.size(); ++i) {
    q.data<float>()[i] = float(i % 11 - 5) / 7;
    k.data<float>()[i] = float(i % 7 - 3) / 9;
  }
  for (dim_t i = 0; i < v.size(); ++i)
    v.data<float>()[i] = float(i % 13 - 6) / 5;
  for (dim_t i = 0; i < g.size(); ++i) {
    g.data<float>()[i] = -float(i % 5 + 1) / 10;
    beta.data<float>()[i] = float(i % 4 + 1) / 5;
  }
  for (dim_t i = 0; i < initial.size(); ++i)
    initial.data<float>()[i] = float(i % 9 - 4) / 8;
  StorageView prefill_state(initial), step_state(initial), prefill;
  ops::GatedDeltaNet()(q, k, v, g, beta, prefill_state, prefill);
  // Slice each batch explicitly: a time slice across batches is not contiguous.
  for (dim_t b = 0; b < batch; ++b) {
    StorageView state({1, heads, kd, vd}, step_state.data<float>() + b * heads * kd * vd);
    for (dim_t t = 0; t < steps; ++t) {
      const auto row = b * steps + t;
      StorageView qt({1, 1, key_heads, kd}, q.data<float>() + row * key_heads * kd);
      StorageView kt({1, 1, key_heads, kd}, k.data<float>() + row * key_heads * kd);
      StorageView vt({1, 1, heads, vd}, v.data<float>() + row * heads * vd);
      StorageView gt({1, 1, heads}, g.data<float>() + row * heads);
      StorageView bt({1, 1, heads}, beta.data<float>() + row * heads);
      StorageView output;
      ops::GatedDeltaNet()(qt, kt, vt, gt, bt, state, output);
      expect_array_eq(output.data<float>(), prefill.data<float>() + row * heads * vd, heads * vd,
                      1e-6f);
    }
  }
  expect_storage_eq(step_state, prefill_state, 1e-6f);
}

TEST(GatedDeltaNetTest, RejectsInvalidShapesAndTypes) {
  StorageView q({1, 1, 2, 2}, 1.f), k(q);
  StorageView v({1, 1, 4, 3}, 1.f);
  StorageView g({1, 1, 4}, 0.f), beta(g);
  StorageView state({1, 4, 2, 3}, 0.f), output;
  ops::GatedDeltaNet op;
  StorageView bad_key({1, 1, 2, 3}, 1.f);
  EXPECT_THROW(op(q, bad_key, v, g, beta, state, output), std::invalid_argument);
  StorageView bad_value({1, 1, 3, 3}, 1.f);
  EXPECT_THROW(op(q, k, bad_value, g, beta, state, output), std::invalid_argument);
  StorageView bad_gate({1, 4}, 1.f);
  EXPECT_THROW(op(q, k, v, bad_gate, beta, state, output), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, g, bad_gate, state, output), std::invalid_argument);
  StorageView bad_state({1, 4, 3, 2}, 0.f);
  EXPECT_THROW(op(q, k, v, g, beta, bad_state, output), std::invalid_argument);
  StorageView bad_type(q.shape(), DataType::INT32);
  EXPECT_THROW(op(bad_type, k, v, g, beta, state, output), std::invalid_argument);
  StorageView bad_output(DataType::INT32);
  EXPECT_THROW(op(q, k, v, g, beta, state, bad_output), std::invalid_argument);
}

TEST(GatedDeltaNetTest, RejectsAliasingIncludingPartialOverlap) {
  StorageView q({1, 1, 1, 2}, 1.f), k(q), v(q);
  StorageView g({1, 1, 1}, 0.f), beta({1, 1, 1}, 0.5f);
  StorageView state({1, 1, 2, 2}, 0.f), output;
  ops::GatedDeltaNet op;
  EXPECT_THROW(op(q, k, v, g, beta, state, state), std::invalid_argument);
  EXPECT_THROW(op(q, k, v, g, beta, state, q), std::invalid_argument);
  StorageView overlapping_query({1, 1, 1, 2}, state.data<float>() + 1);
  EXPECT_THROW(op(overlapping_query, k, v, g, beta, state, output), std::invalid_argument);
  StorageView overlapping_output({1}, state.data<float>() + 3);
  EXPECT_THROW(op(q, k, v, g, beta, state, overlapping_output), std::invalid_argument);
  StorageView input_output_alias({1}, v.data<float>() + 1);
  EXPECT_THROW(op(q, k, v, g, beta, state, input_output_alias), std::invalid_argument);
}

TEST(GatedDeltaNetTest, RejectsAliasesIntoClearedOutputWithoutMutation) {
  StorageView q({1, 1, 1, 2}, 1.f), k(q), v(q);
  StorageView g({1, 1, 1}, 0.f), beta({1, 1, 1}, 0.5f);
  StorageView state({1, 1, 2, 2}, 2.f);
  StorageView output({16}, 3.f);
  auto* allocation = output.data<float>();
  StorageView aliased_query({1, 1, 1, 2}, allocation + 1);
  output.clear();
  const StorageView original_state(state);
  ops::GatedDeltaNet op;
  EXPECT_THROW(op(aliased_query, k, v, g, beta, state, output), std::invalid_argument);
  expect_storage_eq(state, original_state);
  EXPECT_EQ(output.size(), 0);
  for (int i = 0; i < 16; ++i)
    EXPECT_FLOAT_EQ(allocation[i], 3.f);

  StorageView aliased_state({1, 1, 2, 2}, allocation + 5);
  EXPECT_THROW(op(q, k, v, g, beta, aliased_state, output), std::invalid_argument);
  EXPECT_EQ(output.size(), 0);
  for (int i = 0; i < 16; ++i)
    EXPECT_FLOAT_EQ(allocation[i], 3.f);
}

TEST(GatedDeltaNetTest, EmptyTimePreservesState) {
  StorageView q({2, 0, 1, 2}), k(q);
  StorageView v({2, 0, 2, 3});
  StorageView g({2, 0, 2}), beta(g);
  StorageView state({2, 2, 2, 3}, 0.75f);
  const StorageView original(state);
  StorageView output({8}, 9.f);
  ops::GatedDeltaNet()(q, k, v, g, beta, state, output);
  expect_storage_eq(state, original);
  assert_vector_eq(output.shape(), {2, 0, 2, 3});
  EXPECT_EQ(output.size(), 0);
}

TEST(GatedDeltaNetTest, MatchesIndependentDenseTransitionReference) {
  // Express the update as A*S + B, with A = exp(g)*(I-beta*k*k^T)
  // and B = beta*k*v^T. Use double matrix arithmetic, independently of
  // the production residual/rank-one update implementation.
  StorageView q({1, 3, 1, 2}, std::vector<float>{0.4f, -0.7f, 1.2f, 0.3f, -0.8f, 0.6f});
  StorageView k({1, 3, 1, 2}, std::vector<float>{1.3f, -0.2f, 0.4f, 1.1f, -0.6f, 0.9f});
  StorageView v({1, 3, 2, 3}, std::vector<float>{1, -2, 3, 4, 0.5f, -1, -0.5f, 2, 1, -3, 0.2f, 0.7f,
                                                 2, 0, -1, 1, -0.4f, 3});
  StorageView g({1, 3, 2}, std::vector<float>{-0.2f, -0.7f, -0.9f, -0.1f, -0.4f, -0.3f});
  StorageView beta({1, 3, 2}, std::vector<float>{0.2f, 0.8f, 0.6f, 0.4f, 0.9f, 0.1f});
  StorageView state({1, 2, 2, 3},
                    std::vector<float>{1, 2, -1, 0.5f, -0.2f, 3, -2, 0.3f, 1, 4, -1, 0.7f});
  std::vector<double> expected_state(state.data<float>(), state.data<float>() + state.size());
  std::vector<float> expected_output(18);
  for (int t = 0; t < 3; ++t) {
    for (int h = 0; h < 2; ++h) {
      const double gate = beta.data<float>()[t * 2 + h];
      const double decay = std::exp(double(g.data<float>()[t * 2 + h]));
      double transition[2][2];
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          transition[i][j] = decay
                             * ((i == j ? 1. : 0.)
                                - gate * k.data<float>()[t * 2 + i] * k.data<float>()[t * 2 + j]);
      const auto previous = expected_state;
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
          double result = gate * k.data<float>()[t * 2 + i] * v.data<float>()[(t * 2 + h) * 3 + j];
          for (int l = 0; l < 2; ++l)
            result += transition[i][l] * previous[(h * 2 + l) * 3 + j];
          expected_state[(h * 2 + i) * 3 + j] = result;
        }
      }
      for (int j = 0; j < 3; ++j) {
        double result = 0;
        for (int i = 0; i < 2; ++i)
          result += q.data<float>()[t * 2 + i] * expected_state[(h * 2 + i) * 3 + j];
        expected_output[(t * 2 + h) * 3 + j] = result / std::sqrt(2.);
      }
    }
  }
  StorageView output;
  ops::GatedDeltaNet()(q, k, v, g, beta, state, output);
  expect_array_eq(output.data<float>(), expected_output.data(), 18, 2e-6f);
  for (dim_t i = 0; i < state.size(); ++i)
    EXPECT_NEAR(state.data<float>()[i], expected_state[i], 2e-6);
}
