#include <gtest/gtest.h>
#include <sstream>
#include <thread>
#include <vector>
#include "ctranslate2/profiler.h"

using namespace ctranslate2;

#ifdef CT2_ENABLE_PROFILING
TEST(ProfilerTest, CountsNestedScopesAndReset) {
  init_profiling(Device::CPU, 1);
  {
    PROFILE("parent");
    for (int i = 0; i < 3; ++i) { PROFILE("child"); }
  }
  std::ostringstream out;
  dump_profiling(out);
  EXPECT_NE(out.str().find("calls=3"), std::string::npos);
  EXPECT_NE(out.str().find("calls=1"), std::string::npos);
  EXPECT_NE(out.str().find("ct2_profile_format=1"), std::string::npos);
  init_profiling(Device::CPU, 1);
  std::ostringstream reset;
  dump_profiling(reset);
  EXPECT_TRUE(reset.str().empty());
  EXPECT_THROW(init_profiling(Device::CPU, 0), std::invalid_argument);
}

TEST(ProfilerTest, AggregatesThreadsAndShapes) {
  init_profiling(Device::CPU, 2);
  auto work = [] {
    for (int i = 0; i < 10; ++i) {
      PROFILE_WITH_SHAPES("test", std::vector<int>({2, 3}), std::vector<int>({3, 4}));
    }
  };
  std::thread a(work), b(work);
  a.join(); b.join();
  std::ostringstream out;
  dump_profiling(out);
  EXPECT_NE(out.str().find("test[2x3,3x4]"), std::string::npos);
  EXPECT_NE(out.str().find("calls=20"), std::string::npos);
}
#else
TEST(ProfilerTest, DisabledMacrosDoNotEvaluateArguments) {
  int count = 0;
  PROFILE(std::to_string(++count));
  PROFILE_WITH_SHAPES("test", ++count);
  EXPECT_EQ(count, 0);
  EXPECT_THROW(init_profiling(Device::CPU, 1), std::runtime_error);
}
#endif
