#include <string>

#include "edgellm/edgellm.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Name is edgellm", "[library]")
{
  auto const exported = exported_class {};
  REQUIRE(std::string("edgellm") == exported.name());
}
