#include "catch.hpp"

struct Foo
{
    bool is_bar() const
    {
        return true;
    }
};

// Check out https://github.com/catchorg/Catch2 for more information about how to use Catch2
TEST_CASE( "Foo is always Bar", "[Foobar]" )
{
    Foo foo;

    REQUIRE(foo.is_bar());
    CHECK(foo.is_bar());
}
