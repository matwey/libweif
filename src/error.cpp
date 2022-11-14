#include <error.h>

namespace weif {

error::error(const std::string& what): std::runtime_error(what) {}
error::error(const char* what): std::runtime_error(what) {}

mismatched_grids::mismatched_grids() noexcept:
	error("Mismatched grids") {}

} // weif
