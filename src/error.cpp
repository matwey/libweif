#include <error.h>

namespace weif {

error::error(const std::string& what): std::runtime_error(what) {}
error::error(const char* what): std::runtime_error(what) {}

mismatched_grids::mismatched_grids() noexcept:
	error("Mismatched grids") {}

gsl_error::gsl_error(const int gsl_errno) noexcept:
	error(gsl_strerror(gsl_errno)) {}

} // weif
