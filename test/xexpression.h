#ifndef _TEST_XEXPRESSION_H
#define _TEST_XEXPRESSION_H


#include <cppunit/SourceLine.h>
#include <cppunit/TestAssert.h>

#include <limits>
#include <sstream>
#include <type_traits>

#include <xtensor/xarray.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xio.hpp>

template<class T1, class T2>
void checkXArrayClose(const xt::xarray<T1>& a1, const xt::xarray<T2>& a2, CppUnit::SourceLine sourceLine,
	double rtol = std::numeric_limits<std::common_type_t<T1, T2>>::epsilon(),
	double atol = std::numeric_limits<std::common_type_t<T1, T2>>::epsilon()) {

	using namespace xt::print_options;

	if (xt::allclose(a1, a2, rtol, atol))
		return;

	constexpr auto digits1 = std::numeric_limits<T1>::digits10;
	constexpr auto digits2 = std::numeric_limits<T2>::digits10;

	std::ostringstream expected, actual, difference;

	expected   << precision(digits1) << a1;
	actual     << precision(digits2) << a2;
	difference << "Abs.diff: "
	           << xt::abs(a1 - a2);

	::CppUnit::AdditionalMessage additional_message(difference.str());

	::CppUnit::Asserter::failNotEqual(expected.str(), actual.str(),
		sourceLine, additional_message, "close assertion failed");
}

template<class E1, class E2>
void checkXExpressionClose(const xt::xexpression<E1>& e1, const xt::xexpression<E2>& e2, CppUnit::SourceLine sourceLine,
	double rtol = std::numeric_limits<std::common_type_t<typename E1::value_type, typename E2::value_type>>::epsilon(),
	double atol = std::numeric_limits<std::common_type_t<typename E1::value_type, typename E2::value_type>>::epsilon()) {

	/* materialize xexpressions */
	checkXArrayClose<typename E1::value_type, typename E2::value_type>(e1, e2, sourceLine, rtol, atol);
}

#define XT_ASSERT_XEXPRESSION_CLOSE( expected, actual, ... ) \
	checkXExpressionClose( expected, actual, CPPUNIT_SOURCELINE(), __VA_ARGS__ )

#endif // _TEST_XEXPRESSION_H
