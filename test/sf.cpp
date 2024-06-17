/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2012-2024  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <limits>

#include <cppunit/TestAssert.h>
#include <cppunit/TestCase.h>
#include <cppunit/Portability.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp> // IWYU pragma: keep

#include <weif/sf/mono.h>


class test_sf_suite: public CppUnit::TestCase {
CPPUNIT_TEST_SUITE(test_sf_suite);
CPPUNIT_TEST(test_mono1);
CPPUNIT_TEST(test_mono2);
CPPUNIT_TEST(test_mono_vec1);
CPPUNIT_TEST(test_mono_vec2);
CPPUNIT_TEST_SUITE_END();

void test_mono1() {
	using namespace weif::sf;

	constexpr auto delta = 2 * std::numeric_limits<double>::epsilon();
	const mono<double> sf{};

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(0.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.095491502812526298199441616733455781377, sf(0.1), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.34549150281252632112045392152558323154, sf(0.2), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6545084971874738447375492290593730548, sf(0.3), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.90450849718747375305350000989086585172, sf(0.4), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, sf(0.5), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.90450849718747354804173350339350550526, sf(0.6), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.6545084971874735130215429278894416858, sf(0.7), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.3454915028125261552624507709406269452, sf(0.8), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.095491502812526246946499990109134148279, sf(0.9), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(1.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(2.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(4.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(6.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(8.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(10.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(12.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(14.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(16.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(18.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(20.0), delta);
}

void test_mono2() {
	using namespace weif::sf;

	constexpr auto delta = 20 * std::numeric_limits<double>::epsilon();
	const mono<double> sf{};

	CPPUNIT_ASSERT_DOUBLES_EQUAL(9.8696044010893586188344909998761511353, sf.regular(0.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(9.5491502812526287597755108880997723681, sf.regular(0.1), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(8.6372875703131570690797949511052857407, sf.regular(0.2), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(7.2723166354163738996012036107550687387, sf.regular(0.3), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(5.6531781074217103289555253451729588831, sf.regular(0.4), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, sf.regular(0.5), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(2.5125236032985368895928452526622876377, sf.regular(0.6), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.3357316269132109968426578503834077779, sf.regular(0.7), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.53983047314457205766435726165511493315, sf.regular(0.8), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.11789074421299536078350533327467707859, sf.regular(0.9), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(1.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(2.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(4.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(6.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(8.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(10.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(12.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(14.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(16.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(18.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf.regular(20.0), delta);
}

void test_mono_vec1() {
	using namespace weif::sf;

	constexpr auto delta = 2 * std::numeric_limits<double>::epsilon();
	const xt::xarray<double> expected = {
		0.0,
		0.095491502812526298199441616733455781377,
		0.0,
		0.0
	};
	const xt::xarray<double> args = {0.0, 0.1, 1.0, 10.0};
	const mono<double> sf{};
	xt::xarray<double> actual = sf(args);

	CPPUNIT_ASSERT(xt::allclose(expected, actual, delta));
}

void test_mono_vec2() {
	using namespace weif::sf;

	constexpr auto delta = 20 * std::numeric_limits<double>::epsilon();
	const xt::xarray<double> expected = {
		9.8696044010893586188344909998761511353,
		9.5491502812526287597755108880997723681,
		0.0,
		0.0
	};
	const xt::xarray<double> args = {0.0, 0.1, 1.0, 10.0};
	const mono<double> sf{};
	xt::xarray<double> actual = sf.regular(args);

	CPPUNIT_ASSERT(xt::allclose(expected, actual, delta));
}

};
CPPUNIT_TEST_SUITE_REGISTRATION(test_sf_suite);

int main(int argc, char **argv) {
	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest(registry.makeTest());
	return !runner.run("", false);
}

