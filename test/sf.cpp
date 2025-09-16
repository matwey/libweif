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

#include <xtensor/io/xio.hpp>
#include <xtensor/containers/xarray.hpp> // IWYU pragma: keep

#include <weif/sf/mono.h>
#include <weif/sf/gauss.h>

#include "xexpression.h"


class test_sf_suite: public CppUnit::TestCase {
CPPUNIT_TEST_SUITE(test_sf_suite);
CPPUNIT_TEST(test_mono1);
CPPUNIT_TEST(test_mono2);
CPPUNIT_TEST(test_mono_vec1);
CPPUNIT_TEST(test_mono_vec2);
CPPUNIT_TEST(test_gauss1);
CPPUNIT_TEST(test_gauss2);
CPPUNIT_TEST(test_gauss3);
CPPUNIT_TEST(test_gauss4);
CPPUNIT_TEST(test_gauss_vec1);
CPPUNIT_TEST(test_gauss_vec2);
CPPUNIT_TEST(test_gauss_vec3);
CPPUNIT_TEST(test_gauss_vec4);
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

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
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

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss1() {
	using namespace weif::sf;

	constexpr auto delta = 2 * std::numeric_limits<double>::epsilon();
	const gauss<double> sf{0.0};

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

void test_gauss2() {
	using namespace weif::sf;

	constexpr auto delta = 20 * std::numeric_limits<double>::epsilon();
	const gauss<double> sf{0.0};

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

void test_gauss3() {
	using namespace weif::sf;

	constexpr auto delta = 2 * std::numeric_limits<double>::epsilon();
	const gauss<double> sf{0.1};

	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(0.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.095474508234832555785906908949474601604, sf(0.1), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.34524562062268631772164253286300680725, sf(0.2), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.65346090005469131852090787272639940692, sf(0.3), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.90193633296362587435564624777320852076, sf(0.4), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.99556025079112537972064984724386080498, sf(0.5), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.89873141126276185743481757437463734264, sf(0.6), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.64882516239141285175503316197275844736, sf(0.7), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.34157831716541130557889760766872363269, sf(0.8), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.094124695687912931382087740307528013598, sf(0.9), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(1.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(2.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(4.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(8.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(10.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(12.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(14.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(16.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(18.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sf(20.0), delta);
}

void test_gauss4() {
	using namespace weif::sf;

	constexpr auto delta = 20 * std::numeric_limits<double>::epsilon();
	const gauss<double> sf{0.1};

	CPPUNIT_ASSERT_DOUBLES_EQUAL(9.8696044010893586188344909998761511353, sf.regular(0.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(9.5474508234832545186107178241953935714, sf.regular(0.1), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(8.6311405155671569847919704058049388483, sf.regular(0.2), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(7.2606766672743458339735335991930140782, sf.regular(0.3), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(5.6371020810226610888787367972082317232, sf.regular(0.4), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(3.9822410031645015188825993889754432199, sf.regular(0.5), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(2.4964761423965599759913038267601213341, sf.regular(0.6), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.3241329844722708740164638606665138186, sf.regular(0.7), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.53371612057095510571263494483332827774, sf.regular(0.8), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.11620332800976904535514786383332017124, sf.regular(0.9), delta);
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

void test_gauss_vec1() {
	using namespace weif::sf;

	constexpr auto delta = 2 * std::numeric_limits<double>::epsilon();
	const xt::xarray<double> expected = {
		0.0,
		0.095491502812526298199441616733455781377,
		0.0,
		0.0
	};
	const xt::xarray<double> args = {0.0, 0.1, 1.0, 10.0};
	const gauss<double> sf{0.0};
	xt::xarray<double> actual = sf(args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_vec2() {
	using namespace weif::sf;

	constexpr auto delta = 20 * std::numeric_limits<double>::epsilon();
	const xt::xarray<double> expected = {
		9.8696044010893586188344909998761511353,
		9.5491502812526287597755108880997723681,
		0.0,
		0.0
	};
	const xt::xarray<double> args = {0.0, 0.1, 1.0, 10.0};
	const gauss<double> sf{0.0};
	xt::xarray<double> actual = sf.regular(args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_vec3() {
	using namespace weif::sf;

	constexpr auto delta = 2 * std::numeric_limits<double>::epsilon();
	const xt::xarray<double> expected = {
		0.0,
		0.095474508234832555785906908949474601604,
		0.0,
		0.0
	};
	const xt::xarray<double> args = {0.0, 0.1, 1.0, 10.0};
	const gauss<double> sf{0.1};
	xt::xarray<double> actual = sf(args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_vec4() {
	using namespace weif::sf;

	constexpr auto delta = 20 * std::numeric_limits<double>::epsilon();
	const xt::xarray<double> expected = {
		9.8696044010893586188344909998761511353,
		9.5474508234832545186107178241953935714,
		0.0,
		0.0
	};
	const xt::xarray<double> args = {0.0, 0.1, 1.0, 10.0};
	const gauss<double> sf{0.1};
	xt::xarray<double> actual = sf.regular(args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

};
CPPUNIT_TEST_SUITE_REGISTRATION(test_sf_suite);

int main(int argc, char **argv) {
	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest(registry.makeTest());
	return !runner.run("", false);
}

