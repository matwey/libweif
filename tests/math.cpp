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

#include <weif/math.h>

#include "xexpression.h"


class test_math_suite: public CppUnit::TestCase {
CPPUNIT_TEST_SUITE(test_math_suite);
CPPUNIT_TEST(test_jinc_pi1);
CPPUNIT_TEST(test_jinc_pi_vec1);
CPPUNIT_TEST(test_sinc_pi_vec1);
CPPUNIT_TEST_SUITE_END();

void test_jinc_pi1() {
	using namespace weif::math;

	constexpr auto delta = std::numeric_limits<double>::epsilon();

	CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, jinc_pi(0.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.99875052072483995088407208329032034367448, jinc_pi(0.1), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.99500832639235995301101665222954375679445, jinc_pi(0.2), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.98879210848736004857911266164636566401244, jinc_pi(0.3), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.98013288977659371670035875779326717682908, jinc_pi(0.4), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.96907383069949554553581830456612656320181, jinc_pi(0.5), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.95566996021305245396908944143606557677635, jinc_pi(0.6), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.93998783297159698159955096152452021447186, jinc_pi(0.7), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.92210511523542497078257209974936128138697, jinc_pi(0.8), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.90211010239734593734356711925538912797348, jinc_pi(0.9), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.88010117148986703191936440743782982625493, jinc_pi(1.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.57672480775687338720244824226913708691982, jinc_pi(2.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.033021664011774568071592710401637514363668, jinc_pi(4.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.092227952709188536057591601015384586246921, jinc_pi(6.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.05865908671347865609531916289761365288722, jinc_pi(8.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0086945492337722873339497536051718576612593, jinc_pi(10.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.037241184081771268727949619394049526447604, jinc_pi(12.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.019053593528399036157882561026301380802742, jinc_pi(14.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.01129964695766302327983541280725950305621, jinc_pi(16.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.020888320609785510445180601101201575494834, jinc_pi(18.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0066833124175850045578992974193646719982977, jinc_pi(20.0), delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, jinc_pi(std::numeric_limits<double>::infinity()), delta);
};

void test_jinc_pi_vec1() {
	using namespace weif::math;

	constexpr auto delta = std::numeric_limits<double>::epsilon();
	const xt::xarray<double> expected = {
		1.0,
		0.99875052072483995088407208329032034367448,
		0.88010117148986703191936440743782982625493,
		0.0086945492337722873339497536051718576612593,
		0.0
	};
	const xt::xarray<double> args = {0.0, 0.1, 1.0, 10.0, std::numeric_limits<double>::infinity()};
	xt::xarray<double> actual = jinc_pi(args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_sinc_pi_vec1() {
	using namespace weif::math;

	constexpr auto delta = std::numeric_limits<double>::epsilon();
	const xt::xarray<double> expected = {
		1.0,
		0.99833416646828152274465063467924745690004,
		0.84147098480789650665250232163029899962245,
		-0.054402111088936981340474766185137728168366,
		0.0
	};
	const xt::xarray<double> args = {0.0, 0.1, 1.0, 10.0, std::numeric_limits<double>::infinity()};
	xt::xarray<double> actual = sinc_pi(args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

};
CPPUNIT_TEST_SUITE_REGISTRATION(test_math_suite);

int main(int argc, char **argv) {
	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest(registry.makeTest());
	return !runner.run("", false);
}
