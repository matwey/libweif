/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2017-2022  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>

#include <cubic_spline.h>


class test_cubic_spline_suite: public CppUnit::TestCase {
CPPUNIT_TEST_SUITE(test_cubic_spline_suite);
CPPUNIT_TEST(test_spline1);
CPPUNIT_TEST(test_spline2);
CPPUNIT_TEST(test_spline3);
CPPUNIT_TEST(test_spline4);
CPPUNIT_TEST(test_spline5);
CPPUNIT_TEST(test_spline6);
CPPUNIT_TEST(test_spline7);
CPPUNIT_TEST(test_spline8);
CPPUNIT_TEST(test_spline9);
CPPUNIT_TEST(test_spline10);
CPPUNIT_TEST(test_spline11);
CPPUNIT_TEST_SUITE_END();

void test_spline1() {
	const xt::xarray<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
	weif::cubic_spline s{expected};
	const xt::xarray<float> actual = s(xt::arange(0.0f, 4.0f, 1.0f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline2() {
	const xt::xarray<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
	weif::cubic_spline s{expected};
	xt::xarray<float> actual = {0.0f, 0.0f, 0.0f, 0.0f};

	for (std::size_t i = 0; i < 4; ++i) {
		actual(i) = s(i);
	}

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline3() {
	const xt::xarray<float> y = {1.0f, 2.0f, 3.0f, 4.0f};
	weif::cubic_spline s{y, weif::cubic_spline<float>::first_order_boundary{1.0f, 1.0f}};
	const xt::xarray<float> expected = {1.5f, 2.5f, 3.5f};
	const xt::xarray<float> actual = s(xt::arange(0.5f, 3.0f, 1.0f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline4() {
	const xt::xarray<float> y = {1.0f, 2.0f, 3.0f, 4.0f};
	weif::cubic_spline s{y, weif::cubic_spline<float>::first_order_boundary{1.0f, 1.0f}};
	const xt::xarray<float> expected = {1.25f, 2.25f, 3.25f};
	const xt::xarray<float> actual = s(xt::arange(0.25f, 3.0f, 1.0f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline5() {
	const xt::xarray<float> y = {1.0f, 2.0f, 3.0f, 4.0f};
	weif::cubic_spline s{y};
	const xt::xarray<float> expected = {1.5f, 2.5f, 3.5f};
	const xt::xarray<float> actual = s(xt::arange(0.5f, 3.0f, 1.0f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline6() {
	const xt::xarray<float> y = {1.0f, 2.0f, 3.0f, 4.0f};
	weif::cubic_spline s{y};
	const xt::xarray<float> expected = {1.25f, 2.25f, 3.25f};
	const xt::xarray<float> actual = s(xt::arange(0.25f, 3.0f, 1.0f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline7() {
	const xt::xarray<float> y = {0.0f, 1.0f};
	weif::cubic_spline s{y};
	const xt::xarray<float> expected = {0.0f, 0.25f, 0.5f, 0.75f};
	const xt::xarray<float> actual = s(xt::arange(0.0f, 1.0f, 0.25f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline8() {
	const xt::xarray<float> y = {0.0f, 1.0f};
	weif::cubic_spline s{y, weif::cubic_spline<float>::first_order_boundary{0.0f, 0.0f}};
	const xt::xarray<float> expected = {0.0f, 0.15625f, 0.5f, 0.84375f};
	const xt::xarray<float> actual = s(xt::arange(0.0f, 1.0f, 0.25f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline9() {
	const xt::xarray<float> y = {0.0f, 1.0f};
	weif::cubic_spline s{y, weif::cubic_spline<float>::first_order_boundary{1.0f, 1.0f}};
	const xt::xarray<float> expected = {0.0f, 0.25f, 0.5f, 0.75f};
	const xt::xarray<float> actual = s(xt::arange(0.0f, 1.0f, 0.25f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline10() {
	const xt::xarray<float> expected = {0.0f, 1.0f, 4.0f, 9.0f};
	weif::cubic_spline s{expected, weif::cubic_spline<float>::second_order_boundary{2.0f, 2.0f}};
	const xt::xarray<float> actual = s(xt::arange(0.0f, 4.0f, 1.0f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_spline11() {
	const xt::xarray<double> y = {0.0f, 1.0f, 4.0f, 9.0f};
	weif::cubic_spline s{y, weif::cubic_spline<double>::second_order_boundary{2.0f, 2.0f}};
	const xt::xarray<double> expected = {0.25f, 2.25f, 6.25f};
	const xt::xarray<double> actual = s(xt::arange(0.5f, 3.0f, 1.0f));

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

};
CPPUNIT_TEST_SUITE_REGISTRATION(test_cubic_spline_suite);

int main(int argc, char **argv) {
	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest(registry.makeTest());
	return !runner.run("", false);
}
