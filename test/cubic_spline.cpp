/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2017-2022  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <cppunit/TestAssert.h>
#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

#include <xtensor/containers/xarray.hpp> // IWYU pragma: keep
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xio.hpp> // IWYU pragma: keep

#include <weif/detail/cubic_spline.h>

#include "xexpression.h"


using weif::detail::cubic_spline;
using weif::detail::first_order_boundary;
using weif::detail::second_order_boundary;

using value_type = float;


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
CPPUNIT_TEST(test_spline12);
CPPUNIT_TEST(test_spline13);
CPPUNIT_TEST(test_spline14);
CPPUNIT_TEST_SUITE_END();

void test_spline1() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> expected = {1.0f, 2.0f, 3.0f, 4.0f};
	cubic_spline s{expected};
	const xt::xarray<value_type> actual = s(xt::arange(0.0f, 4.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline2() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> expected = {1.0f, 2.0f, 3.0f, 4.0f};
	cubic_spline s{expected};
	xt::xarray<value_type> actual = {0.0f, 0.0f, 0.0f, 0.0f};

	for (std::size_t i = 0; i < 4; ++i) {
		actual(i) = s(i);
	}

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline3() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> y = {1.0f, 2.0f, 3.0f, 4.0f};
	cubic_spline s{y, first_order_boundary{1.0f, 1.0f}};
	const xt::xarray<value_type> expected = {1.5f, 2.5f, 3.5f};
	const xt::xarray<value_type> actual = s(xt::arange(0.5f, 3.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline4() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> y = {1.0f, 2.0f, 3.0f, 4.0f};
	cubic_spline s{y, first_order_boundary{1.0f, 1.0f}};
	const xt::xarray<value_type> expected = {1.25f, 2.25f, 3.25f};
	const xt::xarray<value_type> actual = s(xt::arange(0.25f, 3.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline5() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> y = {1.0f, 2.0f, 3.0f, 4.0f};
	cubic_spline s{y};
	const xt::xarray<value_type> expected = {1.5f, 2.5f, 3.5f};
	const xt::xarray<value_type> actual = s(xt::arange(0.5f, 3.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline6() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> y = {1.0f, 2.0f, 3.0f, 4.0f};
	cubic_spline s{y};
	const xt::xarray<value_type> expected = {1.25f, 2.25f, 3.25f};
	const xt::xarray<value_type> actual = s(xt::arange(0.25f, 3.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline7() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> y = {0.0f, 1.0f};
	cubic_spline s{y};
	const xt::xarray<value_type> expected = {0.0f, 0.25f, 0.5f, 0.75f};
	const xt::xarray<value_type> actual = s(xt::arange(0.0f, 1.0f, 0.25f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline8() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> y = {0.0f, 1.0f};
	cubic_spline s{y, first_order_boundary{0.0f, 0.0f}};
	const xt::xarray<value_type> expected = {0.0f, 0.15625f, 0.5f, 0.84375f};
	const xt::xarray<value_type> actual = s(xt::arange(0.0f, 1.0f, 0.25f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline9() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> y = {0.0f, 1.0f};
	cubic_spline s{y, first_order_boundary{1.0f, 1.0f}};
	const xt::xarray<value_type> expected = {0.0f, 0.25f, 0.5f, 0.75f};
	const xt::xarray<value_type> actual = s(xt::arange(0.0f, 1.0f, 0.25f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline10() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> expected = {0.0f, 1.0f, 4.0f, 9.0f};
	cubic_spline s{expected, second_order_boundary{2.0f, 2.0f}};
	const xt::xarray<value_type> actual = s(xt::arange(0.0f, 4.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline11() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	const xt::xarray<value_type> y = {0.0f, 1.0f, 4.0f, 9.0f};
	cubic_spline s{y, second_order_boundary{2.0f, 2.0f}};
	const xt::xarray<value_type> expected = {0.25f, 2.25f, 6.25f};
	const xt::xarray<value_type> actual = s(xt::arange(0.5f, 3.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline12() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	cubic_spline s{xt::arange<value_type>(0.0f, 4.0f, 1.0f) + 1.0f};
	const xt::xarray<value_type> expected = {1.0f, 2.0f, 3.0f, 4.0f};
	const xt::xarray<value_type> actual = s(xt::arange(0.0f, 4.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline13() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	cubic_spline s{xt::arange<value_type>(0.0f, 4.0f, 1.0f) + 1.0f,
		first_order_boundary{1.0f, 1.0f}};
	const xt::xarray<value_type> expected = {1.5f, 2.5f, 3.5f};
	const xt::xarray<value_type> actual = s(xt::arange(0.5f, 3.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

void test_spline14() {
	constexpr auto delta = std::numeric_limits<value_type>::epsilon();

	cubic_spline s{xt::square(xt::arange<value_type>(0.0f, 4.0f, 1.0f)),
		second_order_boundary{2.0f, 2.0f}};
	const xt::xarray<value_type> expected = {0.0f, 1.0f, 4.0f, 9.0f};
	const xt::xarray<value_type> actual = s(xt::arange(0.0f, 4.0f, 1.0f));

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
};

};
CPPUNIT_TEST_SUITE_REGISTRATION(test_cubic_spline_suite);

int main(int argc, char **argv) {
	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest(registry.makeTest());
	return !runner.run("", false);
}
