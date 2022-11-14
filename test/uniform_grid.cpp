/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Copyright (C) 2017-2022  Matwey V. Kornilov <matwey.kornilov@gmail.com>
 */

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>

#include <uniform_grid.h>


class test_uniform_grid_suite: public CppUnit::TestCase {
CPPUNIT_TEST_SUITE(test_uniform_grid_suite);
CPPUNIT_TEST(test_construct1);
CPPUNIT_TEST(test_construct2);
CPPUNIT_TEST(test_construct3);
CPPUNIT_TEST(test_intersect1);
CPPUNIT_TEST(test_intersect2);
CPPUNIT_TEST(test_intersect3);
CPPUNIT_TEST(test_intersect4);
CPPUNIT_TEST_SUITE_END();

void test_construct1() {
	weif::uniform_grid ug{0.5f, 1.0f, 4};

	const xt::xarray<float> expected = {0.5f, 1.5f, 2.5f, 3.5f};
	const xt::xarray<float> actual = ug;

	CPPUNIT_ASSERT_EQUAL(expected, actual);
};

void test_construct2() {
	const xt::xarray<float> expected = {0.5f, 1.5f, 2.5f, 3.5f};
	weif::uniform_grid ug(expected.cbegin(), expected.cend());
	const xt::xarray<float> actual = ug;

	CPPUNIT_ASSERT_EQUAL(expected, actual);
	CPPUNIT_ASSERT_EQUAL(0.5f, ug.origin());
	CPPUNIT_ASSERT_EQUAL(1.0f, ug.delta());
};

void test_construct3() {
	const xt::xarray<float> non_uniform = {0.5f, 1.5f, 2.5f, 4.0f};

	CPPUNIT_ASSERT_THROW(weif::uniform_grid ug(non_uniform.cbegin(), non_uniform.cend()), weif::non_uniform_grid);
};

void test_intersect1() {
	weif::uniform_grid ug{0.5f, 1.0f, 4};

	CPPUNIT_ASSERT_EQUAL(ug, ug.intersect(ug));
};

void test_intersect2() {
	weif::uniform_grid ug{0.5f, 1.0f, 4};
	weif::uniform_grid ug2{10.5f, 1.0f, 4};
	const auto i_ug = ug2.intersect(ug);

	CPPUNIT_ASSERT_EQUAL(std::size_t(0), i_ug.size());
};

void test_intersect3() {
	weif::uniform_grid ug{0.5f, 1.0f, 4};
	weif::uniform_grid ug2{1.5f, 1.0f, 4};
	const auto i_ug = ug2.intersect(ug);

	CPPUNIT_ASSERT_EQUAL(1.5f, i_ug.origin());
	CPPUNIT_ASSERT_EQUAL(std::size_t(3), i_ug.size());
};

void test_intersect4() {
	weif::uniform_grid ug{0.5f, 1.0f, 4};
	weif::uniform_grid ug2{0.0f, 1.0f, 4};

	CPPUNIT_ASSERT_THROW(ug2.intersect(ug), weif::mismatched_grids);
};

};
CPPUNIT_TEST_SUITE_REGISTRATION(test_uniform_grid_suite);

int main(int argc, char **argv) {
	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest(registry.makeTest());
	return !runner.run("", false);
}
