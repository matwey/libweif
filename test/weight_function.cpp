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

#include <weif/af/point.h>
#include <weif/af/circular.h>
#include <weif/af/gauss.h>
#include <weif/sf/mono.h>
#include <weif/sf/gauss.h>
#include <weif/detail/weight_function_base.h>
#include <weif/weight_function.h>

#include "xexpression.h"


class test_dimensionless_weight_function_suite: public CppUnit::TestCase {
CPPUNIT_TEST_SUITE(test_dimensionless_weight_function_suite);
CPPUNIT_TEST(test_mono_point_vec1);
CPPUNIT_TEST(test_mono_circular_vec1);
CPPUNIT_TEST(test_mono_gauss_vec1);
CPPUNIT_TEST(test_gauss_point_vec1);
CPPUNIT_TEST(test_gauss_point_vec2);
CPPUNIT_TEST(test_gauss_point_vec3);
CPPUNIT_TEST_SUITE_END();

void test_mono_point_vec1() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	constexpr double c = 1.9991032874390479724456646360827626800501;
	const xt::xarray<double> expected = {c, c, c, c, c, c, c, c, c, c, c};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function(sf::mono<double>{}, af::point<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_mono_circular_vec1() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	const xt::xarray<double> expected = {
		0.0,
		0.0095424267805903033901469619621608955428732,
		0.057751681372150197649916026729741548607505,
		0.18275258941523022772990138044815375061858,
		0.44924254632329663701006876363839208048182,
		0.86287430440237028413258369255107941758679,
		1.2614994482444274348527859556314005305702,
		1.5739245403642390458147778288821298254394,
		1.7957566887471521401764802750648900234123,
		1.9370991581536685585369784254993146893821,
		1.9991032874390479724456646360827626800501
	};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function(sf::mono<double>{}, af::circular<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_mono_gauss_vec1() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	const xt::xarray<double> expected = {
		0.0,
		0.027137581375996065171658183879625752019635,
		0.17476188516742233327728999104451119932163,
		0.51712345955734487864103083184261596938139,
		0.95171316166228320405710655849369735433704,
		1.3214145058928385116073278751937442745622,
		1.5899308811559572801316408232226560409783,
		1.7741515511024605903854063708844472103969,
		1.8952868631631815236581760163353700577009,
		1.9684370590292808924194571977468804543152,
		1.9991032874390479724456646360827626800501
	};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function(sf::mono<double>{}, af::gauss<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec1() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	constexpr double c = 1.9133847737114990689173989228583762413866;
	const xt::xarray<double> expected = {c, c, c, c, c, c, c, c, c, c, c};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function(sf::gauss{0.1}, af::point<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec2() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.00009;
	constexpr double c = 1.9865386625648359962669433391220293434374;
	const xt::xarray<double> expected = {c, c, c, c, c, c, c, c, c, c, c};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function(sf::gauss{0.01}, af::point<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec3() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	constexpr double c = 1.9991032874390479724456646360827626800501;
	const xt::xarray<double> expected = {c, c, c, c, c, c, c, c, c, c, c};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function(sf::gauss{0.0}, af::point<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

};
CPPUNIT_TEST_SUITE_REGISTRATION(test_dimensionless_weight_function_suite);

class test_dimensionless_weight_function_2d_suite: public CppUnit::TestCase {
CPPUNIT_TEST_SUITE(test_dimensionless_weight_function_2d_suite);
CPPUNIT_TEST(test_mono_point_vec1);
CPPUNIT_TEST(test_mono_circular_vec1);
CPPUNIT_TEST(test_mono_gauss_vec1);
CPPUNIT_TEST(test_gauss_point_vec1);
CPPUNIT_TEST(test_gauss_point_vec2);
CPPUNIT_TEST(test_gauss_point_vec3);
CPPUNIT_TEST_SUITE_END();

void test_mono_point_vec1() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	constexpr double c = 1.9991032874390479724456646360827626800501;
	const xt::xarray<double> expected = {c, c, c, c, c, c, c, c, c, c, c};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function_2d(sf::mono<double>{}, af::point<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_mono_circular_vec1() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	const xt::xarray<double> expected = {
		0.0,
		0.0095424267805903033901469619621608955428732,
		0.057751681372150197649916026729741548607505,
		0.18275258941523022772990138044815375061858,
		0.44924254632329663701006876363839208048182,
		0.86287430440237028413258369255107941758679,
		1.2614994482444274348527859556314005305702,
		1.5739245403642390458147778288821298254394,
		1.7957566887471521401764802750648900234123,
		1.9370991581536685585369784254993146893821,
		1.9991032874390479724456646360827626800501
	};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function_2d(sf::mono<double>{}, af::circular<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_mono_gauss_vec1() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	const xt::xarray<double> expected = {
		0.0,
		0.027137581375996065171658183879625752019635,
		0.17476188516742233327728999104451119932163,
		0.51712345955734487864103083184261596938139,
		0.95171316166228320405710655849369735433704,
		1.3214145058928385116073278751937442745622,
		1.5899308811559572801316408232226560409783,
		1.7741515511024605903854063708844472103969,
		1.8952868631631815236581760163353700577009,
		1.9684370590292808924194571977468804543152,
		1.9991032874390479724456646360827626800501
	};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function_2d(sf::mono<double>{}, af::gauss<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec1() {
	using namespace weif;
	using namespace weif::detail;

	const auto delta = std::pow(std::numeric_limits<double>::epsilon(), 2.0/3.0);
	constexpr double c = 1.9133847737114990689173989228583762413866;
	const xt::xarray<double> expected = {c, c, c, c, c, c, c, c, c, c, c};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function_2d(sf::gauss{0.1}, af::point<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec2() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.00009;
	constexpr double c = 1.9865386625648359962669433391220293434374;
	const xt::xarray<double> expected = {c, c, c, c, c, c, c, c, c, c, c};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function_2d(sf::gauss{0.01}, af::point<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec3() {
	using namespace weif;
	using namespace weif::detail;

	constexpr double delta = 0.0003;
	constexpr double c = 1.9991032874390479724456646360827626800501;
	const xt::xarray<double> expected = {c, c, c, c, c, c, c, c, c, c, c};
	const xt::xarray<double> args = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	xt::xarray<double> actual = dimensionless_weight_function_2d(sf::gauss{0.0}, af::point<double>{}, args);

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

};
CPPUNIT_TEST_SUITE_REGISTRATION(test_dimensionless_weight_function_2d_suite);

class test_weight_function_suite: public CppUnit::TestCase {
CPPUNIT_TEST_SUITE(test_weight_function_suite);
CPPUNIT_TEST(test_mono_point_vec1);
CPPUNIT_TEST(test_mono_circular_vec1);
CPPUNIT_TEST(test_mono_gauss_vec1);
CPPUNIT_TEST(test_gauss_point_vec1);
CPPUNIT_TEST(test_gauss_point_vec2);
CPPUNIT_TEST(test_gauss_point_vec3);
CPPUNIT_TEST_SUITE_END();

void test_mono_point_vec1() {
	using namespace weif;

	constexpr double lambda = 550;
	constexpr double aperture_scale = 10;
	constexpr double delta = 0.0003;
	const xt::xarray<double> args = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, std::numeric_limits<double>::infinity()};
	const weight_function<double> wf(sf::mono<double>{}, lambda, af::point<double>{}, aperture_scale, 1024);
	const xt::xarray<double> actual = wf(args);
	const xt::xarray<double> expected = {
		0.0,
		68541193203.074699841774822250721611368818,
		122126522328.85717429491402679001511623388,
		217604724387.4327644368299751377039892409,
		387727540036.09136175337735358778328922997,
		690851936811.72176262852279766865892823018,
		1230958209860.6672123780939793934380752668,
		2193318182498.3903949367943120230915638022,
		std::numeric_limits<double>::infinity()
	};

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_mono_circular_vec1() {
	using namespace weif;

	constexpr double lambda = 550;
	constexpr double aperture_scale = 10;
	constexpr double delta = 0.0003;
	const xt::xarray<double> args = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, std::numeric_limits<double>::infinity()};
	const weight_function<double> wf(sf::mono<double>{}, lambda, af::circular<double>{}, aperture_scale, 1024);
	const xt::xarray<double> actual = wf(args);
	const xt::xarray<double> expected = {
		0.0,
		46095950091.596612607102260600389647472116,
		96324603994.200757824334431948993478379833,
		188826153859.60382074969531148828372231882,
		356304606621.6182453863281127220806838675,
		657076804976.76374836331145577965938358833,
		1195089206023.7645592517542497518071268875,
		2155584522441.6070284117170416038147092549,
		std::numeric_limits<double>::infinity()
	};

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_mono_gauss_vec1() {
	using namespace weif;

	constexpr double lambda = 550;
	constexpr double aperture_scale = 10;
	constexpr double delta = 0.0003;
	const xt::xarray<double> args = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, std::numeric_limits<double>::infinity()};
	const weight_function<double> wf(sf::mono<double>{}, lambda, af::gauss<double>{}, aperture_scale, 1024);
	const xt::xarray<double> actual = wf(args);
	const xt::xarray<double> expected = {
		0.0,
		56249590280.866457678494046451327661849971,
		108481158127.85498284498602986817102027147,
		202756532197.33789636818245702232651548152,
		371809348522.62117416810705090698809709724,
		673981231127.59302811145653407195965757201,
		1213239250923.8783447887393524181491460868,
		2174843669172.5074791409300099826737177478,
		std::numeric_limits<double>::infinity()
	};

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec1() {
	using namespace weif;

	constexpr double lambda = 550;
	constexpr double aperture_scale = 10;
	constexpr double delta = 0.0003;
	const xt::xarray<double> args = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, std::numeric_limits<double>::infinity()};
	const weight_function<double> wf(sf::gauss<double>{0.1}, lambda, af::point<double>{}, aperture_scale, 1024);
	const xt::xarray<double> actual = wf(args);
	const xt::xarray<double> expected = {
		0.0,
		65602250904.597050646406921163723673195889,
		116889922476.05285401762004624912925507766,
		208274164194.87824759193409320142826356136,
		371102371805.93516868840319737989353425846,
		661229254681.49447084461203044649636985679,
		1178176590785.2707525509042661373340188365,
		2099272028947.1056206850214772391575422684,
		std::numeric_limits<double>::infinity()
	};

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec2() {
	using namespace weif;

	constexpr double lambda = 550;
	constexpr double aperture_scale = 10;
	constexpr double delta = 0.0003;
	const xt::xarray<double> args = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, std::numeric_limits<double>::infinity()};
	const weight_function<double> wf(sf::gauss<double>{0.01}, lambda, af::point<double>{}, aperture_scale, 1024);
	const xt::xarray<double> actual = wf(args);
	const xt::xarray<double> expected = {
		0.0,
		68110402865.007298333543217481077655456327,
		121358941208.91419145929804825708522411656,
		216237050315.7809055527283703256332597153,
		385290621881.55452245239758058086156528313,
		686509842291.54217979545993553961021704594,
		1223221476976.5228483744362581034970207527,
		2179532891680.2335466733829593944849746025,
		std::numeric_limits<double>::infinity()
	};

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}

void test_gauss_point_vec3() {
	using namespace weif;

	constexpr double lambda = 550;
	constexpr double aperture_scale = 10;
	constexpr double delta = 0.0003;
	const xt::xarray<double> args = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, std::numeric_limits<double>::infinity()};
	const weight_function<double> wf(sf::gauss<double>{0.0}, lambda, af::point<double>{}, aperture_scale, 1024);
	const xt::xarray<double> actual = wf(args);
	const xt::xarray<double> expected = {
		0.0,
		68541193203.074699841774822250721611368818,
		122126522328.85717429491402679001511623388,
		217604724387.4327644368299751377039892409,
		387727540036.09136175337735358778328922997,
		690851936811.72176262852279766865892823018,
		1230958209860.6672123780939793934380752668,
		2193318182498.3903949367943120230915638022,
		std::numeric_limits<double>::infinity()
	};

	XT_ASSERT_XEXPRESSION_CLOSE(expected, actual, delta);
}


};
CPPUNIT_TEST_SUITE_REGISTRATION(test_weight_function_suite);

int main(int argc, char **argv) {
	CppUnit::TextUi::TestRunner runner;
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	runner.addTest(registry.makeTest());
	return !runner.run("", false);
}
