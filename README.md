# libweif

**libweif** is a C++ library for calculating weight functions for optical turbulence in the atmosphere.

## Documentation

Complete documentation is available at **[https://libweif.readthedocs.io](https://libweif.readthedocs.io)**.

The documentation includes:
- [Tutorials](https://libweif.readthedocs.io/en/latest/tutorial.html) – step‑by‑step examples of common use cases.
- [How‑to guides](https://libweif.readthedocs.io/en/latest/howto.html) – building, installing, and integrating the library.
- [API reference](https://libweif.readthedocs.io/en/latest/api_reference.html) – detailed class and function descriptions.
- [Internals](https://libweif.readthedocs.io/en/latest/internals.html) – design decisions, versioning policy, and unit conventions.

## Quick Installation

### Dependencies

- C++ compiler with C++20 support (gcc 10+, clang 10+)
- CMake 3.10+
- Boost.Math and Boost.Program_options
- FFTW3
- RapidCSV
- XTensor 0.26+
- CPPUnit (for tests)

### Build from Source

```bash
git clone https://github.com/matwey/libweif
cd libweif
cmake -DCMAKE_BUILD_TYPE=Release -B build .
cmake --build build
```

Run the test suite to verify the build:

```bash
ctest --test-dir build
```

Install system‑wide (optional):

```bash
sudo cmake --install build
```

For detailed installation instructions, see the [Installation guide](https://libweif.readthedocs.io/en/latest/howto.html#installation).

## Minimal Example

```cpp
#include <weif/weight_function.h>
#include <weif/af/circular.h>
#include <weif/sf/mono.h>

int main() {
	// Create a monochromatic spectral filter
	weif::sf::mono<double> spectral_filter;

	// Create a circular aperture filter
	weif::af::circular<double> aperture_filter;

	// Construct the weight function for wavelength 500 nm,
	// aperture diameter 10 mm
	weif::weight_function<double> wf(spectral_filter, 500.0, aperture_filter, 10.0);

	// Evaluate the weight at altitude 5 km
	double value = wf(5.0);

	return 0;
}
```

More examples are available in the `examples/` directory and in the [Tutorials](https://libweif.readthedocs.io/en/latest/tutorial.html).

## Contributing

Contributions are welcome. Please see the [Contributing guide](https://libweif.readthedocs.io/en/latest/howto.html#howto_1contributing) for details.

## License

libweif is licensed under the **GNU General Public License v3.0 or later**.  
See the [LICENSE](LICENSE) file for details.
