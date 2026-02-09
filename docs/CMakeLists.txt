OPTION(SPHINX_OUTPUT "documentation output path" "${CMAKE_BINARY_DIR}/docs/html")


find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

set(DOXYGEN_GENERATE_HTML NO)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_EXCLUDE_SYMBOLS detail)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/doxygen")
doxygen_add_docs(doxygen ${SOURCES} include)

sphinx_add_docs(docs ALL
	DEFINE "breathe_projects.libweif=${DOXYGEN_OUTPUT_DIRECTORY}/xml"
	DEPENDS doxygen
	OUTPUT_DIRECTORY "${SPHINX_OUTPUT}")
