OPTION(SPHINX_OUTPUT "documentation output path" "${CMAKE_BINARY_DIR}/docs/html")


find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

set(DOX_SOURCES
	docs/index.dox)

set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_EXCLUDE_SYMBOLS detail)
set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_CLASS_GRAPH NO)
set(DOXYGEN_COLLABORATION_GRAPH NO)
set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs/doxygen")
doxygen_add_docs(doxygen src include ${DOX_SOURCES}
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

sphinx_add_docs(docs ALL
	DEFINE "breathe_projects.libweif=${DOXYGEN_OUTPUT_DIRECTORY}/xml"
	DEPENDS doxygen
	OUTPUT_DIRECTORY "${SPHINX_OUTPUT}")
