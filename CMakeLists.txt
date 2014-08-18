project( itkEigenSolverTraits )

cmake_minimum_required( VERSION 2.8 )

set( CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${CMAKE_SOURCE_DIR}/cmake
)

find_package( ITK REQUIRED )
include( ${ITK_USE_FILE} )

find_package( Eigen3 REQUIRED )

include_directories( BEFORE
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/include
)

add_subdirectory( test )
