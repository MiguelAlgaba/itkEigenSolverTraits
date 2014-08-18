set( _tests
  itkParameterizationQuadEdgeMeshFilterTest
)

foreach( _t ${_tests} )
  add_executable( ${_t} ${_tests} )
  target_link_libraries( ${_t} ${ITK_LIBRARIES} )
endforeach()
