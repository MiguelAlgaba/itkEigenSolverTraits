set( _tests
  itkParameterizationQuadEdgeMeshFilterTest
)

foreach( _t ${_tests} )
  add_executable( ${_t} ${_tests} )
  target_link_libraries( ${_t} ${ITK_LIBRARIES} )
endforeach()

set( INPUTDATA ${CMAKE_SOURCE_DIR}/data )

foreach( border 0 1 )
  foreach( coeff 0 1 2 3 4 )
    foreach( solver 0 1 2 )
      add_test(NAME itkParameterizationQuadEdgeMeshFilterTest_b${border}_c${coeff}_s${solver}
            COMMAND itkParameterizationQuadEdgeMeshFilterTest
              ${INPUTDATA}/mushroom.vtk ${border} ${coeff} ${solver}
              ${TEMP}/param_border${border}_coeff${coeff}_solver${solver}.vtk
      )
    endforeach( solver )
  endforeach( coeff )
endforeach( border )
