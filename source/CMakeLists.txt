
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

# SET (_src
#   matrix_creator.cc
#   vector_creator.cc
#   )

FILE(GLOB _header
  ${CMAKE_SOURCE_DIR}/include/mandy/*.h
  )

