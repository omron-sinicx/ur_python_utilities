cmake_minimum_required(VERSION 3.8)
project(ur_gripper_gz)

find_package(ament_cmake REQUIRED)

# Hand-E meshes from robotiq-cri (PickNik vendor stack has no Hand-E model).
set(ROBOTIQ_CRI_MESHES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/robotiq-cri/robotiq_description/meshes)

install(DIRECTORY config launch urdf srdf
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY ${ROBOTIQ_CRI_MESHES}/hande
  DESTINATION share/${PROJECT_NAME}/meshes
)
install(FILES ${ROBOTIQ_CRI_MESHES}/robotiq_85_coupler.stl
  DESTINATION share/${PROJECT_NAME}/meshes
)

ament_package()
