cmake_minimum_required(VERSION 3.8)
project(ur_gripper_gz)

find_package(ament_cmake REQUIRED)

# Hand-E meshes from robotiq-cri (PickNik vendor stack has no Hand-E model).
# REALPATH resolves '..' so ament_cmake_symlink_install gets an absolute path.
get_filename_component(ROBOTIQ_CRI_MESHES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../robotiq-cri/robotiq_description/meshes
  REALPATH)

install(DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY ${ROBOTIQ_CRI_MESHES}/hande
  DESTINATION share/${PROJECT_NAME}/meshes
)
install(FILES ${ROBOTIQ_CRI_MESHES}/robotiq_85_coupler.stl
  DESTINATION share/${PROJECT_NAME}/meshes
)

ament_package()
