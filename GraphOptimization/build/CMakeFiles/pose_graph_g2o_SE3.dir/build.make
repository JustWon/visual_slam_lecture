# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/parallels/Desktop/visual_slam_lecture/GraphOptimization

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/parallels/Desktop/visual_slam_lecture/GraphOptimization/build

# Include any dependencies generated for this target.
include CMakeFiles/pose_graph_g2o_SE3.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pose_graph_g2o_SE3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pose_graph_g2o_SE3.dir/flags.make

CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.o: CMakeFiles/pose_graph_g2o_SE3.dir/flags.make
CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.o: ../pose_graph_g2o_SE3.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/parallels/Desktop/visual_slam_lecture/GraphOptimization/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.o -c /home/parallels/Desktop/visual_slam_lecture/GraphOptimization/pose_graph_g2o_SE3.cpp

CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/parallels/Desktop/visual_slam_lecture/GraphOptimization/pose_graph_g2o_SE3.cpp > CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.i

CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/parallels/Desktop/visual_slam_lecture/GraphOptimization/pose_graph_g2o_SE3.cpp -o CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.s

# Object files for target pose_graph_g2o_SE3
pose_graph_g2o_SE3_OBJECTS = \
"CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.o"

# External object files for target pose_graph_g2o_SE3
pose_graph_g2o_SE3_EXTERNAL_OBJECTS =

pose_graph_g2o_SE3: CMakeFiles/pose_graph_g2o_SE3.dir/pose_graph_g2o_SE3.cpp.o
pose_graph_g2o_SE3: CMakeFiles/pose_graph_g2o_SE3.dir/build.make
pose_graph_g2o_SE3: CMakeFiles/pose_graph_g2o_SE3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/parallels/Desktop/visual_slam_lecture/GraphOptimization/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pose_graph_g2o_SE3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pose_graph_g2o_SE3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pose_graph_g2o_SE3.dir/build: pose_graph_g2o_SE3

.PHONY : CMakeFiles/pose_graph_g2o_SE3.dir/build

CMakeFiles/pose_graph_g2o_SE3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pose_graph_g2o_SE3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pose_graph_g2o_SE3.dir/clean

CMakeFiles/pose_graph_g2o_SE3.dir/depend:
	cd /home/parallels/Desktop/visual_slam_lecture/GraphOptimization/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parallels/Desktop/visual_slam_lecture/GraphOptimization /home/parallels/Desktop/visual_slam_lecture/GraphOptimization /home/parallels/Desktop/visual_slam_lecture/GraphOptimization/build /home/parallels/Desktop/visual_slam_lecture/GraphOptimization/build /home/parallels/Desktop/visual_slam_lecture/GraphOptimization/build/CMakeFiles/pose_graph_g2o_SE3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pose_graph_g2o_SE3.dir/depend

