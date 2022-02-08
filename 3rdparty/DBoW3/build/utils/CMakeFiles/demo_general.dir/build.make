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
CMAKE_SOURCE_DIR = /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build

# Include any dependencies generated for this target.
include utils/CMakeFiles/demo_general.dir/depend.make

# Include the progress variables for this target.
include utils/CMakeFiles/demo_general.dir/progress.make

# Include the compile flags for this target's objects.
include utils/CMakeFiles/demo_general.dir/flags.make

utils/CMakeFiles/demo_general.dir/demo_general.cpp.o: utils/CMakeFiles/demo_general.dir/flags.make
utils/CMakeFiles/demo_general.dir/demo_general.cpp.o: ../utils/demo_general.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object utils/CMakeFiles/demo_general.dir/demo_general.cpp.o"
	cd /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/utils && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo_general.dir/demo_general.cpp.o -c /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/utils/demo_general.cpp

utils/CMakeFiles/demo_general.dir/demo_general.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_general.dir/demo_general.cpp.i"
	cd /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/utils/demo_general.cpp > CMakeFiles/demo_general.dir/demo_general.cpp.i

utils/CMakeFiles/demo_general.dir/demo_general.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_general.dir/demo_general.cpp.s"
	cd /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/utils && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/utils/demo_general.cpp -o CMakeFiles/demo_general.dir/demo_general.cpp.s

# Object files for target demo_general
demo_general_OBJECTS = \
"CMakeFiles/demo_general.dir/demo_general.cpp.o"

# External object files for target demo_general
demo_general_EXTERNAL_OBJECTS =

utils/demo_general: utils/CMakeFiles/demo_general.dir/demo_general.cpp.o
utils/demo_general: utils/CMakeFiles/demo_general.dir/build.make
utils/demo_general: src/libDBoW3.so.0.0.1
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_viz.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.2.0
utils/demo_general: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.2.0
utils/demo_general: utils/CMakeFiles/demo_general.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_general"
	cd /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/utils && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_general.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
utils/CMakeFiles/demo_general.dir/build: utils/demo_general

.PHONY : utils/CMakeFiles/demo_general.dir/build

utils/CMakeFiles/demo_general.dir/clean:
	cd /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/utils && $(CMAKE_COMMAND) -P CMakeFiles/demo_general.dir/cmake_clean.cmake
.PHONY : utils/CMakeFiles/demo_general.dir/clean

utils/CMakeFiles/demo_general.dir/depend:
	cd /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3 /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/utils /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/utils /home/parallels/Desktop/visual_slam_lecture/3rdparty/DBoW3/build/utils/CMakeFiles/demo_general.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : utils/CMakeFiles/demo_general.dir/depend

