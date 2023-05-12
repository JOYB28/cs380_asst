Introduction to Computer Graphics (Spring 2023) - Assignment 5
=====
Instructor: Minhyuk Sung (mhsung@kaist.ac.kr)

Last Update: May. q, 2023 by Eunji Hong (hong969@kaist.ac.kr)

## Install Libraries
In the programming assignments of this course, **GLUT** and **OpenGL** libraries are required to compile code. Please follow one of the installation instructions below according to your OS.

### Windows

1. Follow the instruction below to install VS Code and MinGW and to create "tasks.json".

	https://code.visualstudio.com/docs/cpp/config-mingw

2.  Install GLEW and FreeGLUT using pacman in MinGW

	https://packages.msys2.org/package/mingw-w64-x86_64-glew?repo=mingw64

	https://packages.msys2.org/package/mingw-w64-x86_64-freeglut?repo=mingw64

	pacman -S mingw-w64-x86_64-glew mingw-w64-x86_64-freeglut

3. Edit “args” in the “tasks.json” created in 1. as follows:

```
"args": [
        "-fdiagnostics-color=always",
        "-g",
        "${fileDirname}\\**.cpp",
        "-lopengl32",
        "-lglu32",
        "-lglew32",
        "-lfreeglut",
        "-o",
        "${fileDirname}\\${fileBasenameNoExtension}.exe"
      ],
```

### MacOS

1. GLUT should be already installed in Mac OS.
2. Download GLEW .tgz package [here](https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz/). Unzip the package, go to the directory, compile and install it:

```shell
$ cd ${GLEW_DIR_PATH}
$ make && sudo make install 
```
If you use a non-intel Mac and you have already installed GLEW before through brew, you may need to uninstall it and reinstall GLEW as described above, since your machine may have trouble linking GLEW.

```shell
$ brew uninstall glew
```

## Copy assignment 5 files into assignment 4 directory

The basis of assignment 5 is assignment 4. You have to copy files in assignment 5 into assignment 4 directory. Also, you have to change the filename asstX.cpp to asst5.cpp.


## Compile and execute assignment code

If your OS is LINUX, uncomment the two lines below.

```c++
// # include <tr1/memory>
...
// using namespace std::tr1;
```

Run the following command in an assignment directory.

```shell
$ make
```

After compiling the code, execute the output binary file:

```shell
$ ./asst5
```

The name of the output binary file is stated in the first line of the `Makefile` file.

## Recompile code
Run:

```shell
$ make clean && make
```

