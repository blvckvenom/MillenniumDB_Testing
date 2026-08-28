# Project build

MillenniumDB should be able to be built on any x86-64 Linux distribution.
On windows, Windows Subsystem for Linux (WSL) can be used. MacOS is supported if using a Mac chip. For Mac with Intel chips or Windows without WSL, Docker can be used: see [[Docker]].

## System Requirements

- x86-64 Linux distribution (Windows via WSL, MacOS on Apple Silicon supported)
- GCC >= 8.1
- CMake >= 3.12
- Git
- libssl
- ncursesw and less (for CLI)
- Python >= 3.8 with venv (for tests)
- ICU library (libicu-dev)
- Boost 1.82 (must be manually installed - see below)

## Install Dependencies

MillenniumDB needs the following dependencies:

- GCC >= 8.1
- CMake >= 3.12
- Git
- libssl
- ncursesw and less for the CLI
- Python >= 3.8 with venv to run tests
- Boost 1.82 (see the instructions below to install boost for MillenniumDB)

On current Debian and Ubuntu based distributions they can be installed by running:

```bash
sudo apt update && sudo apt install git g++ cmake libssl-dev libncurses-dev less python3 python3-venv libicu-dev
```

On mac:

```bash
brew install cmake ncurses openssl@3 icu4c
```

## Clone the repository

Clone this repository, enter the repository root directory and set `MDB_HOME`:

```bash
git clone git@github.com:MillenniumDB/MillenniumDB.git
cd MillenniumDB
export MDB_HOME=$(pwd)
```

## Install Boost

Download [`boost_1_82_0.tar.gz`](https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz) using a browser or wget:

```bash
wget -q --show-progress https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz
```

and run the following in the directory where boost was downloaded:

```bash
tar -xf boost_1_82_0.tar.gz
mkdir -p $MDB_HOME/third_party/boost_1_82/include
mv boost_1_82_0/boost $MDB_HOME/third_party/boost_1_82/include
rm -r boost_1_82_0.tar.gz boost_1_82_0
```

## Install Documentation Tools (Optional)

For API documentation generation with class diagrams:

```bash
# Ubuntu/Debian
sudo apt install doxygen graphviz

# MacOS
brew install doxygen graphviz

# Generate documentation
doxygen Doxyfile

# View in browser
xdg-open docs/api/html/index.html  # Linux native
open docs/api/html/index.html       # MacOS

# WSL (Windows Subsystem for Linux)
"/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" "$(wslpath -w docs/api/html/index.html)"
```

## Build the Project

Go back into the repository root directory and configure and build MillenniumDB:

```bash
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release && cmake --build build/Release/
```

To use multiple cores during compilation (much faster) use the following command and replace `<n>` with the desired number of threads:

```bash
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release && cmake --build build/Release/ -j <n>
```

If the compilation is successful, then you should be able to run this command to show the help.

```bash
build/Release/bin/mdb help
```

## Build Types

### Release Build (Recommended for Normal Use)

```bash
cmake -B build/Release -D CMAKE_BUILD_TYPE=Release
cmake --build build/Release -j <n>
```

### Debug Build (With Sanitizers)

Includes AddressSanitizer and UndefinedBehaviorSanitizer for development:

```bash
cmake -B build/Debug -D CMAKE_BUILD_TYPE=Debug
cmake --build build/Debug -j <n>
```

### Profile Build (Requires gperftools/tcmalloc)

For performance profiling:

```bash
cmake -B build/Profile -D CMAKE_BUILD_TYPE=Release -D PROFILE=ON
cmake --build build/Profile -j <n>
```
