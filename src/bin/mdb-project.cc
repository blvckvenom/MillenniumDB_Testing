#include "project/project.h"
#include "third_party/cli11/CLI11.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string name;
    std::string file;
    CLI::App app{"Graph projection"};
    app.add_option("name", name)->required();
    app.add_option("file", file)->required();
    CLI11_PARSE(app, argc, argv);
    try {
        auto res = project(name, file);
        std::cout << res;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
