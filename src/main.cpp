#include "parsers/parse_scene.h"
#include "image.h"
#include "render.h"
#include "timer.h"
#include <memory>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cout << "[Usage] ./lajolla_gpu [-o output_file_name] filename.xml" << std::endl;
        return 0;
    }

    std::string outputfile = "";
    std::vector<std::string> filenames;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-o") {
            outputfile = std::string(argv[++i]);
        } else {
            filenames.push_back(std::string(argv[i]));
        }
    }

    for (const std::string &filename : filenames) {
        Timer timer;
        tick(timer);
        std::cout << "Parsing and constructing scene " << filename << "." << std::endl;
        std::unique_ptr<parser::Scene> parsed_scene = parser::parse_scene(filename);
        const Scene scene(*parsed_scene);
        std::cout << "Done. Took " << tick(timer) << " seconds." << std::endl;
        std::cout << "Rendering..." << std::endl;
        Image3 img = render(scene);
        if (outputfile.compare("") == 0) {outputfile = parsed_scene->output_filename;}
        std::cout << "Done. Took " << tick(timer) << " seconds." << std::endl;
        imwrite(outputfile, img);
        std::cout << "Image written to " << outputfile << std::endl;
    }

    return 0;
}

