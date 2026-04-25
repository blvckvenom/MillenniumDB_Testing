// Enhanced tool to inspect and query projection contents
// Usage: projection_inspect_enhanced <db_folder> <projection_name> [options]

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>

#include "graph_models/gql/projection/projection_manager.h"
#include "graph_models/gql/projection/projection_storage.h"
#include "graph_models/gql/conversions.h"
#include "graph_models/object_id.h"
#include "query/query_context.h"
#include "system/system.h"

enum class OutputFormat {
    TABLE,
    CSV,
    JSON
};

enum class DisplayMode {
    ALL,
    NODES_ONLY,
    EDGES_ONLY,
    SUMMARY
};

struct Options {
    OutputFormat format = OutputFormat::TABLE;
    DisplayMode mode = DisplayMode::ALL;
    size_t limit = 100;
    bool show_properties = false;
    bool show_hex = true;
    std::string output_file;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <db_folder> <projection_name> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --format <table|csv|json>   Output format (default: table)\n";
    std::cout << "  --nodes                     Show only nodes\n";
    std::cout << "  --edges                     Show only edges\n";
    std::cout << "  --summary                   Show only summary statistics\n";
    std::cout << "  --limit N                   Limit output to N items (default: 100)\n";
    std::cout << "  --all                       Show all items (no limit)\n";
    std::cout << "  --no-hex                    Show decimal IDs instead of hexadecimal\n";
    std::cout << "  --output FILE               Write output to file instead of stdout\n";
    std::cout << "  --help                      Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " data/dbs/gql/posts my_projection\n";
    std::cout << "  " << program_name << " data/dbs/gql/posts my_projection --nodes --format csv\n";
    std::cout << "  " << program_name << " data/dbs/gql/posts my_projection --edges --limit 50\n";
    std::cout << "  " << program_name << " data/dbs/gql/posts my_projection --format json --output result.json\n";
}

Options parse_options(int argc, char* argv[]) {
    Options opts;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--format" && i + 1 < argc) {
            std::string format = argv[++i];
            if (format == "table") {
                opts.format = OutputFormat::TABLE;
            } else if (format == "csv") {
                opts.format = OutputFormat::CSV;
            } else if (format == "json") {
                opts.format = OutputFormat::JSON;
            } else {
                throw std::runtime_error("Invalid format: " + format);
            }
        } else if (arg == "--nodes") {
            opts.mode = DisplayMode::NODES_ONLY;
        } else if (arg == "--edges") {
            opts.mode = DisplayMode::EDGES_ONLY;
        } else if (arg == "--summary") {
            opts.mode = DisplayMode::SUMMARY;
        } else if (arg == "--limit" && i + 1 < argc) {
            opts.limit = std::stoull(argv[++i]);
        } else if (arg == "--all") {
            opts.limit = SIZE_MAX;
        } else if (arg == "--no-hex") {
            opts.show_hex = false;
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output_file = argv[++i];
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    return opts;
}

std::string format_id(uint64_t id, bool hex) {
    std::stringstream ss;
    if (hex) {
        ss << "0x" << std::hex << id;
    } else {
        ss << id;
    }
    return ss.str();
}

void print_nodes_table(const std::vector<ObjectId>& nodes, size_t limit, bool hex, std::ostream& out) {
    out << "\n";
    out << "Nodes (" << nodes.size() << " total):\n";
    out << std::string(60, '-') << "\n";
    out << std::setw(8) << "Index" << " | "
        << std::setw(20) << "Node ID" << " | "
        << "Type\n";
    out << std::string(60, '-') << "\n";

    size_t count = std::min(nodes.size(), limit);
    for (size_t i = 0; i < count; i++) {
        out << std::setw(8) << (i + 1) << " | "
            << std::setw(20) << format_id(nodes[i].id, hex) << " | ";

        auto type = GQL_OID::get_type(nodes[i]);
        if (type == GQL_OID::Type::NODE) {
            out << "NODE";
        } else {
            out << "UNKNOWN";
        }
        out << "\n";
    }

    if (nodes.size() > limit) {
        out << "... (" << (nodes.size() - limit) << " more)\n";
    }
    out << std::string(60, '-') << "\n";
}

void print_edges_table(
    const std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>>& edges,
    size_t limit,
    bool hex,
    std::ostream& out
) {
    out << "\n";
    out << "Edges (" << edges.size() << " total):\n";
    out << std::string(90, '-') << "\n";
    out << std::setw(8) << "Index" << " | "
        << std::setw(20) << "From Node" << " | "
        << std::setw(20) << "To Node" << " | "
        << std::setw(20) << "Edge ID" << " | "
        << "Direction\n";
    out << std::string(90, '-') << "\n";

    size_t count = std::min(edges.size(), limit);
    for (size_t i = 0; i < count; i++) {
        auto [from, to, edge_id, is_directed] = edges[i];
        out << std::setw(8) << (i + 1) << " | "
            << std::setw(20) << format_id(from.id, hex) << " | "
            << std::setw(20) << format_id(to.id, hex) << " | "
            << std::setw(20) << format_id(edge_id.id, hex) << " | "
            << (is_directed ? "DIRECTED  " : "UNDIRECTED")
            << "\n";
    }

    if (edges.size() > limit) {
        out << "... (" << (edges.size() - limit) << " more)\n";
    }
    out << std::string(90, '-') << "\n";
}

void print_nodes_csv(const std::vector<ObjectId>& nodes, size_t limit, bool hex, std::ostream& out) {
    out << "index,node_id,type\n";

    size_t count = std::min(nodes.size(), limit);
    for (size_t i = 0; i < count; i++) {
        out << (i + 1) << ","
            << format_id(nodes[i].id, hex) << ",";

        auto type = GQL_OID::get_type(nodes[i]);
        if (type == GQL_OID::Type::NODE) {
            out << "NODE";
        } else {
            out << "UNKNOWN";
        }
        out << "\n";
    }
}

void print_edges_csv(
    const std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>>& edges,
    size_t limit,
    bool hex,
    std::ostream& out
) {
    out << "index,from_node,to_node,edge_id,direction\n";

    size_t count = std::min(edges.size(), limit);
    for (size_t i = 0; i < count; i++) {
        auto [from, to, edge_id, is_directed] = edges[i];
        out << (i + 1) << ","
            << format_id(from.id, hex) << ","
            << format_id(to.id, hex) << ","
            << format_id(edge_id.id, hex) << ","
            << (is_directed ? "directed" : "undirected")
            << "\n";
    }
}

void print_nodes_json(const std::vector<ObjectId>& nodes, size_t limit, bool hex, std::ostream& out) {
    out << "{\n";
    out << "  \"type\": \"nodes\",\n";
    out << "  \"total\": " << nodes.size() << ",\n";
    out << "  \"limit\": " << std::min(nodes.size(), limit) << ",\n";
    out << "  \"nodes\": [\n";

    size_t count = std::min(nodes.size(), limit);
    for (size_t i = 0; i < count; i++) {
        out << "    {\n";
        out << "      \"index\": " << (i + 1) << ",\n";
        out << "      \"node_id\": \"" << format_id(nodes[i].id, hex) << "\",\n";

        auto type = GQL_OID::get_type(nodes[i]);
        out << "      \"type\": \"" << (type == GQL_OID::Type::NODE ? "NODE" : "UNKNOWN") << "\"\n";

        out << "    }";
        if (i + 1 < count) {
            out << ",";
        }
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
}

void print_edges_json(
    const std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>>& edges,
    size_t limit,
    bool hex,
    std::ostream& out
) {
    out << "{\n";
    out << "  \"type\": \"edges\",\n";
    out << "  \"total\": " << edges.size() << ",\n";
    out << "  \"limit\": " << std::min(edges.size(), limit) << ",\n";
    out << "  \"edges\": [\n";

    size_t count = std::min(edges.size(), limit);
    for (size_t i = 0; i < count; i++) {
        auto [from, to, edge_id, is_directed] = edges[i];

        out << "    {\n";
        out << "      \"index\": " << (i + 1) << ",\n";
        out << "      \"from_node\": \"" << format_id(from.id, hex) << "\",\n";
        out << "      \"to_node\": \"" << format_id(to.id, hex) << "\",\n";
        out << "      \"edge_id\": \"" << format_id(edge_id.id, hex) << "\",\n";
        out << "      \"direction\": \"" << (is_directed ? "directed" : "undirected") << "\"\n";
        out << "    }";

        if (i + 1 < count) {
            out << ",";
        }
        out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
}

void print_summary(
    const std::string& projection_name,
    const std::vector<ObjectId>& nodes,
    const std::vector<std::tuple<ObjectId, ObjectId, ObjectId, bool>>& edges,
    std::ostream& out
) {
    // Count directed vs undirected edges
    size_t directed_count = 0;
    size_t undirected_count = 0;

    for (const auto& [from, to, edge_id, is_directed] : edges) {
        if (is_directed) {
            directed_count++;
        } else {
            undirected_count++;
        }
    }

    out << "\n";
    out << "==========================================\n";
    out << "Projection Summary: " << projection_name << "\n";
    out << "==========================================\n";
    out << "  Total Nodes:        " << nodes.size() << "\n";
    out << "  Total Edges:        " << edges.size() << "\n";
    out << "    Directed:         " << directed_count << "\n";
    out << "    Undirected:       " << undirected_count << "\n";
    out << "==========================================\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string db_folder = argv[1];
    std::string projection_name = argv[2];

    try {
        Options opts = parse_options(argc, argv);

        // Initialize system
        System system(
            db_folder,
            256 * 1024 * 1024,   // str_static_size
            256 * 1024 * 1024,   // str_dynamic_size
            1024 * 1024 * 1024,  // shared_buffer_size
            256 * 1024 * 1024,   // private_buffer_size
            64 * 1024 * 1024,    // tensor_static_size
            64 * 1024 * 1024,    // tensor_dynamic_size
            1                    // workers
        );

        // Set up QueryContext
        QueryContext query_ctx;
        QueryContext::set_query_ctx(&query_ctx);

        // Get projection directory
        auto& proj_manager = GQL::ProjectionManager::get_instance();
        proj_manager.init(db_folder);

        std::string proj_dir = db_folder + "/projections/" + projection_name;

        // Read projection data
        GQL::ProjectionStorage storage(proj_dir, db_folder);
        storage.open();

        auto nodes = storage.get_all_node_ids();
        auto edges = storage.get_all_edges_info();

        // Prepare output stream
        std::ostream* out = &std::cout;
        std::ofstream file_out;

        if (!opts.output_file.empty()) {
            file_out.open(opts.output_file);
            if (!file_out.is_open()) {
                throw std::runtime_error("Cannot open output file: " + opts.output_file);
            }
            out = &file_out;
        }

        // Display based on mode and format
        if (opts.mode == DisplayMode::SUMMARY) {
            print_summary(projection_name, nodes, edges, *out);
        } else {
            bool show_nodes = (opts.mode == DisplayMode::ALL || opts.mode == DisplayMode::NODES_ONLY);
            bool show_edges = (opts.mode == DisplayMode::ALL || opts.mode == DisplayMode::EDGES_ONLY);

            switch (opts.format) {
                case OutputFormat::TABLE:
                    if (show_nodes) {
                        print_nodes_table(nodes, opts.limit, opts.show_hex, *out);
                    }
                    if (show_edges) {
                        print_edges_table(edges, opts.limit, opts.show_hex, *out);
                    }
                    if (opts.mode == DisplayMode::ALL) {
                        print_summary(projection_name, nodes, edges, *out);
                    }
                    break;

                case OutputFormat::CSV:
                    if (show_nodes) {
                        print_nodes_csv(nodes, opts.limit, opts.show_hex, *out);
                    }
                    if (show_edges) {
                        print_edges_csv(edges, opts.limit, opts.show_hex, *out);
                    }
                    break;

                case OutputFormat::JSON:
                    if (opts.mode == DisplayMode::ALL) {
                        // Combined JSON output
                        *out << "{\n";
                        *out << "  \"projection\": \"" << projection_name << "\",\n";
                        *out << "  \"node_count\": " << nodes.size() << ",\n";
                        *out << "  \"edge_count\": " << edges.size() << ",\n";
                        *out << "  \"nodes\": ";
                        std::stringstream nodes_json;
                        print_nodes_json(nodes, opts.limit, opts.show_hex, nodes_json);
                        *out << nodes_json.str();
                        *out << ",\n  \"edges\": ";
                        std::stringstream edges_json;
                        print_edges_json(edges, opts.limit, opts.show_hex, edges_json);
                        *out << edges_json.str();
                        *out << "\n}\n";
                    } else if (show_nodes) {
                        print_nodes_json(nodes, opts.limit, opts.show_hex, *out);
                    } else if (show_edges) {
                        print_edges_json(edges, opts.limit, opts.show_hex, *out);
                    }
                    break;
            }
        }

        if (file_out.is_open()) {
            file_out.close();
            std::cout << "Output written to: " << opts.output_file << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
