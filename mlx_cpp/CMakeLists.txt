cmake_minimum_required(VERSION 3.24)
project(mlx_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Explicitly set PostgreSQL paths (postgresql@18)
set(PostgreSQL_INCLUDE_DIRS "/opt/homebrew/include/postgresql@18")
set(PostgreSQL_LIBRARIES "/opt/homebrew/lib/postgresql@18/libpq.dylib")

# Paths for MLX
set(MLX_DIR "/Users/bran/src/play/llm/.venv/lib/python3.14/site-packages/mlx")
set(MLX_INCLUDE_DIR "${MLX_DIR}/include")
set(MLX_LIBRARY "${MLX_DIR}/lib/libmlx.dylib")

include_directories(
    ${MLX_INCLUDE_DIR}
    ${PostgreSQL_INCLUDE_DIRS}
)

# Set runtime search path for libmlx.dylib
set(CMAKE_INSTALL_RPATH "${MLX_DIR}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_Declare(
    indicators
    GIT_REPOSITORY https://github.com/p-ranav/indicators.git
    GIT_TAG v2.3
)
FetchContent_Declare(
    ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG v5.0.0
)
FetchContent_MakeAvailable(json indicators ftxui)

add_executable(mlx_cpp_runner
    main.cpp
    tokenizer.cpp
    model.cpp
    batch_generator.cpp
    grpo_dataset.cpp
)

target_link_libraries(mlx_cpp_runner
    PRIVATE
    ${MLX_LIBRARY}
    ${PostgreSQL_LIBRARIES}
    nlohmann_json::nlohmann_json
    indicators::indicators
    ftxui::screen
    ftxui::dom
    ftxui::component
)


