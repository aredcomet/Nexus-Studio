#pragma once
#include <string>
#include <tuple>

// Returns tuple of (display_str, exact_answer_str, problem_type)
std::tuple<std::string, std::string, std::string> generate_math_problem();
