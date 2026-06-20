#include "grpo_dataset.h"
#include <random>
#include <sstream>
#include <stdexcept>

// Thread-local random generator
static std::mt19937& get_rng() {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    return gen;
}

static int randint(int low, int high) {
    std::uniform_int_distribution<> dist(low, high);
    return dist(get_rng());
}

static std::pair<std::string, std::string> generate_mixed_math_problem() {
    int a = randint(2, 12);
    int b = randint(2, 12);
    int c = randint(2, 12);

    int template_idx = randint(3, 6);
    std::string display_str;
    int exact_answer = 0;

    switch (template_idx) {
        case 3:
            display_str = std::to_string(a) + " + " + std::to_string(b) + " × " + std::to_string(c);
            exact_answer = a + b * c;
            break;
        case 4:
            display_str = std::to_string(a) + " × " + std::to_string(b) + " + " + std::to_string(c);
            exact_answer = a * b + c;
            break;
        case 5:
            display_str = std::to_string(a) + " × " + std::to_string(b) + " - " + std::to_string(c);
            exact_answer = a * b - c;
            break;
        case 6:
            display_str = std::to_string(a) + " - " + std::to_string(b) + " + " + std::to_string(c);
            exact_answer = a - b + c;
            break;
        default:
            break;
    }

    return {display_str, std::to_string(exact_answer)};
}

static std::pair<std::string, std::string> generate_bodmas_problem() {
    int a = randint(2, 10);
    int b = randint(2, 12);
    int c = randint(2, 6);
    int d = randint(2, 5);

    int divisor = randint(2, 5);
    int multiplier = randint(2, 6);
    int numerator = divisor * multiplier;

    int template_idx = randint(0, 6);
    std::string display_str;
    int exact_answer = 0;

    switch (template_idx) {
        case 0:
            display_str = std::to_string(a) + " + " + std::to_string(b) + " × " + std::to_string(c);
            exact_answer = a + b * c;
            break;
        case 1:
            display_str = "(" + std::to_string(a) + " + " + std::to_string(b) + ") × (" + std::to_string(c) + " - " + std::to_string(d) + ")";
            exact_answer = (a + b) * (c - d);
            break;
        case 2:
            display_str = std::to_string(numerator) + " ÷ " + std::to_string(divisor) + " + " + std::to_string(a) + " × " + std::to_string(b);
            exact_answer = (numerator / divisor) + a * b;
            break;
        case 3:
            display_str = std::to_string(a) + " × " + std::to_string(b) + " - (" + std::to_string(c) + " + " + std::to_string(d) + ")";
            exact_answer = a * b - (c + d);
            break;
        case 4:
            display_str = std::to_string(a) + " + " + std::to_string(b) + " × (" + std::to_string(numerator) + " ÷ " + std::to_string(divisor) + ") - " + std::to_string(c);
            exact_answer = a + b * (numerator / divisor) - c;
            break;
        case 5:
            display_str = "(" + std::to_string(a) + " × " + std::to_string(b) + ") + (" + std::to_string(numerator) + " ÷ " + std::to_string(divisor) + ")";
            exact_answer = (a * b) + (numerator / divisor);
            break;
        case 6:
            display_str = std::to_string(a) + " × " + std::to_string(b) + " + " + std::to_string(c) + "^2 - " + std::to_string(d);
            exact_answer = a * b + c * c - d;
            break;
        default:
            break;
    }

    return {display_str, std::to_string(exact_answer)};
}

std::tuple<std::string, std::string, std::string> generate_math_problem() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(get_rng()) < 0.5f) {
        auto prob = generate_mixed_math_problem();
        return {prob.first, prob.second, "mixed"};
    } else {
        auto prob = generate_bodmas_problem();
        return {prob.first, prob.second, "bodmas"};
    }
}
