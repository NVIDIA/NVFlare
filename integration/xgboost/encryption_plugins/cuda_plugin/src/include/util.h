#pragma once
#include <string>
#include <vector>

std::vector<std::pair<int, int>> distribute_work(size_t num_jobs, size_t num_workers);

uint32_t to_int(double d);

double to_double(uint32_t i);

std::string get_string(std::vector<std::pair<std::string_view, std::string_view>> const &args,
  std::string_view const &key,std::string_view default_value = "");

bool get_bool(std::vector<std::pair<std::string_view, std::string_view>> const &args,
    const std::string &key, bool default_value = false);

int get_int(std::vector<std::pair<std::string_view, std::string_view>> const &args,
    const std::string &key, int default_value = 0);
