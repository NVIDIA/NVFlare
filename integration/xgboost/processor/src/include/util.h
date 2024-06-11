#pragma once
#include <string>
#include <vector>
#include <map>

std::vector<std::pair<int, int>> distribute_work(size_t num_jobs, size_t num_workers);

uint32_t to_int(double d);

double to_double(uint32_t i);

std::string get_string(const std::map<std::string, std::string>& params, const std::string key,
                       std::string default_value);

bool get_bool(const std::map<std::string, std::string>& params, const std::string key,
              bool default_value=false);

int get_int(const std::map<std::string, std::string>& params, const std::string key,
            int default_value=0);

