//
// Created by hogiggle on 2020/9/21.
//
#include <iostream>
#include <ctime>
#include <vector>
#include "vector_vector_emplace_check.h"

using namespace std;

void Vector_Vector_Check::check() {
  vector<vector<int>> v1{};
  vector<vector<int>> v2{};
  vector<vector<int>> v3{};
  vector<vector<int>> v4{};

  int n = 10000000;
  clock_t start, end;
  start = clock();
  for (int i = 0; i < n; i++) {
    v1.push_back(vector<int>(100, 0));
  }
  end = clock();
  std::cout << "push_back copy: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

  start = clock();
  for (int i = 0; i < n; i++) {
    v2.push_back(std::move(vector<int>(100, 0)));
  }
  end = clock();
  cout << "push_back move: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

  // emplace_back
  start = clock();
  for (int i = 0; i < n; i++) {
    v3.emplace_back(vector<int>(100, 0));
  }
  end = clock();
  cout << "emplace_back copy: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

  start = clock();
  for (int i = 0; i < n; i++) {
    v4.emplace_back(std::move(vector<int>(100, 0)));
  }
  end = clock();
  cout << "emplace_back move: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
}