//
// Created by hogiggle on 2020/9/24.
//

#ifndef CPLUS_LEETCODE_RANDOM_ENGINE_H
#define CPLUS_LEETCODE_RANDOM_ENGINE_H

#include <random>
#include <iostream>
using namespace std;

class Random {
  void check() {
    std::random_device rd;
    std::mt19937 mt(rd()); //engine
    for (int i = 0; i < 10; i++) {
      std::cout << mt() << endl;
    }

    std::default_random_engine e; //engine
    std::uniform_int_distribution<int> int_u(0, 9);
    std::uniform_real_distribution<double> double_u(0.0, 1.0);
    for (int i = 0; i < 10; i++) {
      cout << int_u(e) << endl;
      cout << double_u(e) << endl;
    }
  }
};

#endif //CPLUS_LEETCODE_RANDOM_ENGINE_H
