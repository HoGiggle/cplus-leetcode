//
// Created by hogiggle on 2020/8/26.
//

#include "rvalue_check.h"
#include <vector>
void RString::check() {
  cout << endl;
  RString str;
  str = RString("hello");
  cout << endl;
  std::vector<RString> vec{};
  vec.push_back(RString("world"));
  cout << endl;

  cout << &str << ", " << &vec[0] << endl;
}