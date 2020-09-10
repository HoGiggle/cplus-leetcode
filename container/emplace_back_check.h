//
// Created by hogiggle on 2020/8/26.
//

#ifndef HELLO_EMPLACE_BACK_CHECK_H
#define HELLO_EMPLACE_BACK_CHECK_H

#include <utility>
#include <vector>
#include <string>
#include <iostream>
using namespace std;
class Student {
public:
  Student() {
    _name = "";
    _age = 0;
  }

  Student(std::string name, int age) {
    _name = std::move(name);
    _age = age;
    cout << "Constructor is called" << endl;
  }

  Student(const Student& stu) {
    _name = stu._name;
    _age = stu._age;
    cout << "Copy constructor is called" << endl;
  }

  Student(Student&& stu) {
    _name = std::move(stu._name);
    _age = stu._age;
    cout << "Moving constructor is called" << endl;
  }

  std::string get_name() const {
    return _name;
  }

  int get_age() const {
    return _age;
  }

  void check();
//  Student& operator=(const Student& stu);

private:
  std::string _name;
  int _age;
};


#endif //HELLO_EMPLACE_BACK_CHECK_H
