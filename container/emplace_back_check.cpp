//
// Created by hogiggle on 2020/8/26.
//

#include "emplace_back_check.h"

void Student::check() {
  std::vector<Student> cls;
  cout << "emplace_back:" << endl;
  cls.emplace_back("lydia", 17);

  cout << endl;
  std::vector<Student> cls2;
  cout << "push_back:" << endl;
  cls2.push_back(Student("giggle", 18));
  cls2.emplace_back(Student("giggle", 18));

  cout << endl << "details:" << endl;
  for (const Student& stu : cls) {
    cout << stu.get_name() << ", " << stu.get_age() << endl;
  }
  for (const Student& stu : cls2) {
    cout << stu.get_name() << ", " << stu.get_age() << endl;
  }
}

