//
// Created by hogiggle on 2020/8/26.
//

#ifndef HELLO_RVALUE_CHECK_H
#define HELLO_RVALUE_CHECK_H

#include <iostream>
#include <cstring>

using namespace std;
class RString {
public:
  RString(){
    _data = nullptr;
    _len = 0;
    cout << "Default constructor is called" << endl;
  }

  RString(const char* p) {
    _len = strlen(p);
    _init_data(p);
    cout << "Constructor is called" << endl;
  }

  RString(const RString& other) {
    _len = other._len;
    _init_data(other._data);
    cout << "Copy constructor is called" << endl;
  }

  RString& operator= (const RString& other) {
    if (this != &other) {
      _len = other._len;
      _init_data(other._data);
    }
    cout << "Assignment operator is called" << endl;
    return *this;
  }

  RString(RString&& other) {
    _len = other._len;
    _data = other._data;
    other._len = 0;
    other._data = nullptr;
    cout << "Move constructor is called" << endl;
  }

  RString& operator= (RString&& other) {
    if (this != &other) {
      _len = other._len;
      _data = other._data;
      other._len = 0;
      other._data = nullptr;
    }
    cout << "Move assignment operator is called" << endl;
    return *this;
  }

  ~RString() {
    cout << "Destructor is called" << endl;
    delete []_data;
  }

  std::size_t size() {
    return _len;
  }

  void check();
private:
  char* _data;
  int _len;
  void _init_data(const char* src) {
    _data = new char[_len + 1];
    memcpy(_data, src, _len);
    _data[_len] = '\0';
  }
};


#endif //HELLO_RVALUE_CHECK_H
