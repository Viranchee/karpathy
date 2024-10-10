// Making micrograd in C++ using operator overloading

#include <iostream>
#include <valarray>
#include <vector>
using namespace std;

class Value {
  typedef vector<Value> Children;
  typedef char Op;

public:
  float data;
  Children children;
  Op op;

  Value(float data, Children childs, Op operation)
      : data(data), children(childs), op(operation) {}
  ~Value() {}

  Value operator+(const Value &other) {
    return Value(data + other.data, children, '+');
  }

  // Value operator-(const Value &other) { return Value(data - other.data); }

  // Value operator*(const Value &other) { return Value(data * other.data); }

  // Value operator/(const Value &other) { return Value(data / other.data); }

  // Value operator-() { return Value(-data); }

  friend std::ostream &operator<<(std::ostream &os, const Value &value) {
    os << value.data;
    return os;
  }

  friend std::istream &operator>>(std::istream &is, Value &value) {
    is >> value.data;
    return is;
  }
};

int main() {
  std::cout << "Hello, World!" << std::endl;
  return 0;
}