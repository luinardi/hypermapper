#ifndef HPVM_HYPERMAPPER_H
#define HPVM_HYPERMAPPER_H
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

class HMInputParam;
void fatalError(const std::string &msg);

enum ParamType { Real, Integer, Ordinal, Categorical };

std::ostream &operator<<(std::ostream &out, const ParamType &PT) {
  switch (PT) {
  case Real:
    out << "Real";
    break;
  case Integer:
    out << "Integer";
    break;
  case Ordinal:
    out << "Ordinal";
    break;
  case Categorical:
    out << "Categorical";
    break;
  }
  return out;
}

std::string getTypeAsString(const ParamType &PT) {
  std::string TypeString;
  switch (PT) {
  case Real:
    TypeString = "real";
    break;
  case Integer:
    TypeString = "integer";
    break;
  case Ordinal:
    TypeString = "ordinal";
    break;
  case Categorical:
    TypeString = "categorical";
    break;
  }

  return TypeString;
}

class HMInputParam {
private:
  std::string Name;
  std::string const Key;
  ParamType Type;
  int startVal;
  int endVal;
  std::vector<int> Range;
  static int count;
  int Value;

public:
  HMInputParam(std::string _Name = "", ParamType _Type = ParamType::Integer)
      : Name(_Name), Type(_Type), startVal(0), endVal(0),
        Key("x" + std::to_string(count++)) {}

  std::string getName() const { return Name; }
  void setName(std::string _Name) { Name = _Name; }

  ParamType getType() const { return Type; }
  void setType(ParamType _Type) { Type = _Type; }

  void setStartVal(int _Val) { startVal = _Val; }
  int getStartVal() const { return startVal; }

  void setEndVal(int _Val) { endVal = _Val; }
  int getEndVal() const { return endVal; }

  void setRange(std::vector<int> const &_Range) { Range = _Range; }
  std::vector<int> getRange() const { return Range; }

  std::string getKey() const { return Key; }

  int getVal() const {return Value;}
  void setVal(int _Value) {Value = _Value;}

  bool operator==(const std::string &_Key) {
    if (Key == _Key) {
      return true;
    } else {
      return false;
    }
  }

  bool operator==(const HMInputParam &IP) {
    if (Key == IP.getKey()) {
      return true;
    } else {
      return false;
    }
  }
  friend std::ostream &operator<<(std::ostream &out, const HMInputParam &IP) {
    out << IP.getKey() << ":";
    out << "\n  Name: " << IP.Name;
    out << "\n  Type: " << IP.Type;
    if (IP.getType() == ParamType::Ordinal ||
        IP.getType() == ParamType::Categorical) {
      out << "\n  Range: {";
      char separator[1] = "";
      for (auto i : IP.getRange()) {
        out << separator << i;
        separator[0] = ',';
      }
      out << "}";
    } else if (IP.getType() == ParamType::Integer) {
      out << "\n  Range: [" << IP.getStartVal() << ", " << IP.getEndVal()
          << "]";
    }
    return out;
  }
};

struct HMObjective {
  int f1_value;
  int f2_value;
  bool valid;
};

#endif
