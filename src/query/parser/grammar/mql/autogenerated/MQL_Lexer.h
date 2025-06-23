
// Generated from MQL_Lexer.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"




class  MQL_Lexer : public antlr4::Lexer {
public:
  enum {
    K_ACYCLIC = 1, K_AND = 2, K_ANGULAR = 3, K_ANY = 4, K_AS = 5, K_AVG = 6, 
    K_ALL = 7, K_ASC = 8, K_BY = 9, K_BOOL = 10, K_COUNT = 11, K_CREATE = 12, 
    K_DELETE = 13, K_DESCRIBE = 14, K_DESC = 15, K_DIMENSIONS = 16, K_DISTINCT = 17, 
    K_EDGE = 18, K_EUCLIDEAN = 19, K_FROM = 20, K_INCOMING = 21, K_INDEX = 22, 
    K_INSERT = 23, K_INTEGER = 24, K_INTO = 25, K_IS = 26, K_FALSE = 27, 
    K_FLOAT = 28, K_GROUP = 29, K_GROUPS = 30, K_IDENTITY = 31, K_LABELS = 32, 
    K_LABEL = 33, K_LIMIT = 34, K_MANHATTAN = 35, K_MATCH = 36, K_MAX = 37, 
    K_MIN = 38, K_NFKD_CASEFOLD = 39, K_NORMALIZE = 40, K_OBJECTS = 41, 
    K_OFFSET = 42, K_ON = 43, K_OPTIONAL = 44, K_ORDER = 45, K_OR = 46, 
    K_OUTGOING = 47, K_PREFIX = 48, K_PROPERTIES = 49, K_PROPERTY = 50, 
    K_NOT = 51, K_NULL = 52, K_SHORTEST = 53, K_SHOW = 54, K_SIMPLE = 55, 
    K_TENSOR_DISTANCE = 56, K_TEXT_SEARCH = 57, K_REGEX = 58, K_RETURN = 59, 
    K_SET = 60, K_SUM = 61, K_STRING = 62, K_PROJECT = 63, K_STORE = 64, 
    K_TENSOR = 65, K_TEXT = 66, K_TRUE = 67, K_TOKENIZE = 68, K_TRAILS = 69, 
    K_VALUES = 70, K_WALKS = 71, K_WITH = 72, K_WHERE = 73, K_WS_KEEP_PUNCT = 74, 
    K_WS_RM_PUNCT = 75, K_WS_SPLIT_PUNCT = 76, TRUE_PROP = 77, FALSE_PROP = 78, 
    ANON_ID = 79, EDGE_ID = 80, KEY = 81, TYPE = 82, TYPE_VAR = 83, VARIABLE = 84, 
    STRING = 85, UNSIGNED_INTEGER = 86, UNSIGNED_FLOAT = 87, UNSIGNED_SCIENTIFIC_NOTATION = 88, 
    NAME = 89, LEQ = 90, GEQ = 91, EQ = 92, NEQ = 93, LT = 94, GT = 95, 
    SINGLE_EQ = 96, PATH_SEQUENCE = 97, PATH_ALTERNATIVE = 98, PATH_NEGATION = 99, 
    STAR = 100, PERCENT = 101, QUESTION_MARK = 102, PLUS = 103, MINUS = 104, 
    L_PAR = 105, R_PAR = 106, LCURLY_BRACKET = 107, RCURLY_BRACKET = 108, 
    LSQUARE_BRACKET = 109, RSQUARE_BRACKET = 110, COMMA = 111, COLON = 112, 
    WHITE_SPACE = 113, SINGLE_LINE_COMMENT = 114, UNRECOGNIZED = 115
  };

  enum {
    WS_CHANNEL = 2
  };

  explicit MQL_Lexer(antlr4::CharStream *input);

  ~MQL_Lexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

