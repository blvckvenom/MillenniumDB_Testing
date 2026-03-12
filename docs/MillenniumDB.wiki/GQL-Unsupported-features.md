# Unsupported features

These features are not supported. The list follows the same order as in the GQL Standard.

* Session management
* Transaction management
* Variable definitions
* Object expressions
* Catalog-modifying statements
* Data-modifying statements
* Query statements
  * Composite query expressions
    * We do not support query conjunction or set operators `UNION`, `EXCEPT`, `INTERSECT`.
  * Group by clause
  * Select statement
* Common elements
  * At schema
  * Use graph clause
  * Graph pattern binding table
  * Insert graph pattern
* Object references
* Type elements
* Predicates
  * Value type predicate
  * Normalized predicate
  * Exists predicate
  * Source/destination predicate
  * All different predicate
  * Same predicate
  * Property exists predicate
* Value expressions
  * Dynamic parameter specification
  * Let value expression
  * Value query expression
  * Element id function
  * Path value expression
  * Path value constructor
  * List value expression
  * Record constructor
  * Duration value expression
  * Vector value expression
