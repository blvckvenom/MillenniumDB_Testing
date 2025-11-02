# SPARQL Implementation Status

## Currently Unsupported SPARQL Features

- Updates other that [INSERT DATA](https://www.w3.org/TR/2013/REC-sparql11-update-20130321/#insertData) and [DELETE DATA](https://www.w3.org/TR/2013/REC-sparql11-update-20130321/#deleteData)
- Named graphs
- The `FROM` clause
- The `GRAPH` keyword
- Regular expression flags other than `i`
- Property paths with Negated Property Sets

## Deviations from the SPARQL Specification

We purposely made the decision not to follow the standard in some details explained below:

- Alternative property paths are not transformed into UNION.
- **Language tag** (`@`) handling is **case sensitive** for `JOIN`s and related operators, but in **expressions** it is **case insensitive**.
- We do **not** store the exact **lexical representation** of numeric datatypes, only the **numeric value**. For example, `"01"^^xsd:integer` and `"1"^^xsd:integer` are identical in MillenniumDB.
  - This implies that expressions that work with the lexical representation may result in a different value. For example `STR(1e0)` should be `"1e0"` according to the standard, but MillenniumDB will evaluate it as `"1.0E0"`.
- We do not differentiate between `"0"^^xsd:boolean` and `false` / `"false"^^xsd:boolean` or between `"1"^^xsd:boolean` and `true` / `"true"^^xsd:boolean`.
- Our implementation uses **ECMAScript** regular expressions, not **Perl** regular expressions.
- The regular path expression `?s :P* ?o` won't return all the nodes in the database that appears as a subject or object in some triple as the standard says. Instead it will only return the nodes that appears as a subject in a triple with predicate `:P`.

### Case Sensitivity of Language Tags

The standard specifies that language tag comparison is case insensitive.
However our implementation for index scans, joins, etc, is case sensitive.
The reason for that is that those operations work at the ObjectId level, comparing
Ids directly for performance reasons. The alternative would be to normalize all
language tags (to lower-case for example), however in that case we would loose
information about the case of language tags.
This means the following strings are different when operating at the ObjectId level:

```text
"asdf"@en
"asdf"@eN
"asdf"@En
"asdf"@EN
```

At the expressions level, however, they are equivalent.

### Representation of Numeric Types

In the SPARQL standard an exact lexical representation of numeric types is used.
In our implementation we use special encodings for known datatypes and do not store
the exact strings. That means the following are identical in our implementation:

```text
"1"^^xsd:integer
"01"^^xsd:integer
"+1"^^xsd:integer
```

```text
"1"^^xsd:double
"01"^^xsd:double
"+1"^^xsd:double
"1.00"^^xsd:double
```

When printing them, applying the function STR(), etc, the original exact representation is lost.
Applying `STR()` to any of those values from the example would yield `"1"` (for integers) or `"1.0E0"` (for doubles).
Also, for operators such as DISTINCT, those are identical values.
This happens with all numeric types.

### Representation of Booleans

In our implementation the following are equivalent:

```text
false
"0"^^xsd:boolean
"false"^^xsd:boolean
```

They are all represented as `"false"^^xsd:boolean.`
The same happens for `true`:

```text
true
"1"^^xsd:boolean
"true"^^xsd:boolean
```

They are all represented as `"true"^^xsd:boolean.`

### Regular Expressions

Our implementation supports `ECMAScript` regular expressions and we only support the `i` flag currently.
The SPARQL standard uses `Perl` regular expressions.
