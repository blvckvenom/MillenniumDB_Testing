#pragma once

#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <string>

/**
 * @file ablation_registry.h
 * @brief One place that resolves every environment switch, and says so.
 *
 * WHY THIS EXISTS
 *
 * The build has 60+ environment switches that select between two code paths.
 * That makes an ablation study possible in principle: flip one, run, compare.
 * In practice it was not, for a reason that only shows up when you try to
 * verify an arm rather than run it.
 *
 * Nothing in the log said which path a run had taken. An A/B therefore rested
 * on the operator having exported the right variable, with no record that the
 * process agreed. Two concrete ways that bites:
 *
 *   1. A typo passes silently. `MDB_PROJECTION_NODE_MEMBERSHIP` only checks
 *      whether the value differs from "binary", so `bitmpa` selects the bitmap
 *      path and looks exactly like a correct run.
 *   2. A switch behind conditional compilation reads the variable and changes
 *      nothing. Seven of them sit behind CUDA guards. Flipping one in a build
 *      without CUDA produces two arms that are the same code, and whatever
 *      difference the clock shows is noise reported as a result.
 *
 * The cure is not more discipline at the call site. It is that resolving a
 * switch and declaring it are THE SAME ACT, so a path cannot be taken without
 * a line saying it was.
 *
 * WHAT IT GUARANTEES
 *
 *   - Resolved ONCE per name and memoised. A switch cannot change meaning
 *     halfway through a run, which is the failure mode that makes a long
 *     measurement uninterpretable after the fact.
 *   - Emits `[ABLATION] <name>=<resolved> source=env|default` on first use, so
 *     the log carries the value the CODE settled on and not the one the
 *     operator meant. When the raw string differs from the resolved value the
 *     line carries it too, which is what makes the typo above visible.
 *   - `active()` marks a switch that the build cannot honour, so a knob that
 *     is inert in this binary says so instead of looking effective.
 *
 * USAGE
 *
 *     static const bool parallel = Ablation::flag("MDB_PROJECTION_PARALLEL_SCAN", true);
 *     static const long workers  = Ablation::number("MDB_GNN_L4_WORKERS", 4);
 *     static const auto backend  = Ablation::choice("MDB_PROJECTION_SORTER", "classic",
 *                                                   {"classic", "radix"});
 *
 * `choice` takes the accepted values. An unrecognised one falls back to the
 * default AND says so, instead of being accepted by not matching the other
 * branch, which is how the typo above went unnoticed.
 */
namespace Ablation {

/// Boolean switch. Present-and-not-"0"/"false"/"no" reads as true.
bool flag(const char* name, bool fallback);

/// Numeric knob. A value that does not parse falls back and is reported.
long number(const char* name, long fallback);

/// Multiple choice. `accepted` is a null-terminated list of valid values; an
/// unrecognised value falls back to `fallback` and is reported as rejected.
std::string choice(const char* name, const char* fallback,
                   std::initializer_list<const char*> accepted);

/// Free-form string (paths and the like). No validation, still declared.
std::string text(const char* name, const char* fallback);

/**
 * @brief Declares that this binary cannot honour @p name.
 *
 * Call from the #else side of a conditional-compilation guard. The switch then
 * reports `honoured=no`, which is the difference between "the arm ran the other
 * path" and "the arm ran the same path twice".
 */
void inert(const char* name, const char* reason);

/// Every switch resolved so far, one `[ABLATION]` line each, in resolve order.
void report(std::ostream& os);

/// Number of switches resolved so far. For tests.
std::size_t resolved_count();

}  // namespace Ablation
