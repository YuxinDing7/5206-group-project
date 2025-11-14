#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_perf.sh [DATAFILE] [THREADS] [REPEAT]
# Examples:
#   ./scripts/run_perf.sh data_N800_D32_K16.json 8 5
#   ./scripts/run_perf.sh        # uses defaults

DATAFILE=${1:-data_N800_D32_K16.json}
THREADS=${2:-8}
REPEAT=${3:-5}

SRC_BASE="Kmeans_multiThread_base.cpp"
SRC_MULTI="Kmeans_multiThread_multi.cpp"
BIN_BASE="kmeans_base"
BIN_MULTI="kmeans_multi"

CXXFLAGS="-std=c++17 -O3 -march=native -g -fno-omit-frame-pointer -pthread -fopenmp"
BIND_RANGE="0-$((THREADS-1))"

echo "Parameters: DATAFILE=$DATAFILE THREADS=$THREADS REPEAT=$REPEAT BIND=$BIND_RANGE"

# Compile if sources exist
function compile_if_exists() {
  local src=$1
  local bin=$2
  if [[ -f "$src" ]]; then
    echo "Compiling $src -> $bin"
    g++ $CXXFLAGS -o "$bin" "$src"
  else
    echo "Source $src not found, skipping compilation of $bin"
  fi
}

compile_if_exists "$SRC_BASE" "$BIN_BASE"
compile_if_exists "$SRC_MULTI" "$BIN_MULTI"

# Check binaries
if [[ ! -x "./$BIN_BASE" && ! -x "./$BIN_MULTI" ]]; then
  echo "No binaries to run. Ensure at least one of $SRC_BASE or $SRC_MULTI exists and compiled." >&2
  exit 1
fi

export OMP_NUM_THREADS=$THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

TASKSET_CMD="taskset -c $BIND_RANGE"

# Warmup to reduce cold-cache effects
echo "Warmup runs (no perf)"
if [[ -x "./$BIN_BASE" ]]; then
  echo "Warmup base..."
  $TASKSET_CMD ./$BIN_BASE "$DATAFILE" >/dev/null || true
fi
if [[ -x "./$BIN_MULTI" ]]; then
  echo "Warmup multi..."
  $TASKSET_CMD ./$BIN_MULTI "$DATAFILE" >/dev/null || true
fi

# perf stat quick comparison
PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branches,branch-misses"

if [[ -x "./$BIN_BASE" ]]; then
  echo "\n=== perf stat for base (repeats=$REPEAT) ==="
  perf stat -r $REPEAT -e $PERF_EVENTS $TASKSET_CMD ./$BIN_BASE "$DATAFILE"
fi

if [[ -x "./$BIN_MULTI" ]]; then
  echo "\n=== perf stat for multi (repeats=$REPEAT) ==="
  perf stat -r $REPEAT -e $PERF_EVENTS $TASKSET_CMD ./$BIN_MULTI "$DATAFILE"
fi

# Optional: more detailed cache events
DETAILED_EVENTS="L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses"
if [[ -x "./$BIN_BASE" ]]; then
  echo "\n=== Detailed cache events (base) ==="
  perf stat -r $REPEAT -e $DETAILED_EVENTS $TASKSET_CMD ./$BIN_BASE "$DATAFILE"
fi
if [[ -x "./$BIN_MULTI" ]]; then
  echo "\n=== Detailed cache events (multi) ==="
  perf stat -r $REPEAT -e $DETAILED_EVENTS $TASKSET_CMD ./$BIN_MULTI "$DATAFILE"
fi

# perf record (sampling) - creates perf_*.data files
# Note: perf record may require sudo on some systems
if [[ -x "./$BIN_BASE" ]]; then
  echo "\n=== perf record (base) ==="
  perf record -F 99 -g -o perf_base.data -- $TASKSET_CMD ./$BIN_BASE "$DATAFILE" || echo "perf record (base) failed"
fi
if [[ -x "./$BIN_MULTI" ]]; then
  echo "\n=== perf record (multi) ==="
  perf record -F 99 -g -o perf_multi.data -- $TASKSET_CMD ./$BIN_MULTI "$DATAFILE" || echo "perf record (multi) failed"
fi

# Provide commands to view reports
cat <<'EOF'

Done. Next steps:
  - Interactive report:    perf report -i perf_base.data
  - Text report:           perf report -i perf_base.data --stdio > perf_base_report.txt
  - Generate flamegraph:   sudo perf script -i perf_base.data > out_base.perf
                           ../FlameGraph/stackcollapse-perf.pl out_base.perf > out_base.folded
                           ../FlameGraph/flamegraph.pl out_base.folded > flame_base.svg

Notes:
  - perf record may require root (sudo) on some systems.
  - Ensure `perf` is installed and the JSON header (nlohmann/json.hpp) is available when compiling.
  - Adjust BIND range or THREADS to match your machine topology for fair comparison.
EOF
