#!/usr/bin/env bash
# Relance automatiquement une commande si son log ne progresse plus pendant STALL secondes (blocage GPU du portable).
# Usage : scripts/run_with_watchdog.sh <log> <stall_s> <commande...>
LOG=$1; STALL=$2; shift 2
for attempt in $(seq 1 20); do
  echo "=== tentative $attempt $(date +%H:%M) ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1 &
  PID=$!
  while kill -0 $PID 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ $age -gt $STALL ]; then
      echo "=== WATCHDOG : log figé depuis ${age}s, relance ===" >> "$LOG"
      kill -USR1 $PID 2>/dev/null; sleep 5
      kill -KILL $PID 2>/dev/null; sleep 10
      break
    fi
  done
  wait $PID 2>/dev/null && { echo "=== TERMINÉ $(date +%H:%M) ===" >> "$LOG"; exit 0; }
done
