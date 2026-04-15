#!/usr/bin/env bash
# Test-Case Runner — Einzelne Subreddits durchlaufen und Ergebnisse speichern.
#
# Nutzung:
#   bash scripts/run_testcases.sh
#   bash scripts/run_testcases.sh art technology         # nur bestimmte Subs
#   LIMIT=500 MAX_NUDGES=5 bash scripts/run_testcases.sh
#
set -euo pipefail

# --- Konfiguration (überschreibbar via Umgebungsvariablen) ---
DEFAULT_SUBS=("art" "technology" "personalfinance")
LIMIT="${LIMIT:-1000}"
MAX_NUDGES="${MAX_NUDGES:-8}"
BASE_DIR="${BASE_DIR:-results/testcases}"

# Falls CLI-Argumente gegeben → diese statt Default nutzen
if [[ $# -gt 0 ]]; then
    SUBS=("$@")
else
    SUBS=("${DEFAULT_SUBS[@]}")
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  OODA-Loop Test-Cases                        ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Subreddits:  ${SUBS[*]}"
echo "║  Limit:       $LIMIT Posts pro Sub"
echo "║  Max Nudges:  $MAX_NUDGES"
echo "║  Output:      $BASE_DIR/{sub}/"
echo "╚══════════════════════════════════════════════╝"
echo ""

TOTAL=${#SUBS[@]}
DONE=0
FAILED=0

for sub in "${SUBS[@]}"; do
    DONE=$((DONE + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " [$DONE/$TOTAL] Test-Case: r/$sub"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if venv/bin/python src/community_crew.py \
        --subreddits "$sub" \
        --limit "$LIMIT" \
        --max-nudges "$MAX_NUDGES" \
        --output-dir "$BASE_DIR/$sub"; then
        echo ""
        echo " ✅ r/$sub abgeschlossen → $BASE_DIR/$sub/"
    else
        FAILED=$((FAILED + 1))
        echo ""
        echo " ❌ r/$sub fehlgeschlagen (Exit-Code $?)"
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Alle Test-Cases abgeschlossen: $((TOTAL - FAILED))/$TOTAL erfolgreich"
echo " Ergebnisse unter: $BASE_DIR/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
