# A65 basis: deck R8r (gold -4, thematic fade reversed F2, semis differentiated), scenario-specific persistence,
# thematic estimation error = equity, trend-2022-neutral panel, gates v4. Source after env57.sh.
ENGINE_HOME=${ENGINE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export DECK_FILE=Deck_R8r_20261030.csv ENGINEV2_LAM_FILE=$ENGINE_HOME/data/decks/Lambda_R8r_20261030.json \
  ENGINEV2_PANEL=$ENGINE_HOME/data/panel_trend2022neutral.csv UNC_THEM=0 UNC_BLOCK=0 \
  ENGINEV2_LAM=0.067 WORST_GATE=0.0427 STRESS_FLOOR=0.0344 STRESS_SHIFT_FILE=$ENGINE_HOME/data/stress/StressG2_R8r.json
