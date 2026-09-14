# Every pending cache run for one model, in order of value. Idempotent: a
# finished run (its JSON exists) or one still running is skipped, so after any
# crash the fix is to start this again.
#
#   tmux new -s q20 -d "bash scripts/pod/queue.sh 20 2>&1 | tee out/_q20.log"
#   tmux new -s q15 -d "bash scripts/pod/queue.sh 15 2>&1 | tee out/_q15.log"

M="${1:?model: 15|20}"
. "$(dirname "$0")/lib.sh"

C="--n-eval 200 --gen-length 512 --eval-steps 256 --kv-cache"
BL="--kv-policy block"
N1="--stale-prefix --kv-policy every_n --kv-refresh-every 1"

if [ "$M" = 15 ]; then
  e() { n=$1; shift; run "$n" $NEED15 "out/$n.json" $EVAL15 "$@" --out "out/$n.json"; }
  # static.sh
  e l15_S_k4   $C $BL --kv-bits 4 --kv-static key
  e l15_S_dyn3 $C $BL --kv-bits 3
  e l15_S_k3   $C $BL --kv-bits 3 --kv-static key
  e l15_S_kv3  $C $BL --kv-bits 3 --kv-static kv
  # stats: tensor dumps are minutes, the decision dump is the long one
  for MR in 0.5 0.0; do
    t=ke_l15_m${MR/./}
    run $t $NEED15 out/$t.json $PY15 scripts/check_key_error.py $M15 --samples 32 --bits 4 3 --mask-ratio $MR --dump out/$t.json
  done
  run br_l15 $NEED15 out/br_l15.json $PY15 scripts/check_block_reuse.py $M15 --kv-rope post --samples 16 --bits 16 4 3 --policies every_n:1 --dump-margins out/br_l15.json
  # onegroup.sh
  e l15_G1_4 $C $BL --kv-group-size 4096 --kv-bits 4
  e l15_G1_3 $C $BL --kv-group-size 4096 --kv-bits 3
  # axis3.sh
  for pol in n1 block; do
    P=$([ $pol = n1 ] && echo "$N1" || echo "$BL")
    e l15_A3_${pol}_tok    $C --kv-bits 3 --kv-rope post $P --kv-key-axis channel
    e l15_A3_${pol}_tok_r4 $C --kv-bits 3 --kv-rope post $P --kv-key-axis channel --rotate-qk
    e l15_A3_${pol}_ch     $C --kv-bits 3 --kv-rope post $P --kv-key-axis token
    e l15_A3_${pol}_ch_r4  $C --kv-bits 3 --kv-rope post $P --kv-key-axis token --rotate-qk
  done
else
  e() { n=$1; shift; run "$n" $NEED20 "out/$n.json" $EVAL20 "$@" --out "out/$n.json"; }
  e l20_S_k4   $C $BL --kv-bits 4 --kv-static key
  e l20_S_dyn3 $C $BL --kv-bits 3
  e l20_S_k3   $C $BL --kv-bits 3 --kv-static key
  e l20_S_kv3  $C $BL --kv-bits 3 --kv-static kv
  for MR in 0.5 0.0; do
    t=ke_l20_m${MR/./}
    run $t $NEED20 out/$t.json $PY20 scripts/check_key_error.py $M20 --samples 32 --bits 4 3 --mask-ratio $MR --dump out/$t.json
    run ${t}_nonorm $NEED20 out/${t}_nonorm.json $PY20 scripts/check_key_error.py $M20 --samples 32 --bits 4 3 --mask-ratio $MR --skip-qk-norm --dump out/${t}_nonorm.json
  done
  run br_l20 $NEED20 out/br_l20.json $PY20 scripts/check_block_reuse.py $M20 --samples 16 --bits 16 4 3 --policies every_n:1 --dump-margins out/br_l20.json
  e l20_G1_4 $C $BL --kv-group-size 4096 --kv-bits 4
  e l20_G1_3 $C $BL --kv-group-size 4096 --kv-bits 3
  e l20_A3_tok    $C --kv-bits 3 $BL --kv-key-axis channel
  e l20_A3_tok_r4 $C --kv-bits 3 $BL --kv-key-axis channel --rotate-qk
  e l20_A3_ch_r4  $C --kv-bits 3 $BL --kv-key-axis token --rotate-qk
fi
echo "$(date +%H:%M) очередь $M прошла"
