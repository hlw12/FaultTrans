#!/bin/bash
set -e
BAK=/root/autodl-tmp/Test_on_Event/_backup_pre_unified
# Restore event-specific H5 pipelines that the unified reader cannot parse.
cp -f $BAK/Chichi/pred_binary.npy /root/autodl-tmp/Test_on_Event/chichi/Chichi/pred_binary.npy
cp -f $BAK/Chichi/pred_prob.npy /root/autodl-tmp/Test_on_Event/chichi/Chichi/pred_prob.npy
cp -f $BAK/Chichi/pgv_fields.npy /root/autodl-tmp/Test_on_Event/chichi/Chichi/pgv_fields.npy
cp -f $BAK/Menyuan/pred_binary.npy /root/autodl-tmp/Test_on_Event/menyuan/Menyuan/pred_binary.npy
cp -f $BAK/Menyuan/pred_prob.npy /root/autodl-tmp/Test_on_Event/menyuan/Menyuan/pred_prob.npy
cp -f $BAK/Menyuan/pgv_fields.npy /root/autodl-tmp/Test_on_Event/menyuan/Menyuan/pgv_fields.npy
echo restored
ls -l /root/autodl-tmp/Test_on_Event/chichi/Chichi/pred_binary.npy /root/autodl-tmp/Test_on_Event/menyuan/Menyuan/pred_binary.npy
