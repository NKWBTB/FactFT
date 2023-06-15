augment_policy="NoAugment NegAugmentOnly AllAugment"
exp_dir=exp/

for policy in $augment_policy;
do
    for k in {0..4};
    do
    mkdir -p $exp_dir/k_5_Fold/$policy/$k
    python model.py \
        --output_folder $exp_dir/$policy/$k \
        --lora \
        --load_policy k_5_Fold \
        --fold $k \
        --augment_policy $policy \
        | tee $exp_dir/k_5_Fold/$policy/$k/log.txt
    done
done
