data_names="cogensumm xsumfaith polytope factcc summeval frank"
load_policy="Train_1_Val_n Train_n_Val_1 Train_1_Val_1"
augment_policy="NoAugment NegAugmentOnly AllAugment"
exp_dir=exp/

for policy in $load_policy;
do
    for name in $data_names;
    do
    mkdir -p $exp_dir/$policy/$name
    python model.py \
        --output_folder $exp_dir/$policy/$name \
        --data_name $name \
        --lora \
        --load_policy $policy \
        --augment_policy NoAugment \
        | tee $exp_dir/$policy/$name/log.txt
    done
done
