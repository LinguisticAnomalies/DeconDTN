


# for dt in "SHAC"
# do
#   for isample in 1152 11063
#   do
#     for methodUsed in 'noaug' 'MMD' 'GDRO' 'GradientReverse'
#     do
#       python HiddenSpaceSeparationMetric.py --dataset=$dt --isample=$isample --methodUsed=$methodUsed
#     done
#   done
# done



for dt in "CD" "HateSpeech"
do
  for isample in 566 6621
  do
    for methodUsed in 'noaug' 'MMD' 'GDRO' 'GradientReverse'
    do
      python HiddenSpaceSeparationMetric.py --dataset=$dt --isample=$isample --methodUsed=$methodUsed
    done
  done
done