declare -a arr=("beta_actin" "myosin_iib" "membrane_caax_63x" "desmoplakin" "sec61_beta", "st6gal1", "fibrillarin", "lamin_b1", "dic_lamin_b1" "alpha_tubulin" "tom20" "zo1" "timelapse_wt2_s2")

for i in "${arr[@]}"
do
   curl -O http://downloads.allencell.org/publication-data/label-free-prediction/$i.tar.gz
   tar -C ./data -xvzf ./$i.tar.gz
done
