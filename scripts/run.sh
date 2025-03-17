base_dir="FakeCloud/FakePCD"
sources=("real" "pointflow" "diffusion" "shapegf")

shapes=("airplane" "car" "chair" "airplane car chair")
for model in pointnet dgcnn
do
    for shape in "${shapes[@]}"; do
        python "$base_dir/closeworld.py" --shapes $shape \
        --sources "${sources[@]}" \
        --model $model \
        --eval False
    done
done