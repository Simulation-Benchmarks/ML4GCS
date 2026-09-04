from pdb import set_trace as st
import process_map_files
from process_map_files import get_maps_and_distance, get_result_name_and_year

name, year = get_result_name_and_year(35)

print("name = ", name)
print("year = ", year)

image1, image2, distance = get_maps_and_distance(
    35, 37, "../spe11b/spe11b_tmco2_dt50y.npz"
)

print("image1.shape = ", image1.shape)
print("image1 = ", image1)
print("image2 = ", image2)
print("distance = ", distance)


boh = process_map_files.load_array_from_npz(npz_path="../spe11b/spe11b_tmco2_dt50y.npz")
st()
print("")
