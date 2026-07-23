from PIL import Image
import os

gif_path = "assets/gradient_descent_multi/descens_gradient_minims_locals.gif"
output_dir = "frames/"
os.makedirs(output_dir, exist_ok=True)

gif = Image.open(gif_path)

for frame in range(gif.n_frames):
    gif.seek(frame)
    gif.save(f"{output_dir}frame_{frame:03d}.png")

print(f"{gif.n_frames} frames extrets a {output_dir}")