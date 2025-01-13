import os
from PIL import Image
from natsort import natsorted

def create_gif_from_folder(folder_path, output_path, duration=100, loop=0):
    """
    Create a GIF from images in a folder.

    Args:
        folder_path (str): Path to the folder containing images.
        output_path (str): Path to save the generated GIF.
        duration (int): Duration for each frame in milliseconds.
        loop (int): Number of times the GIF should loop (0 means infinite).
    """
    # Get all image files in the folder
    images = [
        os.path.join(folder_path, f) for f in natsorted(os.listdir(folder_path))
        if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))
    ]

    if not images:
        print("No images found in the folder.")
        return

    # Open images and add to a list
    frames = [Image.open(img) for img in images]

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

    print(f"GIF saved to {output_path}")

# Example usage
folder_path = "ddpm_vis"
output_path = "ddpm_vis.gif"
duration = 200  # 200 milliseconds per frame
loop = 0  # Infinite loop

create_gif_from_folder(folder_path, output_path, duration, loop)