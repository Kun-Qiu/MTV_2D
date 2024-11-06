import os

from PIL import Image


def split_image_into_chunks(input_directory, output_directory, chunk_size=(64, 64), image_size=(256, 256)):
    # Check if output directory exists, create it if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all valid image files in the directory
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(valid_extensions)]

    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        with Image.open(image_path) as img:
            img = img.convert("L")
            # Split the image into chunks of 64x64
            img_name = os.path.splitext(image_file)[0]
            for i in range(0, image_size[0], chunk_size[0]):
                for j in range(0, image_size[1], chunk_size[1]):
                    # Define the box for cropping
                    box = (j, i, j + chunk_size[0], i + chunk_size[1])
                    chunk = img.crop(box)

                    # Save chunk to output directory
                    chunk_filename = f"{img_name}_chunk_{i // chunk_size[0]}_{j // chunk_size[1]}.png"
                    chunk_path = os.path.join(output_directory, chunk_filename)
                    chunk.save(chunk_path)


# Usage example:
input_directory = "Template_Matching_Test/Target"
output_directory = "Template_Matching_Test/Target/patches"
split_image_into_chunks(input_directory, output_directory)
