import os
import cairosvg
from PIL import Image

def create_icon_dataset(svg_folder, output_folder, resolutions):
    """
    Creates an icon image dataset from SVG files at multiple resolutions.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    for svg_file in os.listdir(svg_folder):
        if svg_file.endswith(".svg"):
            file_name = os.path.splitext(svg_file)[0]
            svg_path = os.path.join(svg_folder, svg_file)

            for res in resolutions:
                output_path = os.path.join(output_folder, f"{file_name}_{res}x{res}.png")
                
                # Use cairosvg to render SVG to PNG
                cairosvg.svg2png(url=svg_path, write_to=output_path, parent_width=res, parent_height=res)
                
                # Check for 8K and ensure consistent quality, handle upscaling if necessary
            
                high_res_img = Image.open(output_path)
                if high_res_img.size != (res, res):
                    high_res_img = high_res_img.resize((res, res), Image.LANCZOS)
                    high_res_img.save(output_path)

if __name__ == "__main__":
    svg_source = "teste"
    output_dir = "test"
    resolutions_to_generate = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    
    create_icon_dataset(svg_source, output_dir, resolutions_to_generate)
